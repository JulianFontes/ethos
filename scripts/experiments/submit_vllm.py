"""Submit a distributed profile_signal.py run described by a YAML config (e.g. configs/vllm_run.yaml), with submitit.

Splits the sample into ceil(comments / job_size) jobs, at most max_nodes (bigger jobs instead, all running at once so
checkpoints keep filling up), and submits them as one Slurm array. Each job, on its own A100:
  1. starts a vLLM server for the model (in its own environment, via uvx),
  2. runs profile_signal.py on its share of the comments (--job i --num-jobs N),
  3. stops the server.
A job that hits its time limit is requeued automatically and resumes from the answers already saved.

Run from a cluster login node (scripts/shell/submit_vllm.sh does the setup):
    scripts/shell/submit_vllm.sh --config configs/vllm_run.yaml --dry-run     # show the plan
    scripts/shell/submit_vllm.sh --config configs/vllm_run.yaml
Logs go next to the answers, in <data>/logs/: <array id>_<task>/ (job output) and vllm-<array id>_<task>.log (the server;
appended to when a job is requeued).
"""
import argparse
import math
import os
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import pandas as pd
import submitit
import yaml

ROOT = Path(__file__).resolve().parents[2]


class Job(submitit.helpers.Checkpointable):
    """One array task: serve the model with vLLM, run profile_signal.py on job `job` of `n_jobs`, stop the server.
    Checkpointable: on timeout or preemption submitit resubmits the same call, which resumes where it stopped."""

    def __init__(self, cfg):
        self.cfg = cfg

    def __call__(self, job, n_jobs):
        cfg, env = self.cfg, dict(os.environ)
        port = free_port()   # nodes are shared (one job per GPU slice), so a fixed port can already be taken
        scratch = os.path.expandvars(cfg["scratch"])
        env.update(UV_CACHE_DIR=f"{scratch}/uv-cache", HF_HOME=f"{scratch}/huggingface", PYTHONUNBUFFERED="1",
                   PYTHONPATH=f"{ROOT}/src", VLLM_BASE_URL=f"http://127.0.0.1:{port}/v1",
                   VLLM_USE_FLASHINFER_SAMPLER="0")   # its sampler compiles CUDA code, but nvcc is not on the GPU nodes
        task = (f"{os.environ['SLURM_ARRAY_JOB_ID']}_{os.environ['SLURM_ARRAY_TASK_ID']}" if "SLURM_ARRAY_JOB_ID" in os.environ
                else os.environ.get("SLURM_JOB_ID", "local"))   # same name as the job's submitit folder
        log = log_dir(cfg) / f"vllm-{task}.log"
        server = subprocess.Popen(
            ["uvx", "--managed-python", "--python", "3.12", "--from", "vllm==0.33.0", "vllm", "serve", cfg["model"], "--port", str(port),
             "--max-model-len", "4096", "--gpu-memory-utilization", "0.90",
             "--max-num-seqs", str(cfg["workers"])],   # vLLM runs at most this many at once; match the client
            env=env, stdout=open(log, "a"), stderr=subprocess.STDOUT)
        try:
            print(f"starting vLLM ({cfg['model']}); the first run downloads it, which can take several minutes", flush=True)
            for _ in range(180):                                   # up to 30 min
                if server.poll() is not None:
                    raise RuntimeError(f"vLLM exited, see {log}")
                try:
                    urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5)
                    break
                except OSError:
                    time.sleep(10)
            else:
                raise RuntimeError(f"vLLM did not start within 30 min, see {log}")
            print("vLLM ready", flush=True)
            subprocess.run([sys.executable, str(ROOT / "scripts/experiments/profile_signal.py"), *run_args(cfg),
                            "--job", str(job), "--num-jobs", str(n_jobs)], cwd=ROOT, env=env, check=True)
        finally:
            server.terminate()


def free_port():
    """A TCP port nothing on this node is listening on."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def log_dir(cfg):
    """Job and vLLM logs live with the run's answers: <data>/logs/."""
    return ROOT / cfg["data"] / "logs"


def run_args(cfg):
    """profile_signal.py options from the config."""
    comments = ["--ids", *map(str, cfg["ids"])] if cfg.get("ids") else ["--all"]
    return [*comments, "--model", f"vllm:{cfg['model']}", "--sample", cfg["sample"], "--data", cfg["data"],
            "--samples", str(cfg["samples"]), "--workers", str(cfg["workers"]),
            "--reasoning-effort", cfg["reasoning_effort"]] + (["--single-values"] if cfg["single_values"] else [])


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--dry-run", action="store_true", help="show the plan without submitting")
    args = p.parse_args()
    cfg = yaml.safe_load(open(ROOT / args.config))

    n_comments = len(cfg["ids"]) if cfg.get("ids") else len(pd.read_csv(ROOT / cfg["sample"], usecols=["id"]))
    n_jobs = math.ceil(n_comments / cfg["job_size"])
    if n_jobs > cfg["max_nodes"]:
        print(f"{n_jobs} jobs of {cfg['job_size']} would exceed max_nodes {cfg['max_nodes']}; "
              f"using {cfg['max_nodes']} jobs of ~{math.ceil(n_comments / cfg['max_nodes'])}")
        n_jobs = cfg["max_nodes"]
    hours, minutes, _ = map(int, cfg["time"].split(":"))
    print(f"{n_comments} comments -> {n_jobs} jobs, one {cfg['partition']} node each, up to {cfg['time']} per job "
          f"(requeued automatically if it runs out)")
    print("each job runs: profile_signal.py " + " ".join(run_args(cfg)) + " --job <i> --num-jobs " + str(n_jobs))
    merge = (f"uv run python scripts/experiments/profile_signal.py --merge --sample {cfg['sample']} "
             f"--data {cfg['data']} --checkpoint-every {cfg['checkpoint_every']}")
    if args.dry_run:
        print(f"\n(dry run: nothing submitted)\nCombine the jobs any time with:\n  {merge}")
        return

    log_dir(cfg).mkdir(parents=True, exist_ok=True)
    executor = submitit.AutoExecutor(folder=log_dir(cfg) / "%j")
    executor.update_parameters(name="ethos-vllm", slurm_partition=cfg["partition"], slurm_account=cfg["account"],
                               gpus_per_node=1, cpus_per_task=8, mem_gb=64, timeout_min=60 * hours + minutes,
                               slurm_array_parallelism=n_jobs)
    jobs = executor.map_array(Job(cfg), range(n_jobs), [n_jobs] * n_jobs)
    print(f"\nsubmitted array {jobs[0].job_id.split('_')[0]} ({n_jobs} jobs); logs in {log_dir(cfg).relative_to(ROOT)}/")
    print(f"Combine the jobs any time (also while running) with:\n  {merge}")


if __name__ == "__main__":
    main()
