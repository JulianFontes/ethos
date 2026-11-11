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
Logs: logs/submitit/<job id>/ (job output) and logs/submitit/vllm-<job id>.log (the server).
"""
import argparse
import math
import os
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
        scratch = os.path.expandvars(cfg["scratch"])
        env.update(UV_CACHE_DIR=f"{scratch}/uv-cache", HF_HOME=f"{scratch}/huggingface", PYTHONUNBUFFERED="1",
                   PYTHONPATH=f"{ROOT}/src", VLLM_BASE_URL="http://127.0.0.1:8000/v1")
        log = ROOT / "logs" / "submitit" / f"vllm-{os.environ.get('SLURM_JOB_ID', 'local')}.log"
        server = subprocess.Popen(
            ["uvx", "--python", "3.12", "--from", "vllm", "vllm", "serve", cfg["model"], "--port", "8000",
             "--max-model-len", "4096", "--gpu-memory-utilization", "0.90"],
            env=env, stdout=open(log, "w"), stderr=subprocess.STDOUT)
        try:
            print(f"starting vLLM ({cfg['model']}); the first run downloads it, which can take several minutes", flush=True)
            for _ in range(180):                                   # up to 30 min
                if server.poll() is not None:
                    raise RuntimeError(f"vLLM exited, see {log}")
                try:
                    urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=5)
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

    (ROOT / "logs" / "submitit").mkdir(parents=True, exist_ok=True)
    executor = submitit.AutoExecutor(folder=ROOT / "logs" / "submitit" / "%j")
    executor.update_parameters(name="ethos-vllm", slurm_partition=cfg["partition"], slurm_account=cfg["account"],
                               gpus_per_node=1, cpus_per_task=8, mem_gb=64, timeout_min=60 * hours + minutes,
                               slurm_array_parallelism=n_jobs)
    jobs = executor.map_array(Job(cfg), range(n_jobs), [n_jobs] * n_jobs)
    print(f"\nsubmitted array {jobs[0].job_id.split('_')[0]} ({n_jobs} jobs); logs in logs/submitit/")
    print(f"Combine the jobs any time (also while running) with:\n  {merge}")


if __name__ == "__main__":
    main()
