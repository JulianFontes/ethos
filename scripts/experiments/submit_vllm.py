"""Submit a distributed profile_signal.py run described by a YAML config (e.g. configs/qwen3_8b/single_values.yaml), with submitit.

Splits the sample into ceil(comments / job_size) jobs, at most max_nodes (bigger jobs instead, all running at once so
checkpoints keep filling up), and submits them as one Slurm array. Each job, on its own A100:
  1. starts a vLLM server for the model (in its own environment, via uvx),
  2. runs profile_signal.py on its share of the comments (--job i --num-jobs N),
  3. stops the server.
With a `steering:` section (vectors from scripts/shell/fit_steering_vectors.sh, values, alphas), the server gets
plugins/vllm_steer and step 2 repeats for each (value, alpha): the job switches the server's steering vector in place
(ethos.steering.in_vllm.select_steering, no restart) and every condition is tagged "|steer=<value>@<alpha>".
Finished settings are marked in <data>/done/ and skipped when a job resumes.
The run folder gets steering_layers.png and steering.json (which layers, vectors and strengths).
A job that hits its time limit is requeued automatically and resumes from the answers already saved.

Run from a cluster login node (scripts/shell/submit_vllm.sh does the setup):
    scripts/shell/submit_vllm.sh --config configs/qwen3_8b/single_values.yaml --dry-run     # show the plan
    scripts/shell/submit_vllm.sh --config configs/qwen3_8b/single_values.yaml
Logs go next to the answers, in <data>/logs/: <array id>_<task>/ (job output) and vllm-<array id>_<task>.log (the server;
appended to when a job is requeued).
"""
import argparse
import math
import os
import subprocess
import sys
import time

import pandas as pd
import submitit
import yaml

from ethos.paths import ROOT
from ethos.steering.in_vllm import save_run_figure, select_steering, serve, steering_server, steering_settings


class Job(submitit.helpers.Checkpointable):
    """One array task: serve the model with vLLM, run profile_signal.py on job `job` of `n_jobs`, stop the server.
    Checkpointable: on timeout or preemption submitit resubmits the same call, which resumes where it stopped."""

    def __init__(self, cfg):
        self.cfg = cfg

    def __call__(self, job, n_jobs):
        cfg = self.cfg
        task = (f"{os.environ['SLURM_ARRAY_JOB_ID']}_{os.environ['SLURM_ARRAY_TASK_ID']}" if "SLURM_ARRAY_JOB_ID" in os.environ
                else os.environ.get("SLURM_JOB_ID", "local"))   # same name as the job's submitit folder
        log = log_dir(cfg) / f"vllm-{task}.log"
        steering = cfg.get("steering")
        done_dir = ROOT / cfg["data"] / "done"                      # finished (job, setting) markers
        settings = steering_settings(steering)
        todo = [s for s in settings if not (done_dir / f"job_{job}of{n_jobs}_{s[0] or 'unsteered'}").exists()]
        print(f"{len(settings) - len(todo)} of {len(settings)} settings already done", flush=True)
        if not todo:
            return
        env, port, server = self.start_server(log)                  # one server; steering is switched in place
        try:
            start = time.time()
            for k, (tag, vector, alpha) in enumerate(todo):
                if steering:
                    layers = select_steering(f"http://127.0.0.1:{port}", vector, alpha)
                    elapsed = time.time() - start                   # overall progress: settings done, time left
                    left = f", ~{elapsed / k * (len(todo) - k) / 60:.0f} min left" if k else ""
                    print(f"=== setting {len(settings) - len(todo) + k + 1}/{len(settings)} "
                          f"({elapsed / 60:.0f} min{left}): {tag or 'unsteered'}, alpha {alpha or 0}, "
                          f"layers {layers} ===", flush=True)
                subprocess.run([sys.executable, str(ROOT / "scripts/experiments/profile_signal.py"), *run_args(cfg),
                                "--job", str(job), "--num-jobs", str(n_jobs)] + (["--steer", tag] if tag else []),
                               cwd=ROOT, env=env, check=True)
                if steering:
                    done_dir.mkdir(parents=True, exist_ok=True)
                    (done_dir / f"job_{job}of{n_jobs}_{tag or 'unsteered'}").touch()
        finally:
            server.terminate()
            server.wait()

    def start_server(self, log):
        """Start this run's vLLM server (with the steering plugin and zeroed steered layers if the run steers)."""
        cfg = self.cfg
        env, uvx_args, serve_args = steering_server(cfg.get("steering"))   # nothing if the run doesn't steer
        return serve(cfg["model"], log, cfg["scratch"], cfg["workers"], env, uvx_args, serve_args)   # workers =
                                                                           # vLLM --max-num-seqs, matching the client


def log_dir(cfg):
    """Job and vLLM logs live with the run's answers: <data>/logs/."""
    return ROOT / cfg["data"] / "logs"


def run_args(cfg):
    """profile_signal.py options from the config."""
    comments = ["--ids", *map(str, cfg["ids"])] if cfg.get("ids") else ["--all"]
    return [*comments, "--model", f"vllm:{cfg['model']}", "--sample", cfg["sample"], "--data", cfg["data"],
            "--samples", str(cfg["samples"]), "--workers", str(cfg["workers"]),
            "--reasoning-effort", cfg["reasoning_effort"]] + (["--single-values"] if cfg["single_values"] else []) \
        + (["--profiles", cfg["profiles"]] if cfg.get("profiles") else []) \
        + (["--weightings", *cfg["weightings"]] if cfg.get("weightings") else []) \
        + (["--any-values"] if cfg.get("any_values") else []) \
        + (["--conditions", *cfg["conditions"]] if cfg.get("conditions") else []) \
        + [arg for key in ("max_tokens", "attempts", "passes") if key in cfg
           for arg in (f"--{key.replace('_', '-')}", str(cfg[key]))]


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
    if cfg.get("steering"):
        settings = steering_settings(cfg["steering"])
        print(f"steering: {len(settings)} settings per job, all on one vLLM server (steering switched in place): "
              + ", ".join(tag or "unsteered" for tag, _, _ in settings))
        print(f"layer figure: {save_run_figure(cfg['model'], cfg['steering'], cfg['data']).relative_to(ROOT)}")
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
