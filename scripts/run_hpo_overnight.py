"""Run mBART, M2M100 and NLLB sequentially on one GPU, up to 4h each.

Usage (with the training environment active):
    python scripts/run_hpo_overnight.py

Uses this Python interpreter for every child process. No shell or GNU timeout
is needed. Runs inside an existing GPU allocation; does not reserve a GPU.
"""

import argparse
from datetime import datetime
import math
import os
from pathlib import Path
import signal
import subprocess
import sys


REPO_DIR = Path(__file__).resolve().parents[1]
HPO_SCRIPT = REPO_DIR / "src" / "hpo_training.py"
MODELS = ("mbart", "m2m100", "nllb")


def stop_process(process, grace_seconds):
    """Stop the active training process before starting another model."""
    def send(force=False):
        try:
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGKILL if force else signal.SIGTERM)
            elif force:
                process.kill()
            else:
                process.terminate()
        except ProcessLookupError:
            pass

    send()
    try:
        process.wait(timeout=grace_seconds)
    except subprocess.TimeoutExpired:
        send(force=True)
        process.wait()


def run_command(command, log_path, timeout_seconds=None):
    """Return 124 on timeout; reserve shutdown grace inside the time budget."""
    grace = min(60.0, timeout_seconds / 10) if timeout_seconds is not None else 60.0
    with Path(log_path).open("ab") as log:
        process = subprocess.Popen(command, cwd=REPO_DIR, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=(os.name == "posix"))
        try:
            wait_seconds = timeout_seconds - grace if timeout_seconds is not None else None
            return process.wait(timeout=wait_seconds)
        except subprocess.TimeoutExpired:
            stop_process(process, grace)
            return 124
        except KeyboardInterrupt:
            stop_process(process, grace)
            raise


def run_overnight(log_dir, hours_per_model=4.0):
    if not math.isfinite(hours_per_model) or hours_per_model <= 0:
        raise ValueError("hours_per_model must be a finite positive number")
    log_dir = Path(log_dir).expanduser().resolve()
    log_dir.mkdir(parents=True, exist_ok=True)
    commands = {
        model: [sys.executable, "-u", str(HPO_SCRIPT), "--config",
                str(REPO_DIR / "src" / "configs" / "hpo" / f"hpo_{model}_st.yaml")]
        for model in MODELS
    }

    # Validate all configurations and the GPU before starting the first study.
    for model, command in commands.items():
        log_path = log_dir / f"{model}.check.log"
        print(f"Checking {model}...", flush=True)
        if run_command(command + ["--check-only"], log_path) != 0:
            print(f"Validation failed for {model}. See {log_path}", flush=True)
            return 1

    status = 0
    for model, command in commands.items():
        log_path = log_dir / f"{model}.log"
        print(f"Starting {model}; limit={hours_per_model:g}h; log={log_path}", flush=True)
        result = run_command(command, log_path, timeout_seconds=hours_per_model * 3600)
        message = f"{datetime.now().astimezone().isoformat()} {model} exit={result}"
        print(message, flush=True)
        with (log_dir / "status.log").open("a", encoding="utf-8") as log:
            log.write(message + "\n")
        if result == 124:
            print(f"{model} reached its time limit. Completed trials remain in best_result.json.", flush=True)
        elif result != 0:
            print(f"{model} failed. See {log_path}. Continuing to the next model.", flush=True)
        if result != 0:
            status = 1
    return status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-dir", type=Path,
                        default=Path(os.getenv("HPO_LOG_DIR", str(REPO_DIR / "hpo_logs" / "night_v1"))))
    parser.add_argument("--hours-per-model", type=float, default=4.0,
                        help="Process time limit per model, including shutdown grace (default: 4)")
    args = parser.parse_args()
    if not math.isfinite(args.hours_per_model) or args.hours_per_model <= 0:
        parser.error("--hours-per-model must be a finite positive number")
    try:
        return run_overnight(args.log_dir, args.hours_per_model)
    except KeyboardInterrupt:
        print("Interrupted; no further model will be started.", flush=True)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
