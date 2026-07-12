#!/usr/bin/env python3
"""
Launch a brain_factory training job on a RunPod GPU pod.

The launcher:
  1. Snapshots local code at the current git HEAD commit hash
  2. Creates a RunPod pod whose startup command waits for the code to appear
  3. rsyncs the local repo into /root/<commit_hash>/ on the pod via SSH
  4. Training starts automatically once rsync completes
  5. Terminates the pod when training finishes (or --max-duration is reached)

Usage:
  # Run until training finishes (no time cap)
  python brain_factory/scripts/launch_runpod_job.py \
      --project dummy/training/dummy --run node_1_gpu_1

  # Cap at 4 hours, then terminate regardless
  python brain_factory/scripts/launch_runpod_job.py \
      --project groot/training/groot --run node_1_gpu_1 --max-duration 4h

  # 8 GPUs, extra Hydra overrides
  python brain_factory/scripts/launch_runpod_job.py \
      --project groot/training/groot --run node_1_gpu_1 \
      --gpu-count 8 --max-duration 12h \
      --overrides experiment_name=groot_run_001 monitor.save_every=500

Environment variables:
  RUNPOD_API_KEY   RunPod API key (alternative to --api-key)
"""

import argparse
import logging
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from typing import Optional

import requests

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RUNPOD_GRAPHQL_URL = "https://api.runpod.io/graphql"
DEFAULT_POLL_INTERVAL = 30  # seconds
SSH_READY_TIMEOUT = 300     # seconds to wait for SSH to become available

# The network volume attached to this project (console.runpod.io/user/storage/vatikxob8a).
# RunPod mounts network volumes at /runpod-volume inside the container.
DEFAULT_NETWORK_VOLUME_ID = "vatikxob8a"
NETWORK_VOLUME_MOUNT = "/runpod-volume"

VENV_ACTIVATE = "/workspace/ego_av_gen_venv/.venv/bin/activate"

# RunPod desiredStatus values that mean the pod is no longer doing work.
TERMINAL_STATUSES = {"EXITED", "TERMINATED", "STOPPED", "DEAD"}
FAILED_STATUSES = {"FAILED"}

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Duration parser
# ---------------------------------------------------------------------------


def parse_duration(value: str) -> int:
    """
    Parse a human-readable duration string into seconds.

    Accepted formats:
      30s  → 30 seconds
      90m  → 5400 seconds
      4h   → 14400 seconds
      1.5h → 5400 seconds
    Bare numbers are treated as minutes.
    """
    value = value.strip().lower()
    try:
        if value.endswith("h"):
            return int(float(value[:-1]) * 3600)
        if value.endswith("m"):
            return int(float(value[:-1]) * 60)
        if value.endswith("s"):
            return int(float(value[:-1]))
        return int(float(value) * 60)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"Invalid duration '{value}'. Use formats like 4h, 90m, 3600s."
        )


# ---------------------------------------------------------------------------
# Git helpers
# ---------------------------------------------------------------------------


def get_local_git_info() -> tuple[str, str]:
    """Return (commit_hash, repo_root) for the current working tree."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        root = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except subprocess.CalledProcessError as exc:
        log.error(f"git command failed: {exc.stderr.strip()}")
        sys.exit(1)
    return commit, root


# ---------------------------------------------------------------------------
# RunPod GraphQL client
# ---------------------------------------------------------------------------


class RunPodClient:
    """Thin wrapper around the RunPod GraphQL API."""

    def __init__(self, api_key: str) -> None:
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            }
        )

    def _gql(self, query: str, variables: Optional[dict] = None) -> dict:
        payload: dict = {"query": query}
        if variables:
            payload["variables"] = variables
        resp = self.session.post(RUNPOD_GRAPHQL_URL, json=payload, timeout=30)
        resp.raise_for_status()
        body = resp.json()
        if "errors" in body:
            raise RuntimeError(f"RunPod API error: {body['errors']}")
        return body["data"]

    # ------------------------------------------------------------------
    # Pod lifecycle
    # ------------------------------------------------------------------

    def create_pod(
        self,
        *,
        name: str,
        image_name: str,
        gpu_type_id: str,
        gpu_count: int = 1,
        container_disk_gb: int = 50,
        network_volume_id: Optional[str] = None,
        env: Optional[dict] = None,
        startup_cmd: Optional[str] = None,
        ports: Optional[str] = None,
    ) -> dict:
        """
        Spin up a RunPod on-demand pod and return its info dict.

        startup_cmd is passed as dockerArgs (the container CMD override).
        When this command exits, the pod transitions to EXITED/STOPPED.
        """
        mutation = """
        mutation LaunchPod($input: PodFindAndDeployOnDemandInput!) {
            podFindAndDeployOnDemand(input: $input) {
                id
                name
                desiredStatus
                imageName
                gpuCount
                costPerHr
            }
        }
        """
        env_vars = [{"key": k, "value": v} for k, v in (env or {}).items()]

        pod_input: dict = {
            "name": name,
            "imageName": image_name,
            "gpuTypeId": gpu_type_id,
            "gpuCount": gpu_count,
            "containerDiskInGb": container_disk_gb,
            "startSsh": True,
            "env": env_vars,
        }
        if network_volume_id:
            pod_input["networkVolumeId"] = network_volume_id
        if startup_cmd:
            pod_input["dockerArgs"] = startup_cmd
        if ports:
            pod_input["ports"] = ports

        data = self._gql(mutation, {"input": pod_input})
        return data["podFindAndDeployOnDemand"]

    def get_pod(self, pod_id: str) -> dict:
        """Return current pod metadata including desiredStatus, runtime, and SSH ports."""
        query = """
        query GetPod($input: PodFilter!) {
            pod(input: $input) {
                id
                name
                desiredStatus
                lastStatusChange
                runtime {
                    uptimeInSeconds
                    ports {
                        ip
                        isIpPublic
                        privatePort
                        publicPort
                        type
                    }
                    gpus {
                        id
                        gpuUtilPercent
                        memoryUtilPercent
                    }
                }
            }
        }
        """
        data = self._gql(query, {"input": {"podId": pod_id}})
        return data["pod"]

    def stop_pod(self, pod_id: str) -> None:
        """Stop (pause) a pod without terminating it — keeps disk alive."""
        mutation = """
        mutation StopPod($input: PodStopInput!) {
            podStop(input: $input) {
                id
                desiredStatus
            }
        }
        """
        self._gql(mutation, {"input": {"podId": pod_id}})

    def terminate_pod(self, pod_id: str) -> None:
        """Permanently terminate a pod and release all resources."""
        mutation = """
        mutation TerminatePod($input: PodTerminateInput!) {
            podTerminate(input: $input)
        }
        """
        self._gql(mutation, {"input": {"podId": pod_id}})


# ---------------------------------------------------------------------------
# Training command builder
# ---------------------------------------------------------------------------


def build_startup_cmd(project: str, run: str, overrides: list[str], commit_hash: str) -> str:
    """
    Build the container startup command.

    The command polls until /root/<commit_hash>/ appears (rsync'd by the
    launcher), then activates the venv and runs training.  The container
    exits when training finishes, which triggers auto-termination.
    """
    code_dir = f"/root/{commit_hash}"
    override_str = " ".join(overrides)
    train_cmd = f"python -m brain_factory.main projects={project} run={run}"
    if override_str:
        train_cmd += f" {override_str}"

    steps = [
        f"while [ ! -d {code_dir} ]; do sleep 2; done",
        f"source {VENV_ACTIVATE}",
        f"cd {code_dir}",
        train_cmd,
    ]
    inner = " && ".join(steps)
    return f"bash -c '{inner}'"


# ---------------------------------------------------------------------------
# SSH / rsync helpers
# ---------------------------------------------------------------------------


def wait_for_ssh(
    client: RunPodClient,
    pod_id: str,
    poll_interval: int,
    timeout: int = SSH_READY_TIMEOUT,
) -> tuple[str, int]:
    """
    Poll until the pod exposes a public SSH port.

    Returns (host, port) once available, or raises TimeoutError.
    """
    log.info(f"Waiting for pod SSH to become available (timeout {timeout}s)...")
    deadline = datetime.now(timezone.utc) + timedelta(seconds=timeout)

    while datetime.now(timezone.utc) < deadline:
        try:
            pod = client.get_pod(pod_id)
            runtime = pod.get("runtime") or {}
            for port in runtime.get("ports") or []:
                if port.get("privatePort") == 22 and port.get("isIpPublic"):
                    host = port["ip"]
                    public_port = port["publicPort"]
                    log.info(f"SSH ready at {host}:{public_port}")
                    return host, public_port
        except Exception as exc:
            log.warning(f"Pod poll error while waiting for SSH: {exc}")

        time.sleep(poll_interval)

    raise TimeoutError(f"SSH not available after {timeout}s")


def rsync_code(repo_root: str, commit_hash: str, ssh_host: str, ssh_port: int) -> None:
    """rsync the local repo into /root/<commit_hash>/ on the pod."""
    dest = f"root@{ssh_host}:/root/{commit_hash}/"
    ssh_opts = f"ssh -p {ssh_port} -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null"
    cmd = [
        "rsync", "-avz", "--progress",
        "--exclude=.git",
        "--exclude=__pycache__",
        "--exclude=*.pyc",
        "--exclude=.venv",
        "-e", ssh_opts,
        f"{repo_root}/",
        dest,
    ]
    log.info(f"rsyncing {repo_root}/ → {dest}")
    subprocess.run(cmd, check=True)
    log.info("rsync complete — training will start on pod.")


# ---------------------------------------------------------------------------
# Job monitor
# ---------------------------------------------------------------------------


def monitor_until_done(
    client: RunPodClient,
    pod_id: str,
    deadline: Optional[datetime],
    poll_interval: int,
) -> str:
    """
    Poll the pod until it reaches a terminal state or the deadline passes.

    Returns one of: 'completed' | 'timeout' | 'failed' | 'interrupted'
    """
    log.info(f"Monitoring pod {pod_id} (poll every {poll_interval}s)")

    if deadline:
        secs_left = int((deadline - datetime.now(timezone.utc)).total_seconds())
        log.info(
            f"Time limit active — deadline {deadline.strftime('%Y-%m-%d %H:%M:%S UTC')} "
            f"({secs_left}s from now)"
        )
    else:
        log.info("No time limit — monitoring until pod exits.")

    consecutive_errors = 0

    while True:
        if deadline and datetime.now(timezone.utc) >= deadline:
            log.warning("Time limit reached.")
            return "timeout"

        try:
            pod = client.get_pod(pod_id)
            consecutive_errors = 0
        except Exception as exc:
            consecutive_errors += 1
            log.warning(
                f"Status poll failed ({consecutive_errors}x): {exc}. "
                f"Retrying in {poll_interval}s..."
            )
            if consecutive_errors >= 5:
                log.error("Too many consecutive poll failures — giving up.")
                return "failed"
            time.sleep(poll_interval)
            continue

        status = (pod.get("desiredStatus") or "UNKNOWN").upper()
        runtime = pod.get("runtime") or {}
        uptime = runtime.get("uptimeInSeconds", 0)

        log.info(f"Pod status={status}  uptime={uptime}s")

        if status in FAILED_STATUSES:
            log.error(f"Pod entered failed state: {status}")
            return "failed"

        if status in TERMINAL_STATUSES:
            log.info(f"Pod reached terminal state ({status}) — training complete.")
            return "completed"

        if deadline:
            secs_to_deadline = (deadline - datetime.now(timezone.utc)).total_seconds()
            sleep_for = min(poll_interval, max(1, int(secs_to_deadline)))
        else:
            sleep_for = poll_interval

        time.sleep(sleep_for)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:  # noqa: C901
    parser = argparse.ArgumentParser(
        description="Launch a brain_factory training job on RunPod.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # --- Required ---
    parser.add_argument(
        "--project",
        required=True,
        help="Hydra project path, e.g. dummy/training/dummy or groot/training/groot",
    )
    parser.add_argument(
        "--run",
        required=True,
        help="Hydra run config, e.g. node_1_gpu_1",
    )

    # --- Time limit ---
    parser.add_argument(
        "--max-duration",
        metavar="DURATION",
        type=parse_duration,
        default=None,
        help=(
            "Maximum wall-clock time before forced termination. "
            "Formats: 4h | 90m | 3600s.  Omit to run until training finishes."
        ),
    )

    # --- Compute ---
    parser.add_argument(
        "--gpu-type",
        default="NVIDIA A100 80GB HBM3",
        help="RunPod GPU type ID (default: NVIDIA A100 80GB HBM3)",
    )
    parser.add_argument(
        "--gpu-count",
        type=int,
        default=1,
        help="Number of GPUs per pod (default: 1)",
    )
    parser.add_argument(
        "--image",
        default="pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel",
        help=(
            "Docker image for the pod. "
            "Default: pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel"
        ),
    )
    parser.add_argument(
        "--disk-gb",
        type=int,
        default=50,
        help="Container disk size in GB (default: 50)",
    )
    parser.add_argument(
        "--network-volume-id",
        default=DEFAULT_NETWORK_VOLUME_ID,
        help=(
            f"RunPod network volume ID to attach (default: {DEFAULT_NETWORK_VOLUME_ID}). "
            f"Mounted at {NETWORK_VOLUME_MOUNT} inside the container. "
            "Pass an empty string to disable."
        ),
    )

    # --- Misc ---
    parser.add_argument(
        "--api-key",
        default=None,
        help="RunPod API key. Falls back to RUNPOD_API_KEY env var.",
    )
    parser.add_argument(
        "--pod-name",
        default=None,
        help=(
            "Human-readable pod name. "
            "Defaults to brain-factory-<project-slug>-<timestamp>."
        ),
    )
    parser.add_argument(
        "--poll-interval",
        type=int,
        default=DEFAULT_POLL_INTERVAL,
        help=f"Seconds between status polls (default: {DEFAULT_POLL_INTERVAL})",
    )
    parser.add_argument(
        "--overrides",
        nargs="*",
        default=[],
        metavar="KEY=VALUE",
        help="Extra Hydra overrides forwarded to brain_factory.main",
    )
    parser.add_argument(
        "--no-terminate",
        action="store_true",
        help=(
            "Skip pod termination after training — leave the pod running "
            "so you can SSH in and inspect outputs."
        ),
    )

    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Resolve API key
    # ------------------------------------------------------------------
    api_key = args.api_key or os.environ.get("RUNPOD_API_KEY")
    if not api_key:
        log.error(
            "RunPod API key is required. "
            "Set the RUNPOD_API_KEY environment variable or pass --api-key."
        )
        sys.exit(1)

    # ------------------------------------------------------------------
    # Snapshot local git state
    # ------------------------------------------------------------------
    commit_hash, repo_root = get_local_git_info()
    code_dir = f"/root/{commit_hash}"

    # ------------------------------------------------------------------
    # Derived values
    # ------------------------------------------------------------------
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    project_slug = args.project.replace("/", "-")
    pod_name = args.pod_name or f"brain-factory-{project_slug}-{timestamp}"

    deadline: Optional[datetime] = None
    if args.max_duration is not None:
        deadline = datetime.now(timezone.utc) + timedelta(seconds=args.max_duration)

    network_volume_id = args.network_volume_id or None

    startup_cmd = build_startup_cmd(args.project, args.run, args.overrides, commit_hash)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    log.info("=" * 60)
    log.info("brain_factory  →  RunPod launcher")
    log.info("=" * 60)
    log.info(f"  Project       : {args.project}")
    log.info(f"  Run config    : {args.run}")
    log.info(f"  GPU           : {args.gpu_type} x{args.gpu_count}")
    log.info(f"  Image         : {args.image}")
    log.info(f"  Pod name      : {pod_name}")
    log.info(f"  Commit hash   : {commit_hash}")
    log.info(f"  Code dir      : {code_dir}  (rsync'd after pod starts)")
    log.info(f"  Repo root     : {repo_root}")
    if network_volume_id:
        log.info(f"  Network vol   : {network_volume_id} → {NETWORK_VOLUME_MOUNT}")
    else:
        log.info("  Network vol   : none")
    if deadline:
        log.info(f"  Time limit    : {args.max_duration}s → {deadline.strftime('%H:%M:%S UTC')}")
    else:
        log.info("  Time limit    : none (run until completion)")
    log.info("=" * 60)

    client = RunPodClient(api_key)

    # ------------------------------------------------------------------
    # Create pod
    # ------------------------------------------------------------------
    log.info(f"Creating pod '{pod_name}'...")
    try:
        pod = client.create_pod(
            name=pod_name,
            image_name=args.image,
            gpu_type_id=args.gpu_type,
            gpu_count=args.gpu_count,
            container_disk_gb=args.disk_gb,
            network_volume_id=network_volume_id,
            startup_cmd=startup_cmd,
        )
    except Exception as exc:
        log.error(f"Failed to create pod: {exc}")
        sys.exit(1)

    pod_id: str = pod["id"]
    cost = pod.get("costPerHr", "?")
    log.info(f"Pod created  id={pod_id}  status={pod['desiredStatus']}  cost=${cost}/hr")
    log.info(f"Dashboard : https://www.runpod.io/console/pods/{pod_id}")

    # ------------------------------------------------------------------
    # Wait for SSH → rsync code
    # ------------------------------------------------------------------
    termination_reason = "unknown"
    try:
        ssh_host, ssh_port = wait_for_ssh(client, pod_id, args.poll_interval)
        rsync_code(repo_root, commit_hash, ssh_host, ssh_port)
    except TimeoutError as exc:
        log.error(f"SSH wait timed out: {exc}")
        termination_reason = "failed"
    except subprocess.CalledProcessError as exc:
        log.error(f"rsync failed: {exc}")
        termination_reason = "failed"
    except KeyboardInterrupt:
        log.warning("Interrupted by user (Ctrl-C) during setup.")
        termination_reason = "interrupted"

    # ------------------------------------------------------------------
    # Monitor
    # ------------------------------------------------------------------
    if termination_reason == "unknown":
        try:
            termination_reason = monitor_until_done(
                client,
                pod_id,
                deadline,
                args.poll_interval,
            )
        except KeyboardInterrupt:
            log.warning("Interrupted by user (Ctrl-C).")
            termination_reason = "interrupted"

    log.info(f"Job ended — reason: {termination_reason}")

    # ------------------------------------------------------------------
    # Terminate pod
    # ------------------------------------------------------------------
    if args.no_terminate:
        log.info(f"--no-terminate set. Pod {pod_id} left running.")
        log.info(f"SSH / inspect: https://www.runpod.io/console/pods/{pod_id}")
    else:
        log.info(f"Terminating pod {pod_id}...")
        try:
            client.terminate_pod(pod_id)
            log.info("Pod terminated successfully.")
        except Exception as exc:
            log.error(f"Termination failed: {exc}")
            log.error(
                f"Please terminate pod {pod_id} manually at "
                f"https://www.runpod.io/console/pods"
            )

    # ------------------------------------------------------------------
    # Exit code
    # ------------------------------------------------------------------
    exit_codes = {
        "completed": 0,
        "timeout": 0,
        "failed": 1,
        "interrupted": 130,
        "unknown": 1,
    }
    sys.exit(exit_codes.get(termination_reason, 1))


if __name__ == "__main__":
    main()
