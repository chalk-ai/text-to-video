#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.12,<3.14"
# dependencies = ["chalkcompute==2.13.1"]
# ///
"""Deploy LTX-Video text-to-video (server.py + index.html) as a GPU ScalingGroup.

The 13B distilled model needs about 40 GB of GPU memory, so the default is one L40S
(48 GB). The replica downloads about 45 GB of weights when it boots, which keeps the
image small enough to build and push quickly.

Credentials come from `chalk config` in the current directory (or the CHALK_* env vars):
    ./deploy.py [--env-id <environment>] [--gpu nvidia-l40s]
Then open the printed URL, describe a shot, and press "Generate clip".
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def load_chalk_env(env_id: str | None) -> None:
    if "CHALK_CLIENT_ID" not in os.environ:
        cfg = json.loads(subprocess.check_output(["chalk", "config", "--json"]))
        os.environ["CHALK_API_SERVER"] = cfg["apiServer"]["value"]
        os.environ["CHALK_CLIENT_ID"] = cfg["clientId"]["value"]
        os.environ["CHALK_CLIENT_SECRET"] = cfg["clientSecret"]["value"]
        os.environ.setdefault("CHALK_ENVIRONMENT_ID", cfg["environment"]["value"])
    if env_id:
        os.environ["CHALK_ENVIRONMENT_ID"] = env_id


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="ltx-video")
    # Defaults to the environment in `chalk config` (or CHALK_ENVIRONMENT_ID).
    parser.add_argument("--env-id", default=None)
    parser.add_argument("--gpu", default="nvidia-l40s")
    parser.add_argument("--ready-timeout", type=float, default=2400.0)
    args = parser.parse_args()

    load_chalk_env(args.env_id)
    from chalkcompute import Image, ReadinessProbe, ScalingGroup

    image = (
        Image.debian_slim(python_version="3.12")
        .pip_install(
            [
                "torch==2.8.0",
                "diffusers==0.35.2",
                "transformers==4.57.6",
                "accelerate",
                "sentencepiece",
                "huggingface-hub==0.36.2",
                "hf_xet",
                "imageio[ffmpeg]",
                "fastapi==0.116.1",
                "uvicorn[standard]==0.35.0",
            ]
        )
        .env({"HF_XET_HIGH_PERFORMANCE": "1"})
        .add_local_file(str(HERE / "server.py"), "/app/server.py", strategy="copy")
        .add_local_file(str(HERE / "index.html"), "/app/index.html", strategy="copy")
        .workdir("/app")
    )

    sg = ScalingGroup(
        image=image,
        name=args.name,
        gpu=args.gpu,
        # Sized to fit a g6e.2xlarge (8 vCPU, 64 GiB) next to the pod's sidecars; any larger
        # request makes Karpenter launch a much bigger and pricier instance for the same GPU.
        cpu="4",
        memory="40Gi",
        env={"MODEL_DIR": "/models", "OUTPUT_DIR": "/outputs", "PORT": "8080"},
        entrypoint=["python", "/app/server.py"],
        port=8080,
        # Booting downloads the weights and loads them onto the GPU before uvicorn listens;
        # give the probe room for that instead of restarting the replica mid-download.
        readiness_probe=ReadinessProbe.http("/status", period_seconds=10, failure_threshold=360),
        # One replica owns the queue; more replicas would each need their own queue and
        # sticky routing for polling.
        min_replicas=1,
        max_replicas=1,
    )
    t0 = time.monotonic()
    try:
        sg.deploy(build_timeout=2400.0, ready_timeout=args.ready_timeout)
    finally:
        print(f"deploy returned after {time.monotonic() - t0:.0f}s")
    print("web_url:", sg.web_url)


if __name__ == "__main__":
    main()
