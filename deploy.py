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

Credentials come from your existing `chalk login` (or the CHALK_* env vars):
    ./deploy.py
Then open the printed URL, describe a shot, and press "Generate clip".
"""

from __future__ import annotations

import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main() -> None:
    from chalkcompute import Image, ReadinessProbe, ScalingGroup

    image = (
        Image.debian_slim(python_version="3.12")
        # torch and its CUDA wheels are most of the image; a layer of their own is cached across
        # changes to everything else.
        .pip_install(["torch==2.8.0"])
        .pip_install(
            [
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
        name='ltx-video',
        gpu='nvidia-l40s',
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
        sg.deploy(build_timeout=2400.0, ready_timeout=2400.0)
    finally:
        print(f"deploy returned after {time.monotonic() - t0:.0f}s")
    print("web_url:", sg.web_url)


if __name__ == "__main__":
    main()
