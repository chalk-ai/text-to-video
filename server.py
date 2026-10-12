"""Text-to-video with LTX-Video 13B distilled, served as a small job queue from one GPU.

Runs inside the container that deploy.py builds for a Chalk ScalingGroup; the page at `/`
is index.html.

HTTP API:
  POST /api/generate  {"prompt": str, "seed": int?}  -> {"job_id", "position"}
  GET  /api/jobs/{id}                                 -> {"status", "position", "stage",
                                                          "progress", "elapsed_s", "video_url"}
  GET  /videos/{id}.mp4                                  the finished clip
  GET  /api/gallery                                      pre-rendered example clips

One GPU renders one clip at a time; later requests wait in a FIFO queue. Each client
address may hold one queued or running job, and finished clips are only reachable through
their unguessable job id.
"""

from __future__ import annotations

import asyncio
import collections
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncIterator

import imageio.v2 as imageio
import numpy as np
import torch
from diffusers import LTXConditionPipeline, LTXLatentUpsamplePipeline
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from huggingface_hub import snapshot_download
from pydantic import BaseModel, Field

MODEL_REPO = os.environ.get("LTX_MODEL", "Lightricks/LTX-Video-0.9.8-13B-distilled")
UPSAMPLER_REPO = os.environ.get("LTX_UPSAMPLER", "Lightricks/ltxv-spatial-upscaler-0.9.7")
MODEL_DIR = Path(os.environ.get("MODEL_DIR", "/models"))
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "/outputs"))
PORT = int(os.environ.get("PORT", "8080"))
# Optional cap for testing on a larger GPU than production, e.g. 0.6 of an 80 GB card.
GPU_MEMORY_FRACTION = float(os.environ.get("GPU_MEMORY_FRACTION", "1.0"))
INDEX_HTML = Path(__file__).with_name("index.html")

# The first pass renders at BASE_* and the latent upsampler doubles it. Both sides must be
# multiples of 32 (the VAE's spatial compression), and frames must be 8k + 1.
BASE_WIDTH, BASE_HEIGHT = 384, 256
NUM_FRAMES = 97
FPS = 24
# Timestep schedules published for the distilled checkpoints: a full 8-step pass, then a
# short refinement pass over the upsampled latents.
FIRST_PASS_TIMESTEPS = [1000, 993, 987, 981, 975, 909, 725, 0.03]
REFINE_TIMESTEPS = [1000, 909, 725, 421, 0]
NEGATIVE_PROMPT = "worst quality, inconsistent motion, blurry, jittery, distorted"

MAX_PROMPT_CHARS = 600
MAX_QUEUE = 12
JOB_TTL_S = 3600
GALLERY_PROMPTS = [
    (
        "A red fox trots through fresh snow in a quiet birch forest at dawn. Soft golden light "
        + "filters between the white trunks, and small clouds of breath rise from the fox as it "
        + "pauses, lifts its head and looks toward the camera. Shallow depth of field, cinematic."
    ),
    (
        "Waves roll onto a black sand beach at sunset. White foam spreads over the dark sand "
        + "and slides back into turquoise water, while orange and pink clouds glow above jagged "
        + "sea stacks on the horizon. Slow, steady camera drifting left."
    ),
    (
        "A close-up of a barista pouring steamed milk into a cup of espresso, drawing a leaf "
        + "pattern in the crema. Warm cafe lighting, steam curling upward, shallow depth of "
        + "field, soft background bokeh."
    ),
]

log = logging.getLogger("ltx")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


@dataclass
class Job:
    id: str
    prompt: str
    seed: int
    client: str
    gallery: bool
    created: float = field(default_factory=time.time)
    status: str = "queued"  # queued | running | done | error
    stage: str = ""
    progress: float = 0.0
    started: float | None = None
    finished: float | None = None
    error: str | None = None

    def video_path(self) -> Path:
        return OUTPUT_DIR / f"{self.id}.mp4"


class Generator:
    """Owns the pipelines. `generate` must only run on one thread at a time."""

    def __init__(self) -> None:
        super().__init__()
        t0 = time.monotonic()
        if GPU_MEMORY_FRACTION < 1.0:
            torch.cuda.set_per_process_memory_fraction(GPU_MEMORY_FRACTION)
        model_path = snapshot_download(
            MODEL_REPO,
            cache_dir=MODEL_DIR,
            # The repo also carries a stray copy of every component under vae/; skip it.
            allow_patterns=[
                "model_index.json",
                "scheduler/*",
                "text_encoder/*",
                "tokenizer/*",
                "transformer/*",
                "vae/config.json",
                "vae/diffusion_pytorch_model.safetensors",
            ],
        )
        # The pipeline reuses the main model's VAE, so only the upsampler weights are needed.
        upsampler_path = snapshot_download(
            UPSAMPLER_REPO,
            cache_dir=MODEL_DIR,
            allow_patterns=["model_index.json", "latent_upsampler/*"],
        )
        log.info("weights ready in %.0fs", time.monotonic() - t0)
        self.pipe = LTXConditionPipeline.from_pretrained(model_path, torch_dtype=torch.bfloat16)
        self.upsample = LTXLatentUpsamplePipeline.from_pretrained(
            upsampler_path, vae=self.pipe.vae, torch_dtype=torch.bfloat16
        )
        self.pipe.to("cuda")
        self.upsample.to("cuda")
        self.pipe.vae.enable_tiling()
        log.info("pipelines on GPU in %.0fs", time.monotonic() - t0)

    def generate(self, job: Job) -> None:
        stages = [("Drafting motion", len(FIRST_PASS_TIMESTEPS)), ("Refining detail", len(REFINE_TIMESTEPS))]
        total = sum(n for _, n in stages) + 2  # + upsample + decode/encode

        def on_step(base: int, stage: str):
            def callback(_pipe: Any, step: int, _t: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
                job.stage = stage
                job.progress = min(0.99, (base + step + 1) / total)
                return kwargs

            return callback

        common: dict[str, Any] = dict(
            conditions=None,
            prompt=job.prompt,
            negative_prompt=NEGATIVE_PROMPT,
            num_frames=NUM_FRAMES,
            decode_timestep=0.05,
            decode_noise_scale=0.025,
            image_cond_noise_scale=0.0,
            guidance_scale=1.0,
            guidance_rescale=0.7,
        )
        with torch.inference_mode():
            job.stage = stages[0][0]
            latents = self.pipe(
                **common,
                width=BASE_WIDTH,
                height=BASE_HEIGHT,
                timesteps=FIRST_PASS_TIMESTEPS,
                generator=torch.Generator().manual_seed(job.seed),
                output_type="latent",
                callback_on_step_end=on_step(0, stages[0][0]),
            ).frames
            job.stage = "Upscaling"
            upscaled = self.upsample(latents=latents, adain_factor=1.0, output_type="latent").frames
            first = len(FIRST_PASS_TIMESTEPS) + 1
            job.progress = first / total
            frames = self.pipe(
                **common,
                width=BASE_WIDTH * 2,
                height=BASE_HEIGHT * 2,
                denoise_strength=0.999,
                timesteps=REFINE_TIMESTEPS,
                latents=upscaled,
                generator=torch.Generator().manual_seed(job.seed),
                output_type="np",
                callback_on_step_end=on_step(first, stages[1][0]),
            ).frames[0]
        job.stage = "Encoding"
        write_mp4(frames, job.video_path())


def write_mp4(frames: np.ndarray, path: Path) -> None:
    """Writes float frames in [0, 1] as browser-playable H.264 (yuv420p, faststart)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.mp4")
    pixels = (np.clip(frames, 0, 1) * 255).round().astype(np.uint8)
    writer = imageio.get_writer(
        tmp,
        fps=FPS,
        codec="libx264",
        quality=8,
        pixelformat="yuv420p",
        ffmpeg_params=["-movflags", "+faststart"],
    )
    for frame in pixels:
        writer.append_data(frame)
    writer.close()
    tmp.rename(path)


class Service:
    def __init__(self, generator: Generator) -> None:
        super().__init__()
        self.generator = generator
        self.jobs: dict[str, Job] = {}
        self.queue: collections.deque[Job] = collections.deque()
        self.wake = asyncio.Event()
        self.gallery: list[Job] = []

    def position(self, job: Job) -> int:
        try:
            return self.queue.index(job) + 1
        except ValueError:
            return 0

    def submit(self, prompt: str, seed: int, client: str, gallery: bool) -> Job:
        job = Job(id=uuid.uuid4().hex, prompt=prompt, seed=seed, client=client, gallery=gallery)
        self.jobs[job.id] = job
        self.queue.append(job)
        self.wake.set()
        return job

    async def run(self) -> None:
        while True:
            while not self.queue:
                self.wake.clear()
                await self.wake.wait()
            job = self.queue[0]
            job.status, job.started = "running", time.time()
            log.info("job %s started (%d waiting)", job.id[:8], len(self.queue) - 1)
            try:
                await asyncio.to_thread(self.generator.generate, job)
                job.status, job.progress = "done", 1.0
            except Exception as e:
                log.exception("job %s failed", job.id[:8])
                job.status, job.error = "error", type(e).__name__
                torch.cuda.empty_cache()
            finally:
                job.finished = time.time()
                self.queue.popleft()
            log.info("job %s %s in %.1fs", job.id[:8], job.status, job.finished - job.started)
            self.expire()

    def expire(self) -> None:
        cutoff = time.time() - JOB_TTL_S
        for job_id, job in list(self.jobs.items()):
            if not job.gallery and job.finished is not None and job.finished < cutoff:
                job.video_path().unlink(missing_ok=True)
                del self.jobs[job_id]

    def busy_client(self, client: str) -> bool:
        return any(j.client == client for j in self.queue)


state: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    # uvicorn accepts no connections until this returns, so the replica only goes ready
    # once the weights are on the GPU.
    service = Service(await asyncio.to_thread(Generator))
    for i, prompt in enumerate(GALLERY_PROMPTS):
        service.gallery.append(service.submit(prompt, seed=1000 + i, client="gallery", gallery=True))
    state["service"] = service
    worker = asyncio.create_task(service.run())
    yield
    worker.cancel()


app = FastAPI(lifespan=lifespan)


class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=3, max_length=MAX_PROMPT_CHARS)
    seed: int | None = None


def client_address(request: Request) -> str:
    # Behind the scaling group's gateway the peer is the proxy; the first forwarded hop is
    # the browser.
    forwarded = request.headers.get("x-forwarded-for", "")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def job_json(service: Service, job: Job) -> dict[str, Any]:
    now = job.finished or time.time()
    return {
        "job_id": job.id,
        "status": job.status,
        "position": service.position(job),
        "stage": job.stage,
        "progress": round(job.progress, 3),
        "elapsed_s": round(now - job.started, 1) if job.started else None,
        "render_s": round(job.finished - job.started, 1) if job.finished and job.started else None,
        "video_url": f"videos/{job.id}.mp4" if job.status == "done" else None,
        "error": job.error,
    }


@app.get("/")
async def index() -> HTMLResponse:
    return HTMLResponse(INDEX_HTML.read_text())


@app.get("/status")
async def status() -> JSONResponse:
    service: Service = state["service"]
    return JSONResponse({"ok": True, "model": MODEL_REPO, "queue": len(service.queue)})


@app.post("/api/generate")
async def generate(body: GenerateRequest, request: Request) -> JSONResponse:
    service: Service = state["service"]
    service.expire()
    client = client_address(request)
    if service.busy_client(client):
        raise HTTPException(429, "You already have a clip rendering. Wait for it to finish.")
    if len(service.queue) >= MAX_QUEUE:
        raise HTTPException(503, "The queue is full right now. Try again in a minute.")
    seed = body.seed if body.seed is not None else int.from_bytes(os.urandom(4), "little")
    job = service.submit(" ".join(body.prompt.split()), seed, client, gallery=False)
    return JSONResponse(job_json(service, job))


@app.get("/api/jobs/{job_id}")
async def job_status(job_id: str) -> JSONResponse:
    service: Service = state["service"]
    service.expire()
    job = service.jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "Unknown or expired job.")
    return JSONResponse(job_json(service, job))


@app.get("/api/gallery")
async def gallery() -> JSONResponse:
    service: Service = state["service"]
    return JSONResponse([{"prompt": j.prompt, **job_json(service, j)} for j in service.gallery])


@app.get("/videos/{job_id}.mp4")
async def video(job_id: str) -> FileResponse:
    service: Service = state["service"]
    job = service.jobs.get(job_id)
    if job is None or job.status != "done":
        raise HTTPException(404, "No such clip.")
    return FileResponse(job.video_path(), media_type="video/mp4")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=PORT, proxy_headers=False)
