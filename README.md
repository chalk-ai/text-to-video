# Text to video on Chalk

Describe a shot in a browser tab and get back a four-second clip, rendered on a single GPU by
Lightricks' [LTX-Video](https://huggingface.co/Lightricks/LTX-Video) 13B distilled model. The
page, the job queue and the model are all served from one replica deployed with
[`chalkcompute`](https://pypi.org/project/chalkcompute/).

```
prompt ──POST /api/generate──▶ FIFO queue ──▶ LTX-Video on one GPU ──▶ MP4 ──▶ browser
```

## Files

| File | Role |
| --- | --- |
| `deploy.py` | Builds the image (torch, diffusers, ffmpeg) and deploys a `chalkcompute.ScalingGroup` on one L40S. |
| `server.py` | FastAPI app in the container: the job queue, the JSON API, and the page at `/`. |
| `index.html` | Prompt box, suggestions, progress, the finished clip, and a gallery of example renders. |

## Run it

You need [`uv`](https://docs.astral.sh/uv/) and a Chalk environment that can schedule GPU
workloads. Credentials come from the SDK’s ambient credentials from `chalk login`. You can also set
`CHALK_API_SERVER`, `CHALK_CLIENT_ID`, `CHALK_CLIENT_SECRET` and `CHALK_ENVIRONMENT_ID` yourself.

```bash
./deploy.py --gpu nvidia-l40s
```

The replica downloads about 45 GB of weights from Hugging Face when it boots, then loads them onto
the GPU. Baking them into the image instead would make every build push a 50 GB layer. The
readiness probe waits for the model before the replica takes traffic. When the deploy finishes,
the script prints a `web_url`. Open it and press **Generate clip**.

## How a clip is rendered

1. **Draft.** The distilled transformer denoises 97 frames of latent video at 384×256 in 8 steps.
2. **Upscale.** The spatial latent upsampler doubles the resolution to 768×512.
3. **Refine.** Four more denoising steps restore detail, and the VAE decodes the frames.
4. **Encode.** ffmpeg writes a 24 fps H.264 MP4 (`yuv420p`, `faststart`) that browsers play inline.

Peak GPU memory is about 41 GB, so the model fits on a 48 GB L40S with the text encoder, the
transformer and the VAE all resident.

## API

| Method | Path | Body / result |
| --- | --- | --- |
| `POST` | `/api/generate` | `{"prompt": "...", "seed": 123}` (seed optional) → `{"job_id", "status", "position", ...}` |
| `GET` | `/api/jobs/{job_id}` | `{"status": "queued" \| "running" \| "done" \| "error", "position", "stage", "progress", "video_url"}` |
| `GET` | `/videos/{job_id}.mp4` | the finished clip |
| `GET` | `/api/gallery` | example prompts rendered at boot |

```python
import time

import httpx


def text_to_video(web_url: str, prompt: str, out: str = "clip.mp4") -> str:
    with httpx.Client(base_url=web_url, timeout=30) as http:
        job = http.post("/api/generate", json={"prompt": prompt}).raise_for_status().json()
        while job["status"] in ("queued", "running"):
            time.sleep(1)
            job = http.get(f"/api/jobs/{job['job_id']}").json()
        if job["status"] != "done":
            raise RuntimeError(job["error"])
        with open(out, "wb") as f:
            f.write(http.get("/" + job["video_url"]).content)
    return out
```

## Notes

- **One clip at a time.** The GPU renders jobs in order. Each client address may have one job
  queued or running, the queue holds at most 12, and finished clips expire after an hour. A clip is
  only reachable by its random job id, so one visitor never sees another's renders.
- **Prompts.** Write them like a shot list: subject, motion, light, camera. LTX-Video responds
  much better to a detailed paragraph than to a few words.
- **The URL is public.** `authenticated` is off so a browser can open the page directly, which
  means anyone with the link can use the GPU. Delete the group when you are done:

  ```bash
  uv run --with chalkcompute python -c \
    "from chalkcompute import ScalingGroup; ScalingGroup.from_name('ltx-video').delete()"
  ```

## License

The code in this repository is Apache-2.0.

The LTX-Video weights are **not** covered by that license. They are released by Lightricks under
the [LTXV Open Weights License](https://huggingface.co/Lightricks/LTX-Video/blob/main/LTX-Video-Open-Weights-License-0.X.txt).
Among other terms, it restricts use for organizations above a revenue threshold without a separate
commercial license, and it requires anyone you host the model for to follow its use restrictions
and to be told that clips are machine-generated. The page in this repository shows that notice.
Read the license before you deploy this for others.

## Local authentication

The SDK reads credentials, API host, and the selected environment from your existing
`chalk login` in `~/.chalk.yml`, including project-scoped and default logins.
For unattended runs, `CHALK_CLIENT_ID` and `CHALK_CLIENT_SECRET` remain supported. Where available, `--env-id`
overrides the target environment; otherwise the SDK uses your login or environment settings.
