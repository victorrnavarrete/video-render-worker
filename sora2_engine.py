"""
Sora 2 (OpenAI) video generation engine.
Provides image-to-video generation as an alternative to Veo 3.

Models:
  - sora-2:     valid seconds = 4, 8, 12
  - sora-2-pro: valid seconds = 10, 15, 25

Resolutions (both models):
  - 720x1280  (9:16 portrait)
  - 1280x720  (16:9 landscape)

Image input must match video output resolution exactly.
"""

import os
import io
import asyncio
import httpx
from PIL import Image


# Sora 2 supported resolutions (720p only for standard model)
SORA_SIZES = {
    "9:16": "720x1280",
    "16:9": "1280x720",
}
SORA_FALLBACK_SIZE = "1280x720"

# Duration → model mapping
# sora-2:     4, 8, 12
# sora-2-pro: 10, 15, 25
SORA_STANDARD_DURATIONS = [4, 8, 12]
SORA_PRO_DURATIONS = [10, 15, 25]


def map_aspect_to_sora_size(aspect_ratio: str) -> str:
    """Map aspect ratio string to Sora 2 resolution. Fallback to 1280x720."""
    return SORA_SIZES.get(aspect_ratio, SORA_FALLBACK_SIZE)


def pick_sora_model_and_duration(requested_seconds: int) -> tuple[str, int]:
    """
    Pick the best model + duration combo for the requested seconds.
    - Up to 12s → sora-2 (standard) with nearest valid duration
    - Above 12s → sora-2-pro with nearest valid duration (10, 15, 25)
    """
    if requested_seconds <= 12:
        best = min(SORA_STANDARD_DURATIONS, key=lambda d: abs(d - requested_seconds))
        return "sora-2", best
    else:
        best = min(SORA_PRO_DURATIONS, key=lambda d: abs(d - requested_seconds))
        return "sora-2-pro", best


def resize_image_for_sora(image_bytes: bytes, target_size: str) -> bytes:
    """Resize image to exact Sora output resolution.
    Sora requires input image dimensions to match the video output size."""
    try:
        w, h = map(int, target_size.split("x"))
        img = Image.open(io.BytesIO(image_bytes))
        if img.mode != "RGB":
            img = img.convert("RGB")

        img = img.resize((w, h), Image.LANCZOS)

        output = io.BytesIO()
        img.save(output, format="PNG")
        print(f"Sora image resized to {w}x{h} ({len(output.getvalue())} bytes)")
        return output.getvalue()
    except Exception as e:
        print(f"Sora image resize warning (using original): {e}")
        return image_bytes


def build_sora_prompt(base_prompt: str) -> str:
    """
    Wrap the base prompt with Sora-specific identity/clothing preservation
    instructions. Sora tends to alter clothing — we reinforce preservation.
    """
    sora_prefix = (
        "CRITICAL: Preserve EXACTLY the person's identity, face, skin, hair, "
        "and clothing from the reference image. Do NOT change, add, or remove "
        "any clothing items. The outfit must remain identical to the source image "
        "throughout the entire video — same colors, same style, same fit. "
        "Photorealistic quality. No AI artifacts.\n\n"
    )
    return sora_prefix + base_prompt


async def call_sora(
    image_bytes: bytes,
    prompt: str,
    aspect_ratio: str = "9:16",
    duration_seconds: int = 8,
) -> bytes:
    """
    Generate video via OpenAI Sora 2 API.
    1. POST /v1/videos (multipart) to start generation
    2. Poll GET /v1/videos/{id} until completed
    3. Download GET /v1/videos/{id}/content
    Returns raw video bytes.
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise Exception("OPENAI_API_KEY not configured")

    sora_size = map_aspect_to_sora_size(aspect_ratio)
    sora_model, sora_duration = pick_sora_model_and_duration(duration_seconds)
    sora_prompt = build_sora_prompt(prompt)

    print(f"Sora: model={sora_model}, size={sora_size}, duration={sora_duration}s, "
          f"requested={duration_seconds}s, prompt_len={len(sora_prompt)}")

    headers = {
        "Authorization": f"Bearer {api_key}",
    }

    # Step 1: Submit generation (multipart form data)
    async with httpx.AsyncClient(timeout=600) as client:
        files = {
            "input_reference": ("image.png", image_bytes, "image/png"),
        }
        form_data = {
            "model": sora_model,
            "prompt": sora_prompt,
            "size": sora_size,
            "seconds": str(sora_duration),
        }

        print(f"Submitting to Sora API: model={sora_model}, seconds={sora_duration}, size={sora_size}")
        submit_res = await client.post(
            "https://api.openai.com/v1/videos",
            headers=headers,
            files=files,
            data=form_data,
        )

        if submit_res.status_code not in (200, 201):
            raise Exception(f"Sora submit failed [{submit_res.status_code}]: {submit_res.text}")

        submit_data = submit_res.json()
        video_id = submit_data.get("id")
        if not video_id:
            raise Exception(f"Sora submit returned no video id: {submit_data}")

        print(f"Sora generation started: video_id={video_id}")

        # Step 2: Poll until completed or failed
        poll_url = f"https://api.openai.com/v1/videos/{video_id}"
        max_polls = 120  # 120 * 15s = 30 minutes max
        for i in range(max_polls):
            await asyncio.sleep(15)

            poll_res = await client.get(poll_url, headers=headers)
            poll_res.raise_for_status()
            poll_data = poll_res.json()

            status = poll_data.get("status", "unknown")
            print(f"Sora poll [{i+1}]: status={status}")

            if status == "completed":
                break
            elif status == "failed":
                error_msg = poll_data.get("error", "Unknown error")
                raise Exception(f"Sora generation failed: {error_msg}")
        else:
            raise Exception("Sora generation timed out after 30 minutes")

        # Step 3: Download video content
        print("Downloading Sora video...")
        download_url = f"https://api.openai.com/v1/videos/{video_id}/content"
        download_res = await client.get(download_url, headers=headers)
        download_res.raise_for_status()

        video_bytes = download_res.content
        print(f"Sora video downloaded: {len(video_bytes)} bytes")
        return video_bytes
