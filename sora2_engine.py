"""
Sora 2 (OpenAI) video generation engine.
Provides image-to-video generation with synchronized audio.

Models:
  - sora-2:     standard quality
  - sora-2-pro: higher quality

Valid seconds (both models): 4, 8, 12, 16, 20

Resolutions (both models):
  - 720x1280  (9:16 portrait)
  - 1280x720  (16:9 landscape)

Image input must match video output resolution exactly.
"""

import os
import io
import uuid
import asyncio
import httpx
from PIL import Image


# Sora 2 supported resolutions (720p only for standard model)
SORA_SIZES = {
    "9:16": "720x1280",
    "16:9": "1280x720",
}
SORA_FALLBACK_SIZE = "1280x720"

# Valid durations for both sora-2 and sora-2-pro
SORA_VALID_DURATIONS = [4, 8, 12, 16, 20]


def map_aspect_to_sora_size(aspect_ratio: str) -> str:
    """Map aspect ratio string to Sora 2 resolution. Fallback to 1280x720."""
    return SORA_SIZES.get(aspect_ratio, SORA_FALLBACK_SIZE)


def pick_sora_model_and_duration(requested_seconds: int) -> tuple[str, int]:
    """
    Pick the best model + duration combo for the requested seconds.
    Both sora-2 and sora-2-pro accept 4, 8, 12, 16, 20.
    Use sora-2-pro for higher quality.
    """
    best = min(SORA_VALID_DURATIONS, key=lambda d: abs(d - requested_seconds))
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


def build_sora_prompt(base_prompt: str, custom_instructions: str = None) -> str:
    """
    Clean and adapt the Veo3 prompt for Sora 2.
    - Extracts SPEECH and custom instructions and places them at the TOP
    - Removes verbose/Veo3-specific sections
    - Keeps camera, face, body instructions
    """
    import re

    # 1. Extract custom instructions
    user_instructions = (custom_instructions or "").strip()
    if not user_instructions:
        match = re.search(
            r"IMPORTANT USER INSTRUCTIONS\s*\(MUST FOLLOW\)\s*\n([\s\S]*?)(?=\n\n[A-Z]|\Z)",
            base_prompt,
        )
        if match:
            user_instructions = match.group(1).strip()
            print(f"Sora: extracted custom instructions from prompt ({len(user_instructions)} chars)")

    # 2. Extract SPEECH section BEFORE removing it (to place at top)
    speech_text = ""
    speech_match = re.search(
        r"SPEECH.*?\n([\s\S]*?)(?=\n\n[A-Z]|\Z)",
        base_prompt,
    )
    if speech_match:
        speech_text = speech_match.group(0).strip()
        print(f"Sora: extracted SPEECH block ({len(speech_text)} chars)")

    # 3. Remove sections that are Veo3-specific or redundant for Sora
    cleaned = base_prompt
    sections_to_remove = [
        r"AUDIO RULE[\s\S]*?(?=\n\n[A-Z]|\Z)",
        r"AUDIO REMINDER[\s\S]*?(?=\n\n[A-Z]|\Z)",
        r"SPEECH.*?\n[\s\S]*?(?=\n\n[A-Z]|\Z)",
        r"LIPSYNC REQUIREMENTS[\s\S]*?(?=\n\n[A-Z]|\Z)",
        r"IDENTITY LOCK \(HIGHEST PRIORITY\)[\s\S]*?(?=\n\n[A-Z]|\Z)",
        r"REALISM REQUIREMENTS[\s\S]*?(?=\n\n[A-Z]|\Z)",
        r"OUTPUT[\s\S]*?(?=\n\n[A-Z]|\Z)",
        r"IMPORTANT USER INSTRUCTIONS\s*\(MUST FOLLOW\)[\s\S]*?(?=\n\n[A-Z]|\Z)",
    ]
    for pattern in sections_to_remove:
        cleaned = re.sub(pattern, "", cleaned)

    cleaned = re.sub(r"photorealistic\. 4K quality\..*?video\s*$", "", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()

    # 4. Build Sora prompt — SPEECH at the very top for maximum priority
    parts = []

    # Speech FIRST — Sora weights early instructions more heavily
    if speech_text:
        parts.append(speech_text)

    # Identity/clothing block
    parts.append(
        "Animate this reference image into a realistic video. "
        "Preserve EXACTLY the person's identity, face, skin, hair, "
        "and clothing from the image. Do NOT change any clothing items — "
        "same outfit, same colors, same style throughout the entire video. "
        "Any product in the image must remain EXACTLY as shown — no deformation, "
        "no morphing, no bending. Product label, text, logo, and packaging must "
        "stay sharp, legible, and unchanged. The product is a rigid physical object. "
        "Photorealistic quality. Natural lighting. Smooth motion."
    )

    # Custom instructions right after identity
    if user_instructions:
        parts.append(f"MANDATORY USER INSTRUCTIONS (MUST FOLLOW EXACTLY):\n{user_instructions}")
        print(f"Sora: custom instructions placed near top ({len(user_instructions)} chars)")

    # Remaining behavioral instructions (camera, face, body)
    if cleaned:
        parts.append(cleaned)

    sora_prompt = "\n\n".join(parts)

    # 5. Smart truncation: preserve top (speech + identity) and trim from the end
    MAX_SORA_PROMPT = 2000
    if len(sora_prompt) > MAX_SORA_PROMPT:
        sora_prompt = sora_prompt[:MAX_SORA_PROMPT].rsplit("\n", 1)[0]

    print(f"Sora prompt built: {len(sora_prompt)} chars (original: {len(base_prompt)} chars, "
          f"speech: {len(speech_text)} chars, custom: {len(user_instructions)} chars)")
    return sora_prompt


async def call_sora(
    image_bytes: bytes,
    prompt: str,
    aspect_ratio: str = "9:16",
    duration_seconds: int = 8,
    custom_instructions: str = None,
    model_override: str = None,
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
    _, sora_duration = pick_sora_model_and_duration(duration_seconds)
    sora_model = model_override if model_override in ("sora-2", "sora-2-pro") else "sora-2"
    sora_prompt = build_sora_prompt(prompt, custom_instructions=custom_instructions)

    print(f"Sora: model={sora_model}, size={sora_size}, duration={sora_duration}s, "
          f"requested={duration_seconds}s, prompt_len={len(sora_prompt)}")

    headers = {
        "Authorization": f"Bearer {api_key}",
    }

    # Step 1: Upload reference image to Supabase Storage to get a public URL
    # (Sora API now requires input_reference as a JSON object with image_url)
    supabase_url = os.environ.get("SUPABASE_URL")
    supabase_key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
    image_public_url = None

    if supabase_url and supabase_key:
        file_name = f"sora-ref-{uuid.uuid4().hex[:12]}.png"
        upload_url = f"{supabase_url}/storage/v1/object/creative-media/{file_name}"
        async with httpx.AsyncClient(timeout=30) as upload_client:
            upload_res = await upload_client.post(
                upload_url,
                headers={
                    "apikey": supabase_key,
                    "Authorization": f"Bearer {supabase_key}",
                    "Content-Type": "image/png",
                },
                content=image_bytes,
            )
            if upload_res.status_code in (200, 201):
                image_public_url = f"{supabase_url}/storage/v1/object/public/creative-media/{file_name}"
                print(f"Sora reference image uploaded: {image_public_url}")
            else:
                print(f"WARNING: Supabase upload failed ({upload_res.status_code}), falling back to multipart")

    # Step 1b: Submit generation and poll
    async with httpx.AsyncClient(timeout=600) as client:
        if image_public_url:
            # JSON format with image_url (new API format)
            json_body = {
                "model": sora_model,
                "prompt": sora_prompt,
                "size": sora_size,
                "seconds": str(sora_duration),
                "input_reference": {
                    "image_url": image_public_url,
                },
            }
            headers["Content-Type"] = "application/json"

            print(f"Submitting to Sora API (JSON): model={sora_model}, seconds={sora_duration}, size={sora_size}")
            submit_res = await client.post(
                "https://api.openai.com/v1/videos",
                headers=headers,
                json=json_body,
            )
        else:
            # Fallback: multipart form data (legacy format)
            files = {
                "input_reference": ("image.png", image_bytes, "image/png"),
            }
            form_data = {
                "model": sora_model,
                "prompt": sora_prompt,
                "size": sora_size,
                "seconds": str(sora_duration),
            }

            print(f"Submitting to Sora API (multipart): model={sora_model}, seconds={sora_duration}, size={sora_size}")
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
