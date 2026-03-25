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


def build_sora_prompt(base_prompt: str, custom_instructions: str = None) -> str:
    """
    Clean and adapt the Veo3 prompt for Sora 2.
    - Extracts and prioritizes custom instructions (placed at the top)
    - Removes verbose audio/technical sections that trigger Sora moderation
    - Adds clothing/identity preservation instructions
    - Keeps only the essential behavioral/visual instructions
    """
    import re

    # 1. Extract custom instructions BEFORE stripping (they'd survive regex
    #    but could be truncated at the end). Use structured param if available.
    user_instructions = (custom_instructions or "").strip()
    if not user_instructions:
        # Fallback: extract from prompt blob
        match = re.search(
            r"IMPORTANT USER INSTRUCTIONS\s*\(MUST FOLLOW\)\s*\n([\s\S]*?)(?=\n\n[A-Z]|\Z)",
            base_prompt,
        )
        if match:
            user_instructions = match.group(1).strip()
            print(f"Sora: extracted custom instructions from prompt ({len(user_instructions)} chars)")

    # 2. Remove sections that are Veo3-specific or trigger Sora moderation
    cleaned = base_prompt
    sections_to_remove = [
        r"AUDIO RULE \(MANDATORY\)[\s\S]*?(?=\n\n[A-Z]|\Z)",
        r"AUDIO REMINDER[\s\S]*?(?=\n\n[A-Z]|\Z)",
        # LIPSYNC REQUIREMENTS is kept — critical for natural mouth movement
        r"IDENTITY LOCK \(HIGHEST PRIORITY\)[\s\S]*?(?=\n\n[A-Z]|\Z)",
        r"REALISM REQUIREMENTS[\s\S]*?(?=\n\n[A-Z]|\Z)",
        r"OUTPUT[\s\S]*?(?=\n\n[A-Z]|\Z)",
        r"IMPORTANT USER INSTRUCTIONS\s*\(MUST FOLLOW\)[\s\S]*?(?=\n\n[A-Z]|\Z)",
    ]
    for pattern in sections_to_remove:
        cleaned = re.sub(pattern, "", cleaned)

    # Remove Veo3 quality suffix (added by build_veo_prompt)
    cleaned = re.sub(r"photorealistic\. 4K quality\..*?video\s*$", "", cleaned, flags=re.MULTILINE)

    # Remove leftover double newlines
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()

    # 3. Build Sora prompt with custom instructions at the TOP (Sora weights early instructions more)
    sora_prompt = (
        "Animate this reference image into a realistic video. "
        "Preserve EXACTLY the person's identity, face, skin, hair, "
        "and clothing from the image. Do NOT change any clothing items — "
        "same outfit, same colors, same style throughout the entire video. "
        "Any product in the image must remain EXACTLY as shown — no deformation, "
        "no morphing, no bending. Product label, text, logo, and packaging must "
        "stay sharp, legible, and unchanged. The product is a rigid physical object. "
        "Photorealistic quality. Natural lighting. Smooth motion.\n\n"
    )

    # Place user instructions right after identity block — highest priority position
    if user_instructions:
        sora_prompt += f"MANDATORY USER INSTRUCTIONS (MUST FOLLOW EXACTLY — overrides any default behavior):\n{user_instructions}\n\n"
        print(f"Sora: custom instructions placed at top ({len(user_instructions)} chars)")

    if cleaned:
        sora_prompt += cleaned

    # 4. Smart truncation: preserve top (identity + custom instructions) and trim from the end
    MAX_SORA_PROMPT = 2000
    if len(sora_prompt) > MAX_SORA_PROMPT:
        sora_prompt = sora_prompt[:MAX_SORA_PROMPT].rsplit("\n", 1)[0]

    print(f"Sora prompt built: {len(sora_prompt)} chars (original: {len(base_prompt)} chars, "
          f"custom_instructions: {len(user_instructions)} chars)")
    return sora_prompt


async def call_sora(
    image_bytes: bytes,
    prompt: str,
    aspect_ratio: str = "9:16",
    duration_seconds: int = 8,
    custom_instructions: str = None,
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

    # Step 1b: Submit generation
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
