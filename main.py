import os
import json
import time
import base64
import requests
import httpx

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# =========================
# CREATE SERVICE ACCOUNT FILE AT RUNTIME
# =========================

if "GOOGLE_APPLICATION_CREDENTIALS_JSON" in os.environ:
    creds_path = "/tmp/service-account.json"
    with open(creds_path, "w") as f:
        f.write(os.environ["GOOGLE_APPLICATION_CREDENTIALS_JSON"])
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = creds_path

# =========================
# IMPORT VERTEX AI AFTER CREDENTIAL SETUP
# =========================

import vertexai
from vertexai.preview.generative_models import GenerativeModel, Part, GenerationConfig

# =========================
# ENV VARS
# =========================

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_SERVICE_ROLE_KEY"]

GOOGLE_PROJECT = os.environ["GOOGLE_CLOUD_PROJECT"]
GOOGLE_LOCATION = os.environ.get("GOOGLE_CLOUD_LOCATION", "global")

# =========================
# INIT VERTEX AI
# =========================

vertexai.init(
    project=GOOGLE_PROJECT,
    location=GOOGLE_LOCATION
)

model = GenerativeModel("veo-3.1-generate-preview")

# =========================
# FASTAPI
# =========================

app = FastAPI()

class GenerateVideoRequest(BaseModel):
    generation_id: str
    image_url: str
    prompt: str
    aspect_ratio: str
    duration: int
    script_text: str | None = None

# =========================
# SUPABASE UPDATE (REST)
# =========================

def update_generation(generation_id, status, video_url=None):

    url = f"{SUPABASE_URL}/rest/v1/video_generations?id=eq.{generation_id}"

    payload = {
        "status": status
    }

    if video_url:
        payload["video_url"] = video_url
        payload["final_video_url"] = video_url

    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
        "Prefer": "return=minimal"
    }

    response = httpx.patch(url, headers=headers, json=payload)

    if response.status_code not in [200, 204]:
        print("Supabase update failed:", response.text)

# =========================
# DOWNLOAD IMAGE
# =========================

def download_image_bytes(url):

    response = requests.get(url)

    if response.status_code != 200:
        raise Exception("Failed to download image")

    return response.content

# =========================
# GENERATE VIDEO ENDPOINT
# =========================

@app.post("/generate-video")
def generate_video(req: GenerateVideoRequest):

    generation_id = req.generation_id

    try:

        print(f"Starting Veo 3.1 generation: {generation_id}")

        update_generation(generation_id, "processing")

        # Download image
        print("Downloading image...")
        image_bytes = download_image_bytes(req.image_url)

        # Create Veo Part object
        image_part = Part.from_data(
            data=image_bytes,
            mime_type="image/jpeg"
        )

        # Build prompt
        full_prompt = req.prompt
        if req.script_text:
            full_prompt += f"\n\nSpeech: {req.script_text}"

        print("Calling Veo 3.1 via Vertex AI...")

        # Generate video operation
        operation = model.generate_video(
            prompt=full_prompt,
            image=image_part,
            generation_config=GenerationConfig(
                aspect_ratio=req.aspect_ratio
            )
        )

        # Wait until done
        while not operation.done:
            time.sleep(10)
            operation = operation.refresh()

        if not operation.response:
            raise Exception("No response from Veo")

        video = operation.response.generated_videos[0]

        video_bytes = video.video_bytes

        # Upload to Supabase Storage
        print("Uploading video to Supabase Storage...")

        file_path = f"video-final/{generation_id}.mp4"

        upload_url = f"{SUPABASE_URL}/storage/v1/object/creative-media/{file_path}"

        headers = {
            "apikey": SUPABASE_KEY,
            "Authorization": f"Bearer {SUPABASE_KEY}",
            "Content-Type": "video/mp4"
        }

        upload_response = httpx.post(
            upload_url,
            headers=headers,
            content=video_bytes
        )

        if upload_response.status_code not in [200, 201]:
            raise Exception(upload_response.text)

        public_url = f"{SUPABASE_URL}/storage/v1/object/public/creative-media/{file_path}"

        print("Video uploaded:", public_url)

        update_generation(
            generation_id,
            "completed",
            public_url
        )

        return {
            "status": "completed",
            "video_url": public_url
        }

    except Exception as e:

        print("ERROR:", str(e))

        update_generation(generation_id, "failed")

        raise HTTPException(status_code=500, detail=str(e))
