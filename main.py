import os
import time
import base64
import httpx
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import vertexai
from vertexai.preview.generative_models import GenerativeModel, Image

# =========================
# ENV
# =========================

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_SERVICE_ROLE_KEY"]

PROJECT_ID = os.environ["GOOGLE_CLOUD_PROJECT"]
LOCATION = os.environ["GOOGLE_CLOUD_LOCATION"]

# =========================
# INIT VERTEX
# =========================

vertexai.init(
    project=PROJECT_ID,
    location=LOCATION,
)

# =========================
# FASTAPI
# =========================

app = FastAPI()

class GenerateVideoRequest(BaseModel):
    generation_id: str
    image_url: str
    prompt: str
    script_text: str | None = None
    aspect_ratio: str = "9:16"
    duration: int = 5

# =========================
# SUPABASE REST HELPERS
# =========================

def supabase_update(generation_id, payload):

    url = f"{SUPABASE_URL}/rest/v1/video_generations?id=eq.{generation_id}"

    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
        "Prefer": "return=minimal"
    }

    response = httpx.patch(url, headers=headers, json=payload, timeout=60)

    if response.status_code >= 300:
        raise Exception(f"Supabase update failed: {response.text}")

def supabase_upload_video(generation_id, video_bytes):

    path = f"video-final/{generation_id}.mp4"

    url = f"{SUPABASE_URL}/storage/v1/object/creative-media/{path}"

    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "video/mp4"
    }

    response = httpx.post(
        url,
        headers=headers,
        content=video_bytes,
        timeout=300
    )

    if response.status_code >= 300:
        raise Exception(f"Upload failed: {response.text}")

    public_url = f"{SUPABASE_URL}/storage/v1/object/public/creative-media/{path}"

    return public_url

# =========================
# DOWNLOAD IMAGE
# =========================

def download_image_bytes(image_url):

    response = requests.get(image_url, timeout=60)

    if response.status_code != 200:
        raise Exception("Failed to download image")

    return response.content

# =========================
# GENERATE VIDEO
# =========================

@app.post("/generate-video")
def generate_video(req: GenerateVideoRequest):

    generation_id = req.generation_id

    try:

        print("Starting Veo 3.1 generation:", generation_id)

        supabase_update(generation_id, {
            "status": "processing"
        })

        # download image
        print("Downloading image...")
        image_bytes = download_image_bytes(req.image_url)

        # create Image object
        veo_image = Image.from_bytes(image_bytes)

        # init model
        print("Calling Veo 3.1 via Vertex AI...")
        model = GenerativeModel("veo-3.1-generate-preview")

        full_prompt = req.prompt

        if req.script_text:
            full_prompt += "\n\nSpeech:\n" + req.script_text

        operation = model.generate_content(
            contents=[full_prompt, veo_image],
            generation_config={
                "video_config": {
                    "duration_seconds": req.duration,
                    "aspect_ratio": req.aspect_ratio
                }
            },
            stream=False
        )

        video_part = None

        for part in operation.candidates[0].content.parts:
            if hasattr(part, "file_data"):
                video_part = part.file_data

        if not video_part:
            raise Exception("No video returned")

        video_uri = video_part.file_uri

        print("Video URI:", video_uri)

        # download video
        print("Downloading video file...")
        video_response = httpx.get(video_uri, timeout=600)

        video_bytes = video_response.content

        # upload to supabase
        print("Uploading to Supabase Storage...")
        public_url = supabase_upload_video(
            generation_id,
            video_bytes
        )

        print("Updating Supabase row...")

        supabase_update(generation_id, {
            "status": "completed",
            "video_url": public_url,
            "final_video_url": public_url
        })

        print("Completed successfully")

        return {
            "status": "completed",
            "video_url": public_url
        }

    except Exception as e:

        print("ERROR:", str(e))

        try:
            supabase_update(generation_id, {
                "status": "failed"
            })
        except:
            pass

        raise HTTPException(status_code=500, detail=str(e))
