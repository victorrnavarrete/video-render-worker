import os
import time
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import vertexai
from vertexai.preview.vision_models import VideoGenerationModel


# =========================
# ENV
# =========================

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")

PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT")
LOCATION = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")

if not PROJECT_ID:
    raise Exception("GOOGLE_CLOUD_PROJECT not set")

vertexai.init(
    project=PROJECT_ID,
    location=LOCATION
)

app = FastAPI()


# =========================
# REQUEST
# =========================

class GenerateVideoRequest(BaseModel):
    generation_id: str
    image_url: str
    script_text: str
    prompt: str
    aspect_ratio: str
    duration: int


# =========================
# SUPABASE UPDATE
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
        "apikey": SUPABASE_SERVICE_ROLE_KEY,
        "Authorization": f"Bearer {SUPABASE_SERVICE_ROLE_KEY}",
        "Content-Type": "application/json",
        "Prefer": "return=minimal"
    }

    requests.patch(url, json=payload, headers=headers)


# =========================
# DOWNLOAD IMAGE
# =========================

def download_image(image_url):

    print("Downloading image...")

    res = requests.get(image_url)

    if res.status_code != 200:
        raise Exception("Failed to download image")

    path = "/tmp/input.jpg"

    with open(path, "wb") as f:
        f.write(res.content)

    return path


# =========================
# GENERATE VIDEO
# =========================

@app.post("/generate-video")
def generate_video(req: GenerateVideoRequest):

    generation_id = req.generation_id

    try:

        print("Starting Veo 3.1 generation:", generation_id)

        update_generation(generation_id, "processing")

        image_path = download_image(req.image_url)

        print("Loading Veo model...")

        model = VideoGenerationModel.from_pretrained(
            "veo-3.1-generate-preview"
        )

        full_prompt = f"""
{req.prompt}

Script:
{req.script_text}

Requirements:
perfect lip sync,
realistic talking avatar,
cinematic realism
"""

        print("Calling Veo...")

        operation = model.generate_video(
            prompt=full_prompt,
            image=image_path,
            aspect_ratio=req.aspect_ratio,
            duration_seconds=req.duration
        )

        print("Polling Veo...")

        video = operation.result()

        video_uri = video.uri

        print("Video URI:", video_uri)

        update_generation(
            generation_id,
            "completed",
            video_uri
        )

        return {
            "status": "completed",
            "video_url": video_uri
        }

    except Exception as e:

        print("ERROR:", str(e))

        update_generation(generation_id, "failed")

        raise HTTPException(status_code=500, detail=str(e))
