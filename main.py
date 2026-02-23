import os
import time
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import vertexai
from vertexai.preview.generative_models import GenerativeModel

from google.oauth2 import service_account

# =========================
# ENV VARIABLES
# =========================

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")

PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT")
LOCATION = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")

# =========================
# INIT VERTEX AI
# =========================

vertexai.init(
    project=PROJECT_ID,
    location=LOCATION
)

# =========================
# FASTAPI
# =========================

app = FastAPI()

# =========================
# REQUEST MODEL
# =========================

class GenerateVideoRequest(BaseModel):
    generation_id: str
    image_url: str
    prompt: str
    aspect_ratio: str = "9:16"
    duration: int = 4


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

    res = requests.patch(url, json=payload, headers=headers)

    if res.status_code >= 300:
        print("Supabase update failed:", res.text)


# =========================
# DOWNLOAD IMAGE
# =========================

def download_image_bytes(image_url):

    res = requests.get(image_url)

    if res.status_code != 200:
        raise Exception("Failed to download image")

    return res.content


# =========================
# GENERATE VIDEO ENDPOINT
# =========================

@app.post("/generate-video")
def generate_video(req: GenerateVideoRequest):

    generation_id = req.generation_id

    try:

        print("Starting Veo 3.1 generation:", generation_id)

        update_generation(generation_id, "processing")

        # =========================
        # DOWNLOAD IMAGE
        # =========================

        print("Downloading image...")
        image_bytes = download_image_bytes(req.image_url)

        # =========================
        # INIT MODEL
        # =========================

        model = GenerativeModel(
            "publishers/google/models/veo-3.1-generate-preview"
        )

        # =========================
        # GENERATE VIDEO OPERATION
        # =========================

        print("Calling Veo 3.1 via Vertex AI...")

        operation = model.generate_videos(
            prompt=req.prompt,
            image=image_bytes,
            aspect_ratio=req.aspect_ratio,
            duration_seconds=req.duration
        )

        # =========================
        # WAIT FOR COMPLETION
        # =========================

        print("Waiting for completion...")

        while not operation.done():
            time.sleep(5)
            operation = operation.refresh()

        response = operation.result()

        # =========================
        # EXTRACT VIDEO URL
        # =========================

        if not response.generated_videos:
            raise Exception("No video returned")

        video = response.generated_videos[0]

        video_url = video.uri

        print("Video ready:", video_url)

        # =========================
        # UPDATE SUPABASE
        # =========================

        update_generation(
            generation_id,
            "completed",
            video_url
        )

        return {
            "status": "completed",
            "video_url": video_url
        }

    except Exception as e:

        print("ERROR:", str(e))

        update_generation(generation_id, "failed")

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
