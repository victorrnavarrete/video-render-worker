import os
import time
import requests
import tempfile

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import vertexai
from vertexai.generative_models import GenerativeModel, Part, Image


# =========================
# ENV VARIABLES
# =========================

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")

GOOGLE_CLOUD_PROJECT = os.environ.get("GOOGLE_CLOUD_PROJECT")
GOOGLE_CLOUD_LOCATION = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")

if not SUPABASE_URL:
    raise Exception("SUPABASE_URL not set")

if not SUPABASE_SERVICE_ROLE_KEY:
    raise Exception("SUPABASE_SERVICE_ROLE_KEY not set")

if not GOOGLE_CLOUD_PROJECT:
    raise Exception("GOOGLE_CLOUD_PROJECT not set")


# =========================
# INIT VERTEX AI
# =========================

vertexai.init(
    project=GOOGLE_CLOUD_PROJECT,
    location=GOOGLE_CLOUD_LOCATION
)

model = GenerativeModel("veo-3.1-generate-preview")


# =========================
# FASTAPI
# =========================

app = FastAPI()


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

    res = requests.patch(url, json=payload, headers=headers)

    if res.status_code >= 300:
        print("Supabase update error:", res.text)


# =========================
# DOWNLOAD IMAGE
# =========================

def download_image(image_url):

    print("Downloading image...")

    res = requests.get(image_url)

    if res.status_code != 200:
        raise Exception("Failed to download image")

    return res.content


# =========================
# GENERATE VIDEO
# =========================

@app.post("/generate-video")
def generate_video(req: GenerateVideoRequest):

    generation_id = req.generation_id

    try:

        print("Starting Veo 3.1 generation:", generation_id)

        update_generation(generation_id, "processing")

        # download image
        image_bytes = download_image(req.image_url)

        print("Creating Veo Image object...")

        veo_image = Image.from_bytes(image_bytes)

        full_prompt = f"""
{req.prompt}

Script:
{req.script_text}

Requirements:
- talking avatar
- perfect lipsync
- photorealistic
- cinematic lighting
"""

        print("Calling Veo 3.1 via Vertex AI...")

        operation = model.generate_videos(
            contents=[
                veo_image,
                full_prompt
            ]
        )

        print("Waiting for completion...")

        while not operation.done:

            time.sleep(10)
            operation = operation.result()

        response = operation.response

        if not response.generated_videos:
            raise Exception("No video generated")

        video_uri = response.generated_videos[0].video.uri

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


# =========================
# HEALTH
# =========================

@app.get("/")
def health():
    return {"status": "ok"}
