import os
import time
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import vertexai
from vertexai.preview.generative_models import GenerativeModel, Image


# =========================
# ENV VARIABLES
# =========================

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")

PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT")
LOCATION = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")

if not PROJECT_ID:
    raise Exception("GOOGLE_CLOUD_PROJECT not set")

# Initialize Vertex AI
vertexai.init(
    project=PROJECT_ID,
    location=LOCATION
)

# =========================
# FASTAPI INIT
# =========================

app = FastAPI()


# =========================
# REQUEST MODEL
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

    res = requests.patch(url, json=payload, headers=headers)

    if res.status_code >= 300:
        print("Supabase update failed:", res.text)


# =========================
# DOWNLOAD IMAGE
# =========================

def download_image_bytes(image_url):

    print("Downloading image...")

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

        print(f"Starting Veo 3.1 generation: {generation_id}")

        update_generation(generation_id, "processing")

        image_bytes = download_image_bytes(req.image_url)

        print("Creating Veo Image object...")

        veo_image = Image.from_bytes(image_bytes)

        print("Initializing Veo model...")

        model = GenerativeModel(
            "projects/{}/locations/{}/publishers/google/models/veo-3.1-generate-preview".format(
                PROJECT_ID,
                LOCATION
            )
        )

        full_prompt = f"""
        {req.prompt}

        Script:
        {req.script_text}

        Requirements:
        - perfect lip sync
        - realistic talking avatar
        - cinematic realism
        - professional lighting
        """

        print("Calling Veo 3.1 via Vertex AI...")

        operation = model.generate_videos(
            prompt=full_prompt,
            image=veo_image,
            config={
                "aspect_ratio": req.aspect_ratio,
                "duration_seconds": req.duration
            }
        )

        print("Polling operation...")

        while not operation.done:

            print("Still processing...")
            time.sleep(10)

            operation = operation.refresh()

        print("Operation completed")

        if not operation.response:
            raise Exception("No response from Veo")

        video = operation.response.generated_videos[0]

        video_url = video.video.uri

        print("Video URL:", video_url)

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

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
