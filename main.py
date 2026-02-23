import os
import time
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from google import genai
from google.genai import types

# =====================================================
# ENVIRONMENT VARIABLES
# =====================================================

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_SERVICE_ROLE_KEY = os.environ["SUPABASE_SERVICE_ROLE_KEY"]

# Vertex AI auth via Service Account JSON
# IMPORTANT: this must point to the JSON file path
GOOGLE_APPLICATION_CREDENTIALS = os.environ.get(
    "GOOGLE_APPLICATION_CREDENTIALS_JSON"
)

if GOOGLE_APPLICATION_CREDENTIALS:
    path = "/tmp/google_credentials.json"
    with open(path, "w") as f:
        f.write(GOOGLE_APPLICATION_CREDENTIALS)
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = path

# Force Vertex AI mode
os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "true"

# Create client
client = genai.Client()

# =====================================================
# FASTAPI
# =====================================================

app = FastAPI()


class GenerateVideoRequest(BaseModel):
    generation_id: str
    image_url: str
    script_text: str
    prompt: str
    aspect_ratio: str
    duration: int


# =====================================================
# SUPABASE UPDATE
# =====================================================

def update_generation(generation_id, status, final_video_url=None):

    url = f"{SUPABASE_URL}/rest/v1/video_generations?id=eq.{generation_id}"

    payload = {
        "status": status
    }

    if final_video_url:
        payload["final_video_url"] = final_video_url

    headers = {
        "apikey": SUPABASE_SERVICE_ROLE_KEY,
        "Authorization": f"Bearer {SUPABASE_SERVICE_ROLE_KEY}",
        "Content-Type": "application/json",
        "Prefer": "return=minimal"
    }

    res = requests.patch(url, json=payload, headers=headers)

    if res.status_code >= 300:
        print("Supabase update failed:", res.text)


# =====================================================
# DOWNLOAD IMAGE
# =====================================================

def download_image_bytes(image_url):

    res = requests.get(image_url)

    if res.status_code != 200:
        raise Exception("Failed to download image")

    return res.content


# =====================================================
# GENERATE VIDEO ENDPOINT
# =====================================================

@app.post("/generate-video")
def generate_video(req: GenerateVideoRequest):

    generation_id = req.generation_id

    try:

        print(f"Starting Veo 3.1 generation: {generation_id}")

        update_generation(generation_id, "processing")

        # Download image
        print("Downloading image...")
        image_bytes = download_image_bytes(req.image_url)

        mime_type = "image/jpeg"

        full_prompt = f"""
{req.prompt}

Script:
{req.script_text}

Create photorealistic talking avatar video.
Natural facial motion.
Accurate lip sync.
"""

        print("Calling Veo 3.1 via Vertex AI...")

        operation = client.models.generate_videos(
            model="veo-3.1-generate-preview",
            prompt=full_prompt,
            image=types.Image(
                image_bytes=image_bytes,
                mime_type=mime_type
            ),
            config=types.GenerateVideosConfig(
                aspect_ratio=req.aspect_ratio
            )
        )

        print("Waiting for completion...")

        while not operation.done:

            time.sleep(10)

            operation = client.operations.get(operation)

        if not operation.response.generated_videos:
            raise Exception("No video returned")

        video = operation.response.generated_videos[0]

        video_url = video.video.uri

        print("Video ready:", video_url)

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
