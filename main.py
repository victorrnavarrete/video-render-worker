import os
import time
import base64
import requests
import httpx
from fastapi import FastAPI, HTTPException

import vertexai
from vertexai.preview.generative_models import Image
from vertexai.preview.generative_models import ImageVideoGenerationModel


# ========================
# CONFIG
# ========================

PROJECT_ID = os.environ["GOOGLE_CLOUD_PROJECT"]
LOCATION = os.environ["GOOGLE_CLOUD_LOCATION"]

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_SERVICE_ROLE_KEY"]

app = FastAPI()


# ========================
# INIT VERTEX AI
# ========================

vertexai.init(
    project=PROJECT_ID,
    location=LOCATION
)


# ========================
# SUPABASE UPDATE
# ========================

async def update_supabase(generation_id, data):

    url = f"{SUPABASE_URL}/rest/v1/video_generations?id=eq.{generation_id}"

    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
        "Prefer": "return=minimal"
    }

    async with httpx.AsyncClient() as client:
        await client.patch(url, headers=headers, json=data)


# ========================
# DOWNLOAD IMAGE
# ========================

def download_image(image_url):

    response = requests.get(image_url)

    if response.status_code != 200:
        raise Exception("Failed to download image")

    return response.content


# ========================
# GENERATE VIDEO
# ========================

@app.post("/generate-video")
async def generate_video(payload: dict):

    generation_id = payload["generation_id"]
    image_url = payload["image_url"]
    prompt = payload.get("prompt", "")

    try:

        print(f"Starting Veo 3.1 generation: {generation_id}")

        await update_supabase(generation_id, {
            "status": "processing"
        })

        # download image
        print("Downloading image...")
        image_bytes = download_image(image_url)

        print("Creating Veo Image object...")
        image = Image.from_bytes(image_bytes)

        print("Loading Veo model...")
        model = ImageVideoGenerationModel.from_pretrained(
            "veo-3.1-generate-preview"
        )

        print("Generating video...")

        operation = model.generate_video(

            image=image,

            prompt=prompt,

            aspect_ratio="9:16",

            duration_seconds=8,

            fps=24,

            person_generation="allow_all"
        )

        print("Waiting for result...")

        result = operation.result()

        video = result.generated_videos[0]

        video_bytes = video.video_bytes

        # upload to supabase storage
        filename = f"{generation_id}.mp4"

        upload_url = f"{SUPABASE_URL}/storage/v1/object/videos/{filename}"

        headers = {
            "apikey": SUPABASE_KEY,
            "Authorization": f"Bearer {SUPABASE_KEY}",
            "Content-Type": "video/mp4"
        }

        upload_response = requests.post(
            upload_url,
            headers=headers,
            data=video_bytes
        )

        if upload_response.status_code not in [200, 201]:
            raise Exception(upload_response.text)

        public_url = f"{SUPABASE_URL}/storage/v1/object/public/videos/{filename}"

        print("Saving URL in Supabase...")

        await update_supabase(generation_id, {
            "status": "completed",
            "video_url": public_url,
            "final_video_url": public_url
        })

        return {"success": True}

    except Exception as e:

        print("ERROR:", str(e))

        await update_supabase(generation_id, {
            "status": "failed",
            "error": str(e)
        })

        raise HTTPException(status_code=500, detail=str(e))
