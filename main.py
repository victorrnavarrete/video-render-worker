import os
import requests
import httpx
import vertexai

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from vertexai.preview.vision_models import VideoGenerationModel, Image

# ENV
PROJECT_ID = os.environ["GOOGLE_CLOUD_PROJECT"]
LOCATION = os.environ["GOOGLE_CLOUD_LOCATION"]

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_SERVICE_ROLE_KEY"]

# INIT VERTEX
vertexai.init(
    project=PROJECT_ID,
    location=LOCATION,
)

# FASTAPI
app = FastAPI()

class GenerateVideoRequest(BaseModel):
    generation_id: str
    image_url: str
    prompt: str


def update_supabase(generation_id, data):

    url = f"{SUPABASE_URL}/rest/v1/video_generations?id=eq.{generation_id}"

    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
        "Prefer": "return=minimal",
    }

    httpx.patch(url, headers=headers, json=data)


@app.post("/generate-video")
async def generate_video(req: GenerateVideoRequest):

    generation_id = req.generation_id

    try:

        print("Starting Veo 3.1 generation:", generation_id)

        update_supabase(generation_id, {"status": "processing"})

        # DOWNLOAD IMAGE
        print("Downloading image...")
        image_bytes = requests.get(req.image_url).content

        veo_image = Image.from_bytes(image_bytes)

        print("Loading Veo model...")

        model = VideoGenerationModel.from_pretrained(
            "veo-3.1-generate-preview"
        )

        print("Generating video...")

        operation = model.generate_video(
            image=veo_image,
            prompt=req.prompt,
            aspect_ratio="9:16",
            duration_seconds=5,
        )

        print("Waiting result...")

        video = operation.result()

        video_bytes = video.video_bytes

        file_path = f"/tmp/{generation_id}.mp4"

        with open(file_path, "wb") as f:
            f.write(video_bytes)

        print("Uploading to Supabase Storage...")

        upload_url = f"{SUPABASE_URL}/storage/v1/object/videos/{generation_id}.mp4"

        headers = {
            "apikey": SUPABASE_KEY,
            "Authorization": f"Bearer {SUPABASE_KEY}",
            "Content-Type": "video/mp4",
        }

        with open(file_path, "rb") as f:
            httpx.post(upload_url, headers=headers, content=f.read())

        public_url = f"{SUPABASE_URL}/storage/v1/object/public/videos/{generation_id}.mp4"

        update_supabase(
            generation_id,
            {
                "status": "completed",
                "video_url": public_url,
                "final_video_url": public_url,
            },
        )

        print("SUCCESS")

        return {"success": True}

    except Exception as e:

        print("ERROR:", str(e))

        update_supabase(
            generation_id,
            {
                "status": "failed",
                "error": str(e),
            },
        )

        raise HTTPException(status_code=500, detail=str(e))
