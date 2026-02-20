import os
import time
import tempfile
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from google import genai
from google.genai import types

# ENV
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY or not GEMINI_API_KEY:
    raise Exception("Missing required environment variables")

# INIT CLIENT
genai_client = genai.Client(api_key=GEMINI_API_KEY)

app = FastAPI()


class GenerateVideoRequest(BaseModel):
    generation_id: str
    image_url: str
    script_text: str
    prompt: str
    aspect_ratio: str
    duration: int


# -------------------------
# Supabase helpers
# -------------------------

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


def upload_to_supabase(generation_id, file_path):

    storage_path = f"video-final/{generation_id}.mp4"

    upload_url = f"{SUPABASE_URL}/storage/v1/object/creative-media/{storage_path}"

    headers = {
        "Authorization": f"Bearer {SUPABASE_SERVICE_ROLE_KEY}",
        "Content-Type": "video/mp4"
    }

    with open(file_path, "rb") as f:
        res = requests.post(upload_url, headers=headers, data=f)

    if res.status_code not in [200, 201]:
        raise Exception(f"Supabase upload failed: {res.text}")

    public_url = f"{SUPABASE_URL}/storage/v1/object/public/creative-media/{storage_path}"

    return public_url


# -------------------------
# Veo generation endpoint
# -------------------------

@app.post("/generate-video")
def generate_video(req: GenerateVideoRequest):

    generation_id = req.generation_id

    try:

        print("Starting Veo 3.1 generation:", generation_id)

        update_generation(generation_id, "processing")

        # Download image locally
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_image:

            img_res = requests.get(req.image_url)

            if img_res.status_code != 200:
                raise Exception("Failed to download image")

            tmp_image.write(img_res.content)

            image_path = tmp_image.name

        print("Uploading image to Gemini Files API...")

        uploaded_file = genai_client.files.upload(file=image_path)

        print("Generating video via Veo 3.1...")

        operation = genai_client.models.generate_videos(
    model="veo-3.1-generate-preview",
    prompt=req.prompt,
    image=uploaded_file,
    config=types.GenerateVideosConfig(
        aspect_ratio=req.aspect_ratio
    )
)

        print("Waiting for generation...")

        while not operation.done:
            time.sleep(10)
            operation = genai_client.operations.get(operation.name)

        if not operation.response.generated_videos:
            raise Exception("No video generated")

        generated_video = operation.response.generated_videos[0]

        # Download video locally
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_video:

            genai_client.files.download(
                file=generated_video.video,
                path=tmp_video.name
            )

            video_path = tmp_video.name

        print("Uploading video to Supabase...")

        final_url = upload_to_supabase(generation_id, video_path)

        update_generation(generation_id, "completed", final_url)

        print("Completed:", final_url)

        return {
            "status": "completed",
            "video_url": final_url
        }

    except Exception as e:

        print("ERROR:", str(e))

        update_generation(generation_id, "failed")

        raise HTTPException(status_code=500, detail=str(e))
