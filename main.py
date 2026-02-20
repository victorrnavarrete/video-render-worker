import os
import time
import requests

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from google import genai
from google.genai import types

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
GEMINI_API_KEY = os.environ.get("VEO3_API_KEY") or os.environ.get("GEMINI_API_KEY")

if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY or not GEMINI_API_KEY:
    raise Exception("Missing SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY, or GEMINI_API_KEY")

client = genai.Client(api_key=GEMINI_API_KEY)

app = FastAPI()


class GenerateVideoRequest(BaseModel):
    generation_id: str
    image_url: str
    script_text: str
    prompt: str
    aspect_ratio: str
    duration: int


def update_generation(generation_id, status, final_video_url=None):

    url = f"{SUPABASE_URL}/rest/v1/video_generations?id=eq.{generation_id}"

    payload = {"status": status}

    if final_video_url:
        payload["final_video_url"] = final_video_url

    headers = {
        "apikey": SUPABASE_SERVICE_ROLE_KEY,
        "Authorization": f"Bearer {SUPABASE_SERVICE_ROLE_KEY}",
        "Content-Type": "application/json",
        "Prefer": "return=minimal",
    }

    requests.patch(url, json=payload, headers=headers)


def upload_video_to_supabase(generation_id, video_bytes):

    storage_path = f"video-final/{generation_id}.mp4"

    url = f"{SUPABASE_URL}/storage/v1/object/creative-media/{storage_path}"

    headers = {
        "Authorization": f"Bearer {SUPABASE_SERVICE_ROLE_KEY}",
        "Content-Type": "video/mp4",
        "x-upsert": "true",
    }

    res = requests.post(url, headers=headers, data=video_bytes)

    if res.status_code >= 300:
        raise Exception(res.text)

    return f"{SUPABASE_URL}/storage/v1/object/public/creative-media/{storage_path}"


def download_image(image_url):

    res = requests.get(image_url)

    if res.status_code != 200:
        raise Exception("Failed to download image")

    temp_path = "/tmp/input_image.jpg"

    with open(temp_path, "wb") as f:
        f.write(res.content)

    return temp_path


@app.post("/generate-video")
def generate_video(req: GenerateVideoRequest):

    try:

        generation_id = req.generation_id

        print("Starting Veo 3.1 generation:", generation_id)

        update_generation(generation_id, "processing")

        # download image locally
        image_path = download_image(req.image_url)

        print("Uploading image to Gemini Files API...")

        uploaded_file = client.files.upload(file=image_path)

        print("Generating video via Veo 3.1...")

        operation = client.models.generate_videos(
            model="veo-3.1-generate-preview",
            prompt=[uploaded_file, req.prompt],
            config=types.GenerateVideosConfig(
                aspect_ratio=req.aspect_ratio
            ),
        )

        print("Polling Veo operation...")

        while not operation.done:

            time.sleep(10)

            operation = client.operations.get(operation)

            print("Still processing...")

        video_file = operation.response.generated_videos[0].video

        print("Downloading video...")

        video_bytes = client.files.download(video_file)

        print("Uploading to Supabase...")

        final_url = upload_video_to_supabase(
            generation_id,
            video_bytes
        )

        update_generation(
            generation_id,
            "completed",
            final_url
        )

        print("Completed:", final_url)

        return {
            "status": "completed",
            "video_url": final_url
        }

    except Exception as e:

        print("ERROR:", str(e))

        update_generation(generation_id, "failed")

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
