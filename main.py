import os
import time
import base64
import requests

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from google import genai
from google.genai import types

# ENV VARIABLES
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
GEMINI_API_KEY = os.environ.get("VEO3_API_KEY") or os.environ.get("GEMINI_API_KEY")

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

    res = requests.patch(url, json=payload, headers=headers)

    if res.status_code >= 300:
        print("Supabase update error:", res.text)


def upload_video_to_supabase(generation_id, video_bytes):

    storage_url = f"{SUPABASE_URL}/storage/v1/object/creative-media/video-final/{generation_id}.mp4"

    headers = {
        "Authorization": f"Bearer {SUPABASE_SERVICE_ROLE_KEY}",
        "Content-Type": "video/mp4",
        "x-upsert": "true",
    }

    res = requests.post(storage_url, headers=headers, data=video_bytes)

    if res.status_code >= 300:
        raise Exception(res.text)

    public_url = f"{SUPABASE_URL}/storage/v1/object/public/creative-media/video-final/{generation_id}.mp4"

    return public_url


def download_image_as_base64(image_url):

    print("Downloading image...")

    res = requests.get(image_url)

    if res.status_code != 200:
        raise Exception("Failed to download image")

    mime = res.headers.get("Content-Type", "image/jpeg")

    base64_bytes = base64.b64encode(res.content).decode("utf-8")

    return base64_bytes, mime


@app.post("/generate-video")
def generate_video(req: GenerateVideoRequest):

    try:

        generation_id = req.generation_id

        print("Starting Veo 3.1 generation:", generation_id)

        update_generation(generation_id, "processing")

        # DOWNLOAD IMAGE
        image_base64, mime_type = download_image_as_base64(req.image_url)

        # GENERATE VIDEO
        print("Calling Veo 3.1...")

        operation = genai_client.models.generate_videos(
            model="veo-3.1-generate-preview",
            prompt=req.prompt,
            image={
                "bytesBase64Encoded": image_base64,
                "mimeType": mime_type
            },
            config=types.GenerateVideosConfig(
                aspect_ratio=req.aspect_ratio
            ),
        )

        print("Polling operation...")

        while not operation.done:

            time.sleep(10)

            operation = genai_client.operations.get(operation)

            print("Still processing...")

        if not operation.response or not operation.response.generated_videos:
            raise Exception("No video returned")

        video_uri = operation.response.generated_videos[0].video.uri

        print("Downloading video from Gemini...")

        video_bytes = genai_client.files.download(video_uri)

        print("Uploading to Supabase...")

        final_url = upload_video_to_supabase(
            generation_id,
            video_bytes,
        )

        print("Completed:", final_url)

        update_generation(
            generation_id,
            "completed",
            final_url,
        )

        return {
            "status": "completed",
            "video_url": final_url,
        }

    except Exception as e:

        print("ERROR:", str(e))

        update_generation(generation_id, "failed")

        raise HTTPException(
            status_code=500,
            detail=str(e),
        )
