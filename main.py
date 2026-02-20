import os
import time
import base64
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from google import genai
from google.genai import types

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY") or os.environ.get("VEO3_API_KEY")

if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY or not GEMINI_API_KEY:
    raise Exception("Missing required environment variables")

client = genai.Client(api_key=GEMINI_API_KEY)

app = FastAPI()


class GenerateVideoRequest(BaseModel):
    generation_id: str
    image_url: str
    script_text: str | None = ""
    prompt: str
    aspect_ratio: str = "9:16"
    duration: int = 5


def update_generation(generation_id, status, final_video_url=None):

    url = f"{SUPABASE_URL}/rest/v1/video_generations?id=eq.{generation_id}"

    payload = {"status": status}

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
        raise Exception(res.text)


def download_image_bytes(url):

    res = requests.get(url, timeout=60)

    if res.status_code != 200:
        raise Exception("Failed to download image")

    return res.content


def detect_mime(image_bytes):

    if image_bytes[:3] == b"\xff\xd8\xff":
        return "image/jpeg"

    if image_bytes[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"

    return "image/jpeg"


@app.post("/generate-video")
def generate_video(req: GenerateVideoRequest):

    generation_id = req.generation_id

    try:

        print(f"Starting Veo 3.1 generation: {generation_id}")

        update_generation(generation_id, "processing")

        print("Downloading image...")
        image_bytes = download_image_bytes(req.image_url)

        mime_type = detect_mime(image_bytes)

        print("Creating Veo Image object...")

        veo_image = types.Image.from_bytes(
            data=image_bytes,
            mime_type=mime_type
        )

        full_prompt = req.prompt

        if req.script_text:
            full_prompt += f"\n\nCharacter speaking naturally in Brazilian Portuguese:\n{req.script_text}"

        print("Generating video via Veo 3.1...")

        operation = client.models.generate_videos(
            model="veo-3.1-generate-preview",
            prompt=full_prompt,
            image=veo_image,
            config=types.GenerateVideosConfig(
                aspect_ratio=req.aspect_ratio
            )
        )

        print("Waiting for Veo completion...")

        while not operation.done:
            time.sleep(10)
            operation = client.operations.get(operation)

        if not operation.response or not operation.response.generated_videos:
            raise Exception("No video generated")

        video_file = operation.response.generated_videos[0].video

        local_path = f"/tmp/{generation_id}.mp4"

        client.files.download(
            file=video_file,
            path=local_path
        )

        print("Uploading video to Supabase Storage...")

        storage_url = f"{SUPABASE_URL}/storage/v1/object/creative-media/video-final/{generation_id}.mp4"

        headers = {
            "Authorization": f"Bearer {SUPABASE_SERVICE_ROLE_KEY}",
            "apikey": SUPABASE_SERVICE_ROLE_KEY,
            "Content-Type": "video/mp4",
            "x-upsert": "true"
        }

        with open(local_path, "rb") as f:

            upload_res = requests.post(
                storage_url,
                headers=headers,
                data=f
            )

        if upload_res.status_code >= 300:
            raise Exception(upload_res.text)

        public_url = f"{SUPABASE_URL}/storage/v1/object/public/creative-media/video-final/{generation_id}.mp4"

        update_generation(
            generation_id,
            "completed",
            public_url
        )

        print("SUCCESS:", public_url)

        return {
            "status": "completed",
            "video_url": public_url
        }

    except Exception as e:

        print("ERROR:", str(e))

        try:
            update_generation(generation_id, "failed")
        except:
            pass

        raise HTTPException(status_code=500, detail=str(e))
