import os
import time
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from google import genai
from google.genai import types

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY") or os.environ.get("VEO3_API_KEY")

app = FastAPI()

client = genai.Client(api_key=GEMINI_API_KEY)

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
        "Prefer": "return=minimal"
    }

    res = requests.patch(url, json=payload, headers=headers)

    if res.status_code >= 300:
        raise Exception(res.text)


@app.post("/generate-video")
def generate_video(req: GenerateVideoRequest):

    try:

        generation_id = req.generation_id

        print("Starting Veo 3.1 generation:", generation_id)

        update_generation(generation_id, "processing")

        # Download image locally
        print("Downloading image...")

        image_bytes = requests.get(req.image_url).content

        with open("/tmp/input.jpg", "wb") as f:
            f.write(image_bytes)

        # Upload to Gemini Files API
        print("Uploading image to Gemini Files API...")

        uploaded_file = client.files.upload(file="/tmp/input.jpg")

        motion_prompt = f"{req.prompt}\n\n{req.script_text}"

        print("Generating video via Veo 3.1...")

        operation = client.models.generate_videos(
            model="veo-3.1-generate-preview",
            prompt=[uploaded_file, motion_prompt],
            config=types.GenerateVideosConfig(
                aspect_ratio=req.aspect_ratio
            )
        )

        print("Polling Veo operation...")

        while not operation.done:

            time.sleep(10)

            operation = client.operations.get(operation.name)

        if not operation.response.generated_videos:

            raise Exception("No video returned")

        video = operation.response.generated_videos[0]

        print("Downloading video...")

        client.files.download(
            file=video.video,
            path="/tmp/output.mp4"
        )

        # Upload to Supabase Storage
        print("Uploading video to Supabase...")

        storage_url = f"{SUPABASE_URL}/storage/v1/object/creative-media/video-final/{generation_id}.mp4"

        headers = {
            "Authorization": f"Bearer {SUPABASE_SERVICE_ROLE_KEY}",
            "Content-Type": "video/mp4"
        }

        with open("/tmp/output.mp4", "rb") as f:

            res = requests.put(storage_url, headers=headers, data=f)

        if res.status_code >= 300:

            raise Exception(res.text)

        public_url = f"{SUPABASE_URL}/storage/v1/object/public/creative-media/video-final/{generation_id}.mp4"

        update_generation(generation_id, "completed", public_url)

        return {
            "status": "completed",
            "video_url": public_url
        }

    except Exception as e:

        print("ERROR:", str(e))

        update_generation(req.generation_id, "failed")

        raise HTTPException(status_code=500, detail=str(e))
