import os
import time
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from google import genai
from google.genai import types


# ENV VARS
SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_SERVICE_ROLE_KEY = os.environ["SUPABASE_SERVICE_ROLE_KEY"]
GEMINI_API_KEY = os.environ.get("VEO3_API_KEY") or os.environ.get("GEMINI_API_KEY")


# INIT CLIENT
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
        "Prefer": "return=minimal"
    }

    requests.patch(url, json=payload, headers=headers)


@app.post("/generate-video")
def generate_video(req: GenerateVideoRequest):

    generation_id = req.generation_id

    try:

        print("Starting Veo 3.1 generation:", generation_id)

        update_generation(generation_id, "processing")

        # STEP 1 — DOWNLOAD IMAGE
        print("Downloading image...")
        img_response = requests.get(req.image_url)

        if img_response.status_code != 200:
            raise Exception("Failed to download image")

        image_bytes = img_response.content

        # STEP 2 — CREATE IMAGE OBJECT (CORRECT WAY)
        print("Creating Veo Image object...")
        veo_image = types.Image(
            mime_type="image/jpeg",
            data=image_bytes
        )

        # STEP 3 — BUILD PROMPT
        full_prompt = f"""
{req.prompt}

Speech:
{req.script_text}

Photorealistic talking avatar.
Natural lip sync.
Realistic face, hands, body.
"""

        # STEP 4 — GENERATE VIDEO
        print("Generating video via Veo 3.1...")

        operation = client.models.generate_videos(
            model="veo-3.1-generate-preview",
            prompt=full_prompt,
            image=veo_image,
        )

        # STEP 5 — POLL
        print("Polling Veo operation...")

        while not operation.done:

            time.sleep(10)
            operation = client.operations.get(operation)

        if not operation.response.generated_videos:
            raise Exception("No video returned")

        video_file = operation.response.generated_videos[0].video

        # STEP 6 — DOWNLOAD VIDEO
        local_path = f"/tmp/{generation_id}.mp4"

        print("Downloading generated video...")
        client.files.download(file=video_file, path=local_path)

        # STEP 7 — UPLOAD TO SUPABASE
        print("Uploading to Supabase...")

        upload_url = f"{SUPABASE_URL}/storage/v1/object/creative-media/video-final/{generation_id}.mp4"

        with open(local_path, "rb") as f:

            upload_headers = {
                "Authorization": f"Bearer {SUPABASE_SERVICE_ROLE_KEY}",
                "Content-Type": "video/mp4"
            }

            upload_res = requests.post(
                upload_url,
                headers=upload_headers,
                data=f
            )

        if upload_res.status_code not in [200, 201]:
            raise Exception(upload_res.text)

        final_url = f"{SUPABASE_URL}/storage/v1/object/public/creative-media/video-final/{generation_id}.mp4"

        update_generation(generation_id, "completed", final_url)

        print("Generation completed:", final_url)

        return {
            "status": "completed",
            "video_url": final_url
        }

    except Exception as e:

        print("ERROR:", str(e))

        update_generation(generation_id, "failed")

        raise HTTPException(status_code=500, detail=str(e))
