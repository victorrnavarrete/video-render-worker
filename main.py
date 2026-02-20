import os
import time
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from google import genai
from google.genai import types

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_SERVICE_ROLE_KEY = os.environ["SUPABASE_SERVICE_ROLE_KEY"]
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY") or os.environ.get("VEO3_API_KEY")

if not SUPABASE_URL.startswith("http"):
    raise Exception("SUPABASE_URL must start with https://")

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

    res = requests.patch(url, json=payload, headers=headers)

    if res.status_code >= 300:
        raise Exception(f"Supabase update failed: {res.text}")


@app.get("/")
def health():
    return {"status": "ok"}


@app.post("/generate-video")
def generate_video(req: GenerateVideoRequest):

    generation_id = req.generation_id

    try:

        print("Starting Veo 3.1 generation:", generation_id)

        update_generation(generation_id, "processing")

        # STEP 1: download image
        print("Downloading image...")
        image_response = requests.get(req.image_url)

        if image_response.status_code != 200:
            raise Exception("Failed to download image")

        image_bytes = image_response.content

        # STEP 2: upload to Gemini Files API
        print("Uploading image to Gemini Files API...")

        uploaded_file = client.files.upload(
            file=image_bytes,
            mime_type="image/jpeg"
        )

        # STEP 3: build prompt string ONLY
        final_prompt = f"{req.prompt}\n\nSpeech:\n{req.script_text}"

        print("Generating video via Veo 3.1...")

        operation = client.models.generate_videos(
            model="veo-3.1-generate-preview",
            prompt=final_prompt,
            image=uploaded_file,
            config=types.GenerateVideosConfig(
                aspect_ratio=req.aspect_ratio
            )
        )

        # STEP 4: poll operation
        while not operation.done:

            print("Waiting Veo 3.1...")
            time.sleep(10)

            operation = client.operations.get(operation)

        if not operation.response.generated_videos:
            raise Exception("No video generated")

        video_file = operation.response.generated_videos[0].video

        # STEP 5: download video
        local_path = f"/tmp/{generation_id}.mp4"

        client.files.download(
            file=video_file,
            path=local_path
        )

        # STEP 6: upload to Supabase Storage
        print("Uploading final video to Supabase...")

        storage_url = f"{SUPABASE_URL}/storage/v1/object/creative-media/video-final/{generation_id}.mp4"

        with open(local_path, "rb") as f:

            upload = requests.post(
                storage_url,
                headers={
                    "Authorization": f"Bearer {SUPABASE_SERVICE_ROLE_KEY}",
                    "Content-Type": "video/mp4"
                },
                data=f
            )

        if upload.status_code not in [200, 201]:
            raise Exception(upload.text)

        public_url = f"{SUPABASE_URL}/storage/v1/object/public/creative-media/video-final/{generation_id}.mp4"

        update_generation(generation_id, "completed", public_url)

        print("SUCCESS:", public_url)

        return {"video_url": public_url}

    except Exception as e:

        print("ERROR:", str(e))

        update_generation(generation_id, "failed")

        raise HTTPException(status_code=500, detail=str(e))
