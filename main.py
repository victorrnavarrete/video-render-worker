import os
import time
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from google import genai
from google.genai import types

# ENV VARS
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY") or os.environ.get("VEO3_API_KEY")

if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY or not GEMINI_API_KEY:
    raise Exception("Missing required environment variables")

# INIT GEMINI CLIENT
client = genai.Client(api_key=GEMINI_API_KEY)

app = FastAPI()

# REQUEST MODEL
class GenerateVideoRequest(BaseModel):
    generation_id: str
    image_url: str
    script_text: str | None = None
    prompt: str
    aspect_ratio: str = "9:16"
    duration: int = 6


# UPDATE SUPABASE
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

    res = requests.patch(url, json=payload, headers=headers, timeout=30)

    if res.status_code not in [200, 204]:
        print("Supabase update failed:", res.text)


@app.get("/")
def health():
    return {"status": "ok"}


@app.post("/generate-video")
def generate_video(req: GenerateVideoRequest):

    generation_id = req.generation_id

    try:

        print(f"Starting Veo 3.1 generation: {generation_id}")

        update_generation(generation_id, "processing")

        # DOWNLOAD IMAGE
        print("Downloading image...")
        img_res = requests.get(req.image_url, timeout=60)
        img_res.raise_for_status()

        image_bytes = img_res.content

        # BUILD FINAL PROMPT
        final_prompt = req.prompt
        if req.script_text:
            final_prompt += f"\n\nThe person says in Portuguese:\n{req.script_text}"

        print("Generating video via Veo 3.1...")

        operation = client.models.generate_videos(
            model="veo-3.1-generate-preview",
            prompt=final_prompt,
            image=types.Image(
                bytes=image_bytes,
                mime_type="image/jpeg"
            ),
            config=types.GenerateVideosConfig(
                aspect_ratio=req.aspect_ratio
            )
        )

        # POLL OPERATION
        while not operation.done:

            print("Waiting for Veo operation...")
            time.sleep(10)

            operation = client.operations.get(operation.name)

        # GET RESULT
        if not operation.response or not operation.response.generated_videos:
            raise Exception("No video generated")

        video_file = operation.response.generated_videos[0].video

        print("Downloading generated video...")

        output_path = f"/tmp/{generation_id}.mp4"

        client.files.download(
            file=video_file,
            path=output_path
        )

        # UPLOAD TO SUPABASE STORAGE
        storage_url = f"{SUPABASE_URL}/storage/v1/object/video-final/{generation_id}.mp4"

        with open(output_path, "rb") as f:

            upload_headers = {
                "apikey": SUPABASE_SERVICE_ROLE_KEY,
                "Authorization": f"Bearer {SUPABASE_SERVICE_ROLE_KEY}",
                "Content-Type": "video/mp4"
            }

            upload_res = requests.post(
                storage_url,
                headers=upload_headers,
                data=f,
                timeout=120
            )

        if upload_res.status_code not in [200, 201]:
            raise Exception(upload_res.text)

        public_url = f"{SUPABASE_URL}/storage/v1/object/public/video-final/{generation_id}.mp4"

        update_generation(
            generation_id,
            "completed",
            public_url
        )

        print("Video generation completed:", public_url)

        return {
            "status": "completed",
            "video_url": public_url
        }

    except Exception as e:

        print("ERROR:", str(e))

        update_generation(generation_id, "failed")

        raise HTTPException(status_code=500, detail=str(e))
