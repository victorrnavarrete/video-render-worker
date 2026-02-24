import os
import time
import requests
import httpx

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# =========================
# CREATE SERVICE ACCOUNT FILE AT RUNTIME
# =========================

if "GOOGLE_APPLICATION_CREDENTIALS_JSON" in os.environ:
    creds_path = "/tmp/service-account.json"
    with open(creds_path, "w") as f:
        f.write(os.environ["GOOGLE_APPLICATION_CREDENTIALS_JSON"])
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = creds_path

# =========================
# GOOGLE GENAI CLIENT (CORRECT SDK)
# =========================

from google import genai
from google.genai import types

client = genai.Client(vertexai=True)

# =========================
# ENV
# =========================

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_SERVICE_ROLE_KEY"]

# =========================
# FASTAPI
# =========================

app = FastAPI()

class GenerateVideoRequest(BaseModel):
    generation_id: str
    image_url: str
    prompt: str
    aspect_ratio: str
    duration: int
    script_text: str | None = None

# =========================
# SUPABASE UPDATE
# =========================

def update_generation(generation_id, status, video_url=None):

    url = f"{SUPABASE_URL}/rest/v1/video_generations?id=eq.{generation_id}"

    payload = {"status": status}

    if video_url:
        payload["video_url"] = video_url
        payload["final_video_url"] = video_url

    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
        "Prefer": "return=minimal"
    }

    httpx.patch(url, headers=headers, json=payload)

# =========================
# DOWNLOAD IMAGE
# =========================

def download_image(path="/tmp/input.jpg", url=None):

    r = requests.get(url)

    if r.status_code != 200:
        raise Exception("Image download failed")

    with open(path, "wb") as f:
        f.write(r.content)

    return path

# =========================
# GENERATE VIDEO
# =========================

@app.post("/generate-video")
def generate_video(req: GenerateVideoRequest):

    generation_id = req.generation_id

    try:

        print("Starting Veo 3.1 generation:", generation_id)

        update_generation(generation_id, "processing")

        # Download image locally
        print("Downloading image...")
        image_path = download_image(url=req.image_url)

        # Upload image to Gemini Files API
        print("Uploading image...")
        image_file = client.files.upload(file=image_path)

        # Build prompt
        prompt = req.prompt
        if req.script_text:
            prompt += "\n\nSpeech: " + req.script_text

        print("Generating video...")

        operation = client.models.generate_videos(
            model="veo-3.1-generate-preview",
            prompt=prompt,
            image=image_file,
            config=types.GenerateVideosConfig(
                aspect_ratio=req.aspect_ratio
            )
        )

        # Poll
        while not operation.done:

            time.sleep(10)
            operation = client.operations.get(operation)

        if not operation.response.generated_videos:
            raise Exception("No video returned")

        video = operation.response.generated_videos[0]

        video_path = f"/tmp/{generation_id}.mp4"

        print("Downloading video...")
        client.files.download(
            file=video.video,
            path=video_path
        )

        # Upload to Supabase Storage
        print("Uploading to Supabase...")

        with open(video_path, "rb") as f:

            video_bytes = f.read()

        storage_path = f"video-final/{generation_id}.mp4"

        upload_url = f"{SUPABASE_URL}/storage/v1/object/creative-media/{storage_path}"

        headers = {
            "apikey": SUPABASE_KEY,
            "Authorization": f"Bearer {SUPABASE_KEY}",
            "Content-Type": "video/mp4"
        }

        res = httpx.post(upload_url, headers=headers, content=video_bytes)

        if res.status_code not in [200, 201]:
            raise Exception(res.text)

        public_url = f"{SUPABASE_URL}/storage/v1/object/public/creative-media/{storage_path}"

        print("Completed:", public_url)

        update_generation(
            generation_id,
            "completed",
            public_url
        )

        return {"video_url": public_url}

    except Exception as e:

        print("ERROR:", str(e))

        update_generation(generation_id, "failed")

        raise HTTPException(status_code=500, detail=str(e))
