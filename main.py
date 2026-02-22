import os
import time
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from google import genai

# ========================
# ENV
# ========================

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY") or os.environ.get("VEO3_API_KEY")

if not SUPABASE_URL:
    raise Exception("SUPABASE_URL missing")

if not SUPABASE_SERVICE_ROLE_KEY:
    raise Exception("SUPABASE_SERVICE_ROLE_KEY missing")

if not GEMINI_API_KEY:
    raise Exception("GEMINI_API_KEY missing")

client = genai.Client(api_key=GEMINI_API_KEY)

# ========================
# FASTAPI
# ========================

app = FastAPI()

class GenerateVideoRequest(BaseModel):
    generation_id: str
    image_url: str
    script_text: str
    prompt: str
    aspect_ratio: str
    duration: int


# ========================
# SUPABASE UPDATE
# ========================

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


# ========================
# GENERATE VIDEO
# ========================

@app.post("/generate-video")
def generate_video(req: GenerateVideoRequest):

    generation_id = req.generation_id

    try:

        print("Starting Veo 3.1 generation:", generation_id)

        update_generation(generation_id, "processing")

        # ========================
        # DOWNLOAD IMAGE
        # ========================

        print("Downloading image...")

        img_res = requests.get(req.image_url)

        if img_res.status_code != 200:
            raise Exception("Image download failed")

        local_image_path = f"/tmp/{generation_id}.jpg"

        with open(local_image_path, "wb") as f:
            f.write(img_res.content)

        # ========================
        # UPLOAD IMAGE TO GEMINI FILE API
        # ========================

        print("Uploading image to Gemini Files API...")

        uploaded_file = client.files.upload(file=local_image_path)

        # ========================
        # BUILD PROMPT STRING
        # ========================

        full_prompt = f"""
{req.prompt}

Speech:
{req.script_text}

Photorealistic talking avatar. Natural lip sync.
"""

        # ========================
        # CALL VEO 3.1
        # ========================

        print("Generating video via Veo 3.1...")

        operation = client.models.generate_videos(
            model="veo-3.1-generate-preview",
            prompt=full_prompt,
            image=uploaded_file
        )

        print("Operation started:", operation.name)

        # ========================
        # POLL UNTIL COMPLETE
        # ========================

        while not operation.done:

            print("Waiting...")
            time.sleep(10)

            operation = client.operations.get(operation.name)

        if not operation.response.generated_videos:
            raise Exception("No video returned")

        video_file = operation.response.generated_videos[0].video

        # ========================
        # DOWNLOAD VIDEO
        # ========================

        local_video_path = f"/tmp/{generation_id}.mp4"

        print("Downloading video...")

        client.files.download(
            file=video_file,
            path=local_video_path
        )

        # ========================
        # UPLOAD TO SUPABASE STORAGE
        # ========================

        print("Uploading to Supabase Storage...")

        storage_url = f"{SUPABASE_URL}/storage/v1/object/creative-media/video-final/{generation_id}.mp4"

        headers = {
            "Authorization": f"Bearer {SUPABASE_SERVICE_ROLE_KEY}",
            "apikey": SUPABASE_SERVICE_ROLE_KEY,
            "Content-Type": "video/mp4"
        }

        with open(local_video_path, "rb") as f:
            upload_res = requests.post(storage_url, headers=headers, data=f)

        if upload_res.status_code >= 300:
            raise Exception(upload_res.text)

        public_url = f"{SUPABASE_URL}/storage/v1/object/public/creative-media/video-final/{generation_id}.mp4"

        print("SUCCESS:", public_url)

        update_generation(generation_id, "completed", public_url)

        return {
            "status": "completed",
            "video_url": public_url
        }

    except Exception as e:

        print("ERROR:", str(e))

        update_generation(generation_id, "failed")

        raise HTTPException(status_code=500, detail=str(e))
