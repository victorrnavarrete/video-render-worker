import os
import time
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import vertexai
from vertexai.preview.vision_models import TextToVideoGenerationModel

# =========================
# ENVIRONMENT
# =========================

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")

PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT")
LOCATION = os.environ.get("GOOGLE_CLOUD_LOCATION", "global")

# =========================
# INIT VERTEX AI
# =========================

vertexai.init(
    project=PROJECT_ID,
    location=LOCATION,
)

model = TextToVideoGenerationModel.from_pretrained(
    "projects/{}/locations/{}/publishers/google/models/veo-3.1-generate-preview".format(
        PROJECT_ID,
        LOCATION
    )
)

# =========================
# FASTAPI
# =========================

app = FastAPI()

# =========================
# REQUEST MODEL
# =========================

class GenerateVideoRequest(BaseModel):
    generation_id: str
    image_url: str
    script_text: str
    prompt: str
    aspect_ratio: str
    duration: int

# =========================
# SUPABASE UPDATE
# =========================

def update_generation(generation_id, status, video_url=None):

    url = f"{SUPABASE_URL}/rest/v1/video_generations?id=eq.{generation_id}"

    payload = {
        "status": status
    }

    if video_url:
        payload["video_url"] = video_url
        payload["final_video_url"] = video_url

    headers = {
        "apikey": SUPABASE_SERVICE_ROLE_KEY,
        "Authorization": f"Bearer {SUPABASE_SERVICE_ROLE_KEY}",
        "Content-Type": "application/json",
        "Prefer": "return=minimal"
    }

    res = requests.patch(url, json=payload, headers=headers)

    if res.status_code >= 300:
        print("Supabase update error:", res.text)

# =========================
# DOWNLOAD IMAGE
# =========================

def download_image_bytes(image_url):

    res = requests.get(image_url)

    if res.status_code != 200:
        raise Exception("Failed to download image")

    return res.content

# =========================
# GENERATE VIDEO
# =========================

@app.post("/generate-video")
def generate_video(req: GenerateVideoRequest):

    generation_id = req.generation_id

    try:

        print(f"Starting Veo 3.1 generation: {generation_id}")

        update_generation(generation_id, "processing")

        # download image
        print("Downloading image...")
        image_bytes = download_image_bytes(req.image_url)

        full_prompt = f"""
        {req.prompt}

        Script:
        {req.script_text}
        """

        print("Calling Veo 3.1 via Vertex AI...")

        operation = model.generate_videos(
            prompt=full_prompt,
            image=image_bytes,
            aspect_ratio=req.aspect_ratio,
            duration_seconds=req.duration,
        )

        print("Waiting for completion...")

        while not operation.done:

            print("Still generating...")
            time.sleep(10)
            operation = operation.refresh()

        if not operation.response.generated_videos:
            raise Exception("No video generated")

        video = operation.response.generated_videos[0]

        video_bytes = video.video_bytes

        video_path = f"/tmp/{generation_id}.mp4"

        with open(video_path, "wb") as f:
            f.write(video_bytes)

        print("Uploading video to Supabase Storage...")

        storage_url = f"{SUPABASE_URL}/storage/v1/object/video-generations/{generation_id}.mp4"

        headers = {
            "Authorization": f"Bearer {SUPABASE_SERVICE_ROLE_KEY}",
            "Content-Type": "video/mp4"
        }

        upload_res = requests.post(
            storage_url,
            headers=headers,
            data=video_bytes
        )

        if upload_res.status_code not in [200, 201]:
            raise Exception(upload_res.text)

        public_url = f"{SUPABASE_URL}/storage/v1/object/public/video-generations/{generation_id}.mp4"

        print("Video stored:", public_url)

        update_generation(
            generation_id,
            "completed",
            public_url
        )

        return {
            "status": "completed",
            "video_url": public_url
        }

    except Exception as e:

        print("ERROR:", str(e))

        update_generation(
            generation_id,
            "failed"
        )

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
