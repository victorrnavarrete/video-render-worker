import os
import base64
import uuid
import time
import requests

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import vertexai
from vertexai.preview.generative_models import GenerativeModel, Part

from supabase import create_client

# =====================================================
# ENV
# =====================================================

PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT")
LOCATION = os.environ.get("GOOGLE_CLOUD_LOCATION", "global")

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

vertexai.init(
    project=PROJECT_ID,
    location=LOCATION,
)

# =====================================================
# FASTAPI
# =====================================================

app = FastAPI()

# =====================================================
# REQUEST MODEL
# =====================================================

class GenerateVideoRequest(BaseModel):
    generation_id: str
    image_url: str
    prompt: str


# =====================================================
# HEALTH CHECK
# =====================================================

@app.get("/")
def health():
    return {"status": "ok"}


# =====================================================
# VIDEO GENERATION
# =====================================================

@app.post("/generate-video")
def generate_video(req: GenerateVideoRequest):

    generation_id = req.generation_id

    try:

        print(f"Starting Veo 3.1 generation: {generation_id}")

        # =====================================================
        # DOWNLOAD IMAGE
        # =====================================================

        print("Downloading image...")
        img_bytes = requests.get(req.image_url).content

        image_part = Part.from_data(
            data=img_bytes,
            mime_type="image/jpeg"
        )

        # =====================================================
        # MODEL
        # =====================================================

        print("Initializing Veo model...")

        model = GenerativeModel(
            "projects/{}/locations/{}/publishers/google/models/veo-3.1-generate-preview".format(
                PROJECT_ID,
                LOCATION
            )
        )

        # =====================================================
        # GENERATE
        # =====================================================

        print("Calling Veo 3.1 via Vertex AI...")

        start = time.time()

        response = model.generate_content(
            contents=[req.prompt, image_part],
            generation_config={
                "temperature": 0.7,
                "top_p": 0.95,
            },
            stream=False
        )

        latency = int((time.time() - start) * 1000)

        # =====================================================
        # EXTRACT VIDEO
        # =====================================================

        print("Extracting video bytes...")

        video_bytes = None

        for part in response.candidates[0].content.parts:

            if hasattr(part, "inline_data"):
                video_bytes = part.inline_data.data
                break

        if video_bytes is None:
            raise Exception("Video not returned")

        # =====================================================
        # SAVE VIDEO
        # =====================================================

        file_name = f"{generation_id}.mp4"

        supabase.storage \
            .from_("videos") \
            .upload(
                file_name,
                video_bytes,
                {"content-type": "video/mp4"}
            )

        video_url = f"{SUPABASE_URL}/storage/v1/object/public/videos/{file_name}"

        print("Video saved:", video_url)

        # =====================================================
        # UPDATE DB
        # =====================================================

        supabase.table("video_generations") \
            .update({
                "status": "completed",
                "video_url": video_url,
                "final_video_url": video_url,
                "latency_ms": latency
            }) \
            .eq("id", generation_id) \
            .execute()

        return {
            "success": True,
            "video_url": video_url
        }

    except Exception as e:

        print("ERROR:", str(e))

        supabase.table("video_generations") \
            .update({
                "status": "failed"
            }) \
            .eq("id", generation_id) \
            .execute()

        raise HTTPException(500, str(e))
