import os
import time
import base64
import requests
from fastapi import FastAPI
from pydantic import BaseModel

from supabase import create_client, Client

import vertexai
from vertexai.preview.generative_models import GenerativeModel, Part, Image

# --------------------------------------------------
# ENV
# --------------------------------------------------

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_SERVICE_ROLE_KEY"]

GOOGLE_PROJECT = os.environ["GOOGLE_CLOUD_PROJECT"]
GOOGLE_LOCATION = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")

# --------------------------------------------------
# INIT
# --------------------------------------------------

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

vertexai.init(
    project=GOOGLE_PROJECT,
    location=GOOGLE_LOCATION,
)

app = FastAPI()

# --------------------------------------------------
# REQUEST MODEL
# --------------------------------------------------

class GenerateVideoRequest(BaseModel):
    generation_id: str
    image_url: str
    prompt: str


# --------------------------------------------------
# HELPER
# --------------------------------------------------

def download_image_bytes(url: str) -> bytes:
    response = requests.get(url)
    response.raise_for_status()
    return response.content


# --------------------------------------------------
# ROUTE
# --------------------------------------------------

@app.post("/generate-video")
def generate_video(req: GenerateVideoRequest):

    generation_id = req.generation_id
    image_url = req.image_url
    prompt = req.prompt

    print(f"Starting Veo generation: {generation_id}")

    # update status → generating
    supabase.table("video_generations").update({
        "status": "generating"
    }).eq("id", generation_id).execute()

    try:

        # download image
        image_bytes = download_image_bytes(image_url)

        # create Veo image object
        veo_image = Image.from_bytes(image_bytes)

        model = GenerativeModel("veo-3.1-generate-preview")

        print("Calling Veo...")

        operation = model.generate_content(
            [
                Part.from_image(veo_image),
                prompt
            ],
            generation_config={
                "response_modalities": ["VIDEO"]
            }
        )

        print("Waiting for result...")

        video_bytes = operation.candidates[0].content.parts[0].inline_data.data

        video_path = f"/tmp/{generation_id}.mp4"

        with open(video_path, "wb") as f:
            f.write(video_bytes)

        # upload to supabase storage
        storage_path = f"videos/{generation_id}.mp4"

        supabase.storage.from_("videos").upload(
            storage_path,
            video_path,
            {"content-type": "video/mp4"}
        )

        video_url = f"{SUPABASE_URL}/storage/v1/object/public/videos/{storage_path}"

        supabase.table("video_generations").update({
            "status": "completed",
            "video_url": video_url,
            "final_video_url": video_url
        }).eq("id", generation_id).execute()

        print("Success")

        return {"success": True}

    except Exception as e:

        print("ERROR:", str(e))

        supabase.table("video_generations").update({
            "status": "failed"
        }).eq("id", generation_id).execute()

        raise e
