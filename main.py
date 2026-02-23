import os
import time
import base64
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from google.cloud import aiplatform_v1
from google.oauth2 import service_account

from supabase import create_client

# ENV
PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT")
LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION", "global")
CREDENTIALS_PATH = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

# Init clients
credentials = service_account.Credentials.from_service_account_file(
    CREDENTIALS_PATH
)

prediction_client = aiplatform_v1.PredictionServiceClient(
    credentials=credentials
)

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

app = FastAPI()

MODEL = "projects/{}/locations/{}/publishers/google/models/veo-3.1-generate-preview".format(
    PROJECT_ID, LOCATION
)


class GenerateVideoRequest(BaseModel):
    generation_id: str
    image_url: str
    prompt: str


def download_image_as_base64(url):
    response = requests.get(url)

    if response.status_code != 200:
        raise Exception("Failed to download image")

    return base64.b64encode(response.content).decode("utf-8")


@app.post("/generate-video")
async def generate_video(req: GenerateVideoRequest):

    generation_id = req.generation_id

    try:

        print("Starting Veo generation:", generation_id)

        supabase.table("video_generations").update({
            "status": "processing"
        }).eq("id", generation_id).execute()

        image_base64 = download_image_as_base64(req.image_url)

        instance = {
            "prompt": req.prompt,
            "image": {
                "bytesBase64Encoded": image_base64,
                "mimeType": "image/jpeg"
            }
        }

        endpoint = prediction_client.endpoint_path(
            project=PROJECT_ID,
            location=LOCATION,
            endpoint="publishers/google/models/veo-3.1-generate-preview"
        )

        operation = prediction_client.predict(
            endpoint=endpoint,
            instances=[instance],
            parameters={}
        )

        video_uri = operation.predictions[0]["video"]["uri"]

        print("Video URI:", video_uri)

        supabase.table("video_generations").update({
            "status": "completed",
            "video_url": video_uri,
            "final_video_url": video_uri
        }).eq("id", generation_id).execute()

        return {
            "success": True,
            "video_url": video_uri
        }

    except Exception as e:

        print("ERROR:", str(e))

        supabase.table("video_generations").update({
            "status": "failed"
        }).eq("id", generation_id).execute()

        raise HTTPException(status_code=500, detail=str(e))
