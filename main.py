import os
import time
import base64
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from google.cloud import aiplatform
from google.cloud.aiplatform_v1 import PredictionServiceClient
from google.cloud.aiplatform_v1.types import PredictLongRunningRequest
from google.protobuf import struct_pb2

from supabase import create_client

app = FastAPI()

# ENV
PROJECT_ID = os.environ["GOOGLE_CLOUD_PROJECT"]
LOCATION = "global"

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_SERVICE_ROLE_KEY"]

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

client = PredictionServiceClient()

class GenerateVideoRequest(BaseModel):
    generation_id: str
    image_url: str
    prompt: str

@app.post("/generate-video")
def generate_video(req: GenerateVideoRequest):

    generation_id = req.generation_id

    try:

        print(f"Starting Veo 3.1 generation: {generation_id}")

        # download image
        print("Downloading image...")
        image_bytes = requests.get(req.image_url).content

        image_base64 = base64.b64encode(image_bytes).decode("utf-8")

        # build instance struct
        instance = struct_pb2.Struct()
        instance.update({
            "prompt": req.prompt,
            "image": {
                "bytesBase64Encoded": image_base64,
                "mimeType": "image/jpeg"
            }
        })

        endpoint = f"projects/{PROJECT_ID}/locations/global/publishers/google/models/veo-3.1-generate-preview"

        print("Calling Veo 3.1 via PredictionServiceClient...")

        operation = client.predict_long_running(
            endpoint=endpoint,
            instances=[instance]
        )

        print("Waiting for video generation...")

        response = operation.result(timeout=1800)

        video_uri = response.predictions[0]["video"]["uri"]

        print("Video generated:", video_uri)

        # save to Supabase
        supabase.table("video_generations").update({
            "status": "completed",
            "video_url": video_uri,
            "final_video_url": video_uri
        }).eq("id", generation_id).execute()

        return {"video_url": video_uri}

    except Exception as e:

        print("ERROR:", str(e))

        supabase.table("video_generations").update({
            "status": "failed"
        }).eq("id", generation_id).execute()

        raise HTTPException(status_code=500, detail=str(e))
