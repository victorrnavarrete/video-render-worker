import os
import time
import base64
import requests
import httpx

from fastapi import FastAPI
from pydantic import BaseModel

from google.cloud import aiplatform_v1
from google.oauth2 import service_account

# =========================
# CONFIG
# =========================

PROJECT_ID = os.environ["GOOGLE_CLOUD_PROJECT"]
LOCATION = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_SERVICE_ROLE_KEY"]

# credentials file written by Dockerfile startup
GOOGLE_APPLICATION_CREDENTIALS = "/app/service-account.json"

credentials = service_account.Credentials.from_service_account_file(
    GOOGLE_APPLICATION_CREDENTIALS
)

# Vertex client CORRETO
prediction_client = aiplatform_v1.PredictionServiceClient(
    credentials=credentials,
    client_options={
        "api_endpoint": f"{LOCATION}-aiplatform.googleapis.com"
    },
)

# endpoint do modelo Veo
MODEL_ENDPOINT = (
    f"projects/{PROJECT_ID}/locations/{LOCATION}"
    f"/publishers/google/models/veo-3.1-generate-preview"
)

app = FastAPI()

# =========================
# REQUEST MODEL
# =========================

class GenerateVideoRequest(BaseModel):
    generation_id: str
    image_url: str
    prompt: str

# =========================
# SUPABASE UPDATE
# =========================

def update_generation(generation_id, data):

    url = f"{SUPABASE_URL}/rest/v1/video_generations?id=eq.{generation_id}"

    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
        "Prefer": "return=minimal"
    }

    httpx.patch(url, headers=headers, json=data)

# =========================
# DOWNLOAD IMAGE
# =========================

def download_image_bytes(image_url):

    r = requests.get(image_url)

    if r.status_code != 200:
        raise Exception("Failed to download image")

    return r.content

# =========================
# GENERATE VIDEO
# =========================

@app.post("/generate-video")
def generate_video(req: GenerateVideoRequest):

    generation_id = req.generation_id

    try:

        print(f"Starting Veo generation: {generation_id}")

        update_generation(generation_id, {"status": "processing"})

        # download image
        print("Downloading image...")
        image_bytes = download_image_bytes(req.image_url)

        image_base64 = base64.b64encode(image_bytes).decode("utf-8")

        instance = {
            "prompt": req.prompt,
            "image": {
                "bytesBase64Encoded": image_base64,
                "mimeType": "image/jpeg"
            }
        }

        print("Calling PredictLongRunning...")

        operation = prediction_client.predict_long_running(
            endpoint=MODEL_ENDPOINT,
            instances=[instance],
        )

        print("Waiting for video generation...")

        result = operation.result(timeout=1800)

        video_uri = result.predictions[0]["video"]["uri"]

        print("Video URI:", video_uri)

        update_generation(generation_id, {
            "status": "completed",
            "video_url": video_uri,
            "final_video_url": video_uri
        })

        return {"success": True}

    except Exception as e:

        print("ERROR:", str(e))

        update_generation(generation_id, {
            "status": "failed"
        })

        return {"error": str(e)}
