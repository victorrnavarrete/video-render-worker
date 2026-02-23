import os
import time
import base64
import requests
import vertexai

from fastapi import FastAPI
from pydantic import BaseModel

from google.oauth2 import service_account
from google.cloud import aiplatform
from google.cloud.aiplatform.gapic import PredictionServiceClient
from google.protobuf import json_format
from google.protobuf.struct_pb2 import Value

# ENV
PROJECT_ID = os.environ["GOOGLE_CLOUD_PROJECT"]
LOCATION = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")
SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_SERVICE_ROLE_KEY"]

credentials = service_account.Credentials.from_service_account_file(
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"]
)

vertexai.init(
    project=PROJECT_ID,
    location=LOCATION,
    credentials=credentials
)

client = PredictionServiceClient(credentials=credentials)

MODEL = f"projects/{PROJECT_ID}/locations/{LOCATION}/publishers/google/models/veo-3.1-generate-preview"

app = FastAPI()


class GenerateVideoRequest(BaseModel):
    generation_id: str
    image_url: str
    prompt: str


def update_supabase(generation_id, data):
    url = f"{SUPABASE_URL}/rest/v1/video_generations?id=eq.{generation_id}"

    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
        "Prefer": "return=minimal"
    }

    requests.patch(url, headers=headers, json=data)


def download_image_bytes(url):
    response = requests.get(url)
    response.raise_for_status()
    return response.content


def generate_video(generation_id, image_url, prompt):

    print("Downloading image...")

    image_bytes = download_image_bytes(image_url)

    image_base64 = base64.b64encode(image_bytes).decode()

    instance = {
        "prompt": prompt,
        "image": {
            "bytesBase64Encoded": image_base64,
            "mimeType": "image/jpeg"
        }
    }

    instance_proto = json_format.ParseDict(instance, Value())

    endpoint = client.endpoint_path(
        project=PROJECT_ID,
        location=LOCATION,
        endpoint="publishers/google/models/veo-3.1-generate-preview"
    )

    print("Calling Veo 3.1 via Vertex AI...")

    operation = client.predict_long_running(
        endpoint=endpoint,
        instances=[instance_proto],
    )

    print("Waiting for result...")

    response = operation.result(timeout=1800)

    prediction = response.predictions[0]

    video_uri = prediction.get("videoUri")

    print("Video URI:", video_uri)

    if not video_uri:
        raise Exception("videoUri not returned")

    video_bytes = requests.get(video_uri).content

    filename = f"{generation_id}.mp4"

    upload_url = f"{SUPABASE_URL}/storage/v1/object/video-generations/{filename}"

    headers = {
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "video/mp4"
    }

    upload = requests.post(upload_url, headers=headers, data=video_bytes)

    if upload.status_code not in [200, 201]:
        raise Exception("Supabase upload failed")

    public_url = f"{SUPABASE_URL}/storage/v1/object/public/video-generations/{filename}"

    update_supabase(generation_id, {
        "status": "completed",
        "video_url": public_url,
        "final_video_url": public_url
    })

    print("Done:", public_url)


@app.post("/generate-video")
def generate_video_endpoint(req: GenerateVideoRequest):

    try:

        print("Starting Veo 3.1 generation:", req.generation_id)

        update_supabase(req.generation_id, {
            "status": "processing"
        })

        generate_video(
            req.generation_id,
            req.image_url,
            req.prompt
        )

        return {"status": "success"}

    except Exception as e:

        print("ERROR:", str(e))

        update_supabase(req.generation_id, {
            "status": "failed"
        })

        return {"status": "error", "message": str(e)}
