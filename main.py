import os
import uuid
import base64
import time
import requests
import httpx

from fastapi import FastAPI, Request
from google.cloud import aiplatform_v1
from google.protobuf import struct_pb2

# =========================
# ENV VARS
# =========================

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_SERVICE_ROLE_KEY"]

PROJECT_ID = os.environ["GOOGLE_CLOUD_PROJECT"]
LOCATION = os.environ["GOOGLE_CLOUD_LOCATION"]

MODEL_ID = "veo-3.1-generate-preview"

# =========================
# FASTAPI
# =========================

app = FastAPI()

# =========================
# SUPABASE REST UPDATE
# =========================

async def update_generation(generation_id, data):

    url = f"{SUPABASE_URL}/rest/v1/video_generations?id=eq.{generation_id}"

    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
        "Prefer": "return=minimal"
    }

    async with httpx.AsyncClient(timeout=60) as client:
        await client.patch(url, headers=headers, json=data)


# =========================
# DOWNLOAD IMAGE
# =========================

def download_image(url):

    r = requests.get(url)

    if r.status_code != 200:
        raise Exception("Failed to download image")

    return r.content


# =========================
# GENERATE VIDEO
# =========================

def generate_video(prompt, image_bytes):

    client = aiplatform_v1.PredictionServiceClient()

    endpoint = f"projects/{PROJECT_ID}/locations/{LOCATION}/publishers/google/models/{MODEL_ID}"

    image_base64 = base64.b64encode(image_bytes).decode()

    instance = struct_pb2.Value()

    struct_pb2.Struct().update({
        "prompt": prompt,
        "image": {
            "bytesBase64Encoded": image_base64,
            "mimeType": "image/jpeg"
        }
    })

    instance = struct_pb2.Value(
        struct_value=struct_pb2.Struct(
            fields={
                "prompt": struct_pb2.Value(string_value=prompt),
                "image": struct_pb2.Value(
                    struct_value=struct_pb2.Struct(
                        fields={
                            "bytesBase64Encoded": struct_pb2.Value(string_value=image_base64),
                            "mimeType": struct_pb2.Value(string_value="image/jpeg"),
                        }
                    )
                )
            }
        )
    )

    operation = client.predict(
        endpoint=endpoint,
        instances=[instance]
    )

    response = operation.predictions

    video_base64 = response[0]["video"]

    video_bytes = base64.b64decode(video_base64)

    return video_bytes


# =========================
# ROUTE
# =========================

@app.post("/generate-video")
async def generate_video_route(request: Request):

    body = await request.json()

    generation_id = body["generation_id"]
    image_url = body["image_url"]
    prompt = body["prompt"]

    print(f"Starting Veo 3.1 generation: {generation_id}")

    try:

        await update_generation(generation_id, {
            "status": "processing"
        })

        image_bytes = download_image(image_url)

        video_bytes = generate_video(prompt, image_bytes)

        video_base64 = base64.b64encode(video_bytes).decode()

        video_url = f"data:video/mp4;base64,{video_base64}"

        await update_generation(generation_id, {
            "status": "completed",
            "video_url": video_url,
            "final_video_url": video_url
        })

        return {"success": True}

    except Exception as e:

        print("ERROR:", str(e))

        await update_generation(generation_id, {
            "status": "failed"
        })

        return {"error": str(e)}
