import os
import json
import time
import base64
import asyncio
import httpx

from fastapi import FastAPI
from pydantic import BaseModel

from google.oauth2 import service_account
from google.auth.transport.requests import Request

# =====================================================
# CONFIG
# =====================================================

PROJECT_ID = os.environ["GOOGLE_CLOUD_PROJECT"]
LOCATION = os.environ["GOOGLE_CLOUD_LOCATION"]

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_SERVICE_ROLE_KEY"]

SERVICE_ACCOUNT_JSON = os.environ["GOOGLE_APPLICATION_CREDENTIALS_JSON"]

# =====================================================
# AUTH VIA SERVICE ACCOUNT JSON (ENV)
# CALL VEO 3.1 VIA PREDICTLONGRUNNING
# =====================================================

async def call_veo(image_bytes: bytes, prompt: str):

    endpoint = (
        f"https://{LOCATION}-aiplatform.googleapis.com/v1/"
        f"projects/{PROJECT_ID}/locations/{LOCATION}/"
        f"publishers/google/models/veo-3.1-generate-preview:predictLongRunning"
    )

    image_base64 = base64.b64encode(image_bytes).decode("utf-8")

    payload = {
        "instances": [
            {
                "prompt": prompt,
                "image": {
                    "bytesBase64Encoded": image_base64,
                    "mimeType": "image/jpeg"
                }
            }
        ]
    }

    headers = {
        "Authorization": f"Bearer {ACCESS_TOKEN}",
        "Content-Type": "application/json"
    }

    async with httpx.AsyncClient(timeout=300) as client:

        print("Calling Veo 3.1 via Vertex PredictLongRunning...")

        response = await client.post(
            endpoint,
            headers=headers,
            json=payload
        )

        response.raise_for_status()

        operation = response.json()

        operation_name = operation["name"]

        # CORREÇÃO CRÍTICA: usar operation_name diretamente
        poll_url = f"https://{LOCATION}-aiplatform.googleapis.com/v1/{operation_name}"

        print(f"Polling operation: {poll_url}")

        while True:

            poll = await client.get(
                poll_url,
                headers=headers
            )

            poll.raise_for_status()

            result = poll.json()

            if result.get("done"):

                if "error" in result:

                    raise Exception(result["error"])

                video_uri = result["response"]["videos"][0]["uri"]

                print(f"Video generated: {video_uri}")

                return video_uri

            await asyncio.sleep(5)

# =====================================================
# SAVE TO SUPABASE
# =====================================================

async def update_supabase(generation_id: str, video_url: str):

    url = f"{SUPABASE_URL}/rest/v1/video_generations?id=eq.{generation_id}"

    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
        "Prefer": "return=minimal"
    }

    payload = {
        "status": "completed",
        "video_url": video_url,
        "final_video_url": video_url
    }

    async with httpx.AsyncClient(timeout=60) as client:

        response = await client.patch(
            url,
            headers=headers,
            json=payload
        )

        response.raise_for_status()

# =====================================================
# MAIN ENDPOINT
# =====================================================

@app.post("/generate-video")
async def generate_video(req: GenerateVideoRequest):

    try:

        print(f"Starting Veo 3.1 generation: {req.generation_id}")

        image_bytes = await download_image_bytes(req.image_url)

        video_url = await call_veo(
            image_bytes=image_bytes,
            prompt=req.prompt
        )

        await update_supabase(
            generation_id=req.generation_id,
            video_url=video_url
        )

        return {
            "status": "completed",
            "video_url": video_url
        }

    except Exception as e:

        print("ERROR:", str(e))

        return {
            "status": "failed",
            "error": str(e)
        }
