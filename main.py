import os
import json
import base64
import asyncio
import httpx

from fastapi import FastAPI
from pydantic import BaseModel

from google.oauth2 import service_account
from google.auth.transport.requests import Request

# =====================================================
# INIT FASTAPI FIRST (CRITICAL)
# =====================================================

app = FastAPI()

# =====================================================
# ENV VARIABLES
# =====================================================

PROJECT_ID = os.environ["GOOGLE_CLOUD_PROJECT"]
LOCATION = os.environ["GOOGLE_CLOUD_LOCATION"]

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_SERVICE_ROLE_KEY"]

SERVICE_ACCOUNT_JSON = os.environ["GOOGLE_APPLICATION_CREDENTIALS_JSON"]

# =====================================================
# AUTH - dinamico (token renovado automaticamente)
# =====================================================

credentials_info = json.loads(SERVICE_ACCOUNT_JSON)

credentials = service_account.Credentials.from_service_account_info(
    credentials_info,
    scopes=["https://www.googleapis.com/auth/cloud-platform"],
)

def get_access_token() -> str:
    if not credentials.valid:
        credentials.refresh(Request())
    return credentials.token

# =====================================================
# REQUEST MODEL
# =====================================================

class GenerateVideoRequest(BaseModel):
    generation_id: str
    image_url: str
    prompt: str

# =====================================================
# DOWNLOAD IMAGE
# =====================================================

async def download_image_bytes(url: str):

    async with httpx.AsyncClient(timeout=60) as client:

        response = await client.get(url)

        response.raise_for_status()

        return response.content

# =====================================================
# CALL VEO USING PREDICT LONG RUNNING (corrigido)
# =====================================================

async def call_veo(image_bytes: bytes, prompt: str):

    token = get_access_token()

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }

    submit_url = (
        f"https://{LOCATION}-aiplatform.googleapis.com/v1/"
        f"projects/{PROJECT_ID}/locations/{LOCATION}/"
        f"publishers/google/models/veo-3.1-generate-preview:predictLongRunning"
    )

    image_base64 = base64.b64encode(image_bytes).decode()

    payload = {
        "instances": [
            {
                "prompt": prompt,
                "image": {
                    "bytesBase64Encoded": image_base64,
                    "mimeType": "image/jpeg"
                }
            }
        ],
        "parameters": {
            "sampleCount": 1
        }
    }

    async with httpx.AsyncClient(timeout=600) as client:

        print("Calling Veo 3.1 via Vertex PredictLongRunning...")

        response = await client.post(submit_url, headers=headers, json=payload)

        response.raise_for_status()

        operation = response.json()

        operation_name = operation["name"]

        print("Operation started:", operation_name)

        fetch_url = (
            f"https://{LOCATION}-aiplatform.googleapis.com/v1/"
            f"projects/{PROJECT_ID}/locations/{LOCATION}/"
            f"publishers/google/models/veo-3.1-generate-preview:fetchPredictOperation"
        )

        print("Polling via fetchPredictOperation...")

        while True:

            token = get_access_token()
            headers["Authorization"] = f"Bearer {token}"

            poll_response = await client.post(
                fetch_url,
                headers=headers,
                json={"operationName": operation_name}
            )

            poll_response.raise_for_status()

            result = poll_response.json()

            if result.get("done"):

                videos = result.get("response", {}).get("videos", [])

                if not videos:
                    raise Exception("Nenhum video retornado: " + str(result))

                video_uri = videos[0].get("gcsUri") or videos[0].get("uri")

                if not video_uri:
                    raise Exception("URI do video nao encontrado: " + str(videos[0]))

                print("Video generated:", video_uri)

                return video_uri

            print("Still processing... waiting 10s")

            await asyncio.sleep(10)

# =====================================================
# UPDATE SUPABASE
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

        response = await client.patch(url, headers=headers, json=payload)

        response.raise_for_status()

# =====================================================
# ENDPOINT
# =====================================================

@app.post("/generate-video")
async def generate_video(req: GenerateVideoRequest):

    try:

        print("Starting Veo 3.1 generation:", req.generation_id)

        image_bytes = await download_image_bytes(req.image_url)

        video_url = await call_veo(image_bytes, req.prompt)

        await update_supabase(req.generation_id, video_url)

        return {
            "status": "success",
            "video_url": video_url
        }

    except Exception as e:

        print("ERROR:", str(e))

        return {
            "status": "error",
            "message": str(e)
        }
