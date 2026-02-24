import os
import json
import base64
import asyncio
import uuid
import httpx

from typing import Optional
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
    # Campos opcionais de metadados para enriquecer o prompt
    emotion_type: Optional[str] = None
    camera_mode: Optional[str] = None
    body_motion_type: Optional[str] = None
    eye_focus_type: Optional[str] = None
    micro_expression_level: Optional[str] = None
    background_motion_type: Optional[str] = None
    head_movement_type: Optional[str] = None
    behavior_type: Optional[str] = None
    intent_type: Optional[str] = None
    aspect_ratio: Optional[str] = None

# =====================================================
# BUILD VEO PROMPT (enriquece com metadados cinematicos)
# =====================================================

def build_veo_prompt(req: GenerateVideoRequest) -> str:

    parts = [req.prompt]

    camera_map = {
        "tripod": "static tripod shot, stable camera",
        "selfie": "handheld selfie-style shot, person holding phone",
    }

    body_map = {
        "natural_shift": "subtle natural body movement",
        "selfie_arm_movement": "arm holding phone with natural relaxed movement",
        "presenting_product": "hands presenting product clearly to camera",
    }

    emotion_map = {
        "confident": "confident and engaging expression",
        "neutral": "natural relaxed expression",
    }

    eye_map = {
        "camera_natural": "looking naturally and directly at camera",
        "selfie_screen_focus": "looking at phone screen naturally",
        "product_to_camera": "directing product toward camera with eye contact",
    }

    bg_map = {
        "nature": "natural outdoor background with subtle ambient movement",
        "none": "clean simple background, no distractions",
    }

    micro_map = {
        "subtle_natural": "subtle natural micro-expressions, authentic feel",
        "commercial": "polished commercial-quality expressions, professional look",
    }

    head_map = {
        "natural": "natural slight head movement",
        "selfie_style": "natural selfie-style head positioning",
    }

    intent_map = {
        "casual_ugc": "casual user-generated content style, authentic and relatable",
        "neutral": "natural conversational style",
    }

    if req.camera_mode and req.camera_mode in camera_map:
        parts.append(camera_map[req.camera_mode])

    if req.body_motion_type and req.body_motion_type in body_map:
        parts.append(body_map[req.body_motion_type])

    if req.emotion_type and req.emotion_type in emotion_map:
        parts.append(emotion_map[req.emotion_type])

    if req.eye_focus_type and req.eye_focus_type in eye_map:
        parts.append(eye_map[req.eye_focus_type])

    if req.background_motion_type and req.background_motion_type in bg_map:
        parts.append(bg_map[req.background_motion_type])

    if req.micro_expression_level and req.micro_expression_level in micro_map:
        parts.append(micro_map[req.micro_expression_level])

    if req.head_movement_type and req.head_movement_type in head_map:
        parts.append(head_map[req.head_movement_type])

    if req.intent_type and req.intent_type in intent_map:
        parts.append(intent_map[req.intent_type])

    # Qualidade base sempre aplicada
    parts += [
        "photorealistic",
        "4K quality",
        "sharp focus",
        "smooth motion",
        "no deformation",
        "no morphing",
        "anatomically correct",
        "professional lighting",
        "9:16 vertical video",
    ]

    enhanced = ". ".join(parts)

    print("Enhanced prompt:", enhanced)

    return enhanced

# =====================================================
# DOWNLOAD IMAGE
# =====================================================

async def download_image_bytes(url: str):

    async with httpx.AsyncClient(timeout=60) as client:

        response = await client.get(url)

        response.raise_for_status()

        return response.content

# =====================================================
# UPLOAD VIDEO TO SUPABASE STORAGE
# =====================================================

async def upload_video_to_supabase(video_bytes: bytes) -> str:

    file_name = f"{uuid.uuid4()}.mp4"

    upload_url = f"{SUPABASE_URL}/storage/v1/object/videos/{file_name}"

    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "video/mp4",
    }

    async with httpx.AsyncClient(timeout=120) as client:

        response = await client.post(upload_url, headers=headers, content=video_bytes)

        response.raise_for_status()

    public_url = f"{SUPABASE_URL}/storage/v1/object/public/videos/{file_name}"

    print("Video uploaded to Supabase Storage:", public_url)

    return public_url

# =====================================================
# CALL VEO USING PREDICT LONG RUNNING
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
            "sampleCount": 1,
            "aspectRatio": "9:16",
            "negativePrompt": (
                "deformation, morphing, distortion, blurry, low quality, "
                "unrealistic, artifacts, glitch, flickering, watermark, text, "
                "extra limbs, missing limbs, anatomical errors"
            )
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
                fetch_url, headers=headers,
                json={"operationName": operation_name}
            )

            poll_response.raise_for_status()

            result = poll_response.json()

            if result.get("done"):

                videos = result.get("response", {}).get("videos", [])

                if not videos:
                    raise Exception("Nenhum video retornado: " + str(result))

                video = videos[0]

                # Caso 1: URI no GCS ou URI direta
                video_uri = video.get("gcsUri") or video.get("uri")
                if video_uri:
                    print("Video URI recebido:", video_uri)
                    return video_uri

                # Caso 2: Video retornado como bytes base64
                video_b64 = video.get("bytesBase64Encoded")
                if video_b64:
                    print("Video retornado como bytes, fazendo upload para Supabase...")
                    video_bytes_decoded = base64.b64decode(video_b64)
                    return await upload_video_to_supabase(video_bytes_decoded)

                raise Exception("Formato de video desconhecido: " + str(video))

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

        enhanced_prompt = build_veo_prompt(req)

        video_url = await call_veo(image_bytes, enhanced_prompt)

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
