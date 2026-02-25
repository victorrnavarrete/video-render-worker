import os
import json
import base64
import asyncio
import uuid
import subprocess
import tempfile
import httpx

from typing import Optional, List
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
# REQUEST MODELS
# =====================================================

class GenerateVideoRequest(BaseModel):
    generation_id: str
    image_url: str
    prompt: str
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

class ClipConfig(BaseModel):
    url: str
    trim_start: Optional[float] = 0.0   # segundos a cortar no inicio
    trim_end: Optional[float] = None     # segundos a cortar no fim (None = sem corte)

class MergeVideosRequest(BaseModel):
    sequence_id: str
    video_urls: Optional[List[str]] = None       # retrocompatibilidade
    clips: Optional[List[ClipConfig]] = None     # novo: com info de trim por cena

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

    parts += [
        "photorealistic", "4K quality", "sharp focus", "smooth motion",
        "no deformation", "no morphing", "anatomically correct",
        "professional lighting", "9:16 vertical video",
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

async def upload_video_to_supabase(video_bytes: bytes, file_name: str = None) -> str:
    if not file_name:
        file_name = f"{uuid.uuid4()}.mp4"
    upload_url = f"{SUPABASE_URL}/storage/v1/object/videos/{file_name}"
    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "video/mp4",
    }
    async with httpx.AsyncClient(timeout=300) as client:
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
                "image": {"bytesBase64Encoded": image_base64, "mimeType": "image/jpeg"}
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
                video_uri = video.get("gcsUri") or video.get("uri")
                if video_uri:
                    print("Video URI recebido:", video_uri)
                    return video_uri
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
# ENDPOINT: GENERATE SINGLE VIDEO
# =====================================================

@app.post("/generate-video")
async def generate_video(req: GenerateVideoRequest):
    try:
        print("Starting Veo 3.1 generation:", req.generation_id)
        image_bytes = await download_image_bytes(req.image_url)
        enhanced_prompt = build_veo_prompt(req)
        video_url = await call_veo(image_bytes, enhanced_prompt)
        await update_supabase(req.generation_id, video_url)
        return {"status": "success", "video_url": video_url}
    except Exception as e:
        print("ERROR:", str(e))
        return {"status": "error", "message": str(e)}

# =====================================================
# ENDPOINT: MERGE VIDEOS (com suporte a trim por cena)
# =====================================================

@app.post("/merge-videos")
async def merge_videos(req: MergeVideosRequest):
    try:
        # Normaliza: aceita 'clips' (novo) ou 'video_urls' (retrocompatibilidade)
        if req.clips:
            clip_list = req.clips
        elif req.video_urls:
            clip_list = [ClipConfig(url=u) for u in req.video_urls]
        else:
            return {"status": "error", "message": "Forneça 'clips' ou 'video_urls'"}

        print(f"Starting merge for sequence: {req.sequence_id} | {len(clip_list)} clips")

        with tempfile.TemporaryDirectory() as tmpdir:

            # 1. Download e trim de cada clip
            trimmed_paths = []

            async with httpx.AsyncClient(timeout=120) as client:
                for i, clip in enumerate(clip_list):
                    print(f"Downloading clip {i + 1}/{len(clip_list)}: {clip.url}")
                    response = await client.get(clip.url)
                    response.raise_for_status()

                    raw_path = os.path.join(tmpdir, f"raw_{i:03d}.mp4")
                    with open(raw_path, "wb") as f:
                        f.write(response.content)

                    # Aplica trim se necessario
                    trim_start = clip.trim_start or 0.0
                    needs_trim = trim_start > 0 or clip.trim_end is not None

                    if needs_trim:
                        trimmed_path = os.path.join(tmpdir, f"clip_{i:03d}.mp4")
                        ffmpeg_cmd = ["ffmpeg", "-y", "-i", raw_path]

                        if trim_start > 0:
                            ffmpeg_cmd += ["-ss", str(trim_start)]

                        if clip.trim_end is not None:
                            # trim_end = segundos a cortar do fim
                            # Pega a duracao total via ffprobe
                            probe = subprocess.run(
                                ["ffprobe", "-v", "error", "-show_entries",
                                 "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", raw_path],
                                capture_output=True, text=True
                            )
                            total_duration = float(probe.stdout.strip())
                            end_time = total_duration - clip.trim_end
                            if end_time > trim_start:
                                ffmpeg_cmd += ["-to", str(end_time)]

                        ffmpeg_cmd += ["-c", "copy", trimmed_path]
                        result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)

                        if result.returncode != 0:
                            print(f"Trim warning clip {i}: {result.stderr}")
                            trimmed_path = raw_path  # fallback: usa original
                        else:
                            print(f"Clip {i + 1} trimmed: start={trim_start}s, trim_end={clip.trim_end}s")
                    else:
                        trimmed_path = raw_path

                    trimmed_paths.append(trimmed_path)

            # 2. Cria arquivo de lista para FFmpeg concat
            concat_file = os.path.join(tmpdir, "concat.txt")
            with open(concat_file, "w") as f:
                for path in trimmed_paths:
                    f.write(f"file '{path}'\n")

            # 3. Concatena todos os clips
            output_path = os.path.join(tmpdir, "merged.mp4")
            result = subprocess.run(
                ["ffmpeg", "-y", "-f", "concat", "-safe", "0",
                 "-i", concat_file, "-c", "copy", output_path],
                capture_output=True, text=True
            )
            if result.returncode != 0:
                raise Exception(f"FFmpeg error: {result.stderr}")

            print("FFmpeg merge completed successfully")

            # 4. Upload do video final
            with open(output_path, "rb") as f:
                merged_bytes = f.read()

            file_name = f"sequence_{req.sequence_id}.mp4"
            public_url = await upload_video_to_supabase(merged_bytes, file_name)

            return {"status": "success", "video_url": public_url}

    except Exception as e:
        print(f"ERROR in merge: {str(e)}")
        return {"status": "error", "message": str(e)}
