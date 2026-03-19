import os
import json
import base64
import asyncio
import uuid
import subprocess
import tempfile
import io
import httpx

from PIL import Image
from typing import Optional, List
from fastapi import FastAPI, Request
from pydantic import BaseModel

from sora2_engine import call_sora, map_aspect_to_sora_size, resize_image_for_sora

from fastapi.responses import JSONResponse
from google.oauth2 import service_account
from google.auth.transport.requests import Request as GoogleRequest

# =====================================================
# INIT FASTAPI FIRST (CRITICAL)
# =====================================================

app = FastAPI()

# =====================================================
# AUTH MIDDLEWARE - validates RENDER_WORKER_SECRET
# =====================================================

RENDER_WORKER_SECRET = os.environ.get("RENDER_WORKER_SECRET", "")

def verify_worker_auth(request: Request) -> bool:
    """Check Authorization Bearer or X-Worker-Secret header."""
    if not RENDER_WORKER_SECRET:
        return True  # skip if not configured (dev mode)
    auth = request.headers.get("Authorization", "")
    if auth == f"Bearer {RENDER_WORKER_SECRET}":
        return True
    if request.headers.get("X-Worker-Secret", "") == RENDER_WORKER_SECRET:
        return True
    return False

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
        credentials.refresh(GoogleRequest())
    return credentials.token

# =====================================================
# REQUEST MODELS
# =====================================================

class GenerateVideoRequest(BaseModel):
    generation_id: str
    image_url: str
    prompt: str
    script_text: Optional[str] = None
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
    duration: Optional[int] = None
    video_quality: Optional[str] = None
    model: Optional[str] = None
    engine: Optional[str] = None

class ClipConfig(BaseModel):
    url: str
    trim_start: Optional[float] = 0.0
    trim_end: Optional[float] = None

class MergeVideosRequest(BaseModel):
    sequence_id: str
    video_urls: Optional[List[str]] = None
    clips: Optional[List[ClipConfig]] = None

# =====================================================
# BUILD VEO PROMPT (enriquece com metadados cinematicos)
# =====================================================

def build_veo_prompt(req: GenerateVideoRequest) -> str:
    """
    The edge functions already build a comprehensive, structured prompt with
    all behavioral descriptions (camera, body, emotion, etc.) included.
    We use it directly and only append quality keywords.
    """
    aspect = req.aspect_ratio or "9:16"

    quality_suffix = (
        f"photorealistic. 4K quality. sharp focus. smooth motion. "
        f"no deformation. no morphing. anatomically correct. "
        f"professional lighting. {aspect} video"
    )

    enhanced = req.prompt + "\n\n" + quality_suffix

    print(f"Final prompt ({len(enhanced)} chars):", enhanced[:500])

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
# CROP IMAGE TO ASPECT RATIO (pre-processamento)
# =====================================================

def crop_image_to_ratio(image_bytes: bytes, target_ratio: str = "9:16") -> bytes:
    """
    Crop image to target aspect ratio using center crop.
    Veo image-to-video mode uses source image dimensions,
    so we must pre-process to ensure correct output ratio.
    """
    try:
        w_ratio, h_ratio = map(int, target_ratio.split(":"))
        target_aspect = w_ratio / h_ratio

        img = Image.open(io.BytesIO(image_bytes))

        if img.mode not in ("RGB",):
            img = img.convert("RGB")

        orig_w, orig_h = img.size
        current_aspect = orig_w / orig_h

        print(f"Image preprocessing: {orig_w}x{orig_h} (aspect {current_aspect:.3f}) -> target {target_ratio} ({target_aspect:.3f})")

        if abs(current_aspect - target_aspect) < 0.02:
            print("Image already at target ratio, skipping crop")
            output = io.BytesIO()
            img.save(output, format="JPEG", quality=95)
            return output.getvalue()

        if current_aspect > target_aspect:
            new_w = int(orig_h * target_aspect)
            left = (orig_w - new_w) // 2
            img = img.crop((left, 0, left + new_w, orig_h))
            print(f"Cropped width: {orig_w} -> {new_w} (left={left})")
        else:
            new_h = int(orig_w / target_aspect)
            top = (orig_h - new_h) // 4
            img = img.crop((0, top, orig_w, top + new_h))
            print(f"Cropped height: {orig_h} -> {new_h} (top={top})")

        output = io.BytesIO()
        img.save(output, format="JPEG", quality=95)
        print(f"Image cropped: {len(image_bytes)} -> {len(output.getvalue())} bytes")
        return output.getvalue()

    except Exception as e:
        print(f"Image crop warning (using original): {e}")
        return image_bytes

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
# PARSE VEO ERROR (transforma erros tecnicos em mensagens amigaveis)
# =====================================================

def parse_veo_error(raw_error: str) -> str:
    err = str(raw_error).lower()

    if "third-party content providers" in err or "35561574" in err:
        return (
            "O prompt foi recusado pelo modelo de IA por conter referencias a "
            "marcas, musicas, celebridades ou conteudo protegido. "
            "Tente reformular o script removendo nomes especificos."
        )

    if "safety" in err or "content_filter" in err or "blocked" in err:
        return (
            "O conteudo foi bloqueado pelas politicas de seguranca da IA. "
            "Revise o prompt e tente novamente."
        )

    if "quota" in err or "resource_exhausted" in err:
        return (
            "Limite de uso da API atingido. Aguarde alguns minutos e tente novamente."
        )

    if "invalid_argument" in err or "code: 3" in err:
        return (
            "Parametros invalidos enviados para o modelo. "
            "Verifique o prompt e tente novamente."
        )

    if "deadline_exceeded" in err or "timeout" in err:
        return (
            "A geracao do video demorou muito e foi interrompida. "
            "Tente novamente."
        )

    if "unavailable" in err or "503" in err:
        return (
            "O servico de geracao de video esta temporariamente indisponivel. "
            "Tente novamente em alguns minutos."
        )

    return f"Erro na geracao: {str(raw_error)[:300]}"

# =====================================================
# CALL VEO USING PREDICT LONG RUNNING
# =====================================================

async def call_veo(
    image_bytes: bytes,
    prompt: str,
    aspect_ratio: str = "9:16",
    duration_seconds: int = 8,
    model: str = "veo-3.1-generate-preview",
):

    token = get_access_token()

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }

    submit_url = (
        f"https://{LOCATION}-aiplatform.googleapis.com/v1/"
        f"projects/{PROJECT_ID}/locations/{LOCATION}/"
        f"publishers/google/models/{model}:predictLongRunning"
    )

    image_base64 = base64.b64encode(image_bytes).decode()

    # Clamp duration to Veo3 valid values (4, 6, 8)
    valid_durations = [4, 6, 8]
    clamped_duration = min(valid_durations, key=lambda d: abs(d - duration_seconds))

    payload = {
        "instances": [
            {
                "prompt": prompt,
                "image": {"bytesBase64Encoded": image_base64, "mimeType": "image/jpeg"}
            }
        ],
        "parameters": {
            "sampleCount": 1,
            "aspectRatio": aspect_ratio,
            "durationSeconds": clamped_duration,
            "negativePrompt": (
                "deformation, morphing, distortion, blurry, low quality, "
                "unrealistic, artifacts, glitch, flickering, watermark, text, "
                "extra limbs, missing limbs, anatomical errors, "
                "subtitle, caption, on-screen text, logo, "
                "bokeh, shallow depth of field, blurry background, out of focus background, "
                "background music, soundtrack, musical score, singing, humming, jingle, beats, "
                "sound effects, swoosh, whoosh, impact sound, transition sound, riser, stinger, "
                "chime, ding, pop, click, dramatic audio, cinematic audio, orchestral, synth pad, "
                "artificial reverb, echo effect, audio filter, applause, laughter track"
            )
        }
    }

    print(f"Veo params: model={model}, duration={clamped_duration}s, aspect={aspect_ratio}")

    async with httpx.AsyncClient(timeout=600) as client:

        print(f"Calling Veo 3.1 | aspect_ratio={aspect_ratio}")

        response = await client.post(submit_url, headers=headers, json=payload)

        response.raise_for_status()

        operation = response.json()

        operation_name = operation["name"]

        print("Operation started:", operation_name)

        fetch_url = (
            f"https://{LOCATION}-aiplatform.googleapis.com/v1/"
            f"projects/{PROJECT_ID}/locations/{LOCATION}/"
            f"publishers/google/models/{model}:fetchPredictOperation"
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

                # Checa erro retornado pelo Veo (content policy, etc.)
                if result.get("error"):
                    raise Exception(str(result))

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
# UPDATE SUPABASE - sucesso
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
# UPDATE SUPABASE - falha
# =====================================================

async def update_supabase_failed(generation_id: str, error_message: str):
    """
    Atualiza a cena como 'failed' com mensagem de erro amigavel.
    Sem isso, cenas com erro ficam presas em 'processing_worker' para sempre.
    """
    url = f"{SUPABASE_URL}/rest/v1/video_generations?id=eq.{generation_id}"

    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
        "Prefer": "return=minimal"
    }

    payload = {
        "status": "failed",
        "error_message": error_message[:1000]
    }

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.patch(url, headers=headers, json=payload)
            response.raise_for_status()
            print(f"Supabase updated: generation {generation_id} -> failed")
    except Exception as e:
        print(f"Warning: could not update failed status in Supabase: {e}")

# =====================================================
# ENDPOINT: GENERATE SINGLE VIDEO
# =====================================================

@app.post("/generate-video")
async def generate_video(req: GenerateVideoRequest, request: Request):

    if not verify_worker_auth(request):
        return JSONResponse(status_code=401, content={"status": "error", "message": "Unauthorized"})

    try:

        veo_model = req.model or "veo-3.1-generate-preview"
        duration = req.duration or 8
        selected_engine = req.engine or "veo3"

        print(f"Starting generation: {req.generation_id} | engine={selected_engine} | model={veo_model} | duration={duration}s")

        aspect = req.aspect_ratio or "9:16"

        image_bytes = await download_image_bytes(req.image_url)

        image_bytes = crop_image_to_ratio(image_bytes, aspect)

        enhanced_prompt = build_veo_prompt(req)

        if selected_engine == "sora2":
            # Sora 2 path: resize image to exact output resolution, then call Sora API
            # call_sora picks the right model (sora-2 vs sora-2-pro) based on duration
            sora_size = map_aspect_to_sora_size(aspect)
            sora_image = resize_image_for_sora(image_bytes, sora_size)
            video_bytes = await call_sora(sora_image, enhanced_prompt, aspect, duration)
            video_url = await upload_video_to_supabase(video_bytes)
        else:
            # Veo3 path (default — no changes)
            video_url = await call_veo(
                image_bytes,
                enhanced_prompt,
                aspect_ratio=aspect,
                duration_seconds=duration,
                model=veo_model,
            )

        await update_supabase(req.generation_id, video_url)

        return {"status": "success", "video_url": video_url}

    except Exception as e:

        raw_error = str(e)
        print("ERROR:", raw_error)

        friendly_error = parse_veo_error(raw_error)

        await update_supabase_failed(req.generation_id, friendly_error)

        return {"status": "error", "message": friendly_error}

# =====================================================
# ENDPOINT: MERGE VIDEOS (com suporte a trim por cena)
# =====================================================

@app.post("/merge-videos")
async def merge_videos(req: MergeVideosRequest, request: Request):

    if not verify_worker_auth(request):
        return JSONResponse(status_code=401, content={"status": "error", "message": "Unauthorized"})

    try:

        if req.clips:
            clip_list = req.clips
        elif req.video_urls:
            clip_list = [ClipConfig(url=u) for u in req.video_urls]
        else:
            return {"status": "error", "message": "Forneca 'clips' ou 'video_urls'"}

        print(f"Starting merge for sequence: {req.sequence_id} | {len(clip_list)} clips")

        with tempfile.TemporaryDirectory() as tmpdir:

            trimmed_paths = []

            async with httpx.AsyncClient(timeout=120) as client:

                for i, clip in enumerate(clip_list):

                    print(f"Downloading clip {i + 1}/{len(clip_list)}: {clip.url}")

                    response = await client.get(clip.url)

                    response.raise_for_status()

                    raw_path = os.path.join(tmpdir, f"raw_{i:03d}.mp4")

                    with open(raw_path, "wb") as f:
                        f.write(response.content)

                    trim_start = clip.trim_start or 0.0
                    needs_trim = trim_start > 0 or clip.trim_end is not None

                    if needs_trim:
                        trimmed_path = os.path.join(tmpdir, f"clip_{i:03d}.mp4")
                        ffmpeg_cmd = ["ffmpeg", "-y", "-i", raw_path]

                        if trim_start > 0:
                            ffmpeg_cmd += ["-ss", str(trim_start)]

                        if clip.trim_end is not None:
                            probe = subprocess.run(
                                ["ffprobe", "-v", "error",
                                 "-show_entries", "format=duration",
                                 "-of", "default=noprint_wrappers=1:nokey=1",
                                 raw_path],
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
                            trimmed_path = raw_path
                        else:
                            print(f"Clip {i + 1} trimmed: start={trim_start}s, trim_end={clip.trim_end}s")
                    else:
                        trimmed_path = raw_path

                    trimmed_paths.append(trimmed_path)

            concat_file = os.path.join(tmpdir, "concat.txt")

            with open(concat_file, "w") as f:
                for path in trimmed_paths:
                    f.write(f"file '{path}'\n")

            output_path = os.path.join(tmpdir, "merged.mp4")

            result = subprocess.run(
                ["ffmpeg", "-y", "-f", "concat", "-safe", "0",
                 "-i", concat_file, "-c", "copy", output_path],
                capture_output=True, text=True
            )

            if result.returncode != 0:
                raise Exception(f"FFmpeg error: {result.stderr}")

            print("FFmpeg merge completed successfully")

            with open(output_path, "rb") as f:
                merged_bytes = f.read()

            file_name = f"sequence_{req.sequence_id}.mp4"

            public_url = await upload_video_to_supabase(merged_bytes, file_name)

            return {"status": "success", "video_url": public_url}

    except Exception as e:

        print(f"ERROR in merge: {str(e)}")

        return {"status": "error", "message": str(e)}

    

# =====================================================
# ENV: SCRAPER SECRET
# =====================================================
SCRAPER_SECRET = os.environ.get("SCRAPER_SECRET", "")


# =====================================================
# ENDPOINT: SCRAPE TRENDING (TikTok Creative Center)
# =====================================================
@app.post("/scrape-trending")
async def scrape_trending(request: Request):
    secret = request.headers.get("X-Scraper-Secret", "")
    if not SCRAPER_SECRET or secret != SCRAPER_SECRET:
        return JSONResponse(status_code=401, content={"status": "error", "message": "Unauthorized"})
    try:
        from scraper import run_scraper
        result = await run_scraper()
        return result
    except Exception as e:
        return {"status": "error", "message": str(e)}
