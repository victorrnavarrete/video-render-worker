import os
import time
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# =============================
# ENV VARIABLES
# =============================

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
    raise Exception("Missing Supabase environment variables")

if not GEMINI_API_KEY:
    raise Exception("Missing GEMINI_API_KEY")

# =============================
# APP INIT
# =============================

app = FastAPI()

# =============================
# REQUEST MODEL
# =============================

class GenerateVideoRequest(BaseModel):
    generation_id: str
    image_url: str
    script_text: str | None = None
    prompt: str
    aspect_ratio: str
    duration: int


# =============================
# SUPABASE UPDATE
# =============================

def update_generation(generation_id, status, final_video_url=None):

    url = f"{SUPABASE_URL}/rest/v1/video_generations?id=eq.{generation_id}"

    payload = {
        "status": status
    }

    if final_video_url:
        payload["final_video_url"] = final_video_url

    headers = {
        "apikey": SUPABASE_SERVICE_ROLE_KEY,
        "Authorization": f"Bearer {SUPABASE_SERVICE_ROLE_KEY}",
        "Content-Type": "application/json",
        "Prefer": "return=minimal"
    }

    res = requests.patch(url, json=payload, headers=headers)

    if res.status_code >= 300:
        print("Supabase update error:", res.text)


# =============================
# GEMINI FILE UPLOAD
# =============================

def upload_image_to_gemini(image_url):

    print("Downloading source image...")

    img = requests.get(image_url)

    if img.status_code != 200:
        raise Exception("Failed to download image")

    upload_url = f"https://generativelanguage.googleapis.com/upload/v1beta/files?key={GEMINI_API_KEY}"

    headers = {
        "X-Goog-Upload-Protocol": "raw",
        "X-Goog-Upload-File-Name": "source.jpg",
        "Content-Type": "image/jpeg"
    }

    print("Uploading image to Gemini Files API...")

    res = requests.post(upload_url, headers=headers, data=img.content)

    if res.status_code != 200:
        raise Exception(f"Gemini file upload failed: {res.text}")

    file_uri = res.json()["file"]["uri"]

    print("Gemini file uri:", file_uri)

    return file_uri


# =============================
# START VEO JOB
# =============================

def start_veo_job(file_uri, prompt, aspect_ratio, duration):

    veo_url = f"https://generativelanguage.googleapis.com/v1beta/models/veo-3.0-fast-image-to-video-preview:predictLongRunning?key={GEMINI_API_KEY}"

    payload = {
        "instances": [
            {
                "image": {
                    "gcsUri": file_uri
                },
                "prompt": prompt
            }
        ],
        "parameters": {
            "aspectRatio": aspect_ratio,
            "durationSeconds": duration
        }
    }

    print("Starting Veo job...")

    res = requests.post(veo_url, json=payload)

    if res.status_code != 200:
        raise Exception(res.text)

    operation = res.json()["name"]

    print("Operation:", operation)

    return operation


# =============================
# POLL VEO JOB
# =============================

def poll_veo(operation):

    poll_url = f"https://generativelanguage.googleapis.com/v1beta/{operation}?key={GEMINI_API_KEY}"

    print("Polling Veo...")

    while True:

        res = requests.get(poll_url)

        if res.status_code != 200:
            raise Exception(res.text)

        data = res.json()

        if data.get("done"):

            video_url = data["response"]["video"]["uri"]

            print("Video ready:", video_url)

            return video_url

        print("Still processing...")

        time.sleep(5)


# =============================
# MAIN ENDPOINT
# =============================

@app.post("/generate-video")
def generate_video(req: GenerateVideoRequest):

    generation_id = req.generation_id

    try:

        print("Starting generation:", generation_id)

        update_generation(generation_id, "processing")

        # combine prompt + speech naturally
        full_prompt = req.prompt

        if req.script_text:
            full_prompt += f"\n\nCharacter speaks naturally in Brazilian Portuguese:\n{req.script_text}"

        # upload image
        file_uri = upload_image_to_gemini(req.image_url)

        # start veo job
        operation = start_veo_job(
            file_uri=file_uri,
            prompt=full_prompt,
            aspect_ratio=req.aspect_ratio,
            duration=req.duration
        )

        # poll result
        video_url = poll_veo(operation)

        update_generation(
            generation_id,
            "completed",
            video_url
        )

        return {
            "status": "completed",
            "video_url": video_url
        }

    except Exception as e:

        print("ERROR:", str(e))

        update_generation(generation_id, "failed")

        raise HTTPException(status_code=500, detail=str(e))


# =============================
# HEALTH CHECK
# =============================

@app.get("/")
def health():
    return {"status": "ok"}
