import os
import time
import base64
import httpx
from fastapi import FastAPI, HTTPException
from google.oauth2 import service_account
from google.auth.transport.requests import Request

app = FastAPI()

# =========================
# ENV VARS
# =========================

PROJECT_ID = os.environ["GOOGLE_CLOUD_PROJECT"]
LOCATION = os.environ["GOOGLE_CLOUD_LOCATION"]

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_SERVICE_ROLE_KEY"]

SERVICE_ACCOUNT_FILE = "/app/service-account.json"

MODEL = "veo-3.1-generate-preview"

# =========================
# AUTH
# =========================

credentials = service_account.Credentials.from_service_account_file(
    SERVICE_ACCOUNT_FILE,
    scopes=["https://www.googleapis.com/auth/cloud-platform"],
)

def get_access_token():
    credentials.refresh(Request())
    return credentials.token


# =========================
# SUPABASE REST
# =========================

def update_generation(generation_id, data):
    url = f"{SUPABASE_URL}/rest/v1/video_generations?id=eq.{generation_id}"

    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
        "Prefer": "return=minimal"
    }

    httpx.patch(url, headers=headers, json=data, timeout=60)


# =========================
# DOWNLOAD IMAGE
# =========================

def download_image_bytes(image_url):
    r = httpx.get(image_url, timeout=120)
    r.raise_for_status()
    return r.content


# =========================
# START VEO JOB
# =========================

def start_veo_job(image_bytes, prompt):

    token = get_access_token()

    url = f"https://{LOCATION}-aiplatform.googleapis.com/v1/projects/{PROJECT_ID}/locations/{LOCATION}/publishers/google/models/{MODEL}:predictLongRunning"

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
        ]
    }

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }

    response = httpx.post(url, headers=headers, json=payload, timeout=120)

    if response.status_code != 200:
        raise Exception(response.text)

    operation = response.json()

    return operation["name"]


# =========================
# POLL OPERATION
# =========================

def poll_operation(operation_name):

    token = get_access_token()

    url = f"https://{LOCATION}-aiplatform.googleapis.com/v1/{operation_name}"

    headers = {
        "Authorization": f"Bearer {token}"
    }

    while True:

        response = httpx.get(url, headers=headers, timeout=60)

        if response.status_code != 200:
            raise Exception(response.text)

        data = response.json()

        if data.get("done"):

            if "error" in data:
                raise Exception(data["error"])

            video_uri = data["response"]["predictions"][0]["video"]["uri"]

            return video_uri

        time.sleep(10)


# =========================
# DOWNLOAD VIDEO
# =========================

def download_video(video_uri):

    token = get_access_token()

    headers = {
        "Authorization": f"Bearer {token}"
    }

    response = httpx.get(video_uri, headers=headers, timeout=600)

    response.raise_for_status()

    return response.content


# =========================
# UPLOAD VIDEO TO SUPABASE STORAGE
# =========================

def upload_to_supabase(generation_id, video_bytes):

    path = f"videos/{generation_id}.mp4"

    url = f"{SUPABASE_URL}/storage/v1/object/video-generations/{path}"

    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "video/mp4"
    }

    httpx.post(url, headers=headers, content=video_bytes, timeout=600)

    public_url = f"{SUPABASE_URL}/storage/v1/object/public/video-generations/{path}"

    return public_url


# =========================
# MAIN ENDPOINT
# =========================

@app.post("/generate-video")
async def generate_video(body: dict):

    try:

        generation_id = body["generation_id"]
        image_url = body["image_url"]
        prompt = body.get("prompt", "cinematic realistic motion")

        print("Starting Veo 3.1 generation:", generation_id)

        update_generation(generation_id, {
            "status": "processing"
        })

        print("Downloading image...")
        image_bytes = download_image_bytes(image_url)

        print("Calling Veo 3.1 via Vertex AI...")
        operation_name = start_veo_job(image_bytes, prompt)

        print("Polling operation...")
        video_uri = poll_operation(operation_name)

        print("Downloading video...")
        video_bytes = download_video(video_uri)

        print("Uploading to Supabase Storage...")
        video_url = upload_to_supabase(generation_id, video_bytes)

        print("Saving URL...")
        update_generation(generation_id, {
            "status": "completed",
            "video_url": video_url,
            "final_video_url": video_url
        })

        return {"success": True, "video_url": video_url}

    except Exception as e:

        print("ERROR:", str(e))

        update_generation(body["generation_id"], {
            "status": "failed"
        })

        raise HTTPException(status_code=500, detail=str(e))
