import os
import time
import base64
import requests
import httpx

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

import vertexai
from vertexai.preview.generative_models import GenerativeModel, Image


app = FastAPI()


# =========================
# ENV VARIABLES
# =========================

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")

PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT")
LOCATION = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")

SERVICE_ACCOUNT_JSON = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS_JSON")


# =========================
# WRITE SERVICE ACCOUNT FILE
# =========================

if SERVICE_ACCOUNT_JSON:
    os.makedirs("/app", exist_ok=True)
    with open("/app/service-account.json", "w") as f:
        f.write(SERVICE_ACCOUNT_JSON)

    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/app/service-account.json"


# =========================
# INIT VERTEX AI
# =========================

vertexai.init(
    project=PROJECT_ID,
    location=LOCATION
)


# =========================
# SUPABASE UPDATE FUNCTION
# =========================

def update_supabase(generation_id, data):

    url = f"{SUPABASE_URL}/rest/v1/video_generations?id=eq.{generation_id}"

    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
        "Prefer": "return=minimal"
    }

    httpx.patch(url, headers=headers, json=data)


# =========================
# GENERATE VIDEO ENDPOINT
# =========================

@app.post("/generate-video")
async def generate_video(request: Request):

    body = await request.json()

    generation_id = body.get("generation_id")
    image_url = body.get("image_url")
    prompt = body.get("prompt", "Natural cinematic motion.")

    start_time = time.time()

    try:

        print(f"Starting Veo 3.1 generation: {generation_id}")

        # =========================
        # DOWNLOAD IMAGE
        # =========================

        print("Downloading image...")

        image_response = requests.get(image_url)

        if image_response.status_code != 200:
            raise Exception("Failed to download image")

        image_bytes = image_response.content


        # =========================
        # CREATE IMAGE OBJECT
        # =========================

        veo_image = Image.from_bytes(image_bytes)


        # =========================
        # LOAD MODEL
        # =========================

        print("Loading Veo model...")

        model = GenerativeModel("veo-3.1-generate-preview")


        # =========================
        # GENERATE VIDEO
        # =========================

        print("Calling Veo 3.1 via Vertex AI...")

        operation = model.generate_content(
            contents=[prompt, veo_image],
            stream=False
        )


        # =========================
        # EXTRACT VIDEO BYTES
        # =========================

        video_bytes = None

        for part in operation.candidates[0].content.parts:
            if hasattr(part, "file_data") and part.file_data:
                video_bytes = part.file_data.data
                break


        if not video_bytes:
            raise Exception("No video returned by Veo")


        # =========================
        # SAVE VIDEO TO SUPABASE STORAGE
        # =========================

        print("Uploading video to Supabase Storage...")

        file_name = f"{generation_id}.mp4"

        upload_url = f"{SUPABASE_URL}/storage/v1/object/videos/{file_name}"

        headers = {
            "apikey": SUPABASE_KEY,
            "Authorization": f"Bearer {SUPABASE_KEY}",
            "Content-Type": "video/mp4"
        }

        upload_response = httpx.post(
            upload_url,
            headers=headers,
            content=video_bytes
        )

        if upload_response.status_code not in [200, 201]:
            raise Exception(upload_response.text)


        video_url = f"{SUPABASE_URL}/storage/v1/object/public/videos/{file_name}"


        # =========================
        # UPDATE DATABASE
        # =========================

        latency = int((time.time() - start_time) * 1000)

        update_supabase(generation_id, {
            "status": "completed",
            "video_url": video_url,
            "final_video_url": video_url,
            "latency_ms": latency
        })


        print("Video generation completed successfully")


        return JSONResponse({
            "status": "success",
            "video_url": video_url
        })


    except Exception as e:

        print("ERROR:", str(e))

        update_supabase(generation_id, {
            "status": "failed"
        })

        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )
