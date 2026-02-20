import os
import time
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# ENV VARS
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
GEMINI_API_KEY = os.environ.get("VEO3_API_KEY")  # sua Gemini API Key

app = FastAPI()


# =========================
# REQUEST MODEL
# =========================

class GenerateVideoRequest(BaseModel):
    generation_id: str
    image_url: str
    script_text: str
    prompt: str
    aspect_ratio: str
    duration: int


# =========================
# SUPABASE UPDATE
# =========================

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

    if res.status_code not in [200, 204]:
        print("Supabase update failed:", res.text)


# =========================
# VEO3 GENERATION
# =========================

@app.post("/generate-video")
def generate_video(req: GenerateVideoRequest):

    try:

        print("Starting Veo3 generation:", req.generation_id)

        update_generation(req.generation_id, "processing")

        # Endpoint oficial Veo3
        url = (
            "https://generativelanguage.googleapis.com/v1beta/"
            "models/veo-3.1-generate-preview:predictLongRunning"
        )

        headers = {
            "x-goog-api-key": GEMINI_API_KEY,
            "Content-Type": "application/json"
        }

        # Prompt completo com script embutido
        full_prompt = f"""
{req.prompt}

Script (spoken naturally in Brazilian Portuguese):
{req.script_text}

Requirements:
- Perfect lip sync
- Natural facial motion
- Realistic human behavior
- Cinematic lighting
- Photorealistic
"""

        payload = {
            "instances": [
                {
                    "prompt": full_prompt,
                    "image": {
                        "uri": req.image_url
                    }
                }
            ],
            "parameters": {
                "aspectRatio": req.aspect_ratio,
                "durationSeconds": req.duration
            }
        }

        print("Submitting to Veo3...")

        response = requests.post(url, json=payload, headers=headers, timeout=60)

        if response.status_code != 200:
            raise Exception(response.text)

        operation = response.json()

        operation_name = operation["name"]

        print("Operation:", operation_name)

        # Poll operation
        poll_url = f"https://generativelanguage.googleapis.com/v1beta/{operation_name}"

        while True:

            poll = requests.get(
                poll_url,
                headers=headers,
                timeout=60
            )

            if poll.status_code != 200:
                raise Exception(poll.text)

            data = poll.json()

            if data.get("done"):

                video_uri = (
                    data["response"]["candidates"][0]
                    ["content"]["parts"][0]["fileData"]["uri"]
                )

                print("Video ready:", video_uri)

                update_generation(
                    req.generation_id,
                    "completed",
                    video_uri
                )

                return {
                    "status": "completed",
                    "video_url": video_uri
                }

            print("Still generating...")

            time.sleep(10)

    except Exception as e:

        print("ERROR:", str(e))

        update_generation(req.generation_id, "failed")

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
