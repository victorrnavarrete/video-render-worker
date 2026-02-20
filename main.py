import os
import time
import base64
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
GEMINI_API_KEY = os.environ.get("VEO3_API_KEY")

app = FastAPI()


class GenerateVideoRequest(BaseModel):
    generation_id: str
    image_url: str
    script_text: str
    prompt: str
    aspect_ratio: str
    duration: int


# =====================
# SUPABASE UPDATE
# =====================

def update_generation(generation_id, status, final_video_url=None):

    url = f"{SUPABASE_URL}/rest/v1/video_generations?id=eq.{generation_id}"

    payload = {"status": status}

    if final_video_url:
        payload["final_video_url"] = final_video_url

    headers = {
        "apikey": SUPABASE_SERVICE_ROLE_KEY,
        "Authorization": f"Bearer {SUPABASE_SERVICE_ROLE_KEY}",
        "Content-Type": "application/json",
        "Prefer": "return=minimal"
    }

    requests.patch(url, json=payload, headers=headers)


# =====================
# DOWNLOAD IMAGE BASE64
# =====================

def download_image_base64(url):

    res = requests.get(url)

    if res.status_code != 200:
        raise Exception("Failed to download source image")

    return base64.b64encode(res.content).decode("utf-8")


# =====================
# GENERATE VIDEO
# =====================

@app.post("/generate-video")
def generate_video(req: GenerateVideoRequest):

    try:

        print("Starting Veo3 generation:", req.generation_id)

        update_generation(req.generation_id, "processing")

        image_base64 = download_image_base64(req.image_url)

        endpoint = (
            "https://generativelanguage.googleapis.com/v1beta/"
            "models/veo-3.1-generate-preview:predictLongRunning"
        )

        headers = {
            "x-goog-api-key": GEMINI_API_KEY,
            "Content-Type": "application/json"
        }

        full_prompt = f"""
{req.prompt}

Script (spoken naturally in Brazilian Portuguese):
{req.script_text}

Requirements:
- perfect lip sync
- natural speech timing
- realistic facial movement
- cinematic realism
"""

        payload = {
            "instances": [
                {
                    "prompt": full_prompt,
                    "image": {
                        "inlineData": {
                            "mimeType": "image/jpeg",
                            "data": image_base64
                        }
                    }
                }
            ],
            "parameters": {
                "aspectRatio": req.aspect_ratio,
                "durationSeconds": req.duration
            }
        }

        print("Submitting to Veo3")

        response = requests.post(endpoint, json=payload, headers=headers)

        if response.status_code != 200:
            raise Exception(response.text)

        operation = response.json()["name"]

        print("Operation:", operation)

        poll_url = f"https://generativelanguage.googleapis.com/v1beta/{operation}"

        while True:

            poll = requests.get(poll_url, headers=headers)

            if poll.status_code != 200:
                raise Exception(poll.text)

            data = poll.json()

            if data.get("done"):

                video_uri = (
                    data["response"]["candidates"][0]
                    ["content"]["parts"][0]
                    ["fileData"]["uri"]
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
