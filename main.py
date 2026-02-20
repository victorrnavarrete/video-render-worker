import os
import time
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


# ========================
# Update Supabase
# ========================

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


# ========================
# Upload image to Gemini Files API
# ========================

def upload_image_to_gemini(image_url):

    print("Downloading source image")

    image = requests.get(image_url)

    if image.status_code != 200:
        raise Exception("Failed to download image")

    print("Uploading to Gemini Files API")

    upload_url = "https://generativelanguage.googleapis.com/upload/v1beta/files"

    headers = {
        "x-goog-api-key": GEMINI_API_KEY
    }

    files = {
        "file": ("image.jpg", image.content, "image/jpeg")
    }

    res = requests.post(upload_url, headers=headers, files=files)

    if res.status_code != 200:
        raise Exception(res.text)

    file_uri = res.json()["file"]["uri"]

    print("Uploaded file_uri:", file_uri)

    return file_uri


# ========================
# Generate video
# ========================

@app.post("/generate-video")
def generate_video(req: GenerateVideoRequest):

    try:

        print("Starting generation:", req.generation_id)

        update_generation(req.generation_id, "processing")

        file_uri = upload_image_to_gemini(req.image_url)

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

Script (Brazilian Portuguese):
{req.script_text}

Requirements:
perfect lip sync
photorealistic
cinematic
natural speech timing
"""

        payload = {

            "instances": [

                {
                    "prompt": full_prompt,

                    "image": {

                        "fileData": {

                            "mimeType": "image/jpeg",
                            "fileUri": file_uri

                        }
                    }
                }
            ],

            "parameters": {

                "aspectRatio": req.aspect_ratio,
                "durationSeconds": req.duration

            }
        }

        print("Submitting Veo3 job")

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

            print("Still processing")

            time.sleep(10)

    except Exception as e:

        print("ERROR:", str(e))

        update_generation(req.generation_id, "failed")

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
