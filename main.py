import os
import time
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY") or os.environ.get("VEO3_API_KEY")

if not SUPABASE_URL:
    raise Exception("SUPABASE_URL not set")

if not SUPABASE_SERVICE_ROLE_KEY:
    raise Exception("SUPABASE_SERVICE_ROLE_KEY not set")

if not GEMINI_API_KEY:
    raise Exception("GEMINI_API_KEY or VEO3_API_KEY not set")

app = FastAPI()

MODEL_NAME = "veo-3.0-fast-generate-preview"


class GenerateVideoRequest(BaseModel):
    generation_id: str
    image_url: str
    script_text: str
    prompt: str
    aspect_ratio: str
    duration: int


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

    res = requests.patch(url, json=payload, headers=headers)

    if res.status_code >= 300:
        print("Supabase update error:", res.text)


@app.post("/generate-video")
def generate_video(req: GenerateVideoRequest):

    try:

        generation_id = req.generation_id

        print("Starting Veo3 generation:", generation_id)

        update_generation(generation_id, "processing")

        prompt = f"""
{req.prompt}

The person speaks the following script in Brazilian Portuguese:

"{req.script_text}"

The avatar lip-sync must match speech exactly.
Photorealistic human motion.
Natural facial muscle movement.
Cinematic lighting.
"""

        create_url = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL_NAME}:generateVideo?key={GEMINI_API_KEY}"

        create_payload = {
            "prompt": prompt,
            "config": {
                "aspectRatio": req.aspect_ratio,
                "durationSeconds": req.duration
            }
        }

        create_res = requests.post(create_url, json=create_payload)

        if create_res.status_code >= 300:
            raise Exception(create_res.text)

        operation = create_res.json()

        operation_name = operation["name"]

        print("Operation:", operation_name)

        poll_url = f"https://generativelanguage.googleapis.com/v1beta/{operation_name}?key={GEMINI_API_KEY}"

        while True:

            poll_res = requests.get(poll_url)

            if poll_res.status_code >= 300:
                raise Exception(poll_res.text)

            poll = poll_res.json()

            if poll.get("done"):

                if "error" in poll:
                    raise Exception(poll["error"])

                video_uri = poll["response"]["video"]["uri"]

                print("Video ready:", video_uri)

                update_generation(
                    generation_id,
                    "completed",
                    video_uri
                )

                return {
                    "status": "completed",
                    "video_url": video_uri
                }

            print("Still processing Veo3...")
            time.sleep(5)

    except Exception as e:

        print("ERROR:", str(e))

        update_generation(generation_id, "failed")

        raise HTTPException(status_code=500, detail=str(e))
