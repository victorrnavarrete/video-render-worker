import os
import time
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# ENV VARS
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
VEO3_API_KEY = os.environ.get("VEO3_API_KEY")

if not SUPABASE_URL:
    raise Exception("SUPABASE_URL not set")

if not SUPABASE_SERVICE_ROLE_KEY:
    raise Exception("SUPABASE_SERVICE_ROLE_KEY not set")

if not VEO3_API_KEY:
    raise Exception("VEO3_API_KEY not set")

app = FastAPI()


# REQUEST MODEL
class GenerateVideoRequest(BaseModel):
    generation_id: str
    image_url: str
    script_text: str
    prompt: str
    aspect_ratio: str
    duration: int


# UPDATE SUPABASE STATUS
def update_generation(generation_id: str, status: str, final_video_url: str = None):

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
        raise Exception(f"Supabase update failed: {res.text}")


# GENERATE VIDEO ENDPOINT
@app.post("/generate-video")
def generate_video(req: GenerateVideoRequest):

    generation_id = req.generation_id

    try:

        print(f"Starting Veo3 generation: {generation_id}")

        update_generation(generation_id, "processing")

        # Gemini Veo3 endpoint
        veo_url = f"https://generativelanguage.googleapis.com/v1beta/models/veo-3.0-generate-001:predict?key={VEO3_API_KEY}"

        prompt = f"""
{req.prompt}

Script:
{req.script_text}

Photorealistic talking avatar. Natural lip sync. Real human motion.
"""

        payload = {
            "instances": [
                {
                    "prompt": prompt,
                    "image": req.image_url,
                    "aspectRatio": req.aspect_ratio,
                    "durationSeconds": req.duration
                }
            ]
        }

        headers = {
            "Content-Type": "application/json"
        }

        res = requests.post(veo_url, json=payload, headers=headers)

        if res.status_code != 200:
            raise Exception(res.text)

        data = res.json()

        operation_name = data.get("name")

        if not operation_name:
            raise Exception("No operation returned from Veo3")

        print(f"Operation: {operation_name}")

        # POLL
        operation_url = f"https://generativelanguage.googleapis.com/v1beta/{operation_name}?key={VEO3_API_KEY}"

        while True:

            poll = requests.get(operation_url)

            if poll.status_code != 200:
                raise Exception(poll.text)

            poll_data = poll.json()

            if poll_data.get("done"):

                video_url = poll_data["response"]["video"]["uri"]

                print(f"Video ready: {video_url}")

                update_generation(
                    generation_id,
                    "completed",
                    video_url
                )

                return {
                    "status": "completed",
                    "video_url": video_url
                }

            print("Processing Veo3...")
            time.sleep(5)

    except Exception as e:

        print("ERROR:", str(e))

        try:
            update_generation(generation_id, "failed")
        except:
            pass

        raise HTTPException(status_code=500, detail=str(e))
