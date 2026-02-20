import os
import requests
import time
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.environ.get("SUPABASE_SERVICE_KEY")
VEO3_API_KEY = os.environ.get("VEO3_API_KEY")

app = FastAPI()

class GenerateVideoRequest(BaseModel):
    generation_id: str
    image_url: str
    script_text: str
    prompt: str
    aspect_ratio: str
    duration: int


def update_generation_status(generation_id, status, final_video_url=None):
    url = f"{SUPABASE_URL}/rest/v1/video_generations?id=eq.{generation_id}"

    payload = {
        "status": status
    }

    if final_video_url:
        payload["final_video_url"] = final_video_url

    headers = {
        "apikey": SUPABASE_SERVICE_KEY,
        "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
        "Content-Type": "application/json",
        "Prefer": "return=minimal"
    }

    requests.patch(url, json=payload, headers=headers)


@app.post("/generate-video")
def generate_video(req: GenerateVideoRequest):

    try:

        generation_id = req.generation_id

        print("Starting Veo3 generation:", generation_id)

        update_generation_status(generation_id, "processing")

        # CALL VEO3 VIA FAL.AI
        fal_url = "https://fal.run/fal-ai/google/veo3"

        fal_headers = {
            "Authorization": f"Key {os.environ.get('FAL_API_KEY')}",
            "Content-Type": "application/json"
        }

        fal_payload = {
            "prompt": f"{req.prompt}\n\nScript:\n{req.script_text}",
            "image_url": req.image_url,
            "aspect_ratio": req.aspect_ratio,
            "duration": req.duration
        }

        res = requests.post(fal_url, json=fal_payload, headers=fal_headers)

        if res.status_code != 200:
            raise Exception(res.text)

        job = res.json()

        job_id = job.get("id")

        if not job_id:
            raise Exception("No Veo job id")

        print("Veo job_id:", job_id)

        # POLL RESULT
        status_url = f"https://fal.run/fal-ai/google/veo3/{job_id}"

        while True:

            poll = requests.get(status_url, headers=fal_headers)
            data = poll.json()

            if data.get("status") == "COMPLETED":

                video_url = data["video"]["url"]

                print("Video ready:", video_url)

                update_generation_status(
                    generation_id,
                    "completed",
                    video_url
                )

                return {
                    "status": "completed",
                    "video_url": video_url
                }

            if data.get("status") == "FAILED":
                raise Exception("Veo failed")

            print("Still processing...")
            time.sleep(5)

    except Exception as e:

        print("ERROR:", str(e))

        update_generation_status(generation_id, "failed")

        raise HTTPException(status_code=500, detail=str(e))
