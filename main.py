from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
import time
import os
import uuid

app = FastAPI()

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_SERVICE_ROLE_KEY = os.environ["SUPABASE_SERVICE_ROLE_KEY"]
VEO3_API_KEY = os.environ["VEO3_API_KEY"]

class GenerateVideoRequest(BaseModel):
    generation_id: str
    image_url: str
    script_text: str | None = None
    prompt: str
    aspect_ratio: str = "9:16"
    duration: int = 8


@app.post("/generate-video")
def generate_video(req: GenerateVideoRequest):

    try:

        print("Starting Veo3 generation...")

        # STEP 1 — Submit Veo3 job
        submit = requests.post(
            "https://api.veo3.ai/v1/video/generate",
            headers={
                "Authorization": f"Bearer {VEO3_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "image_url": req.image_url,
                "prompt": req.prompt,
                "script": req.script_text,
                "aspect_ratio": req.aspect_ratio,
                "duration": req.duration
            }
        )

        if submit.status_code != 200:
            raise Exception(submit.text)

        job_id = submit.json()["job_id"]

        print(f"Veo3 job_id: {job_id}")

        # STEP 2 — Poll Veo3
        video_url = None

        for i in range(120):

            poll = requests.get(
                f"https://api.veo3.ai/v1/video/status/{job_id}",
                headers={"Authorization": f"Bearer {VEO3_API_KEY}"}
            )

            data = poll.json()

            print(f"Poll status: {data}")

            if data["status"] == "completed":
                video_url = data["video_url"]
                break

            time.sleep(5)

        if not video_url:
            raise Exception("Timeout waiting Veo3")

        print(f"Video ready: {video_url}")

        # STEP 3 — Download video
        video_bytes = requests.get(video_url).content

        filename = f"{req.generation_id}.mp4"

        # STEP 4 — Upload to Supabase Storage
        upload = requests.post(
            f"{SUPABASE_URL}/storage/v1/object/video-final/{filename}",
            headers={
                "Authorization": f"Bearer {SUPABASE_SERVICE_ROLE_KEY}",
                "Content-Type": "video/mp4"
            },
            data=video_bytes
        )

        public_url = f"{SUPABASE_URL}/storage/v1/object/public/video-final/{filename}"

        print(f"Uploaded to Supabase: {public_url}")

        # STEP 5 — Update DB
        requests.patch(
            f"{SUPABASE_URL}/rest/v1/video_generations?id=eq.{req.generation_id}",
            headers={
                "Authorization": f"Bearer {SUPABASE_SERVICE_ROLE_KEY}",
                "Content-Type": "application/json",
                "apikey": SUPABASE_SERVICE_ROLE_KEY
            },
            json={
                "status": "completed",
                "final_video_url": public_url
            }
        )

        return {"status": "success", "video_url": public_url}

    except Exception as e:

        print(f"ERROR: {e}")

        requests.patch(
            f"{SUPABASE_URL}/rest/v1/video_generations?id=eq.{req.generation_id}",
            headers={
                "Authorization": f"Bearer {SUPABASE_SERVICE_ROLE_KEY}",
                "Content-Type": "application/json",
                "apikey": SUPABASE_SERVICE_ROLE_KEY
            },
            json={"status": "failed"}
        )

        raise HTTPException(status_code=500, detail=str(e))
