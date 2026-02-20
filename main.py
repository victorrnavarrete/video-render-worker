import os
import requests
import time

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


# =========================
# ENV VARIABLES
# =========================

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
VEO3_API_KEY = os.environ.get("VEO3_API_KEY")


if not SUPABASE_URL:
    raise Exception("SUPABASE_URL not set")

if not SUPABASE_SERVICE_ROLE_KEY:
    raise Exception("SUPABASE_SERVICE_ROLE_KEY not set")

if not VEO3_API_KEY:
    raise Exception("VEO3_API_KEY not set")


# =========================
# FASTAPI INIT
# =========================

app = FastAPI(title="Veo3 Video Render Worker")


# =========================
# REQUEST MODEL
# =========================

class GenerateVideoRequest(BaseModel):
    generation_id: str
    image_url: str
    script_text: str | None = None
    prompt: str
    aspect_ratio: str = "9:16"
    duration: int = 8


# =========================
# SUPABASE UPDATE FUNCTION
# =========================

def update_generation_status(
    generation_id: str,
    status: str,
    final_video_url: str | None = None
):

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

    response = requests.patch(url, json=payload, headers=headers)

    if response.status_code not in [200, 204]:
        raise Exception(f"Supabase update failed: {response.text}")


# =========================
# HEALTH CHECK
# =========================

@app.get("/")
def health():
    return {"status": "ok"}


# =========================
# MAIN VIDEO GENERATION
# =========================

@app.post("/generate-video")
def generate_video(req: GenerateVideoRequest):

    generation_id = req.generation_id

    try:

        print(f"Starting Veo3 generation: {generation_id}")

        update_generation_status(generation_id, "processing")

        # =========================
        # BUILD FINAL PROMPT
        # =========================

        final_prompt = req.prompt

        if req.script_text:
            final_prompt += f"\n\nCharacter says in Brazilian Portuguese:\n{req.script_text}"

        # =========================
        # SUBMIT VEO3 JOB (DIRECT)
        # =========================

        veo_submit_url = "https://api.veo.ai/v1/video/image-to-video"

        headers = {
            "Authorization": f"Bearer {VEO3_API_KEY}",
            "Content-Type": "application/json"
        }

        payload = {
            "image_url": req.image_url,
            "prompt": final_prompt,
            "aspect_ratio": req.aspect_ratio,
            "duration": req.duration,
            "quality": "high",
            "identity_preservation": True,
            "speech": True
        }

        submit_response = requests.post(
            veo_submit_url,
            headers=headers,
            json=payload
        )

        if submit_response.status_code != 200:
            raise Exception(
                f"Veo submit failed: {submit_response.text}"
            )

        submit_data = submit_response.json()

        job_id = submit_data.get("id")

        if not job_id:
            raise Exception("No Veo3 job id returned")

        print(f"Veo3 job submitted: {job_id}")

        # =========================
        # POLL RESULT
        # =========================

        poll_url = f"https://api.veo.ai/v1/video/{job_id}"

        video_url = None

        max_wait_seconds = 1800
        poll_interval = 5

        start_time = time.time()

        while True:

            elapsed = time.time() - start_time

            if elapsed > max_wait_seconds:
                raise Exception("Veo3 timeout")

            poll_response = requests.get(
                poll_url,
                headers=headers
            )

            if poll_response.status_code != 200:
                raise Exception(
                    f"Poll failed: {poll_response.text}"
                )

            poll_data = poll_response.json()

            status = poll_data.get("status")

            print(f"Polling Veo3 status: {status}")

            if status == "completed":

                video_url = poll_data.get("video_url")

                if not video_url:
                    raise Exception("No video_url returned")

                print(f"Video ready: {video_url}")

                break

            if status == "failed":
                raise Exception("Veo3 generation failed")

            time.sleep(poll_interval)

        # =========================
        # UPDATE SUPABASE
        # =========================

        update_generation_status(
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

        try:
            update_generation_status(
                generation_id,
                "failed"
            )
        except:
            pass

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
