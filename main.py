import os
import requests
import time
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_SERVICE_KEY = os.environ["SUPABASE_SERVICE_ROLE_KEY"]
VEO3_API_KEY = os.environ["VEO3_API_KEY"]

SUPABASE_STORAGE = f"{SUPABASE_URL}/storage/v1/object"
SUPABASE_REST = f"{SUPABASE_URL}/rest/v1"

app = FastAPI()


class GenerateVideoRequest(BaseModel):
    generation_id: str
    image_url: str
    script_text: str
    prompt: str
    aspect_ratio: str
    duration: int


def update_generation(generation_id, status=None, final_video_url=None):
    headers = {
        "apikey": SUPABASE_SERVICE_KEY,
        "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
        "Content-Type": "application/json"
    }

    data = {}

    if status:
        data["status"] = status

    if final_video_url:
        data["final_video_url"] = final_video_url

    requests.patch(
        f"{SUPABASE_REST}/video_generations?id=eq.{generation_id}",
        headers=headers,
        json=data
    )


def upload_to_supabase(video_bytes, generation_id):

    path = f"video-final/{generation_id}.mp4"

    headers = {
        "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
        "Content-Type": "video/mp4"
    }

    upload_url = f"{SUPABASE_STORAGE}/creative-media/{path}"

    res = requests.put(upload_url, headers=headers, data=video_bytes)

    if res.status_code not in [200, 201]:
        raise Exception("Upload failed")

    public_url = f"{SUPABASE_URL}/storage/v1/object/public/creative-media/{path}"

    return public_url


def generate_with_veo3(req):

    headers = {
        "Authorization": f"Bearer {VEO3_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "prompt": req.prompt + f"\nCharacter speaking: {req.script_text}",
        "image_url": req.image_url,
        "aspect_ratio": req.aspect_ratio,
        "duration": req.duration
    }

    submit = requests.post(
        "https://api.veo3.ai/v1/generate",
        headers=headers,
        json=payload
    )

    if submit.status_code != 200:
        raise Exception("Veo3 submit failed")

    job_id = submit.json()["job_id"]

    # polling

    for _ in range(300):

        status = requests.get(
            f"https://api.veo3.ai/v1/status/{job_id}",
            headers=headers
        ).json()

        if status["status"] == "completed":

            video_url = status["video_url"]

            video_bytes = requests.get(video_url).content

            return video_bytes

        time.sleep(5)

    raise Exception("Timeout Veo3")


@app.post("/generate-video")
def generate_video(req: GenerateVideoRequest):

    try:

        update_generation(req.generation_id, status="processing")

        video_bytes = generate_with_veo3(req)

        public_url = upload_to_supabase(video_bytes, req.generation_id)

        update_generation(
            req.generation_id,
            status="completed",
            final_video_url=public_url
        )

        return {"status": "ok"}

    except Exception as e:

        update_generation(req.generation_id, status="failed")

        raise HTTPException(500, str(e))
