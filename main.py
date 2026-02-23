import os
import time
import base64
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from google import genai
from google.genai import types

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_SERVICE_ROLE_KEY = os.environ["SUPABASE_SERVICE_ROLE_KEY"]

app = FastAPI()

client = genai.Client()  # usa Vertex AI automaticamente quando GOOGLE_GENAI_USE_VERTEXAI=true


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

    if res.status_code not in (200, 204):
        raise Exception(res.text)


def download_image_base64(image_url):

    print("Downloading image...")

    res = requests.get(image_url)

    if res.status_code != 200:
        raise Exception("Failed to download image")

    image_bytes = res.content

    image_base64 = base64.b64encode(image_bytes).decode("utf-8")

    mime_type = "image/jpeg"

    if image_url.lower().endswith(".png"):
        mime_type = "image/png"

    return image_base64, mime_type


@app.post("/generate-video")
def generate_video(req: GenerateVideoRequest):

    generation_id = req.generation_id

    try:

        print(f"Starting Veo 3.1 generation: {generation_id}")

        update_generation(generation_id, "processing")

        image_base64, mime_type = download_image_base64(req.image_url)

        full_prompt = f"{req.prompt}\n\nScript:\n{req.script_text}"

        print("Calling Veo 3.1 via Vertex AI...")

        operation = client.models.generate_videos(
            model="veo-3.1-generate-preview",
            prompt=full_prompt,
            image=types.Image(
    image_bytes=base64.b64decode(image_base64),
    mime_type=mime_type
),
            config=types.GenerateVideosConfig(
                aspect_ratio=req.aspect_ratio
            )
        )

        print("Waiting for Veo generation...")

        while not operation.done:
            time.sleep(10)
            operation = client.operations.get(operation.name)

        if not operation.response.generated_videos:
            raise Exception("No video returned")

        video = operation.response.generated_videos[0]

        output_path = f"/tmp/{generation_id}.mp4"

        print("Downloading video...")

        client.files.download(
            file=video.video,
            path=output_path
        )

        print("Uploading video to Supabase Storage...")

        with open(output_path, "rb") as f:

            upload_url = f"{SUPABASE_URL}/storage/v1/object/creative-media/video-final/{generation_id}.mp4"

            headers = {
                "Authorization": f"Bearer {SUPABASE_SERVICE_ROLE_KEY}",
                "Content-Type": "video/mp4"
            }

            res = requests.put(upload_url, data=f, headers=headers)

            if res.status_code not in (200, 201):
                raise Exception(res.text)

        public_url = f"{SUPABASE_URL}/storage/v1/object/public/creative-media/video-final/{generation_id}.mp4"

        update_generation(generation_id, "completed", public_url)

        print("SUCCESS:", public_url)

        return {"status": "completed", "video_url": public_url}

    except Exception as e:

        print("ERROR:", str(e))

        update_generation(generation_id, "failed")

        raise HTTPException(status_code=500, detail=str(e))
