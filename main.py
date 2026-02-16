from fastapi import FastAPI, Request, Header, HTTPException
import subprocess
import uuid
import os
import requests

app = FastAPI()

RENDER_WORKER_SECRET = os.environ.get("RENDER_WORKER_SECRET")

@app.get("/")
def health():
    return {"status": "ok"}

@app.post("/merge")
async def merge_video_audio(
    request: Request,
    authorization: str = Header(None)
):
    if authorization != f"Bearer {RENDER_WORKER_SECRET}":
        raise HTTPException(status_code=401, detail="Unauthorized")

    data = await request.json()

    video_url = data["video_url"]
    audio_url = data["audio_url"]

    video_file = f"/tmp/{uuid.uuid4()}_video.mp4"
    audio_file = f"/tmp/{uuid.uuid4()}_audio.mp3"
    output_file = f"/tmp/{uuid.uuid4()}_merged.mp4"

    # download video
    with open(video_file, "wb") as f:
        f.write(requests.get(video_url).content)

    # download audio
    with open(audio_file, "wb") as f:
        f.write(requests.get(audio_url).content)

    # merge with ffmpeg
    subprocess.run([
        "ffmpeg",
        "-i", video_file,
        "-i", audio_file,
        "-c:v", "copy",
        "-c:a", "aac",
        "-shortest",
        output_file
    ])

    return {
        "status": "success",
        "final_video_url": output_file
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)

