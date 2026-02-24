FROM python:3.11-slim-bullseye
WORKDIR /app

# Instala FFmpeg para merge de videos
RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt
COPY main.py .
EXPOSE 8080
CMD ["sh", "-c", "echo \"$GOOGLE_APPLICATION_CREDENTIALS_JSON\" > /tmp/service-account.json && uvicorn main:app --host 0.0.0.0 --port 8080"]
