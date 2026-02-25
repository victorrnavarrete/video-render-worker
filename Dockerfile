FROM python:3.11-slim-bullseye
WORKDIR /app

# Instala FFmpeg + dependencias do sistema para Playwright Chromium
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libnss3 \
    libnspr4 \
    libatk1.0-0 \
    libatk-bridge2.0-0 \
    libcups2 \
    libxcomposite1 \
    libxdamage1 \
    libxfixes3 \
    libxrandr2 \
    libgbm1 \
    libxkbcommon0 \
    libpango-1.0-0 \
    libcairo2 \
    libasound2 \
    libatspi2.0-0 \
    libwayland-client0 \
    libx11-6 \
    libx11-xcb1 \
    libxcb1 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Instala o browser Chromium para o Playwright
RUN playwright install chromium

COPY main.py .
EXPOSE 8080
CMD ["sh", "-c", "echo \"$GOOGLE_APPLICATION_CREDENTIALS_JSON\" > /tmp/service-account.json && uvicorn main:app --host 0.0.0.0 --port 8080"]
