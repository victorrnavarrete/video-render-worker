FROM python:3.11-slim-bullseye

WORKDIR /app

COPY requirements.txt .

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .

EXPOSE 8080

CMD ["sh", "-c", "echo $GOOGLE_APPLICATION_CREDENTIALS_JSON > /tmp/service-account.json && export GOOGLE_APPLICATION_CREDENTIALS=/tmp/service-account.json && uvicorn main:app --host 0.0.0.0 --port 8080"]
