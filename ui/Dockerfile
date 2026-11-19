# Build from the repo root:  docker build -f ui/Dockerfile .
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1
WORKDIR /app

COPY ui/requirements.txt ui/requirements.txt
RUN pip install --no-cache-dir -r ui/requirements.txt

COPY ui ./ui

EXPOSE 8501
CMD ["streamlit", "run", "ui/app.py", \
     "--server.address=0.0.0.0", "--server.port=8501", \
     "--server.headless=true", "--browser.gatherUsageStats=false"]
