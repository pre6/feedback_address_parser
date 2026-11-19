# Build from the repo root:  docker build -f trainer/Dockerfile .
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app
WORKDIR /app

COPY trainer/requirements.txt trainer/requirements.txt
RUN pip install --no-cache-dir -r trainer/requirements.txt

COPY shared ./shared
COPY trainer ./trainer

# Default: the full loop (build -> train -> evaluate -> promote -> tell the API to reload).
# data/ and models/ are mounted in by docker-compose, not baked into the image.
CMD ["python", "trainer/retrain.py", "--api-url", "http://api:8000"]
