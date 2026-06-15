FROM python:3.11-slim

WORKDIR /app

# system deps — gcc needed to compile some spaCy wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# copy requirements first — Docker layer caches this separately
# so spaCy doesn't reinstall on every code change
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# copy trainer code
COPY train.py .
COPY config.cfg .

# volumes are mounted at runtime — don't COPY them
# /models        → named Docker volume (read-write, shared with api)
# /training-data → bind mount from host (read-only)

ENV SPACY_VERSION=3.7.2
ENV MODEL_OUTPUT_DIR=/models
ENV TRAINING_DATA_DIR=/training-data

ENTRYPOINT ["python", "train.py"]