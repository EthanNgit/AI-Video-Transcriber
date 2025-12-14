FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
COPY requirements-ml.txt .

RUN pip install --upgrade pip setuptools wheel

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    -r requirements.txt \
    -r requirements-ml.txt

COPY . .

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
