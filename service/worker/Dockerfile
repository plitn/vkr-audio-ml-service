FROM python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    ninja-build \
    pkg-config \
    protobuf-compiler \
    libsndfile1 \
    libgomp1 \
    ffmpeg \
    sox \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements-base.txt requirements-ml.txt constraints.txt ./
RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -r requirements-base.txt \
    && pip install --no-cache-dir -c constraints.txt -r requirements-ml.txt

COPY . .

CMD ["python", "main.py"]
