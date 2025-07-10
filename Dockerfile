# Use docker image for cuda
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

# python, git, pip
RUN apt-get update && apt-get install -y \
    python3.10 python3-pip python3.10-venv git \
    && ln -sf python3.10 /usr/bin/python \
    && ln -sf pip3 /usr/bin/pip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy torch wheel locally
COPY whl/torch-2.5.1-cp310-cp310-manylinux1_x86_64.whl /app/whl/

# Copy requirements file
COPY requirements.txt .

# Install torch wheel locally
RUN pip install /app/whl/torch-2.5.1-cp310-cp310-manylinux1_x86_64.whl

# Install other dependencies without torch to avoid double install
RUN pip install --default-timeout=1000 --no-cache-dir --no-deps -r requirements.txt

# Copy project files
COPY . .

# Copy pre-downloaded ViT-B-32.pt model to avoid re-download on runtime
COPY ViT-B-32.pt /root/.cache/clip/ViT-B-32.pt

# Expose port and start server
CMD ["uvicorn", "main:AI_services", "--host", "0.0.0.0", "--port", "8000"]
