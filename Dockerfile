# Minimal SDXL LoRA Training API
# Fast build, serverless-ready

FROM python:3.11-slim

WORKDIR /app

# Install deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps
RUN pip install --no-cache-dir \
    torch==2.5.1 torchvision==0.20.1 torchaudio --index-url https://download.pytorch.org/whl/cu121 \
    accelerate \
    transformers==4.45.0 \
    diffusers==0.31.0 \
    safetensors \
    huggingface_hub \
    einops \
    tqdm \
    requests \
    numpy \
    scipy \
    peft \
    fastapi \
    uvicorn \
    runpod

# Install kohya-ss train_network (minimal clone)
RUN git clone --depth 1 --branch v25.2.1 https://github.com/bmaltais/kohya_ss.git /kohya \
    && cd /kohya \
    && pip install --no-cache-dir -r requirements.txt \
    && rm -rf /kohya/.git

# Copy app
COPY handler.py .
COPY train_lora.py .

EXPOSE 8000

CMD ["python", "handler.py"]
