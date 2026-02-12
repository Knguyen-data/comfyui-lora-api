# LoRA Training API

Serverless API endpoint for training SDXL LoRA models using kohya-ss.

## Features
- FastAPI REST endpoint
- kohya-ss v25.2.1 training
- Async job processing
- RunPod serverless ready

## RunPod Config
- **Repo:** `https://github.com/Knguyen-data/comfyui-lora-api`
- **Port:** `8000`
- **Dockerfile:** `Dockerfile`
- **Env:** `RUNPOD_API_KEY` (optional, for serverless)

## API Endpoints

### Health
```
GET /health
```

### Train
```
POST /train
{
  "model_path": "/path/to/model.safetensors",
  "instance_data_dir": "/path/to/images",
  "output_dir": "/tmp/output",
  "network_dim": 32,
  "network_alpha": 32,
  "learning_rate": 0.0001,
  "max_train_steps": 1000,
  "batch_size": 1,
  "resolution": "1024,1024"
}
```

### Status
```
GET /status/{job_id}
```

## Response
```json
{
  "job_id": "abc123",
  "status": "started"
}
```
