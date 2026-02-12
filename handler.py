"""
LoRA Training API for RunPod Serverless
Uses kohya-ss train_network.py
"""

import os
import subprocess
import json
import uuid
from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
from typing import Optional

app = FastAPI()

class TrainRequest(BaseModel):
    model_path: str
    instance_data_dir: str
    output_dir: str = "/tmp/output"
    network_dim: int = 32
    network_alpha: int = 32
    learning_rate: float = 0.0001
    max_train_steps: int = 1000
    batch_size: int = 1
    resolution: str = "1024,1024"

class TrainResponse(BaseModel):
    job_id: str
    status: str

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/")
def root():
    return {
        "service": "LoRA Training API",
        "version": "1.0.0",
        "endpoints": ["/health", "/train", "/status/{job_id}"]
    }

@app.post("/train", response_model=TrainResponse)
def train(req: TrainRequest, background_tasks: BackgroundTasks):
    job_id = str(uuid.uuid4())[:8]
    
    def run_training():
        cmd = [
            "accelerate", "launch",
            "--mixed_precision", "fp16",
            "--num_cpu_threads_per_process", "4",
            "networks/train_network.py",
            "--pretrained_model_name_or_path", req.model_path,
            "--instance_data_dir", req.instance_data_dir,
            "--output_dir", req.output_dir,
            "--network_dim", str(req.network_dim),
            "--network_alpha", str(req.network_alpha),
            "--learning_rate", str(req.learning_rate),
            "--max_train_steps", str(req.max_train_steps),
            "--batch_size", str(req.batch_size),
            "--resolution", req.resolution,
            "--network_module", "networks.lora",
            "--network_train_type", "LoRA",
            "--save_precision", "fp16",
            "--optimizer", "AdamW8bit",
            "--cache_latents",
            "--gradient_checkpointing",
            "--output_name", f"lora_{job_id}",
        ]
        
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = "0"
        
        try:
            result = subprocess.run(
                cmd,
                cwd="/kohya",
                capture_output=True,
                text=True,
                timeout=7200,
                env=env
            )
            
            # Find output file
            output_file = None
            lora_path = f"{req.output_dir}/lora_{job_id}.safetensors"
            if os.path.exists(lora_path):
                output_file = lora_path
            
            status = "completed" if result.returncode == 0 else "failed"
            
            with open(f"/tmp/status_{job_id}.json", "w") as f:
                json.dump({
                    "job_id": job_id,
                    "status": status,
                    "output_file": output_file,
                    "returncode": result.returncode,
                }, f)
        except Exception as e:
            with open(f"/tmp/status_{job_id}.json", "w") as f:
                json.dump({"job_id": job_id, "status": "error", "error": str(e)}, f)
    
    background_tasks.add_task(run_training)
    
    return TrainResponse(job_id=job_id, status="started")

@app.get("/status/{job_id}")
def get_status(job_id: str):
    status_file = f"/tmp/status_{job_id}.json"
    if os.path.exists(status_file):
        with open(status_file) as f:
            return json.load(f)
    return {"job_id": job_id, "status": "pending"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
