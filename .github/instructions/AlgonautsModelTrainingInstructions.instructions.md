---
applyTo: 'algonauts_v4.ipynb'
---
Provide project context and coding guidelines that AI should follow when generating code, answering questions, or reviewing changes.

# MultimodalTRIBE Training Setup and Best Practices

This document summarizes recommended settings for training on a 6 GB GPU (Lenovo LOQ with RTX 4050, PyTorch 2.5.1, CUDA 12.1).

## 1. CUDA/PyTorch Environment Settings
- **Enable cuDNN autotune** for fixed-size layers:  
  ```python
  import torch
  torch.backends.cudnn.benchmark = True
  ```  
  *(If input sizes vary a lot, it may incur overhead.)*

- **Enable TF32** on Ampere GPUs:  
  ```python
  torch.backends.cuda.matmul.allow_tf32 = True
  ```  

- **Preallocate memory** to avoid fragmentation:
  ```python
  dummy = torch.zeros(batch_size, max_seq_len, device='cuda')
  output = model(dummy)
  output.sum().backward()
  torch.cuda.empty_cache()
  ```

- **Memory fragmentation setting (optional)**:
  ```bash
  export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32
  ```

## 2. Mixed Precision (AMP)
- Use PyTorch’s native AMP:
  ```python
  from torch.cuda.amp import autocast, GradScaler
  scaler = GradScaler()
  for data, target in loader:
      with autocast():
          loss = model(data).loss(target)
      scaler.scale(loss).backward()
      scaler.step(optimizer)
      scaler.update()
  ```

- **Lightning example**:
  ```python
  from pytorch_lightning import Trainer
  trainer = Trainer(
      accelerator="gpu", devices=1,
      precision=16,
      accumulate_grad_batches=4,
      strategy="deepspeed_stage_2_offload"
  )
  ```

## 3. Batch Size and Sequence Length
- Start with `batch_size=4`, `seq_len=256`, or reduce to `batch=2`, `seq_len=384` if needed.
- Use **gradient accumulation** to scale effective batch size:
  ```python
  TrainingArguments(
      per_device_train_batch_size=4,
      gradient_accumulation_steps=4
  )
  ```

## 4. Memory/Parallelism Optimizations
- **Gradient Checkpointing**:
  ```python
  TrainingArguments(..., gradient_checkpointing=True)
  ```

- **DeepSpeed ZeRO Offload** config:
  ```json
  {
    "zero_optimization": {
      "stage": 2,
      "offload_optimizer": { "device": "cpu" },
      "contiguous_gradients": true
    }
  }
  ```

## 5. PyTorch Lightning / HuggingFace Accelerate
- Mixed Precision:
  - `Trainer(precision=16)` or `precision="bf16"`
  - `accelerate launch --mixed_precision fp16`

- DeepSpeed: 
  - `strategy="deepspeed_stage_2_offload"` in Lightning
  - Include DeepSpeed config in `accelerate`

## 6. Additional Best Practices
- Use `torch.compile(model)` to optimize model
- Monitor with `nvidia-smi` and avoid unnecessary synchronizations