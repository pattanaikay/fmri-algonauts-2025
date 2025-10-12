# 🧠 fMRI–Multimodal Encoding Project Roadmap

## 🎯 Goal
Develop and optimize a multimodal transformer model (TRIBE-style) to predict fMRI responses from text, audio, and video stimuli in the **Algonauts 2025 challenge**.

We’ll proceed in **three phases**:
1. **Toy Dataset** → controlled hyperparameter sweeps and architecture exploration.  
2. **10 GB Subset** → scaling, CUDA optimization, and pipeline validation.  
3. **Full Dataset** → production-scale training and submission-ready model.

---

## 🚀 Phase 1 — Toy Dataset Experiments

**Objective:** Find promising architecture + hyperparameters using synthetic data. Validate pipeline, logging, and CUDA setup.

### 🧩 A. Model Architecture Experiments
| Task ID | Description | Parameters / Notes | Deliverables |
|----------|--------------|--------------------|---------------|
| A1 | Projection dimension sweep | `proj_dim ∈ {128, 256, 512}` | Compare train loss & Val Pearson |
| A2 | Transformer depth | `transformer_layers ∈ {2, 4, 8}` | Monitor generalization vs overfit |
| A3 | Attention heads | `nheads ∈ {4, 8, 12}` | Check effect on temporal fusion |
| A4 | Feed-forward hidden dim | `ff_dim ∈ {2×, 4× d_model}` | Note compute cost trade-offs |
| A5 | Pooling comparison | AvgPool vs AttentionPool | Implement alt pooling layer |
| A6 | Subject embedding ablation | On vs Off (`subj_emb`, `subj_bias`) | Evaluate subject conditioning impact |

### ⚙️ B. Regularization Experiments
| ID | Parameter | Values / Strategy | Deliverable |
|----|------------|------------------|--------------|
| B1 | Transformer dropout | `dropout ∈ {0.1, 0.3, 0.5}` | Prevent overfit |
| B2 | Modality dropout | `modality_dropout_p ∈ {0.0, 0.2, 0.4}` | Check modality robustness |
| B3 | Weight decay | `1e-2` vs `5e-3` vs `1e-3` | Effect on stability |

### 🧮 C. Training Dynamics Experiments
| ID | Parameter | Values | Deliverable |
|----|------------|---------|--------------|
| C1 | Learning rate | `{1e-3, 5e-4, 1e-4}` | LR–loss curves |
| C2 | Warm-up steps | `{50, 200, 500}` | LR schedule tuning |
| C3 | Optimizer type | `AdamW` vs `Adam` | Compare convergence |
| C4 | Batch size | `{8, 16, 32}` | Gradient stability |
| C5 | Scheduler shape | Linear vs Cosine | Smoothness of learning |

### 🧪 D. Ablations
| ID | Experiment | Description |
|----|-------------|-------------|
| D1 | Unimodal | Train text-only, audio-only, video-only |
| D2 | Context window | `fT ∈ {30, 60, 120}` |
| D3 | Combined modality drop | Random 1–2 modality drop at train time |

### ⚡ E. CUDA Optimization Tasks
| ID | Task | Notes |
|----|------|-------|
| E1 | Enable AMP + GradScaler | Mixed precision training |
| E2 | Gradient clipping | `max_norm=1.0` |
| E3 | DataLoader tuning | `num_workers=4–8`, `pin_memory=True` |
| E4 | Measure per-step time + memory | Use `torch.cuda.memory_allocated()` |
| E5 | Log CUDA metrics | Write to TensorBoard (`writer.add_scalar`) |

### 📊 Logging & Analysis
- All experiments logged under separate TensorBoard runs via `run_grid_search()`.  
- Log: `Loss/train_step`, `Loss/train_epoch`, `Val/Pearson`, `LR`, and GPU utilization.  
- After each sweep → record best config (JSON).

**Deliverables**
- `best_toy_config.json`  
- Step-time & memory report  
- TensorBoard screenshots of top 3 configs  

---

## 💾 Phase 2 — 10 GB Subset Experiments

**Objective:** Validate real-data pipeline, check GPU throughput, confirm hyperparam scaling, and ensure stable training.

### A. Data Tasks
| ID | Task | Description |
|----|------|--------------|
| F1 | Create balanced 10 GB subset | Equal samples from multiple stimuli types |
| F2 | Run feature extraction | Video (VideoMAE), Audio (Wav2Vec2), Text (Llama/BERT) |
| F3 | Align to 2 Hz grid + save memmaps | Temporal alignment for multimodal input |

### B. Pilot Training & Benchmarking
| ID | Task | Description |
|----|------|--------------|
| G1 | Load subset features, run top 3 configs | Verify loss ↓ and Val Pearson ↑ |
| G2 | Benchmark step time, memory, throughput | Use CUDA profiler |
| G3 | Test AMP, checkpointing, accumulation | Optimize GPU efficiency |
| G4 | Evaluate on held-out subjects | Mini-validation correlation |

### C. Reporting
| ID | Task | Deliverable |
|----|------|-------------|
| H1 | Compare top configs | Table: Val Pearson vs time/memory |
| H2 | Finalize `best_subset_config.json` | Best hyperparams for full run |
| H3 | Estimate full-data training time | Use step-time extrapolation |

---

## 🧬 Phase 3 — Full Algonauts Dataset Training

**Objective:** Train optimized multimodal transformer on complete dataset for submission.

| ID | Task | Description |
|----|------|-------------|
| I1 | Prepare full dataset pipeline | Ensure all features + metadata ready |
| I2 | Implement distributed/AMP training | Multi-GPU DDP or single-GPU fallback |
| I3 | Train using `best_subset_config.json` | Monitor loss & Val Pearson |
| I4 | Save checkpoints every N epochs | Best model + final model |
| I5 | Evaluate final predictions | Compute correlation maps |
| I6 | Package submission | Follow Algonauts 2025 submission format |

---

## 🧩 Supporting Utilities
- `run_grid_search(base_cfg, grid, run_single_experiment)` → for all sweeps  
- `tensorboard --logdir runs` → visualize runs  
- `analyze_results.py` → aggregate Val Pearson & resource metrics  
- `estimate_runtime.py` → extrapolate full-training time  

---

## 🧭 Organizational Tips
- Use labels: `phase1-toy`, `phase2-subset`, `phase3-full`, `hyperparam`, `architecture`, `cuda`, `data`.  
- Close tickets only after posting **TensorBoard screenshot + config JSON**.  
- Weekly milestone reviews: choose next parameter sweep based on validation correlation trends.
