# Quick Reference: Algonauts V4 Cell Execution Map

## Essential Steps for Model Training

### ⚡ Minimum Viable Run (15 min)
```
✓ Cells 1-7     → Setup (GPU check, imports)
✓ Cell 32       → Step 4: 10% sampling
✓ Cell 33       → Step 5: Load features + fMRI (cached)
✓ Cell 35       → Step 6: Preprocess & align
✓ Cell 36       → Step 7: Train Ridge + Encoder
```
**Output:** Trained Ridge model + per-parcel correlations (mean ~0.30)

---

### 🚀 Full Pipeline Run (35 min)
```
✓ Cells 1-7     → Setup
✓ Cell 32       → Step 4: 10% sampling
✓ Cell 33       → Step 5: Load features + fMRI
✓ Cell 35       → Step 6: Preprocess & align  
✓ Cell 36       → Step 7: Train Ridge + Encoder
✓ Cell 38       → Step 8: Train TRIBE + B-MOR
```
**Output:** TRIBE+B-MOR model + per-parcel correlations (mean ~0.32)

---

### 📊 Analysis Run (Optional)
```
✓ Cells 1-7     → Setup
✓ Cells 8-13    → Step 2: Visualize alignment (10 min, interactive)
✓ Cells 14-31   → Step 3: Feature extraction (20 min, cached)
✓ Cells 32-36   → Steps 4-7: Full training (20 min)
```
**Output:** Full analysis + trained models

---

## Cell Quick Reference Table

| Cell | Step | Name | Runtime | Requires | Outputs |
|------|------|------|---------|----------|---------|
| 1-7 | 0-1 | Environment Setup | 30s | Nothing | GPU ready, imports |
| 8-13 | 2 | Visualization (optional) | 5-10m | Data files | Brain plots |
| 14-31 | 3 | Feature Extraction | 20-30m | Movies/transcripts | Feature arrays |
| **32** | **4** | **10% Sampling** 🔑 | **5s** | **Data dir scan** | **sampled_episodes** |
| **33** | **5** | **Load Features+fMRI** 🔑 | **2-20m** | **Step 4 output** | **features_by_episode, fmri_by_subject** |
| **35** | **6** | **Align & Preprocess** 🔑 | **2m** | **Step 5 output** | **dataset_config** |
| **36** | **7** | **Train Ridge+Encoder** ✨ | **5m** | **dataset_config** | **ridge_cv, model** |
| **38** | **8** | **TRIBE+B-MOR** ✨ | **10-15m** | **dataset_config** | **r_vals** |

🔑 = Key dependency point  
✨ = Main training steps

---

## Data Shape Tracking

```
After Step 4:
  sampled_episodes: List[4] episodes
  sampled_subjects: List[1] subject

After Step 5:
  features_by_episode: {episode_name: {'visual': [T,2048], 'audio': [T,20], 'language': [T,768]}}
  fmri_by_subject: {subject: {episode: [T, 1000]}}

After Step 6:
  dataset_config['X_final']: [N, 256]        (PCA-reduced features)
  dataset_config['y_final']: [N, 1000]       (fMRI responses, 1000 parcels)
  N ≈ 100-300 samples (depends on episodes/subjects × HRF alignment)

After Step 7:
  ridge_cv: Trained model, r_mean ≈ 0.30
  model: Trained encoder, r_mean ≈ 0.28

After Step 8:
  r_vals: [1000] per-parcel correlations
  mean(r_vals) ≈ 0.32
```

---

## Common Modifications

### To use MORE data (50% instead of 10%):
In **Cell 32**, change:
```python
sample_size = max(1, int(np.ceil(n_episodes * 0.1)))  # ← 0.1
```
to:
```python
sample_size = max(1, int(np.ceil(n_episodes * 0.5)))  # ← 0.5
```
Expected runtime: +30 min, better performance (+0.05 r)

### To skip visualization:
Skip **Cells 8-13**, go directly to **Cell 14**

### To use cached features (skip extraction):
**Cell 33** automatically loads from cache if available  
To force re-extraction: delete `feature_cache/` directory

### To reduce GPU memory usage:
In **Cell 33**, add before feature extraction:
```python
batch_size = 2  # ← reduce from 4
```

### To use CPU only:
In **Cell 36**, change:
```python
device = torch.device('cpu')
```

---

## Performance Targets

| Step | Model | n_samples | Mean r | Time |
|------|-------|-----------|--------|------|
| 7 | Ridge | 80-240 | 0.28-0.32 | 3m |
| 7 | SimpleEncoder | 80-240 | 0.26-0.30 | 5m |
| 8 | TRIBE+B-MOR | 80-240 | 0.30-0.34 | 12m |

*r = per-parcel Pearson correlation (challenge metric)*

---

## Troubleshooting Decision Tree

```
❌ CUDA error?
   → Use CPU: torch.device('cpu')
   → Reduce batch_size in Cell 33

❌ Memory error in Step 8?
   → Reduce n_batches in B-MOR call (line ~1800)
   → Or use CPU (much slower but works)

❌ Low correlation (r < 0.15)?
   → Check HRF delay in Step 6 (try 2, 4, 5)
   → Verify feature alignment in Step 5
   → Ensure Step 4 sampling worked

❌ Features not found in Step 5?
   → Verify root_data_dir path
   → Check feature_cache/ exists and is writable
   → Ensure Step 4 completed successfully

❌ Step 8 very slow?
   → Increase n_jobs=8 in B-MOR call
   → Or reduce data size (0.05 instead of 0.1)
   → Use GPU (10x faster)
```

---

## Submission Workflow

### After Step 7 (quick submission):
```python
# Save Ridge predictions
np.save('predictions_ridge.npy', y_val_pred_ridge)
r_scores = np.array(ridge_correlations)
np.save('correlations_ridge.npy', r_scores)
print(f"Mean Pearson r: {np.mean(r_scores):.4f}")

# Create submission package
import zipfile
with zipfile.ZipFile('submission_ridge.zip', 'w') as z:
    z.write('predictions_ridge.npy')
    z.write('correlations_ridge.npy')
# Upload submission_ridge.zip to Codabench
```

### After Step 8 (best submission):
```python
# Save TRIBE predictions
np.save('predictions_tribe.npy', Y_val_pred)
np.save('correlations_tribe.npy', r_vals)
print(f"Mean Pearson r: {np.nanmean(r_vals):.4f}")

# Create submission
import zipfile
with zipfile.ZipFile('submission_tribe.zip', 'w') as z:
    z.write('predictions_tribe.npy')
    z.write('correlations_tribe.npy')
# Upload submission_tribe.zip to Codabench
```

---

## Key Variables to Monitor

**After Step 4:**
```python
len(sampled_episodes)  # Should be 1-2 (10% of data)
len(sampled_subjects)  # Should be 1 (10% of data)
```

**After Step 6:**
```python
dataset_config['X_final'].shape  # Should be (N, 256)
dataset_config['y_final'].shape  # Should be (N, 1000)
N  # Number of samples (100-300 typical)
```

**After Step 7:**
```python
np.mean(ridge_correlations)  # Target: 0.25-0.35
np.mean(model_correlations)  # Target: 0.23-0.33
```

**After Step 8:**
```python
np.nanmean(r_vals)     # Target: 0.28-0.36
np.nanmedian(r_vals)   # Target: 0.30-0.38
np.nanstd(r_vals)      # Target: 0.15-0.25
```

---

## File Locations

```
Root:
  ├── algonauts_v4.ipynb                    ← Main notebook
  ├── NOTEBOOK_EXECUTION_GUIDE.md           ← Full guide
  ├── CHANGES_SUMMARY.md                    ← What changed
  ├── QUICK_REFERENCE.md                    ← This file
  │
  └── Data (external):
      ├── feature_cache/                    ← Cached extracted features
      │   ├── s01e01a_features.npz
      │   └── ...
      ├── outputs/
      │   └── tribe_encoder_real_best.pth   ← Saved TRIBE model
      └── predictions/
          ├── predictions_ridge.npy
          └── predictions_tribe.npy
```

---

## Last Resort: Start Fresh

If notebook state is corrupted:

```python
# In first cell, run:
import importlib
import sys

# Clear all user variables (careful!)
%reset -f

# Restart kernel
%restart

# Then re-run from Cell 1
```

Then follow the **Minimum Viable Run** sequence above.

---

**Last Updated:** December 8, 2025  
**Notebook:** V4 (with real data fixes)  
**Status:** ✅ Ready to train
