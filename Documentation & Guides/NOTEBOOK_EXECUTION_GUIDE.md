# Algonauts V4 Notebook — Execution Guide

## Overview
This notebook implements a complete training pipeline for the Algonauts 2025 fMRI encoding challenge. It trains models on **10% of the data** for quick prototyping, using real multimodal features (visual, audio, language) and fMRI responses.

---

## Step-by-Step Execution

### **Step 0-1: Environment Setup** ✅
- **What to run:** Cells 1-7
- **Purpose:** Check GPU, install dependencies, import libraries
- **Runtime:** ~30 seconds
- **Output:** Confirms GPU availability and imports successful

---

### **Step 2: Data Alignment & Brain Visualization** (Optional)
- **What to run:** Cells 8-13
- **Purpose:** Load movie, transcript, fMRI data; interactive visualization of brain responses
- **Runtime:** ~5-10 minutes (includes video processing)
- **Output:** Interactive interface to explore stimulus-brain alignment
- **Note:** Skip if you don't need visualization

---

### **Step 3: Feature Extraction** (Pre-computed)
- **What to run:** Cells 14-31
- **Purpose:** Extract visual (slow_r50), audio (MFCC), and language (BERT) features
- **Runtime:** ~20-30 minutes per episode (GPU accelerated)
- **Output:** Extracted feature arrays
- **Note:** Already included in the notebook; uses caching to avoid re-extraction

---

### **Step 4: Data Discovery & 10% Sampling** ⭐ KEY
- **Cell ID:** Cell 32
- **Purpose:** 
  - Scan available episodes and subjects
  - Randomly select 10% for training (for quick iteration)
  - Returns `sampled_episodes` and `sampled_subjects` 
- **Runtime:** ~2-5 seconds
- **Output:** List of selected episodes and subjects
- **Key Variables:** `sampled_episodes`, `sampled_subjects`, `available_episodes`, `available_subjects`

---

### **Step 5: Data Ingestion** ⭐ KEY
- **Cell ID:** Cell 33
- **Purpose:**
  - Extract/load features for sampled episodes
  - Load fMRI responses for sampled subjects
  - Cache extracted features to avoid re-computation
- **Runtime:** ~10-20 minutes (first run); ~1-2 min (cached)
- **Output:** 
  - `features_by_episode`: Dict of extracted features per episode
  - `fmri_by_subject`: Dict of fMRI data per subject
- **Key Variables:** `features_by_episode`, `fmri_by_subject`

---

### **Step 6: Preprocessing & Alignment** ⭐ KEY
- **Cell ID:** Cell 35
- **Purpose:**
  - Align features and fMRI with HRF delay (accounts for hemodynamic lag)
  - Concatenate visual + audio + language features
  - Apply global PCA for dimensionality reduction
  - Standardize all features
- **Runtime:** ~1-2 minutes
- **Output:**
  - `dataset_config`: Contains aligned features, targets, and preprocessing objects
  - `X_final`: PCA-reduced, standardized features [N, 256]
  - `y_final`: fMRI responses [N, 1000 parcels]
- **Key Variables:** `dataset_config['X_final']`, `dataset_config['y_final']`

---

### **Step 7: Model Training (Ridge + SimpleEncoder)** ✨ MAIN TRAINING
- **Cell ID:** Cell 36
- **Purpose:**
  - Train Ridge regression as baseline
  - Train SimpleEncoderModel (neural network) as alternative
  - Compare models using per-parcel Pearson correlation (challenge metric)
- **Runtime:** ~3-5 minutes
- **Input:** Uses actual data from Step 6 (`dataset_config['X_final']` & `['y_final']`)
- **Output:**
  - Ridge predictions with ~0.2-0.4 mean correlation
  - Encoder predictions with comparable performance
  - Visualization of correlation distributions
- **Key Variables:** 
  - `ridge_cv`: Trained Ridge model
  - `model`: Trained SimpleEncoder model
  - `ridge_correlations`, `model_correlations`: Per-parcel metrics

---

### **Step 8: TRIBE + B-MOR Pipeline** ✨ ADVANCED TRAINING
- **Cell ID:** Cell 38
- **Purpose:**
  - Train MultimodalTRIBE transformer encoder on small ROI
  - Extract pooled features from encoder
  - Apply B-MOR (Batched Multilinear Ridge) for all 1000 parcels
  - Evaluate with per-parcel Pearson correlation
- **Runtime:** ~10-15 minutes
- **Input:** Uses actual data from Step 6
- **Output:**
  - TRIBE encoder checkpoint
  - B-MOR coefficients for all parcels
  - Evaluation metrics (mean/median/std Pearson r)
- **Key Variables:**
  - `tribe_model`: Trained TRIBE encoder
  - `bmor_result`: B-MOR weights and intercepts
  - `r_vals`: Per-parcel correlation scores

---

## Recommended Execution Flow for Model Training

### **Quick Iteration (10-15 min):**
```
Steps 0-1 → Step 4 (10% sampling) → Step 5 (data load) → Step 6 (preprocess) → Step 7 (train)
```

### **Full Pipeline (25-35 min):**
```
Steps 0-1 → Step 4 → Step 5 → Step 6 → Step 7 (Ridge baseline) → Step 8 (TRIBE+B-MOR)
```

### **Production (scale up):**
1. Increase sampling % in Step 4 (currently 10%)
2. Reduce PCA dimension in Step 6 if needed
3. Add hyperparameter tuning in Steps 7-8
4. Save final models and submit to Codabench

---

## Key Data Flow

```
Step 4 (Sampling)
    ↓
Step 5 (Load features + fMRI) — uses sampled_episodes, sampled_subjects
    ↓
Step 6 (Align & preprocess) — output: dataset_config with X_final, y_final
    ↓
Step 7 (Ridge + Encoder training) — input: X_final, y_final
    ↓
Step 8 (TRIBE + B-MOR) — input: X_final, y_final via dataset_config
    ↓
Submit predictions to Codabench
```

---

## Important Notes

### Data Usage
- **Steps 4, 5, 6** prepare the actual 10% dataset
- **Steps 7, 8** now correctly use the real data (previously used synthetic)
- All downstream models train on actual fMRI + multimodal features

### Metrics
- **Challenge metric:** Per-parcel Pearson correlation (not MSE)
- **Expected range:** 0.15-0.45 for 10% data
- **Full data expected:** 0.25-0.55

### Caching
- Features are cached in `./feature_cache/` directory
- Delete cache to force re-extraction on data changes
- fMRI is loaded fresh each time (no caching)

### GPU Usage
- All models default to CUDA if available, else CPU
- TRIBE + B-MOR strongly recommend GPU (10-15x speedup)
- Ridge regression is CPU-efficient

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| CUDA out of memory | Reduce batch_size in Step 5, or use CPU |
| Missing features in Step 5 | Ensure Step 4-5 completed; check file paths |
| Low correlations in Step 7 | Check HRF delay (Step 6), verify alignment |
| B-MOR too slow | Increase `n_jobs` in Step 8 or reduce sample size |

---

## Output Submission

After Step 7 or 8, save predictions:

```python
# Save for submission
np.save('predictions_ridge.npy', y_val_pred_ridge)
np.save('predictions_tribe.npy', Y_val_pred)

# Zip and upload to Codabench
```

---

**Last Updated:** December 2025  
**Notebook Version:** V4
