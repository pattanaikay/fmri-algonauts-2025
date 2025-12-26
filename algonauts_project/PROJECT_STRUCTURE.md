# MultimodalTRIBE fMRI Encoding Model - Project Structure

## Overview

This is a complete refactoring of the monolithic `algonauts_v4.ipynb` into a production-grade Python package with proper separation of concerns.

## Directory Structure

```
algonauts_project/
├── src/                           # Main source code
│   ├── __init__.py
│   ├── preprocessing/             # ✓ COMPLETED
│   │   ├── __init__.py
│   │   ├── feature_extraction.py  # Visual, Audio, Language feature extraction
│   │   ├── alignment.py           # fMRI loading and HRF alignment
│   │   ├── normalization.py       # Per-modality PCA and standardization
│   │   └── data_loader.py         # Movie/transcript loading utilities
│   ├── models/                    # Model architectures (TODO)
│   │   ├── __init__.py
│   │   ├── simple_encoder.py      # SimpleEncoderModel
│   │   ├── ridge_baseline.py      # Ridge regression baseline
│   │   ├── tribe.py               # MultimodalTRIBE model
│   │   └── bmor.py                # B-MOR (Batched Multilinear Ridge)
│   ├── training/                  # Training infrastructure (TODO)
│   │   ├── __init__.py
│   │   ├── amp_utils.py           # AMP-enabled training with GradScaler
│   │   ├── checkpoint.py          # Extraction checkpoints and validation
│   │   └── trainer.py             # Training orchestration
│   ├── evaluation/                # Evaluation and metrics (TODO)
│   │   ├── __init__.py
│   │   ├── metrics.py             # Pearson correlation, MSE
│   │   └── validator.py           # Feature validation
│   ├── inference/                 # Inference pipelines (TODO)
│   │   ├── __init__.py
│   │   └── predictor.py           # Prediction orchestration
│   ├── submission/                # Submission formatting (TODO)
│   │   ├── __init__.py
│   │   ├── formatter.py           # Format predictions
│   │   └── exporter.py            # Export to ZIP
│   ├── utils/                     # Utilities (TODO)
│   │   ├── __init__.py
│   │   ├── gpu_utils.py           # GPU memory management
│   │   ├── timing_utils.py        # Performance timing
│   │   ├── logging_utils.py       # Structured logging
│   │   └── file_utils.py          # File I/O helpers
│   ├── config.py                  # Configuration management (TODO)
│   └── constants.py               # Constants and hyperparameters (TODO)
├── notebooks/                     # Jupyter notebooks (for reference)
│   ├── algonauts_v4.ipynb         # Original notebook (source of truth)
│   └── exploration.ipynb          # Experimental notebooks
├── scripts/                       # Executable scripts (TODO)
│   ├── run_pipeline.py            # End-to-end pipeline
│   ├── run_training.py            # Training script
│   └── run_inference.py           # Inference script
├── configs/                       # Configuration files (TODO)
│   ├── training.yaml              # Training hyperparameters
│   ├── model.yaml                 # Model architecture config
│   └── data.yaml                  # Data paths and settings
├── tests/                         # Unit tests (TODO)
│   ├── test_preprocessing.py
│   ├── test_models.py
│   └── test_training.py
├── data/                          # Data directory (local cache)
│   ├── feature_cache/
│   ├── extraction_checkpoints/
│   ├── extraction_logs/
│   └── outputs/
├── logs/                          # Training/execution logs
├── REFACTORING_STATUS.md          # Detailed refactoring status
├── README.md                      # Project documentation (TODO)
├── requirements.txt               # Python dependencies (TODO)
└── setup.py                       # Package installation (TODO)
```

## Completion Status

### ✓ COMPLETED (Preprocessing Module)

**Files Created:**
- `src/preprocessing/feature_extraction.py` - Visual, Audio, Language extraction with optimizations
- `src/preprocessing/alignment.py` - fMRI loading and HRF alignment
- `src/preprocessing/normalization.py` - Per-modality PCA and standardization
- `src/preprocessing/data_loader.py` - Movie and transcript loading utilities
- `src/preprocessing/__init__.py` - Module exports

**Optimizations Preserved:**
- FP16 mixed precision (1.5-2x speedup)
- Frozen encoders (50% memory savings)
- Early temporal pooling
- Incremental/randomized PCA
- Per-modality reduction (prevents vision dominance)
- Z-scoring after PCA (preserves geometry)

### 📋 TODO (Priority 1)

**Training Module**
- `src/training/amp_utils.py` - AMP training loop with GradScaler
- `src/training/checkpoint.py` - ExtractionCheckpoint, FeatureValidator
- `src/training/trainer.py` - Training orchestration

**Models Module**
- `src/models/simple_encoder.py` - Encoder-decoder architecture
- `src/models/tribe.py` - MultimodalTRIBE with all optimizations
- `src/models/ridge_baseline.py` - Ridge regression baseline

### 📋 TODO (Priority 2)

**Evaluation & Submission**
- `src/evaluation/metrics.py` - Pearson correlation computation
- `src/evaluation/validator.py` - Data validation
- `src/submission/formatter.py` - Output formatting
- `src/submission/exporter.py` - ZIP export

**Utilities**
- `src/utils/gpu_utils.py` - Memory management
- `src/utils/timing_utils.py` - Time tracking
- `src/utils/logging_utils.py` - Logging
- `src/utils/file_utils.py` - File I/O

### 📋 TODO (Priority 3)

**Configuration & Documentation**
- `src/config.py` - Configuration management
- `src/constants.py` - Hyperparameters
- `README.md` - Documentation
- `requirements.txt` - Dependencies
- `setup.py` - Package setup

**Scripts**
- `scripts/run_pipeline.py` - End-to-end pipeline
- `scripts/run_training.py` - Training runner
- `scripts/run_inference.py` - Inference runner

## Usage Example

```python
# After refactoring is complete:
from src.preprocessing import (
    get_vision_model,
    extract_visual_features,
    extract_audio_features,
    get_language_model,
    extract_language_features,
    load_fmri_for_subject_episode,
    align_features_with_hrf,
    fit_modality_specific_pca,
    apply_per_modality_pca,
    concatenate_and_standardize_pca_features,
)

# Extract features
device = torch.device('cuda')
feature_extractor, model_layer = get_vision_model(device)
visual_features = extract_visual_features(episode_path, 1.49, ...)
audio_features = extract_audio_features(episode_path, 1.49, ...)
language_features = extract_language_features(transcript_path, model, ...)

# Load fMRI
fmri = load_fmri_for_subject_episode(subject, episode, fmri_dir, root_data_dir)

# Align with HRF
visual_aligned, fmri_aligned = align_features_with_hrf(visual_features, fmri)

# Apply per-modality PCA
pca_models, features_reduced = apply_per_modality_pca(features_by_episode)
X_final, feature_names = concatenate_and_standardize_pca_features(features_reduced)
```

## Key Design Decisions

1. **Per-Modality PCA**: Separate PCA per modality prevents vision features (typically high-variance) from dominating audio/language
2. **Z-scoring After PCA**: Applied after dimensionality reduction to preserve geometric properties
3. **FP16 Mixed Precision**: Used consistently across feature extraction for memory efficiency
4. **Frozen Encoders**: Pre-trained models frozen during inference to reduce memory consumption
5. **Modular Structure**: Each preprocessing step is independent and reusable

## Dependencies

Core:
- `torch >= 2.0.0` - PyTorch with CUDA support
- `numpy >= 1.20.0` - Numerical computing
- `scikit-learn >= 1.0.0` - Machine learning utilities
- `transformers >= 4.20.0` - BERT for language features
- `librosa >= 0.10.0` - Audio processing
- `moviepy >= 1.0.3` - Video processing

Optional:
- `pytorch-lightning >= 2.0.0` - Training orchestration
- `joblib >= 1.2.0` - Parallel processing
- `pandas >= 1.5.0` - Data manipulation
- `h5py >= 3.6.0` - HDF5 file handling
- `tqdm >= 4.64.0` - Progress bars

## Notes

- All code extracted from `algonauts_v4.ipynb` maintains exact functionality
- No refactoring or optimization changes to extracted code
- GPU optimizations preserved from original notebook
- Project structure follows industry best practices for reproducible research
