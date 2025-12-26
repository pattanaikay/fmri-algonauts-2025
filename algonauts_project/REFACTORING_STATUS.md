# Project Refactoring Progress

## Completed: Preprocessing Module

The monolithic `algonauts_v4.ipynb` notebook has been systematically refactored into a modular Python project. All preprocessing components have been extracted and organized.

### Files Created

#### 1. `src/preprocessing/feature_extraction.py`
**Purpose:** Extract visual, audio, and language features from raw movie stimuli

**Functions:**
- `get_vision_model(device)` - Load pre-trained slow_r50 video model
- `extract_visual_features(...)` - Extract video features with FP16 mixed precision optimization
- `extract_audio_features(...)` - Extract MFCC audio features
- `get_language_model(device)` - Load pre-trained BERT model with frozen weights
- `extract_language_features(...)` - Extract BERT embeddings with FP16 optimization

**Optimizations Preserved:**
- FP16 mixed precision (1.5-2x speedup, 40% memory savings)
- Frozen encoders (50% memory savings)
- Early temporal pooling for visual features
- MFCC averaging for audio features
- Efficient batch processing

#### 2. `src/preprocessing/normalization.py`
**Purpose:** Per-modality PCA reduction and feature standardization

**Functions:**
- `fit_modality_specific_pca(...)` - Fit PCA per modality with incremental/randomized SVD
- `apply_per_modality_pca(...)` - Apply separate PCA to visual, audio, language
- `concatenate_and_standardize_pca_features(...)` - Combine and Z-score after PCA

**Optimizations Preserved:**
- Incremental PCA for unbounded dataset sizes
- Randomized SVD for speed
- Per-modality reduction (prevents vision feature dominance)
- Z-scoring AFTER PCA (preserves geometric properties)

#### 3. `src/preprocessing/alignment.py`
**Purpose:** Load fMRI data and align with feature timing

**Functions:**
- `load_fmri_for_subject_episode(...)` - Load HDF5 fMRI data
- `align_features_with_hrf(...)` - Align features with HRF delay compensation

**Features:**
- Handles HDF5 "ses-XXX_task-EPISODE" key format
- Configurable HRF delay (typically 3-5 TRs)
- Proper temporal alignment

#### 4. `src/preprocessing/data_loader.py`
**Purpose:** Load and process movie metadata and clips

**Functions:**
- `load_transcript(...)` - Load TSV transcript files
- `get_movie_info(...)` - Extract metadata (duration, FPS, resolution)
- `split_movie_into_chunks(...)` - Get chunk boundaries for TR-aligned extraction
- `extract_movie_segment_with_sound(...)` - Extract audio/video segments
- `display_transcript_and_movie(...)` - Display transcript alongside movie
- `interface_display_transcript_and_movie(...)` - Interactive browsing widget

### Module Integration

Updated `src/preprocessing/__init__.py` to export all functions:

```python
from .feature_extraction import (
    get_vision_model,
    extract_visual_features,
    extract_audio_features,
    get_language_model,
    extract_language_features,
)

from .alignment import (
    load_fmri_for_subject_episode,
    align_features_with_hrf,
)

from .normalization import (
    fit_modality_specific_pca,
    apply_per_modality_pca,
    concatenate_and_standardize_pca_features,
)

from .data_loader import (
    load_transcript,
    get_movie_info,
    split_movie_into_chunks,
    extract_movie_segment_with_sound,
    display_transcript_and_movie,
    interface_display_transcript_and_movie,
)
```

## Next Steps

### Priority 1: Training Module
- `src/training/amp_utils.py` - AMP-enabled training loop with GradScaler
- `src/training/checkpoint.py` - ExtractionCheckpoint and FeatureValidator classes
- `src/training/trainer.py` - Training orchestration with early stopping

### Priority 2: Models Module
- `src/models/simple_encoder.py` - SimpleEncoderModel (encoder-decoder architecture)
- `src/models/ridge_baseline.py` - Ridge regression baseline
- `src/models/tribe.py` - MultimodalTRIBE model
- `src/models/bmor.py` - B-MOR (Batched Multilinear Ridge) implementation

### Priority 3: Evaluation & Submission
- `src/evaluation/metrics.py` - Pearson correlation, MSE
- `src/evaluation/validator.py` - Feature validation
- `src/submission/formatter.py` - Submission format conversion
- `src/submission/exporter.py` - Export to ZIP

### Priority 4: Utilities & Configuration
- `src/utils/gpu_utils.py` - GPU memory preallocation
- `src/utils/timing_utils.py` - Time tracking and extrapolation
- `src/utils/logging_utils.py` - Structured logging
- `src/utils/file_utils.py` - File I/O helpers
- `src/config.py` - Configuration management
- `src/constants.py` - Constants and hyperparameters

### Scripts
- `scripts/run_pipeline.py` - Complete end-to-end pipeline
- `scripts/run_training.py` - Training script
- `scripts/run_inference.py` - Inference/submission script

### Documentation
- `README.md` - Project overview and setup instructions
- `requirements.txt` - Python dependencies
- `setup.py` - Package installation

## Code Preservation

All code from the notebook has been copied exactly as-is. No refactoring, optimization changes, or code modifications have been made. The organization is purely structural—moving logically grouped functions into dedicated modules.

## Testing

Before completing the refactoring:
1. Verify all imports work correctly
2. Test feature extraction pipeline
3. Validate alignment and preprocessing
4. Confirm PCA transformations produce expected shapes

## Notes

- GPU optimizations (FP16, cuDNN autotune, TF32) are preserved in feature extraction
- Per-modality PCA is critical for preventing vision feature dominance
- Z-scoring applied AFTER PCA (not before) preserves geometric properties
- Frozen encoders reduce memory consumption on inference
- All functions maintain their original docstrings and behavior
