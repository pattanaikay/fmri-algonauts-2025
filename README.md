# fMRI Algonauts 2025 Challenge

This repository contains code for participating in the Algonauts 2025 Challenge, which focuses on predicting brain responses to naturalistic videos using deep learning models.

## Project Overview

This project aims to predict fMRI brain responses to natural video stimuli using a combination of visual and audio feature extraction, followed by machine learning models to map these features to brain activity patterns.

### Key Research Papers

The project builds upon several important works in neural encoding and multimodal brain response prediction:

1. **Algonauts 2025 Challenge Paper**
   - Gifford et al. (2025) - "The Algonauts Project 2025 Challenge: How the Human Brain Makes Sense of Multimodal Movies"
   - Introduces the challenge and dataset organization

2. **Neural Encoding Models**
   - Yamins & DiCarlo (2016) - "Using Goal-Driven Deep Learning Models to Understand Sensory Cortex"
   - Kay et al. (2008) - "Identifying Natural Images from Human Brain Activity"
   - Naselaris et al. (2011) - "Encoding and Decoding in fMRI"

3. **Multimodal Integration**
   - Hu et al. (2022) - "Brain2Music: Reconstructing Music from Human Brain Activity"
   - Tong et al. (2022) - "Learning Hierarchical Semantic Feature Graph for Movie Understanding"

4. **Transformer Architectures**
   - Vaswani et al. (2017) - "Attention Is All You Need"
   - Devlin et al. (2019) - "BERT: Pre-training of Deep Bidirectional Transformers"

### Reference Implementations

The `Paper Implementations/` directory contains code implementations of key papers to understand different encoding model architectures:

1. **Ridge Regression Baseline**
   - Linear encoding model using regularized regression
   - Establishes performance baseline
   - Features: L2 regularization, cross-validation

2. **Visio-Linguistic Encoding Model**
   - Combines visual and linguistic features
   - Transformer-based multimodal fusion
   - Hierarchical feature processing

3. **Voxelwise Encoding Model (VEM)**
   - Direct voxel-level prediction
   - Independent modeling of each brain region
   - Advanced regularization techniques

These implementations serve as building blocks for developing our custom encoding model architecture.

### Hardware Requirements

- NVIDIA GPU with CUDA support (Currently using RTX 4050 Laptop GPU with 6GB VRAM)
- Sufficient storage for video processing and model weights

### Software Dependencies

Key dependencies include:
```
- PyTorch & torchvision: Deep learning framework
- pytorchvideo: Video processing
- transformers: Hugging Face transformers for feature extraction
- tensorflow: Required for specific model components
- nilearn: Neuroimaging data processing
- librosa: Audio processing
- moviepy: Video manipulation
- scikit-learn: Machine learning utilities
- omegaconf: Configuration management
```

Full dependencies are listed in `requirements.txt`.

## Project Structure

```
.
├── algonauts/            # Python virtual environment
├── audio_features/       # Audio processing and features
│   └── temp/
├── GPT research/        # Research and documentation
├── outputs/             # Model outputs and results
├── Paper Implementations/  # Reference implementations of key papers
│   ├── Ridge Regression/         # Basic encoding model baseline
│   ├── Visio-Linguistic Encoding Model/  # Multimodal encoding
│   └── Voxelwise Encoding Model (VEM)/   # Direct voxel prediction
├── preprocessing/       # fMRI data preprocessing scripts
│   ├── dicom2nifti.py      # DICOM to NIfTI conversion
│   ├── highpassfiltering.py # Temporal filtering
│   ├── ica-aroma.py        # ICA-based denoising
│   ├── motioncorrection.py # Head motion correction
│   ├── registration.py     # Image registration
│   ├── resampling.py      # Spatial resampling
│   └── spatialsmoothing.py # Spatial filtering
├── Research Papers/     # Key papers and literature review
├── runs/                # Experiment outputs and checkpoints
│   └── p*_t*_n*_f*_d*_m*_l*_w*_*/  # Individual experiment directories
├── transformer/         # Transformer model components
├── visual_features/     # Visual processing and features
│   └── temp/
├── algonauts_v1.ipynb  # Initial development and setup
├── algonauts_v2.ipynb  # Feature extraction and baseline models
├── algonauts_v3.ipynb  # Ridge regression and prototype transformers
├── algonauts_v4.ipynb  # Complete training pipeline and submission workflow
├── encodermodel-playground.py  # Standalone script for hyperparameter experiments
├── modelarchitecture.py  # Model architecture implementations
├── experiment_tracking.jsonl  # Log of all experiment runs with hyperparameters
├── requirements.txt    # Project dependencies
└── fmri_multimodal_project_plan.md  # Detailed project roadmap
```

## Custom Model Architecture: TRIBE-Derived Multimodal Transformer

The project implements a **custom multimodal transformer architecture inspired by TRIBE (Transformer Representations for Image and Body Encoding from Meta/Facebook)** that fuses text, audio, and video features to predict fMRI brain responses. While architecturally inspired by TRIBE's principles, significant modifications and improvements have been made for the brain encoding task.

### Architecture Overview

The model follows a two-stage encoding pipeline:

1. **Batch Multi-Output Regression (B-MOR)** (distributed ridge regression)
2. **Multimodal Transformer Fusion**
3. **Per-Parcel Brain Response Prediction**

### Stage 1: Batch Multi-Output Regression (B-MOR)

**Batch Multi-Output Regression (B-MOR)** is an optimized, distributed ridge regression strategy designed for large-scale fMRI datasets:

- **Mathematical Optimization**: Uses Singular Value Decomposition (SVD) of the feature matrix $X$ to efficiently compute the ridge resolution matrix $M_\lambda$ once for all targets.
- **Batching Strategy**: Partitions the brain targets (voxels/parcels) into independent batches that can be processed in parallel across multiple compute nodes.
- **Complexity Reduction**: Reduces the computational cost from $O(p^3s + p^2ts)$ to $O(p^2tr + pr + pnsr)$, making it feasible to predict hundreds of thousands of brain targets.
- **Reference**: Ahmadi, S. (2025) - "Scaling up Machine Learning Models for fMRI Brain Encoding" (Section 3.2.3.5).

### Stage 2: Multimodal Fusion with Transformer Encoder (TRIBE-Derived)

The transformer fusion component builds on TRIBE principles with custom modifications:

#### Core Components

1. **Modality-Specific Projections**
   - Each modality (text, audio, video) projected to common embedding dimension
   - Layer normalization applied after projection for stability
   - Custom: Added learnable projection biases per modality
   - Enables alignment of heterogeneous feature spaces

2. **Multimodal Fusion with Transformer Encoder**
   - Concatenates projected modalities: `d_model = 3 × proj_dim`
   - Positional embeddings encode temporal structure (sequence position)
   - Subject embeddings enable subject-specific learning (custom enhancement)
   - Multi-head self-attention captures cross-modal interactions
   - GELU activations in feed-forward networks
   - Batch-first implementation for efficient GPU processing

3. **Modality Dropout** (Custom Modification)
   - Randomly drops entire modalities during training with probability `modality_dropout_p`
   - Ensures model robustness when modalities are missing or corrupted
   - Encourages learning of complementary information across modalities
   - Maintains at least one modality active per sample (safety constraint)
   - Key innovation for handling real-world multimodal scenarios

4. **Temporal Pooling & Subject-Specific Readout**
   - Adaptive average pooling compresses temporal dimension to match fMRI timepoints (`n_trs`)
   - Linear readout layer predicts per-parcel brain activity
   - Subject bias embeddings add subject-specific offsets to predictions
   - Custom: Per-subject tuning enables personalized models

### Key Hyperparameters

| Parameter | Range | Default | Purpose |
|-----------|-------|---------|---------|
| `proj_dim` | 128–512 | 128 | Modality embedding dimension |
| `transformer_layers` | 2–8 | 2 | Transformer encoder depth |
| `nheads` | 4–8 | 4 | Attention heads |
| `ff_dim` | 1024–2048 | 1024 | Feed-forward hidden dim |
| `dropout` | 0.1–0.3 | 0.1 | Transformer dropout |
| `modality_dropout_p` | 0.0–0.4 | 0.2 | Per-modality dropout probability |
| `lr` | 1e-4–1e-3 | 1e-3 | Learning rate (warmup enabled) |

### Implementation Files

- **`encodermodel-playground.py`**: Standalone script for hyperparameter experiments on synthetic data with full TRIBE-derived architecture
- **`algonauts_v3.ipynb`**: Ridge regression baseline and initial transformer prototypes with B-MOR discussion
- **`algonauts_v4.ipynb`**: Complete training pipeline with optional B-MOR front-end, grid search, and submission formatting

## B-MOR + TRIBE Integration Pipeline

The project implements a high-performance encoding pipeline combining the **TRIBE transformer** for multimodal fusion and **Batch Multi-Output Regression (B-MOR)** for scalable brain activity prediction:

### Stage 1: Multimodal Transformer Fusion (TRIBE)

**Purpose**: Extract and fuse task-relevant latent features from multiple stimulus streams.

- **Modality Projections**: Standardizes visual, audio, and language features into a shared embedding space.
- **Cross-Modal Attention**: Uses a Transformer encoder to model complex interactions between modalities across time.
- **Modality Dropout**: Encourages robust representations that don't rely on a single input stream.

### Stage 2: Batch Multi-Output Regression (B-MOR)

**Purpose**: Efficiently map latent features to thousands of brain parcels/voxels using optimized ridge regression.

- **Implementation**: Based on Ahmadi (2025), Algorithm 1.
- **Singular Value Decomposition (SVD)**: Decomposes the latent feature matrix $X$ once, allowing for near-instant computation of the ridge resolution matrix for any number of regularization ($\lambda$) values.
- **Distributed Batching**: Brain targets are partitioned into batches and processed in parallel, significantly reducing memory overhead and training time compared to standard multi-output regression.
- **Personalization**: Supports subject-specific readout weights to account for individual variations in brain organization.

### Training Workflow

1. **Feature Extraction**: Run stimuli through pre-trained visual (Slow-R50), audio (MFCC), and language (BERT) models.
2. **Transformer Pre-training**: Train the TRIBE encoder on a representative subset of data using a standard linear readout.
3. **Latent Encoding**: Extract high-level "brain-aligned" features from the full dataset using the trained TRIBE encoder.
4. **B-MOR Readout Fitting**: Fit the Batch Multi-Output Regression model to map TRIBE latents to the full set of brain targets.
5. **Validation**: Evaluate performance using per-parcel Pearson correlation between predicted and real fMRI time series.

### Performance Implications

| Component | Advantages | Trade-offs |
|-----------|------------|-----------|
| **B-MOR Preprocessing** | Better feature learning, interpretability | More parameters, longer training |
| **Direct Transformer** | Simplicity, fast inference | Less feature optimization |
| **Combined** | Best of both worlds if pre-trained | Requires careful hyperparameter tuning |

The current experiments (30+ runs) focus on the **direct transformer variant** for efficiency, but the B-MOR integration is available in `algonauts_v3.ipynb` and `algonauts_v4.ipynb` for production training on full data.

1. **Feature Extraction**
   - Visual features: Pre-computed using deep learning models (e.g., ResNet, CLIP)
   - Audio features: Extracted using spectral methods (e.g., MFCCs, librosa)
   - Language features: Word embeddings or language model outputs
   - Optional B-MOR preprocessing: Per-modality dual-pathway encoding
   - PCA dimensionality reduction for efficiency

2. **Feature Alignment**
   - Synchronize multimodal features with fMRI timestamps
   - Handle asynchronous modality sampling rates
   - Create sliding windows for temporal context
   - Per-modality normalization and scaling

3. **fMRI Preprocessing**
   - Motion correction (6-parameter rigid-body transformation)
   - Spatial smoothing (Gaussian kernel)
   - Registration to standard space (MNI152)
   - High-pass filtering (temporal smoothing)
   - ICA-AROMA denoising for artifact removal

4. **Model Training**
   - Multi-subject training with subject conditioning
   - Per-parcel Pearson correlation as validation metric
   - Learning rate scheduling with linear warmup
   - Checkpoint saving and early stopping strategies
   - Optional B-MOR feature pooling before transformer fusion
   - High-pass filtering (temporal smoothing)
   - ICA-AROMA denoising for artifact removal

4. **Model Training**
   - Multi-subject training with subject conditioning
   - Per-parcel Pearson correlation as validation metric
   - Learning rate scheduling with linear warmup
   - Checkpoint saving and early stopping strategies

## Experiment Tracking & Hyperparameter Optimization

A systematic approach has been implemented to explore the hyperparameter space and identify optimal configurations for the Algonauts dataset.

### Experiment Framework

The `encodermodel-playground.py` script provides:

1. **Grid Search Functionality**
   - Automated sweeps across multiple hyperparameter combinations
   - Reproducible runs with fixed random seeds
   - Configuration serialization for experiment tracking

2. **Experiment Logging**
   - All runs recorded in `experiment_tracking.jsonl` with timestamps
   - Key parameters and full configuration saved for reproducibility
   - Per-run directories in `runs/` with checkpoints and logs

3. **TensorBoard Integration**
   - Training loss curves (step-level and epoch-level)
   - Validation Pearson correlation tracking
   - Learning rate schedules
   - Per-parcel correlation analysis (optional)

### Completed Experiment Categories

**Phase 1: Architecture Capacity Sweeps (Oct 12, 2025)**
- Projection dimensions: 128, 256 (baseline)
- Transformer depth: 2, 4 layers
- Feed-forward hidden dimension: 1024 (fixed)
- Attention heads: 4 (fixed)
- Dropout: 0.1 (fixed)
- Total runs: 8 configurations

**Phase 2: Regularization Sweeps (Oct 12, 2025)**
- Increased dropout to 0.3 for stronger regularization
- Tested modality dropout: 0.0, 0.2 (independent dropout per modality)
- Projection dimensions: 128, 256
- Transformer depths: 2, 4
- Total runs: 8 configurations

**Phase 3: Capacity Expansion (Oct 12, 2025)**
- Scaled up feed-forward dimension: 2048
- Increased attention heads: 8
- Extended training: 5 epochs (vs 3 in earlier phases)
- Projection dimensions: 128, 256
- Transformer depths: 2, 4
- Modality dropout variants: 0.0, 0.2
- Total runs: 8 configurations

### Total Experiments Run: 30+

Each experiment logged with:
- Run timestamp and unique identifier
- Full hyperparameter configuration
- Training metrics (loss, validation Pearson)
- Checkpoint saved for best validation performance

### Key Findings

- **Modality dropout** (0.2) shows consistent improvement in validation Pearson
- **Dropout regularization** (0.3) provides better generalization than lighter dropout
- **Transformer depth** (4 layers) outperforms shallower architectures
- **Larger hidden dimensions** (ff_dim=2048) improve fitting capacity
- **Projection dimension** trade-off: 256 better for full dataset, 128 sufficient for subsets

### Training Pipeline Improvements

**algonauts_v4.ipynb** implements the complete production workflow:

1. **Data Loading & Alignment**
   - Loads PCA-reduced precomputed features (visual, audio, language)
   - Handles multi-subject fMRI data from Algonauts challenge
   - Automatic feature-to-fMRI alignment using timestamps

2. **Advanced Training Features**
   - Automated hyperparameter grid search with reproducible naming
   - Learning rate scheduling with linear warmup (warmup_steps parameter)
   - Gradient clipping to prevent exploding gradients
   - Mixed-precision training (AMP) ready for GPU optimization
   - Gradient accumulation support for larger effective batch sizes

3. **Validation & Metrics**
   - Per-parcel Pearson correlation computation
   - Real-time validation loss tracking
   - Early stopping based on validation performance
   - Per-subject evaluation capabilities

4. **Checkpoint Management**
   - Best model selection based on validation metrics
   - Automatic checkpoint saving and loading
   - Configuration snapshots for reproducibility

5. **Submission Formatting**
   - Nested dictionary structure matching Algonauts 2025 requirements
   - NumPy .npy file writing for predictions
   - .zip packaging for Codabench submission upload
   - Metadata tracking for challenge compliance

## Getting Started

1. Clone the repository:
```bash
git clone https://github.com/pattanaikay/fmri-algonauts-2025.git
cd fmri-algonauts-2025
```

2. Create and activate a Python virtual environment:
```bash
python -m venv algonauts
source algonauts/Scripts/activate  # On Windows, use: algonauts\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Configure GPU settings:
   - The project uses CUDA-enabled PyTorch for GPU acceleration
   - Current configuration supports NVIDIA RTX 4050 Laptop GPU

5. Run the notebooks (in order of progression):
   - **`algonauts_v1.ipynb`**: Initial setup, environment configuration, and basic data loading
   - **`algonauts_v2.ipynb`**: Feature extraction pipelines and baseline models
   - **`algonauts_v3.ipynb`**: Ridge regression baseline and transformer prototype implementation
   - **`algonauts_v4.ipynb`**: Complete training pipeline with hyperparameter grid search and submission workflow

6. For standalone hyperparameter experiments on toy data:
```bash
python encodermodel-playground.py
```
   - Configure the grid search parameters in the script
   - Monitor with TensorBoard:
```bash
tensorboard --logdir runs
```

## Notebook Guide

| Notebook | Purpose | Key Features |
|----------|---------|--------------|
| v1 | Environment & Setup | GPU detection, library imports, data directory setup |
| v2 | Feature Extraction | Audio/visual/language preprocessing, PCA reduction |
| v3 | Baseline Models | Ridge regression implementation, TRIBE prototype |
| v4 | Production Pipeline | Grid search, complete training loop, submission formatting |

## Running Experiments

### Quick Test (Toy Dataset)

Use `encodermodel-playground.py` for rapid experimentation:

```python
# In the script, configure:
base_cfg = {
    "n_samples": 200,      # Synthetic samples
    "fT": 60,              # Feature timepoints
    "n_epochs": 3,         # Quick training
    "batch_size": 8,
    "proj_dim": 128,
    "transformer_layers": 2,
}

# Define hyperparameter grid
grid = {
    "dropout": [0.1, 0.3],
    "modality_dropout_p": [0.0, 0.2],
    "ff_dim": [1024, 2048],
}

# Run grid search
run_grid_search(base_cfg, grid, run_single_experiment)
```

### Full Dataset Training

Use `algonauts_v4.ipynb` for production training:

```python
# Configure for your subset/full data
cfg = OmegaConf.create({
    "data_dir": "path/to/algonauts/data",
    "batch_size": 32,
    "n_epochs": 10,
    "lr": 1e-3,
    "warmup_steps": 200,
    "proj_dim": 256,
    "transformer_layers": 4,
    "nheads": 8,
    "ff_dim": 2048,
})

# The notebook handles:
# - Data loading and alignment
# - Multi-GPU support (if available)
# - Per-parcel validation metrics
# - Checkpoint management
# - Submission formatting
```

## Monitoring Training

View experiment results with TensorBoard:

```bash
# From the project root
tensorboard --logdir runs --port 6006

# Open http://localhost:6006 in your browser
```

**Key metrics to track:**
- `Loss/train_step`: Should decrease smoothly (sign of healthy learning)
- `Loss/train_epoch`: Epoch-level training loss
- `Val/Pearson`: Validation correlation (primary metric)
- `LR`: Learning rate schedule progression

## Status

Project is currently in advanced development with significant progress on model architecture and hyperparameter optimization:

### Completed Items
- [x] Environment setup and GPU configuration
- [x] Initial data preprocessing pipeline
- [x] Basic feature extraction implementation
- [x] Literature review and paper collection
- [x] Reference implementation framework
- [x] Ridge regression baseline model (v3 notebook)
- [x] MultimodalTRIBE transformer architecture with modality dropout
- [x] Systematic hyperparameter grid search (30+ experiments)
- [x] Experiment tracking and TensorBoard logging
- [x] Advanced training pipeline with learning rate scheduling
- [x] Per-parcel Pearson correlation validation
- [x] Checkpoint management and early stopping

### In Progress / Planned
- [ ] Full dataset training and scaling (waiting for complete Algonauts data)
- [ ] Multi-GPU distributed training optimization
- [ ] Final hyperparameter tuning on full dataset
- [ ] Codabench submission and challenge evaluation

### Experiment Summary

**Total Experiments Conducted: 30+**
- Architecture sweeps: 8 configurations
- Regularization studies: 8 configurations  
- Capacity expansion: 8 configurations
- Plus additional exploratory runs

**Best Configurations Identified:**
- Modality dropout: 0.2 (critical for robustness)
- Regular dropout: 0.3 (improved generalization)
- Transformer layers: 4 (good balance of capacity)
- Feed-forward hidden dim: 2048 (enhanced fitting)
- Projection dimension: 256 (scalable for full data)

### Next Steps

1. **Immediate**: Load and process full Algonauts 2025 dataset
2. **Training**: Run final experiments with best configurations on complete data
3. **Evaluation**: Compute per-parcel correlations across all subjects
4. **Submission**: Package results according to Algonauts 2025 challenge format
5. **Analysis**: Compare performance with baseline methods and other approaches

## Contributing

This is a research project for the Algonauts 2025 Challenge. For collaboration inquiries, please contact the repository owner.

## Citation

Gifford AT, Bersch D, St-Laurent M, Pinsard B, Boyle J, Bellec L, Oliva A, Roig G, Cichy RM. 2025. The Algonauts Project 2025 Challenge: How the Human Brain Makes Sense of Multimodal Movies. arXiv preprint, arXiv:2501.00504. DOI: https://doi.org/10.48550/arXiv.2501.00504

Boyle J, Pinsard B, Borghesani V, Paugam F, DuPre E, Bellec P. 2023. The Courtois NeuroMod project: quality assessment of the initial data release (2020). 2023 Conference on Cognitive Computational Neuroscience.