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
├── transformer/         # Transformer model components
├── visual_features/     # Visual processing and features
│   └── temp/
├── algonauts_v1.ipynb  # Main notebook for model development
├── algonauts_v2.ipynb  # Alternative/experimental notebook
└── requirements.txt    # Project dependencies
```

## Data Processing Pipeline

1. Video Processing:
   - Video segmentation and frame extraction
   - Visual feature extraction using deep learning models

2. Audio Processing:
   - Audio extraction from videos
   - Feature extraction using audio processing techniques

3. fMRI Preprocessing:
   - Motion correction
   - Spatial smoothing
   - Registration
   - High-pass filtering
   - ICA-AROMA denoising

4. Model Training:
   - Feature encoding
   - Brain response prediction
   - Model evaluation and optimization

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

5. Run the notebooks:
   - Start with `algonauts_v1.ipynb` for the main development pipeline
   - Use `algonauts_v2.ipynb` for experimental features

## Status

Project is currently in development with the following progress:
- [x] Environment setup and GPU configuration
- [x] Initial data preprocessing pipeline
- [x] Basic feature extraction implementation
- [x] Literature review and paper collection
- [x] Reference implementation framework
- [-] Ridge regression baseline model
- [-] Visio-linguistic encoding model
- [-] Voxelwise encoding model
- [ ] Custom architecture development
- [ ] Complete model training pipeline
- [ ] Model evaluation and optimization
- [ ] Final submission preparation

## Contributing

This is a research project for the Algonauts 2025 Challenge. For collaboration inquiries, please contact the repository owner.

## Citation

Gifford AT, Bersch D, St-Laurent M, Pinsard B, Boyle J, Bellec L, Oliva A, Roig G, Cichy RM. 2025. The Algonauts Project 2025 Challenge: How the Human Brain Makes Sense of Multimodal Movies. arXiv preprint, arXiv:2501.00504. DOI: https://doi.org/10.48550/arXiv.2501.00504

Boyle J, Pinsard B, Borghesani V, Paugam F, DuPre E, Bellec P. 2023. The Courtois NeuroMod project: quality assessment of the initial data release (2020). 2023 Conference on Cognitive Computational Neuroscience.