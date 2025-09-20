# fMRI Algonauts 2025 Challenge

This repository contains code for participating in the Algonauts 2025 Challenge, which focuses on predicting brain responses to naturalistic videos using deep learning models.

## Project Overview

This project aims to predict fMRI brain responses to natural video stimuli using a combination of visual and audio feature extraction, followed by machine learning models to map these features to brain activity patterns.

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
├── preprocessing/       # fMRI data preprocessing scripts
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
- [ ] Complete model training pipeline
- [ ] Model evaluation and optimization
- [ ] Final submission preparation

## Contributing

This is a research project for the Algonauts 2025 Challenge. For collaboration inquiries, please contact the repository owner.

## Citation

Gifford AT, Bersch D, St-Laurent M, Pinsard B, Boyle J, Bellec L, Oliva A, Roig G, Cichy RM. 2025. The Algonauts Project 2025 Challenge: How the Human Brain Makes Sense of Multimodal Movies. arXiv preprint, arXiv:2501.00504. DOI: https://doi.org/10.48550/arXiv.2501.00504

Boyle J, Pinsard B, Borghesani V, Paugam F, DuPre E, Bellec P. 2023. The Courtois NeuroMod project: quality assessment of the initial data release (2020). 2023 Conference on Cognitive Computational Neuroscience.