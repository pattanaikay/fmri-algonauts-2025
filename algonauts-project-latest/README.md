# Algonauts 2025: Modular Encoding Pipeline

This repository is a refactored, modularized version of the **Algonauts 2025** challenge training script. It splits the monolithic workflow into specialized modules for data handling, model architecture, and inference.

## Project Structure

```text
/algonauts-project-latest/
├── data/
│   ├── dataset.py       # Custom PyTorch Datasets & fMRI indexing
│   ├── extraction.py    # Multi-modal feature extraction (Vision, Audio, Lang)
│   └── preprocessing.py # PCA & scaling pipeline
├── models/
│   ├── tribe.py         # Multimodal TRIBE Transformer architecture
│   └── bmor.py          # B-MOR Ridge-based scaling layer
├── utils/
│   ├── config.py        # Global constants, paths, and GPU settings
│   └── metrics.py       # Scoring (Pearson r) & OOD logic
├── train.py             # Main training loop (TRIBE + B-MOR)
├── submission.py        # Inference & Codabench packaging
└── requirements.txt     # Python dependencies
```

## Setup

1. **Environment**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Configuration**:
   Edit `utils/config.py` to point to your `ROOT_DATA_DIR` (e.g., `D:\fmri-algonauts-2025-data`).

## Workflow

1. **Preprocessing**:
   Ensure you have run the preprocessing pipeline to generate `dataset_config.pkl`. This module aligns stimulus features with fMRI responses using an HRF delay.

2. **Training**:
   ```bash
   python train.py
   ```
   This script:
   - Trains the **Multimodal TRIBE** encoder on temporally pooled chunks.
   - Extracts latent representations for all training samples.
   - Fits the **B-MOR** scaling layer to map latents to 1000-parcel brain responses.

3. **Submission**:
   ```bash
   python submission.py
   ```
   This script:
   - Loads the test stimulus (OOD movies/Season 7).
   - Generates predictions for all subjects.
   - Packages the results into a `submission.zip` file compatible with Codabench.

## Features

- **Disk-Backed Loading**: Uses a lazy-loading dataset to handle massive fMRI datasets without exhausting RAM.
- **Mixed Precision**: TRIBE training uses `torch.cuda.amp` for 2x speedup on compatible GPUs.
- **Modular Design**: Easily swap out the visual encoder (e.g., SlowFast) or the language model (e.g., RoBERTa) by modifying the respective modules.
