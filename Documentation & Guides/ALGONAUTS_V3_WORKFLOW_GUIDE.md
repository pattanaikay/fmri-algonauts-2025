# Algonauts 2025 Challenge — Complete Workflow Documentation

**Notebook**: `algonauts_v3.ipynb`

**Purpose**: End-to-end pipeline for fMRI encoding model training, validation, and submission to the Algonauts 2025 challenge on Codabench.

---

## Table of Contents

1. [Overview & Architecture](#overview--architecture)
2. [Step 0: Environment Setup](#step-0-environment-setup)
3. [Step 1: Data Discovery & 10% Sampling](#step-1-data-discovery--10-sampling)
4. [Step 2: Data Ingestion (Feature Extraction & Loading)](#step-2-data-ingestion-feature-extraction--loading)
5. [Step 3: Preprocessing & Alignment](#step-3-preprocessing--alignment)
6. [Step 4: Model Architecture Training](#step-4-model-architecture-training)
7. [Steps 5-8: Validation, Submission, and Advanced Training](#steps-5-8-validation-submission-and-advanced-training)
8. [Key Concepts](#key-concepts)
9. [Data Structures](#data-structures)
10. [GPU Optimization Strategies](#gpu-optimization-strategies)
11. [Troubleshooting](#troubleshooting)

---

## Overview & Architecture

### Problem Statement

The **Algonauts 2025 challenge** asks participants to build encoding models that predict fMRI brain responses to audiovisual stimuli (Friends TV series and Movie10 dataset). The challenge is evaluated on per-parcel Pearson correlation between predicted and actual fMRI responses across 1000 brain parcels.

### Solution Architecture

This notebook implements a **modular, scalable pipeline**:

```
Raw Stimuli (Movies + Transcripts)
    ↓
[STEP 2] Feature Extraction (Visual + Audio + Language)
    ↓ (caching)
[STEP 3] Preprocessing & Alignment
    • HRF delay adjustment
    • Feature standardization
    • Modality concatenation
    • PCA dimensionality reduction
    ↓
[STEP 4] Model Training
    • Ridge Regression (baseline)
    • Neural Network (advanced)
    ↓
[STEP 5-6] Validation & Evaluation
    • Per-parcel Pearson correlation
    • Performance visualization
    ↓
[STEP 7-8] Submission Preparation
    • Nested dictionary formatting
    • .npy serialization
    • .zip compression
    ↓
Codabench Upload & Scoring
```

### Key Design Decisions

1. **10% Sampling Strategy**: Train on 10% of data initially for rapid iteration (typically 1-3 episodes vs. 80+ hours)
2. **Smart Caching**: Extracted features are cached to `.npz` files, avoiding redundant computation
3. **Dual Model Support**: Both Ridge regression (fast, interpretable) and neural networks (flexible, high-capacity)
4. **Per-Parcel Evaluation**: Challenge metric is per-parcel Pearson correlation, computed for all 1000 parcels
5. **Memory Efficiency**: PCA preprocessing reduces 2800+ dims to 256 dims, enabling GPU training on modest hardware

---

## Step 0: Environment Setup

### Purpose
Verify GPU availability, set up PyTorch, and configure the computational device.

### What It Does

**Cell 1-3: GPU Detection**
```python
# Check CUDA availability
cuda_available = torch.cuda.is_available()
# Print GPU properties (name, compute capability, total memory)
# Get current device info
device = torch.device("cuda" if cuda_available else "cpu")
```

**Cell 4-5: System Configuration**
- Python version
- PyTorch version
- CUDA version
- nvidia-smi output (detailed GPU info)

### Output
```
CUDA Device 0:
  Name: NVIDIA RTX 4090
  Compute Capability: 8.9
  Total Memory: 24.00 GB
Current CUDA device: 0
```

### Why It Matters

- **Device selection** determines where tensors and models run
- **GPU capabilities** inform optimization strategies (mixed precision, gradient accumulation, activation checkpointing)
- **Memory info** helps estimate batch sizes and model architecture feasibility
- **Early detection** of CPU-only environments allows fallback strategies

---

## Step 1: Data Discovery & 10% Sampling

### Purpose
Identify available episodes and subjects in the dataset, then select a 10% stratified sample for rapid prototyping.

### What It Does

**[1] Scan for Available Episodes**
```
Location: {root_data_dir}/algonauts_2025.competitors/stimuli/transcripts/friends/s{1-7}/
Pattern: friends_s{season}e{episode}{split}.tsv
Example: friends_s01e01a.tsv
```
- Loops through all season directories (s1-s7)
- For each season, collects all `.tsv` transcript files
- Builds a list of `(episode, season)` tuples
- Total: ~100+ episodes across Friends S1-S7

**[2] Scan for Available Subjects**
```
Location: {root_data_dir}/algonauts_2025.competitors/fmri/sub-{01-04}/func/
Pattern: sub-{id}_task-friends_space-MNI152NLin2009cAsym_atlas-Schaefer18_parcel-1000Par7Net_desc-*.h5
```
- Loops through subject directories (sub-01, sub-02, sub-03, sub-04)
- For each subject, records the fMRI directory path
- Total: 4 subjects with complete fMRI coverage

**[3] Calculate 10% Sample**
```python
sample_size = max(1, int(np.ceil(n_episodes * 0.1)))
# If 100 episodes → sample 10 episodes
# If 14 episodes → sample 2 episodes
```

**[4] Random Selection (Reproducible)**
```python
np.random.seed(42)  # For reproducibility
sampled_episodes = np.random.choice(all_episodes, size=sample_size, replace=False)
sampled_subjects = np.random.choice(all_subjects, size=n_subjects_sample, replace=False)
```

### Output Example
```
✓ Found 104 episodes
✓ Found 4 subjects
✓ Sampling 10 episodes (10%) and 1 subject (25%)

Selected Episodes:
  - s02e04a (Season: s2)
  - s03e08b (Season: s3)
  - s05e01a (Season: s5)
  ... [7 more]

Selected Subjects:
  - sub-01
```

### Why 10% Sampling?

| Aspect | Full Dataset | 10% Sample |
|--------|-------------|-----------|
| **Episodes** | 100+ | 10 |
| **Hours of Video** | 80+ | 8-10 |
| **Total fMRI Samples** | 200k+ | 20k-30k |
| **Training Time** | 2-6 hours | 5-15 minutes |
| **GPU Memory** | 12-24 GB | 4-8 GB |
| **Iteration Speed** | Slow | Fast |

Quick feedback loop enables:
- Architecture experimentation
- Hyperparameter exploration
- Feature engineering validation
- Before committing to full training

---

## Step 2: Data Ingestion (Feature Extraction & Loading)

### Purpose
Extract multimodal features from raw stimuli (movies + transcripts) and load corresponding fMRI responses.

### Architecture Overview

```
For each episode:
    ├─→ Load Movie File (*.mkv)
    │   ├─→ Extract Visual Features (slow_r50) → 2048-dim
    │   └─→ Extract Audio Features (MFCC) → 20-dim
    │
    ├─→ Load Transcript File (*.tsv)
    │   └─→ Extract Language Features (BERT) → 768-dim
    │
    └─→ Cache All Features (*.npz)

For each (subject, episode) pair:
    └─→ Load fMRI Response (*.h5)
        └─→ Extract dataset: ses-003_task-{episode}
            └─→ Shape: (n_samples, 1000) parcels
```

### Detailed Breakdown

#### 2A: Visual Feature Extraction

**Model**: SlowFast R50 (pytorchvideo)
```python
model = torch.hub.load('facebookresearch/pytorchvideo', 'slow_r50', pretrained=True)
feature_extractor = create_feature_extractor(model, return_nodes=['blocks.5.pool'])
```

**Process**:
1. Load movie at 30 FPS (typical frame rate)
2. Split into 1.49-second chunks (aligned with fMRI TR)
3. For each chunk:
   - Extract all frames
   - Stack into tensor: `[batch, channels, frames, height, width]`
   - Preprocess (normalization, resizing)
   - Forward through slow_r50 backbone
   - Extract features from `blocks.5.pool` layer
4. Pool temporal dimension → per-chunk feature vector
5. Result: `[n_chunks, 2048]` array

**Time Complexity**: ~5-10 mins per episode (CPU/GPU dependent)

#### 2B: Audio Feature Extraction

**Method**: Mel-Frequency Cepstral Coefficients (MFCC)
```python
import librosa
# Extract audio from *.mkv
# Resample to sr=22050 Hz
# Compute MFCC with n_mfcc=20 coefficients
# Aggregate per TR (1.49 sec window)
```

**Process**:
1. Extract audio stream from `.mkv` file
2. Resample to 22.05 kHz (standard speech frequency)
3. Compute MFCC:
   - Divide into ~46ms windows with 50% overlap
   - Compute mel-scaled spectrogram
   - Apply discrete cosine transform
   - Extract top 20 coefficients
4. Aggregate per TR:
   - Group MFCC frames by 1.49-sec windows
   - Average within each window
5. Result: `[n_chunks, 20]` array

**Why MFCC?**
- Designed for speech understanding
- Mimics human auditory perception
- Compact (20 dims) while preserving semantics
- Fast to compute

#### 2C: Language Feature Extraction

**Model**: BERT (Bidirectional Encoder Representations from Transformers)
```python
from transformers import BertTokenizer, BertModel
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)
```

**Process**:
1. Load `.tsv` transcript file
2. For each 1.49-sec chunk:
   - Extract `text_per_tr` (words spoken in this chunk)
   - Tokenize: "hello world" → [CLS] hello wo ##rld [SEP]
   - Forward through BERT
   - Extract embeddings from final layer + previous layer
3. Pool across tokens:
   - Take `[CLS]` token pooled output (768-dim) - sentence-level representation
   - Alternative: Mean pooling across all tokens
4. Result: `[n_chunks, 768]` array

**Why BERT?**
- Contextual embeddings (word meaning depends on context)
- Pre-trained on 3.3B words
- 768-dim representation captures rich linguistic information
- State-of-the-art for NLP tasks

#### 2D: fMRI Loading

**Format**: HDF5 (hierarchical data format)
```python
with h5py.File(fmri_file_path, 'r') as f:
    dataset_name = f"ses-003_task-{episode}"  # e.g., ses-003_task-s01e01a
    fmri_data = f[dataset_name][()]  # Load into memory
```

**Structure**:
```
{fmri_file}.h5
├── ses-003_task-s01e01a: [5000, 1000]  # 5000 TRs, 1000 parcels
├── ses-003_task-s01e01b: [4800, 1000]
├── ses-003_task-s01e02a: [5100, 1000]
└── ... (one dataset per episode)
```

**What is it?**
- Brain response measured by functional magnetic resonance imaging
- 1000 brain parcels (Schaefer 2018 atlas)
- 1 sample per TR (1.49 seconds)
- Values represent % signal change from baseline

#### 2E: Smart Caching

**Problem**: Extracting visual features takes ~5-10 mins per episode. With 100 episodes, that's 500+ minutes (~8 hours)!

**Solution**: Cache extracted features
```python
cache_dir = "{root_data_dir}/feature_cache"
cache_file = "{cache_dir}/{episode}_features.npz"

# First run: extract and save
visual_feats = extract_visual_features(...)
audio_feats = extract_audio_features(...)
language_feats = extract_language_features(...)
np.savez(cache_file, visual=visual_feats, audio=audio_feats, language=language_feats)

# Subsequent runs: load from cache
cached = np.load(cache_file, allow_pickle=True)
features = {'visual': cached['visual'], 'audio': cached['audio'], 'language': cached['language']}
```

**Benefit**: 
- Second run: ~10ms (vs. 5-10 mins)
- Speedup: 500-1000x
- Development time: Hours → Minutes

### Output Example

```
STEP 2: DATA INGESTION (Extract & Load Features + fMRI)

[1] Preparing feature extraction tools...
  ✓ Feature extraction tools ready

[2] Extracting features for 3 sampled episode(s)...

  s02e04a:
    Loading cached features for s02e04a
    ✓ Visual: (3400, 2048)
    ✓ Audio: (3400, 20)
    ✓ Language: (3400, 768)

  s03e08b:
    Extracting features for s03e08b...
      Extracting visual features...
      ✓ Visual: (3200, 2048)
      Extracting audio features...
      ✓ Audio: (3200, 20)
      Extracting language features...
      ✓ Language: (3200, 768)
      Cached to .../feature_cache/s03e08b_features.npz

[3] Loading fMRI for 1 sampled subject(s)...

  Loading sub-01:
    ✓ s02e04a: shape (3400, 1000)
    ✓ s03e08b: shape (3200, 1000)

✓ Feature extraction complete for 3 episode(s)
✓ fMRI loading complete for 1 subject(s)
```

---

## Step 3: Preprocessing & Alignment

### Purpose
Align features and fMRI responses in time, normalize modalities, concatenate, and reduce dimensionality.

### The HRF Delay Problem

**Question**: Which fMRI sample corresponds to which movie frame?

**Answer**: Not frame N! Due to **hemodynamic response function (HRF)**.

**What is HRF?**

When a brain region activates:
1. Neurons consume oxygen
2. Blood flow increases to replenish oxygen (vascular response)
3. This **takes time** (~4-6 seconds peak)
4. fMRI measures blood oxygenation, not neural activity directly

**Timeline**:
```
t=0s: Visual stimulus onset
t=0-2s: Neural response (we don't measure this directly)
t=4-6s: fMRI signal peaks (what we observe)
t=8-10s: fMRI signal returns to baseline
```

**Solution**: Apply **HRF delay**
- fMRI sample at timepoint `t` reflects stimulus from `t - hrf_delay`
- With TR=1.49s and typical delay of ~4-6s: `hrf_delay = 3-4 TRs`
- Implementation: `fmri_aligned = fmri[hrf_delay:]`, `features_aligned = features[:n-hrf_delay]`

### Detailed Preprocessing Steps

#### 3A: HRF Alignment

```python
hrf_delay = 3  # ~4.47 seconds

# Original shapes
n_features = 3400  # TRs in movie
n_fmri = 3400  # TRs in fMRI scan

# After alignment
fmri_aligned = fmri[hrf_delay:]  # fmri[3:] → shape (3397, 1000)
features_aligned = features[:n_fmri - hrf_delay]  # features[:3397] → shape (3397, 2048)

# Result: matched pairs
# features[0] → fmri[3]
# features[1] → fmri[4]
# ...
# features[3396] → fmri[3399]
```

**Why hrf_delay=3?**
- TR = 1.49s → hrf_delay=3 → 4.47s delay
- Typical peak response at 4-6s after stimulus
- Empirically validated on similar datasets

#### 3B: Feature Standardization

**Per-modality normalization** (before concatenation)

```python
from sklearn.preprocessing import StandardScaler

for modality in ['visual', 'audio', 'language']:
    feat = features[modality]  # e.g., shape (3397, 2048)
    scaler = StandardScaler()
    feat_scaled = scaler.fit_transform(feat)
    # Now: mean=0, std=1
```

**Why per-modality?**
- Modalities have different scales:
  - Visual: 0-1000+ (large values)
  - Audio: 0-100 (medium values)
  - Language: -50 to +50 (small values)
- Without standardization: visual dominates due to scale
- After standardization: equal importance

#### 3C: Modality Concatenation

```python
X_combined = np.concatenate([
    features_scaled['visual'],      # (3397, 2048)
    features_scaled['audio'],       # (3397, 20)
    features_scaled['language'],    # (3397, 768)
], axis=1)
# Result: (3397, 2836)
```

**Dimensionality**: 2048 + 20 + 768 = **2836 features**

**Issue**: Too many features! 
- Curse of dimensionality
- Overfitting risk
- Slow training
- Doesn't fit in GPU memory easily

**Solution**: PCA dimensionality reduction

#### 3D: PCA Dimensionality Reduction

**Principal Component Analysis**:
```python
from sklearn.decomposition import PCA

pca = PCA(n_components=256)  # Target: 256 components
X_pca = pca.fit_transform(X_combined)  # (3397, 2836) → (3397, 256)

variance_explained = pca.explained_variance_ratio_.sum()
# Typically: 95-98% of variance retained in 256 components
```

**How PCA works**:
1. Center data (subtract mean)
2. Compute covariance matrix
3. Find eigenvectors (principal components)
4. Project onto top-K eigenvectors
5. Result: K-dimensional representation capturing maximum variance

**Why it works**:
- First 256 PCs capture 95%+ of variance
- Removes ~90% of noise (last 2580 PCs are mostly noise)
- Speedup: 11x smaller feature vectors
- Better generalization: fewer parameters to fit

**Variance Preserved**:
```
PC 1:   25% of variance
PC 2:   18% of variance
PC 3:   12% of variance
...
PC 50:  0.1% of variance
...
PC 256: 0.01% of variance
PC 257: 0.005% of variance
PC 2836: ~0.00001% of variance
```

#### 3E: Global Standardization

```python
from sklearn.preprocessing import StandardScaler

scaler_global = StandardScaler()
X_final = scaler_global.fit_transform(X_pca)
# Now: each feature has mean=0, std=1
# Across the dataset (all samples, all features)
```

**Why again?**
- PCA projection changes scale
- Neural networks train better on standardized input
- Helps optimization (gradients more stable)

### Output Example

```
STEP 3: PREPROCESSING & ALIGNMENT

[1] Aligning features and fMRI with HRF delay=3...

  sub-01 / s02e04a:
    Original shapes:
      Visual: (3400, 2048)
      Audio: (3400, 20)
      Language: (3400, 768)
      fMRI: (3400, 1000)
    
    Aligned shapes:
      Combined features: (3397, 2836)
      fMRI: (3397, 1000)

  sub-01 / s03e08b:
    Original shapes:
      Visual: (3200, 2048)
      Audio: (3200, 20)
      Language: (3200, 768)
      fMRI: (3200, 1000)
    
    Aligned shapes:
      Combined features: (3197, 2836)
      fMRI: (3197, 1000)

✓ Aligned 2 (subject, episode) pairs

[2] Combining all data...
  Combined X shape: (6594, 2836)
  Combined y shape: (6594, 1000)

[3] Applying PCA preprocessing...
  Original feature dim: 2836
  PCA reduced dim: 256
  Variance explained: 97.34%

  Final X shape (standardized): (6594, 256)
  Final y shape: (6594, 1000)

✓ Preprocessing complete. Data ready for model architecture.

[4] Dataset Config:
  Total samples: 6594
  Feature dimension: 256
  Output parcels: 1000
```

---

## Step 4: Model Architecture Training

### Purpose
Train two alternative models to predict fMRI responses from preprocessed features.

### Overview

```
Option A: Ridge Regression (Baseline)
├─ Simple, interpretable
├─ Fast training (<1 sec)
├─ Good for 10% subset testing
└─ Expected correlation: 0.15-0.25

Option B: SimpleEncoderModel (Advanced)
├─ Multi-layer neural network
├─ More expressivity
├─ ~5-10 mins training
└─ Expected correlation: 0.20-0.35
```

### Option A: Ridge Regression

#### Theory

**Linear Regression**: predict `y = Xw + b`
- Loss: MSE = mean((y - Xw)^2)
- Solution: w = (X^T X)^{-1} X^T y (closed form)

**Ridge Regression**: predict `y = Xw + b` with L2 penalty
- Loss: MSE + α||w||^2
- Adds regularization term: penalizes large weights
- Prevents overfitting (especially important when features > samples)

#### Implementation

```python
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import train_test_split

# Split data
X_train, X_val, y_train, y_val = train_test_split(
    X_final, y_final, test_size=0.2, random_state=42
)

# Cross-validation to find best alpha
ridge_cv = RidgeCV(alphas=[0.001, 0.01, 0.1, 1.0, 10.0, 100.0], cv=5)
ridge_cv.fit(X_train, y_train)

print(f"Best alpha: {ridge_cv.alpha_}")  # e.g., 1.0

# Predict
y_val_pred = ridge_cv.predict(X_val)
```

**Alpha (regularization strength)**:
| Alpha | Interpretation | When to Use |
|-------|------------------|------------|
| 0.001 | Very weak penalty | Many features, moderate overfitting |
| 0.01  | Weak penalty | Default for well-conditioned problems |
| 0.1   | Medium penalty | Some overfitting detected |
| 1.0   | Strong penalty | High overfitting, high noise |
| 10.0  | Very strong | Extreme regularization |
| 100.0 | Maximum penalty | Near-zero weights (underfitting) |

#### Evaluation Metrics

**Per-Parcel Pearson Correlation**:
```python
from scipy.stats import pearsonr

correlations = []
for parcel_idx in range(1000):
    pred = y_val_pred[:, parcel_idx]
    true = y_val[:, parcel_idx]
    r, p_value = pearsonr(pred, true)
    correlations.append(r)

mean_corr = np.mean(correlations)  # Challenge metric!
```

**Interpretation**:
- **r = 1.0**: Perfect prediction
- **r = 0.5**: Strong correlation (good model!)
- **r = 0.2**: Weak correlation (baseline performance)
- **r = 0.0**: No correlation (random predictions)
- **r < 0.0**: Negative correlation (worse than random!)

**Expected Performance** (10% subset, Ridge):
- Median parcel correlation: 0.15-0.25
- Top 10% parcels: 0.35-0.50 (visual, motor regions easier to predict)
- Bottom 10% parcels: 0.00-0.05 (complex cognitive regions harder)

#### Output Example

```
[2] Option A: Baseline Ridge Regression with Cross-Validation...
  Best alpha: 1.0
  MSE: 0.8234
  Mean per-parcel Pearson correlation: 0.1843 ± 0.1256
```

### Option B: SimpleEncoderModel (Neural Network)

#### Architecture

```python
class SimpleEncoderModel(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=512):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),           # (256) → (512)
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),     # (512) → (256)
            nn.ReLU(),
        )
        self.decoder = nn.Linear(hidden_dim // 2, output_dim)  # (256) → (1000)
    
    def forward(self, x):
        h = self.encoder(x)      # Encode: X → latent representation
        y = self.decoder(h)      # Decode: latent → fMRI predictions
        return y
```

**Diagram**:
```
Input (256)
    ↓
Linear + ReLU (256 → 512)
    ↓
Dropout (p=0.1)
    ↓
Linear + ReLU (512 → 256)
    ↓
Linear (256 → 1000)
    ↓
Output (1000 parcel predictions)
```

**Why this architecture?**
- **Two hidden layers**: Enough capacity for non-linear transformations
- **512 units**: Larger than input (256) creates latent bottleneck
- **Dropout (p=0.1)**: 10% of activations randomly zero during training (regularization)
- **ReLU**: Non-linearity enables learning complex feature interactions

#### Training Loop

```python
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

X_train_t = torch.from_numpy(X_train).float().to(device)
y_train_t = torch.from_numpy(y_train).float().to(device)

for epoch in range(50):
    # Forward pass
    model.train()
    y_pred = model(X_train_t)
    loss = loss_fn(y_pred, y_train_t)
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Validation
    model.eval()
    with torch.no_grad():
        y_val_pred = model(X_val_t)
        val_loss = loss_fn(y_val_pred, y_val_t)
    
    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= 5:
            model.load_state_dict(best_state)
            break
```

**Key Hyperparameters**:
| Parameter | Value | Why? |
|-----------|-------|------|
| Learning rate | 1e-3 | Standard for Adam, allows convergence in 50 epochs |
| Optimizer | Adam | Adaptive learning rates, widely used default |
| Loss function | MSE | Regression task (not classification) |
| Dropout rate | 0.1 | Light regularization (avoid too much information loss) |
| Epochs | 50 | Enough for convergence on 10% subset |
| Early stopping patience | 5 | Stop if val loss doesn't improve for 5 epochs |

#### Training Dynamics

**First Few Epochs** (high loss):
- Model randomly initialized
- Learns basic correlations
- Training loss drops quickly: 2.5 → 1.2 → 0.8 → ...

**Middle Epochs** (medium loss):
- Model overfits to training data
- Training loss still decreasing: 0.7 → 0.65 → ...
- Validation loss plateaus or increases: 0.85 → 0.84 → 0.86 → ...

**Late Epochs** (early stopping):
- Model hasn't improved on validation for 5 epochs
- Stop training, restore best checkpoint
- Prevents overfitting

**Expected Performance** (10% subset, SimpleEncoder):
- Mean parcel correlation: 0.20-0.30
- Slight improvement over Ridge (~5-10% better)
- More volatile predictions (noisier)

#### Output Example

```
[3] Option B: SimpleEncoderModel...
  Training on device: cuda
  Model: SimpleEncoderModel(256 -> 1000)
  
  Epoch  10: train_loss=0.7234, val_loss=0.8456
  Epoch  20: train_loss=0.6123, val_loss=0.8234
  Epoch  30: train_loss=0.5891, val_loss=0.8567
  Epoch  40: train_loss=0.5234, val_loss=0.8899
  Early stopping at epoch 45
  
  MSE: 0.8899
  Mean per-parcel Pearson correlation: 0.2134 ± 0.1389
```

### Model Comparison

```
[4] Model Comparison (on 10% Real Dataset):
  Ridge Regression:        corr=0.1843
  SimpleEncoderModel:      corr=0.2134
  Winner: SimpleEncoder (~16% better)
```

---

## Steps 5-8: Validation, Submission, and Advanced Training

### Step 5: Per-Parcel Validation Visualization

**Goal**: Visualize model performance across all 1000 parcels

**Output**: 
- Histogram of per-parcel correlations
- Scatter plot: predicted vs. true (example parcel)
- Statistics: mean, std, min, max

**Interpretation**:
```
Distribution:
  Most parcels: r=0.10-0.25 (weak)
  Top 10%: r=0.35-0.50 (strong, likely visual/motor)
  Bottom 10%: r=-0.05-0.05 (very weak, complex cognitive)
```

### Step 6: Full Dataset Scaling

**Transition from 10% → 100%**:
1. Use same preprocessing pipeline
2. Train on full Friends S1-S6 (80+ hours)
3. Validate on Friends S7 (withheld)
4. Monitor for convergence with full data
5. Adjust hyperparameters if needed

### Step 7: Hyperparameter Tuning

**Grid Search over Ridge alphas**:
```python
alphas = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
for alpha in alphas:
    ridge = Ridge(alpha=alpha)
    ridge.fit(X_train, y_train)
    val_score = ridge.score(X_val, y_val)
    print(f"Alpha {alpha}: {val_score:.4f}")
```

**Result**: Select alpha with best validation score

**Alternative: Neural Network Tuning**:
- Learning rate: {1e-4, 5e-4, 1e-3, 5e-3}
- Hidden dims: {256, 512, 1024}
- Dropout rate: {0.05, 0.1, 0.2}
- Perform grid/random search

### Step 8: Submission Preparation

**Submission Format** (Codabench requirement):

```python
# Nested dictionary structure
submission_dict = {
    'sub-01': {
        's07e01a': predictions_array_1,  # shape: (N_samples, 1000)
        's07e01b': predictions_array_2,  # shape: (M_samples, 1000)
        ...
    },
    'sub-02': {
        's07e01a': predictions_array_3,
        ...
    },
    ...
}

# Save
np.save('submission.npy', submission_dict, allow_pickle=True)

# Zip
import zipfile
with zipfile.ZipFile('submission.zip', 'w') as zf:
    zf.write('submission.npy')
```

**Upload to Codabench**:
1. Go to https://www.codabench.org/competitions/4313/
2. Select "Participate" → "Model Building Phase"
3. Click "Make a Submission"
4. Upload `submission.zip`
5. Wait for scoring (~10 mins)

---

## Key Concepts

### 1. Temporal Alignment & fMRI TR

**TR (Repetition Time)**: Time between successive fMRI scans
- Algonauts dataset: TR = 1.49 seconds
- Corresponds to one "sample" or "volume"
- Standard fMRI data structure: [n_volumes, spatial_dims]
- 1 hour of video → ~2400 volumes

**Feature Alignment to TR**:
- Split 1.49-sec chunks of movies
- Extract one feature vector per chunk
- Result: features sampled at fMRI rate

### 2. Dimensionality Curse vs. Blessing

**Curse of Dimensionality**:
- More features → more parameters
- With limited data: parameters > samples → overfitting
- Example: 2836 features × 6594 samples = 1M+ parameters (ridge) to fit

**Blessing of Dimensionality**:
- High-dim space has more room for patterns
- Rich representations capture complex relationships
- Can actually help with enough regularization

**Solution**: PCA
- Reduce 2836 → 256 (90% reduction)
- Retain 95%+ variance (information)
- Regularize implicitly (drop noisy components)

### 3. Cross-Modality Fusion

**Why Multiple Modalities?**
- Vision → Visual cortex
- Audio → Auditory cortex
- Language → Language areas (Broca, Wernicke)
- Combinations → Multisensory areas

**Concatenation** (current approach):
```python
X_combined = [visual_features; audio_features; language_features]
```
Pro: Simple, preserves individual modality information
Con: Doesn't learn cross-modality interactions

**Attention-based Fusion** (alternative):
```python
attention_weights = compute_attention(visual, audio, language)
X_fused = attention_weights[0] * visual + ... + attention_weights[2] * language
```
Pro: Learned modality importance
Con: Requires more parameters, more data

### 4. Parcel-wise Prediction

**What is a parcel?**
- Brain region in the Schaefer 2018 atlas
- 1000 total parcels (non-overlapping)
- Variable size (5-100mm³)
- Each parcel assigned a functional network label

**Why predict per-parcel?**
- Spatially structured brain
- Neighboring parcels have similar function
- Challenge metric: average correlation across parcels

**Alternative: Voxel-wise**
- Predict all voxels (~50M for whole brain)
- More fine-grained
- Much harder (more parameters, noisier signal)

### 5. Leave-One-Subject-Out Cross-Validation

**Standard approach for brain data**:
```
FOR each subject in [sub-01, sub-02, sub-03, sub-04]:
    train_subjects = others
    test_subject = current subject
    
    Train model on train_subjects
    Evaluate on test_subject
    Record performance

Average performance across folds
```

**Why subject-level CV?**
- Subjects have individual brain anatomy
- Goal: generalization across subjects
- Subject-specific overfitting would inflate scores

**In this notebook**:
- Using subset of data (10%)
- Full CV expensive → faster prototyping
- Eventually should do full LOSO-CV before submission

---

## Data Structures

### Directory Structure

```
{root_data_dir}/
├── algonauts_2025.competitors/
│   ├── stimuli/
│   │   ├── movies/
│   │   │   ├── friends/
│   │   │   │   ├── s1/
│   │   │   │   │   ├── friends_s01e01a.mkv
│   │   │   │   │   ├── friends_s01e01b.mkv
│   │   │   │   │   └── ...
│   │   │   │   ├── s2/
│   │   │   │   └── ...
│   │   │   └── movie10/
│   │   │       └── ...
│   │   └── transcripts/
│   │       ├── friends/
│   │       │   ├── s1/
│   │       │   │   ├── friends_s01e01a.tsv
│   │       │   │   └── ...
│   │       │   └── ...
│   │       └── movie10/
│   └── fmri/
│       ├── sub-01/
│       │   ├── func/
│       │   │   └── sub-01_task-friends_*.h5
│       │   └── atlas/
│       │       └── sub-01_*_parcellation.nii.gz
│       ├── sub-02/
│       │   └── ...
│       └── ...
│
└── feature_cache/  (created during execution)
    ├── s01e01a_features.npz
    ├── s01e01b_features.npz
    └── ...
```

### Data Format: Features

**Extracted Features** (after STEP 2):
```python
features_by_episode['s01e01a'] = {
    'visual': np.array(shape=(3400, 2048), dtype=float32),
    'audio': np.array(shape=(3400, 20), dtype=float32),
    'language': np.array(shape=(3400, 768), dtype=float32),
}
```

**Cached Features** (*.npz):
```python
np.savez('s01e01a_features.npz',
         visual=visual_array,
         audio=audio_array,
         language=language_array)
# Binary format, compressed
# Size: ~50-100 MB per episode
```

### Data Format: fMRI

**HDF5 Structure**:
```
sub-01_task-friends_space-MNI152NLin2009cAsym_atlas-Schaefer18_parcel-1000Par7Net_desc-s123456_bold.h5
├── ses-003_task-s01e01a: dtype=float32, shape=(3400, 1000)
├── ses-003_task-s01e01b: dtype=float32, shape=(3200, 1000)
├── ses-003_task-s01e02a: dtype=float32, shape=(3450, 1000)
└── ... (one dataset per episode)
```

**Access**:
```python
with h5py.File(fmri_path, 'r') as f:
    keys = list(f.keys())  # ['ses-003_task-s01e01a', ...]
    data = f['ses-003_task-s01e01a'][()]  # Load into memory
    shape = data.shape  # (n_samples, 1000)
```

### Dataset Config (After Preprocessing)

```python
dataset_config = {
    'X_final': np.array(shape=(6594, 256), dtype=float32),      # Preprocessed features
    'y_final': np.array(shape=(6594, 1000), dtype=float32),     # fMRI responses
    'pca': PCA(n_components=256, ...),                           # Fitted scaler
    'scaler_global': StandardScaler(...),                        # Fitted scaler
    'aligned_data': [
        {'subject': 'sub-01', 'episode': 's01e01a', 'X': ..., 'y': ...},
        {'subject': 'sub-01', 'episode': 's01e01b', 'X': ..., 'y': ...},
        ...
    ],
    'n_samples': 6594,
    'n_features': 256,
    'n_parcels': 1000,
}
```

---

## GPU Optimization Strategies

### For RTX 4050 (6GB VRAM)

**Challenge**: 
- Only 6 GB GPU memory
- Training set: 6594 samples × 256 features × 4 bytes = 6.7 MB (features)
- Training set: 6594 samples × 1000 parcels × 4 bytes = 26.4 MB (targets)
- Model parameters: ~500K weights × 4 bytes = 2 MB
- Optimizer state (Adam): 2× model size = 4 MB
- **Total**: ~40 MB usable (well within 6GB)

**Optimization Techniques**:

1. **Batch Gradient Descent** (vs. SGD):
   ```python
   for epoch in range(epochs):
       optimizer.zero_grad()
       y_pred = model(X_train)  # All samples at once
       loss = loss_fn(y_pred, y_train)
       loss.backward()
       optimizer.step()
   ```
   Pros: Stable gradients, single GPU pass
   Cons: Uses all GPU memory at once

2. **Mini-batch Training**:
   ```python
   batch_size = 64
   for batch in DataLoader(dataset, batch_size=batch_size):
       y_pred = model(batch['X'])
       loss = loss_fn(y_pred, batch['y'])
       loss.backward()
       optimizer.step()
   ```
   Pros: Lower memory, faster iterations
   Cons: Noisier gradients

3. **Mixed Precision (AMP)**:
   ```python
   from torch.cuda.amp import autocast, GradScaler
   
   scaler = GradScaler()
   with autocast():
       y_pred = model(X_train)
       loss = loss_fn(y_pred, y_train)
   
   scaler.scale(loss).backward()
   scaler.step(optimizer)
   scaler.update()
   ```
   Pros: 2x speedup, half memory (fp16 vs fp32)
   Cons: Slight accuracy loss (usually negligible)

4. **Gradient Accumulation**:
   ```python
   accumulation_steps = 4
   for i, batch in enumerate(loader):
       y_pred = model(batch['X'])
       loss = loss_fn(y_pred, batch['y']) / accumulation_steps
       loss.backward()  # Accumulate gradients
       
       if (i+1) % accumulation_steps == 0:
           optimizer.step()  # Update after 4 batches
           optimizer.zero_grad()
   ```
   Pros: Effective batch size 4x larger, same memory
   Cons: More backward passes, slower per-epoch

5. **Activation Checkpointing**:
   ```python
   from torch.utils.checkpoint import checkpoint
   
   # In model forward:
   h = checkpoint(self.encoder, x)  # Don't store intermediate activations
   y = self.decoder(h)
   ```
   Pros: ~50% less memory for deep networks
   Cons: Slower (recompute activations in backward)

### Memory Estimate Table

| Strategy | Batch Size | Memory (GB) | Speed |
|----------|-----------|------------|-------|
| Full batch | 6594 | 2.5 | 1.0x (baseline) |
| Mini-batch (64) | 64 | 0.1 | 0.95x |
| Mini-batch (32) | 32 | 0.08 | 0.9x |
| AMP full batch | 6594 | 1.5 | 2.0x |
| AMP + Gradient Accum (4) | 256 | 0.2 | 1.8x |

---

## Troubleshooting

### Issue: "CUDA out of memory"

**Cause**: Tensor too large for GPU memory

**Solutions** (in order):
1. Reduce batch size: `batch_size = 32` (instead of 256)
2. Enable mixed precision: `with autocast():`
3. Use gradient accumulation: `loss.backward() / n_accum`
4. Reduce model size: `hidden_dim = 256` (instead of 512)
5. Use CPU: `device = torch.device('cpu')`

### Issue: "fMRI file not found"

**Cause**: Incorrect path or missing data

**Debug**:
```python
import os
print(os.path.exists(fmri_path))  # Should be True
print(os.listdir(os.path.dirname(fmri_path)))  # List files in directory
```

**Solution**: Verify data directory structure matches expected paths

### Issue: "Feature extraction takes forever"

**Cause**: Not using cached features

**Solution**: Ensure cache directory exists and is writable
```python
cache_dir = os.path.join(root_data_dir, "feature_cache")
os.makedirs(cache_dir, exist_ok=True)
# Run extraction once, subsequent runs use cache
```

### Issue: "Model validation loss not decreasing"

**Cause**: Learning rate too high or too low, or bad initialization

**Solutions**:
1. Lower learning rate: `lr = 5e-4` (instead of 1e-3)
2. Increase epochs: `epochs = 100` (instead of 50)
3. Reduce dropout: `dropout = 0.05` (instead of 0.1)
4. Add batch normalization: `nn.BatchNorm1d(hidden_dim)`

### Issue: "Predictions are all zeros or constant"

**Cause**: Model not training (weights not updating)

**Debug**:
```python
# Check gradients
for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"{name}: grad_norm={param.grad.norm().item()}")
    else:
        print(f"{name}: NO GRADIENT")
```

**Solution**: 
- Verify loss is being computed correctly
- Check learning rate is not too small
- Verify training loop calls `loss.backward()`

### Issue: "Train/val correlation very different"

**Cause**: Overfitting

**Solutions**:
1. Increase dropout: `dropout = 0.2`
2. Increase L2 regularization (Ridge alpha)
3. Use less complex model
4. Get more data (less practical)

### Issue: "Codabench says 'Invalid submission format'"

**Cause**: Wrong dictionary structure or dtype

**Debug**:
```python
# Check structure
print(type(submission_dict))  # dict
print(list(submission_dict.keys()))  # ['sub-01', 'sub-02', ...]
print(type(submission_dict['sub-01']))  # dict
print(list(submission_dict['sub-01'].keys()))  # ['s07e01a', ...]
print(submission_dict['sub-01']['s07e01a'].dtype)  # float32
```

**Solution**:
- Ensure all values are float32: `predictions.astype(np.float32)`
- Ensure sample counts match test fMRI files exactly
- Ensure nested dict structure (not list or other)

---

## Summary: Execution Checklist

### Before Running:
- [ ] GPU available (check Step 0)
- [ ] Data directory exists (`{root_data_dir}/algonauts_2025.competitors/`)
- [ ] At least 10 GB free disk space (for features + models)
- [ ] Python 3.8+ with PyTorch 1.10+

### First Run:
- [ ] Execute Step 1 (data discovery)
- [ ] Execute Step 2 (feature extraction, will take 30-60 mins first time)
- [ ] Execute Step 3 (preprocessing)
- [ ] Execute Step 4 (training)
- [ ] Check Step 4 outputs: correlation ~0.15-0.30?

### After First Run:
- [ ] Feature cache created (verify feature_cache/ directory)
- [ ] Subsequent runs 100x faster
- [ ] Can iterate on Steps 3-4 rapidly
- [ ] Experiment with hyperparameters, architectures, data splits

### Before Submission:
- [ ] Switch to full dataset (all episodes, all subjects)
- [ ] Run Steps 3-4 on full data (~2-6 hours)
- [ ] Validate on held-out Friends S7
- [ ] Format predictions as nested dict
- [ ] Create submission.zip
- [ ] Upload to Codabench
- [ ] Monitor scoring (~10 mins)
- [ ] Check correlation result

---

## References & Further Reading

1. **fMRI Basics**:
   - Van Essen et al. (2013): "The Human Connectome Project"
   - Huettel et al. (2014): "Functional Magnetic Resonance Imaging"

2. **Encoding Models**:
   - Nishimoto et al. (2011): "Reconstructing Visual Experience from Brain Activity"
   - Kay et al. (2008): "Identifying Natural Images from Brain Activity"

3. **Algonauts Challenge**:
   - Official website: https://www.algonauts.org/
   - Paper: Gifford et al. (2022): "Distributed Funtional Organization of Biological Vision"

4. **Tools & Libraries**:
   - PyTorch: https://pytorch.org/
   - Scikit-learn: https://scikit-learn.org/
   - librosa (audio): https://librosa.org/
   - Transformers (BERT): https://huggingface.co/transformers/

---

**Version**: 1.0  
**Last Updated**: December 2025  
**Author**: Algonauts Challenge Team
