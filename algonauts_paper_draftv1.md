MultimodalTRIBE: A Scalable Approach for fMRI Encoding with BMOR-Enhanced Readout
Abstract
We present MultimodalTRIBE, a unified multimodal encoding model designed for the Algonauts 2025 Challenge. The model integrates visual, audio, and language features using a Transformer-based architecture to predict brain responses (fMRI) to movie stimuli. To handle the high-dimensional output space (thousands of brain parcels) and potential scalability issues, we introduce BMOR (Batched Mutillinear Ridge), a specialized readout mechanism that efficiently maps learned representations to voxel-wise activity. Our approach combines deep representation learning with robust linear modeling, achieving a balance between expressivity and computational efficiency.

1. Introduction
The Algonauts 2025 Challenge focuses on predicting human brain responses to naturalistic movie stimuli. Understanding how the brain integrates multimodal information—visual scenes, spoken dialogue, and background sounds—is a fundamental goal in cognitive neuroscience. Traditional encoding models often treat modalities in isolation or rely on simple concatenation.

We propose a method that:

Extracts rich features from state-of-the-art pre-trained models (SlowFast, BERT).
Fuses modalities using a Transformer encoder (MultimodalTRIBE) that learns interactions over time.
Scales to whole-brain prediction using BMOR, an optimized ridge regression framework that handles large-scale target data through batching and parallelization.
2. Methodology
2.1 Feature Extraction
We leverage pre-trained models to extract high-level features from the stimuli:

Visual: We use slow_r50 (SlowFast ResNet50) pre-trained on Kinetics-400. Features are extracted from the blocks.5.pool layer, providing a 2048-dimensional representation for every 1.49s TR.
Audio: Mel-Frequency Cepstral Coefficients (MFCCs) are computed using librosa, resulting in a 20-dimensional feature vector per TR.
Language: bert-base-uncased is used to encode transcripts. We extract both the pooled output and the last hidden state, resulting in high-dimensional text embeddings (768-dim).
2.2 Model Architecture: MultimodalTRIBE
The core of our approach is the MultimodalTRIBE encoder. It is a Transformer-based model designed to fuse the extracted features.

Key Components:
Modality Projection: Each modality (Video, Audio, Text) is projected to a shared embedding dimension (d_model) using linear layers and LayerNorm.
Transformer Encoder: A stack of Transformer encoder layers (default: 2 layers, 4 heads) processes the sequence of multimodal embeddings. This allows the model to capture temporal context and cross-modal interactions.
Subject Embeddings: To account for inter-subject variability, learnable subject embeddings are added to the input sequence.
Modality Dropout: During training, we randomly drop entire modalities (Text, Audio, or Video) to encourage robustness and prevent the model from over-relying on a single dominant modality (e.g., Vision).
# Simplified Architecture Diagram
Input (Txt, Aud, Vid) -> Projections -> Concat -> Transformer -> Pooling -> Features
                                           ^
                                           |
                                      Subject Emb
2.3 Scalable Readout: BMOR (Batched Multilinear Ridge)
Predicting responses for thousands of brain parcels directly from a deep network can be memory-intensive and prone to overfitting. We address this with BMOR.

Two-Stage Training: First, the MultimodalTRIBE encoder is trained end-to-end on a smaller subset of parcels (Region of Interest) to learn rich shared representations.
Feature Pooling: The encoder's output is pooled over time (to match the TR resolution).
Batched Ridge Regression: The fixed representations from the encoder are then used as inputs to a Ridge Regression model. BMOR solves this efficiently by splitting the targets (parcels) into batches and solving them in parallel (joblib), allowing for high-throughput training on limited hardware.
3. Implementation Details
Data Sampling: For rapid prototyping, we implemented a 10% sampling strategy, selecting a subset of episodes and subjects to validate the pipeline before full-scale training.
Optimization:
FP16 Mixed Precision: Used during feature extraction to reduce memory usage and speed up inference.
Caching: All extracted features are cached to disk (.npz) to prevent redundant computation.
Checkpointing: A custom ExtractionCheckpoint system ensures the pipeline can resume from interruptions.
4. Preliminary Results
Verification was performed using per-parcel Pearson correlation on a hold-out test set (Friends Season 7). The pipeline reports mean, median, and max correlation values to track performance improvements.

(Note: Specific numerical results would be inserted here based on actual run logs.)

5. Conclusion
MultimodalTRIBE offers a flexible and scalable framework for fMRI encoding. By decoupling representation learning (Transformer) from the readout (BMOR), we achieve a system that is both expressive enough to capture complex multimodal dynamics and efficient enough to predict whole-brain activity.