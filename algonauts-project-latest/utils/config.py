import os
import torch
import numpy as np

# =============================================================================
# DATA DIRECTORIES
# =============================================================================
ROOT_DATA_DIR = r"D:\fmri-algonauts-2025-data"
ALGONAUTS_DIR = os.path.join(ROOT_DATA_DIR, "algonauts_2025.competitors")
STIMULI_DIR = os.path.join(ALGONAUTS_DIR, "stimuli")
TEST_DATA_DIR = os.path.join(ALGONAUTS_DIR, "testdata")
FMRI_BASE_DIR = os.path.join(ALGONAUTS_DIR, "fmri")
FEATURE_CACHE_DIR = os.path.join(ROOT_DATA_DIR, "feature_cache_v2")
CHECKPOINT_DIR = os.path.join(ROOT_DATA_DIR, "extraction_checkpoints")
LOG_DIR = os.path.join(ROOT_DATA_DIR, "extraction_logs")
MODELS_DIR = os.path.join(ROOT_DATA_DIR, "trained_models")
PREPROCESSING_DIR = os.path.join(ROOT_DATA_DIR, "preprocessing_pipeline")

# Output Directories
OUTPUT_DIR = "./predictions_submission"
PHASE1_OUTPUT_DIR = "./phase1_ridge_submission_ood"

# Ensure dirs exist
for d in [FEATURE_CACHE_DIR, CHECKPOINT_DIR, LOG_DIR, MODELS_DIR, PREPROCESSING_DIR, OUTPUT_DIR]:
    os.makedirs(d, exist_ok=True)

# =============================================================================
# MODEL PARAMETERS
# =============================================================================
PROJ_DIM = 64
N_TRS = 4
N_PARCELS_SMALL = 100
N_PARCELS_TOTAL = 1000
TR_DURATION = 1.49
ALPHAS = np.logspace(1, 5, 10)
SUBJECTS = ["sub-01", "sub-02", "sub-03", "sub-05"]

# =============================================================================
# GPU SETTINGS
# =============================================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# =============================================================================
# OOD TARGET COUNTS
# =============================================================================
TARGET_COUNTS = {
    'chaplin1': 432, 'chaplin2': 405, 
    'mononoke1': 423, 'mononoke2': 426,
    'passepartout1': 422, 'passepartout2': 436,
    'planetearth1': 433, 'planetearth2': 418,
    'pulpfiction1': 468, 'pulpfiction2': 378,
    'wot1': 353, 'wot2': 324
}

MOVIE10_SEQUENCE = (
    [f"bourne{i:02d}" for i in range(1, 11)] +
    [f"wolf{i:02d}"   for i in range(1, 18)] +
    [f"life{i:02d}"   for i in range(1, 6)] +
    [f"figures{i:02d}" for i in range(1, 13)]
)
