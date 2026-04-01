"""
PHASE 1: RIDGE REGRESSION ENCODING MODEL (BASELINE)
===================================================
Optimized for pre-computed PCA and Subject-Specific Training.
Integrates both Friends (S1-S6) and Movie10 datasets for training.

THE FLOW:
---------
1. DATA DISCOVERY: 
   Scans for Friends (S1-S6) and Movie10 episodes for training.
   Seasons 7 is reserved for the leaderboard submission (test set).

2. FEATURE PREPARATION:
   Loads pre-extracted features and applies your saved PCA model once.

3. SUBJECT-SPECIFIC TRAINING:
   For each subject, aligns stimulus with fMRI (HRF delay) and fits 
   1,000 independent RidgeCV models in parallel.

4. INFERENCE & SUBMISSION:
   Predicts Season 7 brain activity and packages it for Codabench.
"""

import os
import glob
import h5py
import pickle
import zipfile
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from joblib import Parallel, delayed

# =============================================================================
# CONFIGURATION
# =============================================================================
ROOT_DATA_DIR = r"D:\fmri-algonauts-2025-data" 
ALGONAUTS_DIR = os.path.join(ROOT_DATA_DIR, "algonauts_2025.competitors")
FEATURE_CACHE_DIR = os.path.join(ROOT_DATA_DIR, "feature_cache_v2")
PCA_MODEL_PATH = os.path.join(ROOT_DATA_DIR, "preprocessing_pipeline","dataset_config.pkl") 
OUTPUT_DIR = "./phase1_ridge_submission"

# Parameters
HRF_DELAY = 3
ALPHAS = np.logspace(1, 5, 10) 
N_JOBS = -1 # Use all CPU cores

# Subject Definitions
SUBJECTS = ["sub-01", "sub-02", "sub-03", "sub-05"]

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_episode_lists():
    """Identifies training episodes (S1-S6 + Movie10) and test episodes (S7)."""
    # 1. Get Friends Episodes
    transcript_root = os.path.join(ALGONAUTS_DIR, "stimuli", "transcripts", "friends")
    friends_eps = []
    for s_dir in sorted(os.listdir(transcript_root)):
        s_path = os.path.join(transcript_root, s_dir)
        if os.path.isdir(s_path):
            friends_eps.extend([f.replace("friends_", "").replace(".tsv", "") 
                               for f in sorted(os.listdir(s_path)) if f.endswith(".tsv")])
    
    # 2. Get Movie10 Episodes
    movie_root = os.path.join(ALGONAUTS_DIR, "stimuli", "transcripts", "movie10")
    movie_eps = []
    if os.path.exists(movie_root):
        for genre in os.listdir(movie_root):
            g_path = os.path.join(movie_root, genre)
            if os.path.isdir(g_path):
                movie_eps.extend([f.replace("movie10_", "").replace(".tsv", "") 
                                 for f in sorted(os.listdir(g_path)) if f.endswith(".tsv")])

    train_eps = [ep for ep in friends_eps if "s7" not in ep] + movie_eps
    test_eps = [ep for ep in friends_eps if "s7" in ep]
    return train_eps, test_eps

def load_and_preprocess_features(ep_name, pca_model):
    """Loads cached features, cleans them, and applies your PCA model."""
    cache_path = os.path.join(FEATURE_CACHE_DIR, f"{ep_name}_features.npz")
    if not os.path.exists(cache_path): return None
    
    with np.load(cache_path, allow_pickle=True) as data:
        vis, aud, lang = data['visual'], data['audio'], data['language']
        
        # Handle case where lang is stored as a 0-d object array (numpy item)
        if lang.dtype == object and lang.shape == ():
            lang = lang.item()
        if vis.dtype == object and vis.shape == ():
            vis = vis.item()

        # Handle Language format (Flatten if 3D)
        if lang.ndim == 3: 
            lang = lang.reshape(lang.shape[0], -1)
            
        # Basic Imputation & Scaling
        imputer = SimpleImputer(strategy='mean')
        vis = imputer.fit_transform(vis) if np.isnan(vis).any() else vis
        aud = imputer.fit_transform(aud) if np.isnan(aud).any() else aud
        lang = imputer.fit_transform(lang) if np.isnan(lang).any() else lang
        
        vis = StandardScaler().fit_transform(vis)
        aud = StandardScaler().fit_transform(aud)
        lang = StandardScaler().fit_transform(lang)
        
        X_combined = np.concatenate([vis, aud, lang], axis=1).astype(np.float32)
        return pca_model.transform(X_combined)

def fit_subject_models(X_train, Y_train):
    """Fits one RidgeCV model per brain parcel in parallel."""
    def fit_single_parcel(p_idx):
        model = RidgeCV(alphas=ALPHAS, scoring='neg_mean_squared_error')
        model.fit(X_train, Y_train[:, p_idx])
        return model

    print(f"  Training 1000 Ridge models in parallel...")
    models = Parallel(n_jobs=N_JOBS)(
        delayed(fit_single_parcel)(i) for i in tqdm(range(Y_train.shape[1]), leave=False)
    )
    return models

# =============================================================================
# MAIN PIPELINE
# =============================================================================

def main():
    # 1. Setup
    train_eps, test_eps = get_episode_lists()
    print(f"✓ Found {len(train_eps)} Train Episodes and {len(test_eps)} Test Episodes.")
    
    if not os.path.exists(PCA_MODEL_PATH):
        print(f"⚠ Error: Export your PCA object to {PCA_MODEL_PATH} from your notebook first!")
        return
    
    with open(PCA_MODEL_PATH, 'rb') as f:
        pca = pickle.load(f)
    print(f"✓ Loaded existing PCA model from {PCA_MODEL_PATH}")

    # 2. Pre-calculate PCA features (Stimulus only - done once to save memory)
    print("\n[STEP 1] Pre-calculating PCA features for all episodes...")
    pca_features_cache = {}
    for ep in tqdm(train_eps + test_eps, desc="Transforming"):
        feat = load_and_preprocess_features(ep, pca)
        if feat is not None: 
            pca_features_cache[ep] = feat

    # 3. Training Loop (Per Subject)
    predictions_dict = {}
    for subject in SUBJECTS:
        print(f"\n>>> PROCESSING SUBJECT: {subject}")
        X_subj, Y_subj = [], []
        fmri_files = glob.glob(os.path.join(ALGONAUTS_DIR, "fmri", subject, "func", "*.h5"))
        
        if not fmri_files:
            print(f"  ⚠ No fMRI files found for {subject}. Skipping...")
            continue

        for h5_path in fmri_files:
            with h5py.File(h5_path, 'r') as f:
                for ep in tqdm(train_eps, desc=f"Aligning {Path(h5_path).name}", leave=False):
                    # Key-matching logic (matches ep ID in HDF5 key)
                    matching_key = [k for k in f.keys() if ep in k]
                    if not matching_key or ep not in pca_features_cache: 
                        continue
                    
                    X = pca_features_cache[ep]
                    Y = f[matching_key[0]][()]
                    
                    # Align: fMRI(t+3) matches Stimulus(t)
                    Y_aligned = Y[HRF_DELAY:]
                    n = min(len(Y_aligned), len(X))
                    X_subj.append(X[:n])
                    Y_subj.append(Y_aligned[:n])

        if not X_subj:
            print(f"  ⚠ No aligned data found for {subject}. Skipping...")
            continue

        # Execute Training (The Voxelwise Encoding)
        X_train_full = np.vstack(X_subj)
        Y_train_full = np.vstack(Y_subj)
        print(f"  Training Matrix: {X_train_full.shape} -> {Y_train_full.shape}")
        subject_models = fit_subject_models(X_train_full, Y_train_full)

        # Inference (Season 7)
        print(f"  Predicting S7...")
        predictions_dict[subject] = {}
        subj_dir = os.path.join(OUTPUT_DIR, subject)
        os.makedirs(subj_dir, exist_ok=True)

        for ep in tqdm(test_eps, desc="Inference", leave=False):
            if ep not in pca_features_cache: continue
            X_test = pca_features_cache[ep]
            
            Y_pred = np.zeros((X_test.shape[0], 1000), dtype=np.float32)
            for p_idx, model in enumerate(subject_models):
                Y_pred[:, p_idx] = model.predict(X_test)
            
            predictions_dict[subject][ep] = Y_pred
            np.save(os.path.join(subj_dir, f"{ep}_predictions.npy"), Y_pred)

    # 4. Finalize Submission
    print("\n[STEP 3] Packaging submission...")
    pickle_path = os.path.join(OUTPUT_DIR, "predictions_dict.pkl")
    with open(pickle_path, 'wb') as f:
        pickle.dump(predictions_dict, f)

    zip_path = os.path.join(OUTPUT_DIR, "ridge_baseline_submission.zip")
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.write(pickle_path, arcname='predictions_dict.pkl')
        for subject in predictions_dict.keys():
            for ep in predictions_dict[subject].keys():
                npy_file = os.path.join(OUTPUT_DIR, subject, f"{ep}_predictions.npy")
                if os.path.exists(npy_file):
                    zf.write(npy_file, arcname=f"{subject}/{ep}_predictions.npy")

    print(f"\n✓ Phase 1 Complete! Submission ready at: {zip_path}")

if __name__ == "__main__":
    main()
