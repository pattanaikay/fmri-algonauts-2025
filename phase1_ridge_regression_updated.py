"""
PHASE 1: RIDGE REGRESSION ENCODING MODEL (UPDATED & PERSISTENT)
==============================================================
This version uses the pre-computed 'dataset_config.pkl' from Step 6.
It SAVES trained models to disk to allow for resumption and later use.

THE FLOW:
---------
1. LOAD CONFIG: Loads X_final, y_final, PCA model, and global scaler via joblib.
2. PARTITION: Slices the global matrices into subject-specific chunks.
3. TRAIN/LOAD: Checks for saved models; otherwise, fits 1,000 RidgeCV models.
4. TEST PREP: Loads Season 7 stimulus, applies PCA, then applies Global Scaler.
5. INFERENCE: Predicts brain activity for Season 7.
6. SUBMIT: Packages results for Codabench.
"""

import os
import glob
import h5py
import pickle
import joblib  
import zipfile
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# =============================================================================
# CONFIGURATION
# =============================================================================
ROOT_DATA_DIR = r"D:\fmri-algonauts-2025-data" 
ALGONAUTS_DIR = os.path.join(ROOT_DATA_DIR, "algonauts_2025.competitors")
# Path based on successful discovery
TEST_DATA_DIR = os.path.join(ALGONAUTS_DIR, "testdata")
FEATURE_CACHE_DIR = os.path.join(ROOT_DATA_DIR, "feature_cache_v2")
DATASET_CONFIG_PATH = os.path.join(ROOT_DATA_DIR, "preprocessing_pipeline", "dataset_config.pkl")

# Output Directories
OUTPUT_DIR = "./phase1_ridge_submission_ood"
MODELS_DIR = os.path.join(OUTPUT_DIR, "trained_models")

# Parameters
ALPHAS = np.logspace(1, 5, 10) 
SUBJECTS = ["sub-01", "sub-02", "sub-03", "sub-05"]

os.makedirs(MODELS_DIR, exist_ok=True)

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

# Exact sample counts required by the competition (Codabench targets)
TARGET_COUNTS = {
    'chaplin1': 432, 'chaplin2': 405, 
    'mononoke1': 423, 'mononoke2': 426,
    'passepartout1': 422, 'passepartout2': 436,
    'planetearth1': 433, 'planetearth2': 418,
    'pulpfiction1': 468, 'pulpfiction2': 378,
    'wot1': 353, 'wot2': 324
}

def get_test_episodes():
    """Identifies ONLY the 12 required OOD test movies for the competition."""
    test_eps = []
    movies_ood_path = os.path.join(TEST_DATA_DIR, "movies", "ood")
    
    # Whitelist of episodes required by Codabench
    required_episodes = set(TARGET_COUNTS.keys())
    
    if os.path.exists(movies_ood_path):
        try:
            genres = [f for f in os.listdir(movies_ood_path) 
                     if os.path.isdir(os.path.join(movies_ood_path, f)) and f != ".datalad"]
            
            for genre in genres:
                genre_path = os.path.join(movies_ood_path, genre)
                # Find all task-*_video.mkv files
                for f in os.listdir(genre_path):
                    if f.startswith("task-") and f.endswith("_video.mkv"):
                        # Extract episode name: task-chaplin1_video.mkv -> chaplin1
                        ep = f.replace("task-", "").replace("_video.mkv", "")
                        # STRICT FILTER: Only include if it is in the required list
                        if ep in required_episodes:
                            test_eps.append(ep)
                            
            print(f"  ✓ Found {len(test_eps)} required OOD episodes.")
        except Exception as e:
            print(f"  ⚠ Error reading movies/ood: {e}")
    
    # Final fallback: Check feature cache if folder scan failed
    if not test_eps:
        print("  ⚠ No required OOD movies found in testdata, checking feature cache...")
        if os.path.exists(FEATURE_CACHE_DIR):
            for f in os.listdir(FEATURE_CACHE_DIR):
                ep = None
                if f.startswith("task-") and f.endswith("_video_features.npz"):
                    ep = f.replace("task-", "").replace("_video_features.npz", "")
                elif f.endswith("_features.npz"):
                    ep = f.replace("_features.npz", "")
                
                if ep in required_episodes:
                    test_eps.append(ep)
        
    return sorted(list(set(test_eps)))

def load_and_preprocess_test_features(ep_name, pca_model, global_scaler):
    """Loads OOD movie features and pads them to match the target competition length."""
    
    # 1. Determine Target Length (Priority: Hardcoded > Transcript > Cache)
    target_samples = TARGET_COUNTS.get(ep_name)
    
    if target_samples is None:
        transcripts_ood_path = os.path.join(TEST_DATA_DIR, "transcripts", "ood")
        if os.path.exists(transcripts_ood_path):
            for genre in os.listdir(transcripts_ood_path):
                # Try multiple naming conventions for transcripts
                possible_ts = [
                    f"{ep_name}.tsv",
                    f"task-{ep_name}_video.tsv",
                    f"task-{ep_name}.tsv"
                ]
                for ts_name in possible_ts:
                    transcript_path = os.path.join(transcripts_ood_path, genre, ts_name)
                    if os.path.exists(transcript_path):
                        df_temp = pd.read_csv(transcript_path, sep='\t')
                        target_samples = len(df_temp)
                        break
                if target_samples: break

    # 2. Load Features
    possible_names = [
        f"{ep_name}_features.npz",
        f"task-{ep_name}_video_features.npz",
        f"task-{ep_name}_features.npz",
    ]
    
    cache_path = None
    for name in possible_names:
        test_path = os.path.join(FEATURE_CACHE_DIR, name)
        if os.path.exists(test_path):
            cache_path = test_path
            break
    
    # Search for partial matches if needed
    if not cache_path and os.path.exists(FEATURE_CACHE_DIR):
        for f in os.listdir(FEATURE_CACHE_DIR):
            if ep_name in f and "video_features.npz" in f:
                cache_path = os.path.join(FEATURE_CACHE_DIR, f)
                break
            
    if not cache_path:
        print(f"    ⚠ Feature cache not found for {ep_name}")
        return None

    with np.load(cache_path, allow_pickle=True) as data:
        # Load modalities
        vis = data.get('visual')
        aud = data.get('audio')
        lang = data.get('language')

        # If language is a scalar object, extract it
        if lang is not None and lang.dtype == object and lang.shape == ():
            lang = lang.item()

        # Check existing samples
        valid_mods = [m for m in [vis, aud, lang] if m is not None]
        if not valid_mods: return None
        
        current_samples = min(m.shape[0] for m in valid_mods)
        
        # Final length determination
        if target_samples is None:
            print(f"    ⚠ Target length unknown for {ep_name}, using current: {current_samples}")
            target_samples = current_samples
        else:
            diff = target_samples - current_samples
            if diff != 0:
                status = "Padding" if diff > 0 else "Trimming"
                print(f"    ⚠ {ep_name}: {status} {current_samples} -> {target_samples} samples")
            else:
                print(f"    ✓ {ep_name}: Length matches target {target_samples}")

        # 3. Helper to Pad or Trim
        def fix_length(feat, dim):
            if feat is None:
                return np.zeros((target_samples, dim), dtype=np.float32)
            
            # Reshape language if needed
            if dim == 768:
                if feat.ndim == 3: feat = feat.reshape(feat.shape[0], -1)
                if feat.shape[1] > 768: feat = feat[:, :768]
                elif feat.shape[1] < 768:
                    pad = np.zeros((feat.shape[0], 768 - feat.shape[1]), dtype=feat.dtype)
                    feat = np.concatenate([feat, pad], axis=1)

            if feat.shape[0] >= target_samples:
                return feat[:target_samples]
            else:
                # Pad with the last frame
                padding = np.tile(feat[-1:], (target_samples - feat.shape[0], 1))
                return np.vstack([feat, padding])

        vis = fix_length(vis, 2048)
        aud = fix_length(aud, 20)
        lang = fix_length(lang, 768)

        # 4. Final Processing
        imputer = SimpleImputer(strategy='mean')
        vis = imputer.fit_transform(vis)
        aud = imputer.fit_transform(aud)
        lang = imputer.fit_transform(lang)
        
        vis = StandardScaler().fit_transform(vis)
        aud = StandardScaler().fit_transform(aud)
        lang = StandardScaler().fit_transform(lang)
        
        X_combined = np.concatenate([vis, aud, lang], axis=1).astype(np.float32)
        X_pca = pca_model.transform(X_combined)
        return global_scaler.transform(X_pca).astype(np.float32)

def fit_subject_models(X_train, Y_train):
    """Fits RidgeCV models for each parcel in multi-target batches."""
    print(f"  Training Ridge models for {Y_train.shape[1]} parcels...")
    batch_size = 100
    n_parcels = Y_train.shape[1]
    all_models = []
    
    for i in tqdm(range(0, n_parcels, batch_size), desc="Training Batches"):
        end = min(i + batch_size, n_parcels)
        model = RidgeCV(alphas=ALPHAS, scoring='neg_mean_squared_error')
        model.fit(X_train, Y_train[:, i:end])
        all_models.append(model)
    return all_models

# =============================================================================
# MAIN PIPELINE
# =============================================================================

def main():
    if not os.path.exists(DATASET_CONFIG_PATH):
        print(f"⚠ Error: {DATASET_CONFIG_PATH} not found!")
        return
    
    print(f"\n[STEP 1] Loading precomputed data config...")
    try:
        config = joblib.load(DATASET_CONFIG_PATH, mmap_mode='r')
        print(f"✓ Successfully loaded {DATASET_CONFIG_PATH}")
    except Exception as e:
        print(f"⚠ Failed to load config: {e}"); return
    
    X_train_all = config['X_final']        
    Y_train_all = config['y_final']        
    pca_model = config['pca']              
    global_scaler = config['scaler_global'] 
    aligned_metadata = config['aligned_data'] 
    
    test_eps = get_test_episodes()
    print(f"✓ Training Data Matrix: {X_train_all.shape}")
    print(f"✓ Test Episodes Found: {len(test_eps)}")

    predictions_dict = {}
    for subject in SUBJECTS:
        print(f"\n>>> PROCESSING SUBJECT: {subject}")
        model_save_path = os.path.join(MODELS_DIR, f"{subject}_ridge_models.joblib")
        
        if os.path.exists(model_save_path):
            print(f"  Found existing models for {subject}. Loading...")
            subject_models = joblib.load(model_save_path)
        else:
            current_row = 0
            X_subj_list, Y_subj_list = [], []
            for entry in tqdm(aligned_metadata, desc="Partitioning Rows", leave=False):
                n_samples = entry['X'].shape[0]
                if entry['subject'] == subject:
                    X_subj_list.append(X_train_all[current_row : current_row + n_samples])
                    Y_subj_list.append(Y_train_all[current_row : current_row + n_samples])
                current_row += n_samples

            if not X_subj_list: continue
            X_subj = np.vstack(X_subj_list)
            Y_subj = np.vstack(Y_subj_list)
            print(f"  Training Matrix: {X_subj.shape} features -> {Y_subj.shape} responses")
            subject_models = fit_subject_models(X_subj, Y_subj)
            joblib.dump(subject_models, model_save_path)

        if test_eps:
            print(f"  Predicting OOD movies...")
            predictions_dict[subject] = {}
            subj_dir = os.path.join(OUTPUT_DIR, subject)
            os.makedirs(subj_dir, exist_ok=True)

            for ep in tqdm(test_eps, desc="Inference", leave=False):
                X_test_final = load_and_preprocess_test_features(ep, pca_model, global_scaler)
                if X_test_final is None: continue
                
                Y_pred_list = []
                for model in subject_models:
                    pred = model.predict(X_test_final)
                    if pred.ndim == 1: pred = pred.reshape(-1, 1)
                    Y_pred_list.append(pred)
                
                Y_pred = np.concatenate(Y_pred_list, axis=1).astype(np.float32)
                predictions_dict[subject][ep] = Y_pred
                np.save(os.path.join(subj_dir, f"{ep}_predictions.npy"), Y_pred)

    if predictions_dict:
        print("\n[STEP 3] Packaging submission...")
        npy_submission_path = os.path.join(OUTPUT_DIR, "predictions.npy")
        np.save(npy_submission_path, predictions_dict)
        
        zip_path = os.path.join(OUTPUT_DIR, "ridge_ood_submission.zip")
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            zf.write(npy_submission_path, arcname='predictions.npy')
        print(f"\n✓ Phase 1 Complete! OOD Submission ZIP: {zip_path}")

if __name__ == "__main__":
    main()
