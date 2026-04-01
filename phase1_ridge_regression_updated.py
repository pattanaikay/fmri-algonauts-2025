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
OUTPUT_DIR = "./phase1_ridge_submission_updated"
MODELS_DIR = os.path.join(OUTPUT_DIR, "trained_models")

# Parameters
ALPHAS = np.logspace(1, 5, 10) 
SUBJECTS = ["sub-01", "sub-02", "sub-03", "sub-05"]

os.makedirs(MODELS_DIR, exist_ok=True)

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_test_episodes():
    """Identifies test episodes (S7) from the testdata folder."""
    test_eps = []
    
    search_paths = [
        TEST_DATA_DIR,
        os.path.join(TEST_DATA_DIR, "transcripts", "s7"),
        os.path.join(TEST_DATA_DIR, "friends", "s7")
    ]
    
    for path in search_paths:
        if os.path.exists(path):
            try:
                files = os.listdir(path)
                for f in files:
                    if f.endswith(".tsv") and ("s7" in f or "s07" in f):
                        ep = f.replace("friends_", "").replace(".tsv", "").replace("movie10_", "")
                        test_eps.append(ep)
            except Exception:
                continue
    
    if not test_eps:
        print("  ⚠ No episodes found in testdata folders, checking feature cache...")
        if os.path.exists(FEATURE_CACHE_DIR):
            test_eps = [f.replace("_features.npz", "") for f in os.listdir(FEATURE_CACHE_DIR) if "s7" in f or "s07" in f]
        
    return sorted(list(set(test_eps)))

def load_and_preprocess_test_features(ep_name, pca_model, global_scaler):
    """Loads S7 features, applies PCA, then applies the global training scaler."""
    cache_path = os.path.join(FEATURE_CACHE_DIR, f"{ep_name}_features.npz")
    if not os.path.exists(cache_path): return None
    
    with np.load(cache_path, allow_pickle=True) as data:
        # Safely load modalities
        def process_modality(name):
            val = data.get(name)
            if val is None: return None
            if hasattr(val, 'dtype') and val.dtype == object and val.shape == ():
                return val.item()
            return val

        vis = process_modality('visual')
        aud = process_modality('audio')
        lang = process_modality('language')

        valid_mods = [m for m in [vis, aud, lang] if m is not None]
        if not valid_mods: return None
        
        n_samples = min(m.shape[0] for m in valid_mods)
        
        # Format and pad missing modalities
        if vis is not None: vis = vis[:n_samples]
        else: vis = np.zeros((n_samples, 2048), dtype=np.float32)
            
        if aud is not None: aud = aud[:n_samples]
        else: aud = np.zeros((n_samples, 20), dtype=np.float32)
            
        if lang is not None:
            if lang.ndim == 3: lang = lang.reshape(lang.shape[0], -1)
            if lang.shape[1] > 768: lang = lang[:, :768]
            elif lang.shape[1] < 768:
                padding = np.zeros((lang.shape[0], 768 - lang.shape[1]), dtype=lang.dtype)
                lang = np.concatenate([lang, padding], axis=1)
            lang = lang[:n_samples]
        else: lang = np.zeros((n_samples, 768), dtype=np.float32)

        # Scale and Transfrom
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
            print(f"  Predicting Season 7...")
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
        
        zip_path = os.path.join(OUTPUT_DIR, "ridge_baseline_submission.zip")
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            zf.write(npy_submission_path, arcname='predictions.npy')
        print(f"\n✓ Phase 1 Complete! Submission ZIP: {zip_path}")

if __name__ == "__main__":
    main()
