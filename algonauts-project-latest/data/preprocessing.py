import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.impute import SimpleImputer

def fit_modality_specific_pca(features_by_episode, modality, n_components=512, use_incremental=True):
    all_features = [f[modality] for f in features_by_episode.values() if f.get(modality) is not None]
    if not all_features: return None, {}, 0.0
    
    X = np.vstack(all_features).astype('float32')
    if use_incremental and X.shape[0] > 100000:
        pca = IncrementalPCA(n_components=n_components, batch_size=4096)
        for i in range(0, X.shape[0], 4096):
            pca.partial_fit(X[i:i+4096])
    else:
        pca = PCA(n_components=min(n_components, X.shape[1]-1), svd_solver='randomized', random_state=42)
        pca.fit(X)
        
    features_pca = {ep: pca.transform(f[modality]).astype('float32') 
                   for ep, f in features_by_episode.items() if modality in f and f[modality] is not None}
    return pca, features_pca, np.sum(pca.explained_variance_ratio_)

def preprocess_test_features(features_dict, pca, scaler_global, hrf_delay=3):
    visual = features_dict.get('visual')
    audio = features_dict.get('audio')
    language = features_dict.get('language')
    
    n_samples = max(v.shape[0] for v in [visual, audio, language] if v is not None)
    if visual is None: visual = np.zeros((n_samples, 2048), dtype=np.float32)
    if audio is None: audio = np.zeros((n_samples, 20), dtype=np.float32)
    if language is None: language = np.zeros((n_samples, 768), dtype=np.float32)
    
    n_aligned = min(visual.shape[0], audio.shape[0], language.shape[0])
    imputer = SimpleImputer(strategy='mean')
    
    def process_mod(mod, dim):
        m = mod[:n_aligned]
        if np.isnan(m).any(): m = imputer.fit_transform(m)
        return StandardScaler().fit_transform(m).astype(np.float32)
        
    X_combined = np.concatenate([process_mod(visual, 2048), process_mod(audio, 20), process_mod(language, 768)], axis=1)
    X_pca = pca.transform(X_combined).astype(np.float32)
    return scaler_global.transform(X_pca).astype(np.float32)
