import numpy as np
import os
import pandas as pd
from .config import TARGET_COUNTS, TEST_DATA_DIR, FEATURE_CACHE_DIR

def pearson_r(y_true, y_pred):
    """
    Compute Pearson correlation between y_true and y_pred.
    Works for multiple targets (parcels).
    """
    yt = y_true - y_true.mean(0)
    yp = y_pred - y_pred.mean(0)
    return (yt * yp).sum(0) / (
        np.sqrt((yt**2).sum(0) * (yp**2).sum(0)) + 1e-8
    )

def get_test_episodes():
    """Identifies ONLY the 12 required OOD test movies for the competition."""
    test_eps = []
    movies_ood_path = os.path.join(TEST_DATA_DIR, "movies", "ood")
    required_episodes = set(TARGET_COUNTS.keys())
    
    if os.path.exists(movies_ood_path):
        try:
            genres = [f for f in os.listdir(movies_ood_path) 
                     if os.path.isdir(os.path.join(movies_ood_path, f)) and f != ".datalad"]
            
            for genre in genres:
                genre_path = os.path.join(movies_ood_path, genre)
                for f in os.listdir(genre_path):
                    if f.startswith("task-") and f.endswith("_video.mkv"):
                        ep = f.replace("task-", "").replace("_video.mkv", "")
                        if ep in required_episodes:
                            test_eps.append(ep)
        except Exception as e:
            print(f"  ⚠ Error reading movies/ood: {e}")
    
    if not test_eps:
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
