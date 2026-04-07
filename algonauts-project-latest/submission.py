import os
import torch
import joblib
import pickle
import zipfile
import numpy as np
from tqdm import tqdm

from utils.config import (
    DEVICE, MODELS_DIR, FEATURE_CACHE_DIR, OUTPUT_DIR, PROJ_DIM, N_TRS, PREPROCESSING_DIR
)
from utils.metrics import get_test_episodes
from models.tribe import MultimodalTRIBE

def main():
    print("Step 1: Loading Trained Models...")
    tribe_path = os.path.join(MODELS_DIR, 'tribe_encoder_real_best.pth')
    bmor_path = os.path.join(MODELS_DIR, "bmor_result.joblib")
    config_path = os.path.join(PREPROCESSING_DIR, "dataset_config.pkl")
    
    if not all(os.path.exists(p) for p in [tribe_path, bmor_path, config_path]):
        raise RuntimeError("Missing model or config files. Run train.py first.")

    dataset_config = joblib.load(config_path)
    aligned_data = dataset_config['aligned_data']
    subjects = sorted(set([d['subject'] for d in aligned_data]))
    subject_map = {s: i for i, s in enumerate(subjects)}
    
    tribe_model = MultimodalTRIBE(
        D_text=8448, D_audio=20, D_video=2048, # Standard dims from ref script
        proj_dim=PROJ_DIM, n_subjects=len(subject_map),
        n_parcels=100, n_trs=N_TRS, max_seq_len=N_TRS
    ).to(DEVICE)
    tribe_model.load_state_dict(torch.load(tribe_path, map_location=DEVICE))
    tribe_model.eval()
    
    bmor_result = joblib.load(bmor_path)
    bmor_coefs = bmor_result['coefs']
    bmor_intercepts = bmor_result['intercepts']
    bmor_scaler = bmor_result['scaler']
    
    print("Step 2: Loading Test Episodes...")
    test_eps = get_test_episodes()
    predictions_dict = {}

    for subject in subjects:
        print(f"  Inference for {subject}...")
        predictions_dict[subject] = {}
        subj_idx = subject_map[subject]
        subject_id = torch.tensor([subj_idx], dtype=torch.long).to(DEVICE)

        for ep in tqdm(test_eps, leave=False):
            feat_path = os.path.join(FEATURE_CACHE_DIR, f"{ep}_features.npz")
            if not os.path.exists(feat_path): continue
            
            feat = np.load(feat_path, allow_pickle=True)
            def unwrap(f, k, dim):
                if k not in f: return np.zeros((10, dim)) # Placeholder
                arr = f[k].item() if f[k].shape == () else f[k]
                return arr

            x_txt = torch.from_numpy(unwrap(feat, "language", 768)).float().to(DEVICE).unsqueeze(0)
            x_aud = torch.from_numpy(unwrap(feat, "audio", 20)).float().to(DEVICE).unsqueeze(0)
            x_vid = torch.from_numpy(unwrap(feat, "visual", 2048)).float().to(DEVICE).unsqueeze(0)
            n_samples = x_vid.shape[1]

            with torch.no_grad():
                latents = tribe_model.encode_only(x_txt, x_aud, x_vid, subject_id).squeeze(0)
                # Interpolate back to original sample count
                latents_upsampled = torch.nn.functional.interpolate(
                    latents.transpose(0, 1).unsqueeze(0), size=n_samples, mode='linear'
                ).squeeze(0).transpose(0, 1).cpu().numpy()

            X_scaled = bmor_scaler.transform(latents_upsampled)
            Y_pred = X_scaled.dot(bmor_coefs.T) + bmor_intercepts[None, :]
            predictions_dict[subject][ep] = Y_pred.astype(np.float32)

    print("Step 3: Packaging Submission...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    pickle_path = os.path.join(OUTPUT_DIR, "predictions_dict.pkl")
    with open(pickle_path, 'wb') as f:
        pickle.dump(predictions_dict, f)

    zip_path = os.path.join(OUTPUT_DIR, "submission.zip")
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.write(pickle_path, arcname='predictions_dict.pkl')
        for subj in predictions_dict:
            for ep, arr in predictions_dict[subj].items():
                npy_path = os.path.join(OUTPUT_DIR, f"{subj}_{ep}.npy")
                np.save(npy_path, arr)
                zf.write(npy_path, arcname=f"{subj}/{ep}_predictions.npy")
                os.remove(npy_path)

    print(f"✓ Submission created at {zip_path}")

if __name__ == "__main__":
    main()
