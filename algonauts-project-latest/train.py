import os
import torch
import joblib
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset
from torch.cuda.amp import autocast, GradScaler

from utils.config import (
    DEVICE, MODELS_DIR, PREPROCESSING_DIR, FEATURE_CACHE_DIR,
    FMRI_BASE_DIR, PROJ_DIM, N_TRS, N_PARCELS_SMALL
)
from data.dataset import RealFMRIDatasetPooled, build_fmri_index, collate_fn_pad_sequences
from models.tribe import MultimodalTRIBE
from models.bmor import fit_bmor_joblib

def train_tribe_encoder_optimized(model, train_loader, val_loader, device='cuda', epochs=10, lr=3e-4, accumulation_steps=2):
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    scaler = GradScaler()
    best_val = float('inf')
    save_path = os.path.join(MODELS_DIR, 'tribe_encoder_real_best.pth')

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for i, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}")):
            x_txt, x_aud, x_vid, subjs, y_small, _ = [b.to(device) if torch.is_tensor(b) else b for b in batch]
            with autocast():
                preds = model(x_txt, x_aud, x_vid, subjs)
                loss = criterion(preds, y_small) / accumulation_steps
            scaler.scale(loss).backward()
            if (i + 1) % accumulation_steps == 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            train_loss += loss.item() * accumulation_steps
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                x_txt, x_aud, x_vid, subjs, y_small, _ = [b.to(device) if torch.is_tensor(b) else b for b in batch]
                val_loss += criterion(model(x_txt, x_aud, x_vid, subjs), y_small).item()
        
        val_avg = val_loss / len(val_loader)
        print(f"Epoch {epoch+1} Val Loss: {val_avg:.6f}")
        if val_avg < best_val:
            best_val = val_avg
            torch.save(model.state_dict(), save_path)
            print(f"✓ Best model saved (epoch {epoch+1})")
    
    model.load_state_dict(torch.load(save_path))
    return model

def get_latents(model, loader, device):
    X, Y = [], []
    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc="Extracting latents"):
            txt, aud, vid, sub, _, y_all = [b.to(device) if torch.is_tensor(b) else b for b in batch]
            latents = model.encode_only(txt, aud, vid, sub)
            X.append(latents.cpu().reshape(-1, latents.shape[-1]).numpy())
            Y.append(y_all.cpu().reshape(-1, y_all.shape[-1]).numpy())
    return np.vstack(X), np.vstack(Y)

def main():
    print("Step 1: Loading Precomputed Data Config...")
    checkpoint_path = os.path.join(PREPROCESSING_DIR, "dataset_config.pkl")
    if not os.path.exists(checkpoint_path):
        raise RuntimeError(f"Run preprocessing first. Missing: {checkpoint_path}")
    
    dataset_config = joblib.load(checkpoint_path)
    aligned_data = dataset_config['aligned_data']
    
    print("Step 2: Building Indexers...")
    fmri_index = build_fmri_index(FMRI_BASE_DIR)
    subjects = sorted(set([d['subject'] for d in aligned_data]))
    subject_map = {s: i for i, s in enumerate(subjects)}
    
    print("Step 3: Creating Dataset...")
    full_ds = RealFMRIDatasetPooled(
        aligned_data=aligned_data,
        subject_map=subject_map,
        feature_root=FEATURE_CACHE_DIR,
        fmri_index=fmri_index,
        n_trs=N_TRS,
        n_parcels_small=N_PARCELS_SMALL
    )
    
    n_train = int(0.8 * len(full_ds))
    train_ds = Subset(full_ds, list(range(n_train)))
    val_ds = Subset(full_ds, list(range(n_train, len(full_ds))))
    
    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, collate_fn=collate_fn_pad_sequences)
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False, collate_fn=collate_fn_pad_sequences)
    
    print("Step 4: Training TRIBE Encoder...")
    sample = full_ds[0]
    model = MultimodalTRIBE(
        D_text=sample[0].shape[1], D_audio=sample[1].shape[1], D_video=sample[2].shape[1],
        proj_dim=PROJ_DIM, n_subjects=len(subject_map), n_parcels=N_PARCELS_SMALL,
        n_trs=N_TRS, max_seq_len=N_TRS
    )
    
    tribe_model = train_tribe_encoder_optimized(model, train_loader, val_loader, device=DEVICE)
    
    print("Step 5: Extracting Latents for B-MOR...")
    X_latents, Y_responses = get_latents(tribe_model, train_loader, device=DEVICE)
    
    # Simple NaN cleanup
    mask = ~(np.isnan(X_latents).any(axis=1) | np.isnan(Y_responses).any(axis=1))
    X_clean, Y_clean = X_latents[mask], Y_responses[mask]
    
    print("Step 6: Fitting B-MOR Layer...")
    bmor_result = fit_bmor_joblib(X_clean, Y_clean, n_jobs=4)
    joblib.dump(bmor_result, os.path.join(MODELS_DIR, "bmor_result.joblib"))
    print("✓ Training Complete!")

if __name__ == "__main__":
    main()
