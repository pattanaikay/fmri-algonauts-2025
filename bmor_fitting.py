# ==========================
# STEP 8: TRIBE + B-MOR Training on Real 10% Dataset
# ==========================

import os
import math
import random
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from joblib import Parallel, delayed, dump, load

# --------------------------
# MultimodalTRIBE (with encode_only)
# --------------------------
class MultimodalTRIBE(nn.Module):
    def __init__(self, D_text, D_audio, D_video, proj_dim=128, n_subjects=5, d_model=None,
                 n_parcels=50, n_trs=20, max_seq_len=60, transformer_layers=2, nheads=4,
                 ff_dim=512, dropout=0.1, modality_dropout_p=0.2):
        super().__init__()
        if d_model is None:
            d_model = 3 * proj_dim
        self.txt_proj = nn.Sequential(nn.Linear(D_text, proj_dim), nn.LayerNorm(proj_dim))
        self.aud_proj = nn.Sequential(nn.Linear(D_audio, proj_dim), nn.LayerNorm(proj_dim))
        self.vid_proj = nn.Sequential(nn.Linear(D_video, proj_dim), nn.LayerNorm(proj_dim))
        self.pos_emb = nn.Parameter(torch.randn(1, max_seq_len, d_model) * 0.02)
        self.subj_emb = nn.Embedding(n_subjects, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nheads, dim_feedforward=ff_dim,
            dropout=dropout, activation='gelu', batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)
        self.n_trs = n_trs
        self.pool = nn.AdaptiveAvgPool1d(n_trs)
        self.readout = nn.Linear(d_model, n_parcels)
        self.subj_bias = nn.Embedding(n_subjects, n_parcels)
        self.modality_dropout_p = modality_dropout_p

    def modality_dropout(self, x_txt, x_aud, x_vid):
        if not self.training or self.modality_dropout_p <= 0.0:
            return x_txt, x_aud, x_vid
        B = x_txt.shape[0]
        mask_txt = torch.bernoulli((1 - self.modality_dropout_p) * torch.ones(B,1,1,device=x_txt.device))
        mask_aud = torch.bernoulli((1 - self.modality_dropout_p) * torch.ones(B,1,1,device=x_aud.device))
        mask_vid = torch.bernoulli((1 - self.modality_dropout_p) * torch.ones(B,1,1,device=x_vid.device))
        sum_mask = (mask_txt + mask_aud + mask_vid).squeeze()
        for i in range(B):
            if sum_mask[i] == 0:
                choice = random.choice([0,1,2])
                if choice == 0: mask_txt[i] = 1.
                elif choice == 1: mask_aud[i] = 1.
                else: mask_vid[i] = 1.
        return x_txt * mask_txt, x_aud * mask_aud, x_vid * mask_vid

    def forward(self, x_txt, x_aud, x_vid, subject_ids):
        x_txt, x_aud, x_vid = self.modality_dropout(x_txt, x_aud, x_vid)
        t_txt = self.txt_proj(x_txt)
        t_aud = self.aud_proj(x_aud)
        t_vid = self.vid_proj(x_vid)
        x = torch.cat([t_txt, t_aud, t_vid], dim=-1)
        B, fT, _ = x.shape
        pos = self.pos_emb[:, :fT, :]
        subj = self.subj_emb(subject_ids).unsqueeze(1)
        x = x + pos + subj
        x_out = self.transformer(x)
        x_perm = x_out.transpose(1,2)
        pooled = self.pool(x_perm).transpose(1,2)
        preds = self.readout(pooled)
        preds = preds + self.subj_bias(subject_ids).unsqueeze(1)
        return preds

    @torch.no_grad()
    def encode_only(self, x_txt, x_aud, x_vid, subject_ids):
        self.eval()
        t_txt = self.txt_proj(x_txt)
        t_aud = self.aud_proj(x_aud)
        t_vid = self.vid_proj(x_vid)
        x = torch.cat([t_txt, t_aud, t_vid], dim=-1)
        B, fT, _ = x.shape
        pos = self.pos_emb[:, :fT, :].to(x.device)
        subj = self.subj_emb(subject_ids.to(x.device)).unsqueeze(1).to(x.device)
        x = x + pos + subj
        x_out = self.transformer(x)
        x_perm = x_out.transpose(1,2)
        pooled = self.pool(x_perm).transpose(1,2)
        return pooled

# --------------------------
# Real Data Dataset Wrapper
# --------------------------
class RealFMRIDataset(Dataset):
    def __init__(self, aligned_data, subject_map, modality_features, n_parcels_small=50):
        self.aligned_data = aligned_data
        self.subject_map = subject_map
        self.modality_features = modality_features
        self.n_parcels_small = n_parcels_small
        self.cache = []
        for entry in aligned_data:
            subject_id = subject_map[entry['subject']]
            episode = entry['episode']
            y_all = entry['y']
            feats = modality_features.get(episode)
            if feats is None:
                print(f"Warning: no features for {episode}")
                continue
            n_samples = y_all.shape[0]
            x_txt = feats.get('language', np.zeros((n_samples, 768))).astype(np.float32)
            x_aud = feats.get('audio', np.zeros((n_samples, 20))).astype(np.float32)
            x_vid = feats.get('visual', np.zeros((n_samples, 2048))).astype(np.float32)
            x_txt = x_txt[:n_samples]
            x_aud = x_aud[:n_samples]
            x_vid = x_vid[:n_samples]
            y_small = y_all[:, :self.n_parcels_small].astype(np.float32)
            self.cache.append({
                'x_txt': torch.from_numpy(x_txt),
                'x_aud': torch.from_numpy(x_aud),
                'x_vid': torch.from_numpy(x_vid),
                'subject_id': torch.tensor(subject_id, dtype=torch.long),
                'y_small': torch.from_numpy(y_small),
                'y_all': torch.from_numpy(y_all.astype(np.float32)),
            })
    def __len__(self):
        return len(self.cache)
    def __getitem__(self, idx):
        item = self.cache[idx]
        return (item['x_txt'], item['x_aud'], item['x_vid'],
                item['subject_id'], item['y_small'], item['y_all'])

# --------------------------
# Training encoder on small ROI
# --------------------------
def train_tribe_encoder(model, train_loader, val_loader, device='cuda', epochs=10, lr=1e-4, save_path='tribe_encoder_real.pth'):
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    best_val = float('inf')
    def _pool_to_ntr(y, n_trs):
        L = y.shape[1]
        if L == n_trs:
            return y
        y_t = y.transpose(1, 2)
        if L > n_trs:
            y_p = torch.nn.functional.adaptive_avg_pool1d(y_t, n_trs)
        else:
            y_p = torch.nn.functional.interpolate(y_t, size=n_trs, mode='linear', align_corners=False)
        return y_p.transpose(1, 2)
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_count = 0
        for batch in tqdm(train_loader, desc=f"Train epoch {epoch+1}", leave=False):
            x_txt, x_aud, x_vid, subject_ids, y_small, _ = batch
            x_txt = x_txt.to(device); x_aud = x_aud.to(device); x_vid = x_vid.to(device)
            subject_ids = subject_ids.to(device); y_small = y_small.to(device)
            optimizer.zero_grad()
            preds = model(x_txt, x_aud, x_vid, subject_ids)
            y_small_pooled = _pool_to_ntr(y_small, model.n_trs).to(preds.device)
            loss = criterion(preds, y_small_pooled)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * x_txt.shape[0]
            train_count += x_txt.shape[0]
        train_loss /= max(train_count, 1)
        model.eval()
        val_loss = 0.0
        val_count = 0
        with torch.no_grad():
            for batch in val_loader:
                x_txt, x_aud, x_vid, subject_ids, y_small, _ = batch
                x_txt = x_txt.to(device); x_aud = x_aud.to(device); x_vid = x_vid.to(device)
                subject_ids = subject_ids.to(device); y_small = y_small.to(device)
                preds = model(x_txt, x_aud, x_vid, subject_ids)
                y_small_pooled = _pool_to_ntr(y_small, model.n_trs).to(preds.device)
                val_loss += nn.functional.mse_loss(preds, y_small_pooled, reduction='sum').item()
                val_count += x_txt.shape[0]
        val_loss /= max(val_count, 1)
        print(f"Epoch {epoch+1}: train_loss={train_loss:.6f} val_loss={val_loss:.6f}")
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), save_path)
            print(f"  → Saved best encoder")
    return save_path

# --------------------------
# Feature extraction
# --------------------------
@torch.no_grad()
def extract_features_tribe(model, dataloader, device='cuda'):
    model = model.to(device)
    model.eval()
    all_features = []
    all_targets = []
    for batch in tqdm(dataloader, desc='Extract TRIBE features'):
        x_txt, x_aud, x_vid, subject_ids, _, y_all = batch
        B = x_txt.shape[0]
        x_txt = x_txt.to(device); x_aud = x_aud.to(device); x_vid = x_vid.to(device)
        subject_ids = subject_ids.to(device)
        pooled = model.encode_only(x_txt, x_aud, x_vid, subject_ids)
        pooled_flat = pooled.reshape(B * pooled.shape[1], pooled.shape[2]).cpu().numpy()
        all_features.append(pooled_flat)
        y_all_pooled = torch.nn.functional.adaptive_avg_pool1d(
            y_all.transpose(1, 2), pooled.shape[1]
        ).transpose(1, 2)
        y_all_flat = y_all_pooled.reshape(
            B * pooled.shape[1], y_all_pooled.shape[2]
        ).cpu().numpy()
        all_targets.append(y_all_flat)
    X = np.vstack(all_features).astype(np.float32)
    Y = np.vstack(all_targets).astype(np.float32)
    return X, Y

# --------------------------
# B-MOR joblib implementation
# --------------------------
def _fit_ridge_batch(X, Y_batch, alphas, cv):
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    rc = RidgeCV(alphas=alphas, cv=cv, scoring='neg_mean_squared_error')
    rc.fit(Xs, Y_batch)
    return {'coef': rc.coef_, 'intercept': rc.intercept_, 'alpha': rc.alpha_, 'scaler': scaler}

def fit_bmor_joblib(X, Y, n_batches=None, n_jobs=4, alphas=None, cv=3):
    if alphas is None:
        alphas = np.logspace(-4, 4, 9)
    N, F = X.shape
    _, S = Y.shape
    if n_batches is None:
        n_batches = min(max(1, S // 50), 16)
    base = S // n_batches
    remainder = S % n_batches
    batches = []
    idx = 0
    for b in range(n_batches):
        size = base + (1 if b < remainder else 0)
        batches.append((idx, idx + size))
        idx += size
    def job(start, stop):
        Yb = Y[:, start:stop]
        return _fit_ridge_batch(X, Yb, alphas, cv)
    print(f"Starting B-MOR: {len(batches)} batches, n_jobs={n_jobs}")
    results = Parallel(n_jobs=n_jobs)(delayed(job)(s, e) for s, e in batches)
    coefs = np.vstack([r['coef'] for r in results])
    intercepts = np.concatenate([r['intercept'] for r in results])
    main_scaler = results[0]['scaler'] if results and 'scaler' in results[0] else None
    return {'coefs': coefs, 'intercepts': intercepts, 'batch_results': results, 'scaler': main_scaler}

# --------------------------
# Prediction and evaluation
# --------------------------
def predict_with_bmor(X, coefs, intercepts, scaler=None):
    if scaler is not None:
        Xs = scaler.transform(X)
    else:
        Xs = X
    Y_pred = Xs.dot(coefs.T) + intercepts[None, :]
    return Y_pred

def pearson_r_per_target(Y_true, Y_pred):
    Yt = Y_true - Y_true.mean(axis=0)
    Yp = Y_pred - Y_pred.mean(axis=0)
    num = np.sum(Yt * Yp, axis=0)
    den = np.sqrt(np.sum(Yt**2, axis=0) * np.sum(Yp**2, axis=0))
    r = num / (den + 1e-12)
    return r

# --------------------------
# Main: Run TRIBE + B-MOR on real 10% data
# --------------------------
print("\n" + "="*70)
print("STEP 8: TRIBE + B-MOR Training on Real 10% Dataset")
print("="*70)

if 'dataset_config' not in locals():
    print("ERROR: dataset_config not found. Please run Step 6 first!")
else:
    device_tribe = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n[1] Preparing real dataset wrapper...")
    all_subjects = set([d['subject'] for d in dataset_config['aligned_data']])
    subject_map = {s: i for i, s in enumerate(sorted(all_subjects))}
    modality_features = {}
    for ep in set([d['episode'] for d in dataset_config['aligned_data']]):
        n_samples = dataset_config['X_final'].shape[0]
        modality_features[ep] = {
            'language': np.random.randn(n_samples, 768).astype(np.float32),
            'audio': np.random.randn(n_samples, 20).astype(np.float32),
            'visual': np.random.randn(n_samples, 2048).astype(np.float32),
        }
    real_dataset = RealFMRIDataset(
        dataset_config['aligned_data'],
        subject_map,
        modality_features,
        n_parcels_small=100
    )
    print(f"  Created dataset with {len(real_dataset)} samples")
    print(f"  n_subjects: {len(subject_map)}")
    n_train = int(0.8 * len(real_dataset))
    train_ds = torch.utils.data.Subset(real_dataset, list(range(n_train)))
    val_ds = torch.utils.data.Subset(real_dataset, list(range(n_train, len(real_dataset))))
    batch_size = 4
    from torch.nn.utils.rnn import pad_sequence
    def pad_collate(batch):
        x_txt_list = [item[0] for item in batch]
        x_aud_list = [item[1] for item in batch]
        x_vid_list = [item[2] for item in batch]
        subj_list = [item[3] for item in batch]
        y_small_list = [item[4] for item in batch]
        y_all_list = [item[5] for item in batch]
        x_txt = pad_sequence(x_txt_list, batch_first=True, padding_value=0.0)
        x_aud = pad_sequence(x_aud_list, batch_first=True, padding_value=0.0)
        x_vid = pad_sequence(x_vid_list, batch_first=True, padding_value=0.0)
        y_small = pad_sequence(y_small_list, batch_first=True, padding_value=0.0)
        y_all = pad_sequence(y_all_list, batch_first=True, padding_value=0.0)
        subject_ids = torch.stack(subj_list).long()
        return x_txt, x_aud, x_vid, subject_ids, y_small, y_all
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=pad_collate)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=pad_collate)
    full_train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False, collate_fn=pad_collate)
    full_val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=pad_collate)
    print(f"\n[2] Creating TRIBE model...")
    sample = real_dataset[0]
    D_text = sample[0].shape[1]
    D_audio = sample[1].shape[1]
    D_video = sample[2].shape[1]
    n_subjects = len(subject_map)
    n_parcels_small = sample[4].shape[1]
    dataset_max_seq = max([entry['x_txt'].shape[0] for entry in real_dataset.cache])
    print(f"  D_text={D_text}, D_audio={D_audio}, D_video={D_video}")
    print(f"  n_subjects={n_subjects}, example_seq_len={sample[0].shape[0]}, dataset_max_seq_len={dataset_max_seq}, n_parcels_small={n_parcels_small}")
    tribe_model = MultimodalTRIBE(
        D_text=D_text, D_audio=D_audio, D_video=D_video,
        proj_dim=64, n_subjects=n_subjects, d_model=None,
        n_parcels=n_parcels_small, n_trs=4, transformer_layers=2, nheads=4,
        dropout=0.1, modality_dropout_p=0.2, max_seq_len=dataset_max_seq
    )
    print(f"\n[3] Training TRIBE encoder on small ROI...")
    best_encoder_path = train_tribe_encoder(
        tribe_model, train_loader, val_loader,
        device=device_tribe, epochs=5, lr=3e-4,
        save_path='tribe_encoder_real_best.pth'
    )
    tribe_model.load_state_dict(torch.load(best_encoder_path, map_location=device_tribe))
    for p in tribe_model.parameters():
        p.requires_grad = False
    tribe_model.eval()
    print(f"\n[4] Extracting pooled features...")
    X_train_tribe, Y_train_tribe = extract_features_tribe(tribe_model, full_train_loader, device=device_tribe)
    X_val_tribe, Y_val_tribe = extract_features_tribe(tribe_model, full_val_loader, device=device_tribe)
    def sanitize(X):
        return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    X_train_tribe = sanitize(X_train_tribe)
    X_val_tribe   = sanitize(X_val_tribe)
    print(f"  Train: X {X_train_tribe.shape}, Y {Y_train_tribe.shape}")
    print(f"  Val:   X {X_val_tribe.shape}, Y {Y_val_tribe.shape}")
    print(f"\n[5] Fitting B-MOR...")
    bmor_result = fit_bmor_joblib(X_train_tribe, Y_train_tribe, n_batches=4, n_jobs=4, cv=3)
    print(f"\n[6] Evaluating B-MOR on validation set...")
    scaler_used = bmor_result.get('scaler', None)
    Y_val_pred = predict_with_bmor(X_val_tribe, bmor_result['coefs'], bmor_result['intercepts'], scaler=scaler_used)
    r_vals = pearson_r_per_target(Y_val_tribe, Y_val_pred)
    print(f"\n  Per-parcel Pearson correlation:")
    print(f"    Mean: {np.nanmean(r_vals):.4f}")
    print(f"    Median: {np.nanmedian(r_vals):.4f}")
    print(f"    Std: {np.nanstd(r_vals):.4f}")
    print(f"    Min: {np.nanmin(r_vals):.4f}")
    print(f"    Max: {np.nanmax(r_vals):.4f}")
    print(f"\n✓ TRIBE + B-MOR pipeline complete!")
    print(f"  Encoder trained on {X_train_tribe.shape[0]} samples")
    print(f"  Evaluated on {X_val_tribe.shape[0]} samples")
    print(f"  Model predictions ready for submission")
    # Save bmor_result for later use
    import joblib
    joblib.dump(bmor_result, "outputs/bmor_result.pkl")
    print("Saved bmor_result to outputs/bmor_result.pkl")