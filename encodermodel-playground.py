import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from torch.utils.data import Dataset, DataLoader

# ---------- Model (from your code) ----------
class MultimodalTRIBE(nn.Module):
    def __init__(self,
                 D_text, D_audio, D_video,
                 proj_dim=128,          # smaller for toy test
                 n_subjects=5,
                 d_model=None,
                 n_parcels=50,          # fewer parcels for toy
                 n_trs=20,              # fewer TRs per window
                 max_seq_len=60,
                 transformer_layers=2,
                 nheads=4,
                 ff_dim=512,
                 dropout=0.1,
                 modality_dropout_p=0.2):
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
            dropout=dropout, activation='gelu'
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
        mask_txt = torch.bernoulli((1 - self.modality_dropout_p) *
                                   torch.ones(B,1,1,device=x_txt.device))
        mask_aud = torch.bernoulli((1 - self.modality_dropout_p) *
                                   torch.ones(B,1,1,device=x_aud.device))
        mask_vid = torch.bernoulli((1 - self.modality_dropout_p) *
                                   torch.ones(B,1,1,device=x_vid.device))
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

        x_t = x.transpose(0,1)
        x_out = self.transformer(x_t)
        x_out = x_out.transpose(0,1)

        x_perm = x_out.transpose(1,2)
        pooled = self.pool(x_perm).transpose(1,2)

        preds = self.readout(pooled)
        preds = preds + self.subj_bias(subject_ids).unsqueeze(1)
        return preds

# ---------- Synthetic Dataset ----------
class ToyDataset(Dataset):
    def __init__(self, n_samples=200, fT=60, D_text=300, D_audio=64, D_video=128,
                 n_trs=20, n_parcels=50, n_subjects=5):
        self.n_samples = n_samples
        self.fT = fT
        self.D_text, self.D_audio, self.D_video = D_text, D_audio, D_video
        self.n_trs, self.n_parcels, self.n_subjects = n_trs, n_parcels, n_subjects

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        x_txt = torch.randn(self.fT, self.D_text)
        x_aud = torch.randn(self.fT, self.D_audio)
        x_vid = torch.randn(self.fT, self.D_video)
        subj = torch.randint(0, self.n_subjects, (1,)).item()
        # ground truth: synthetic random target
        y = torch.randn(self.n_trs, self.n_parcels)
        return x_txt, x_aud, x_vid, subj, y

# ---------- Training Loop ----------
def pearson_corr(preds, targets, eps=1e-8):
    pred = preds - preds.mean(0)
    targ = targets - targets.mean(0)
    num = (pred * targ).sum(0)
    den = torch.sqrt((pred**2).sum(0) * (targ**2).sum(0)).clamp(min=eps)
    return (num / den).mean().item()

# Settings
device = "cuda" if torch.cuda.is_available() else "cpu"
dataset = ToyDataset()
loader = DataLoader(dataset, batch_size=8, shuffle=True)

model = MultimodalTRIBE(D_text=300, D_audio=64, D_video=128,
                        proj_dim=128, n_subjects=5,
                        n_parcels=50, n_trs=20).to(device)
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)

# Train
for epoch in range(3):  # just a few epochs for test
    model.train()
    total_loss = 0
    for x_txt, x_aud, x_vid, subj, y in loader:
        x_txt, x_aud, x_vid, y = x_txt.to(device), x_aud.to(device), x_vid.to(device), y.to(device)
        subj = subj.to(device)
        preds = model(x_txt, x_aud, x_vid, subj)
        loss = F.mse_loss(preds, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch} | Loss {total_loss/len(loader):.4f}")

# Validation sanity check
model.eval()
x_txt, x_aud, x_vid, subj, y = dataset[0]
x_txt, x_aud, x_vid, y = x_txt.unsqueeze(0).to(device), x_aud.unsqueeze(0).to(device), x_vid.unsqueeze(0).to(device), y.unsqueeze(0).to(device)
subj = torch.tensor([subj], device=device)
preds = model(x_txt, x_aud, x_vid, subj).squeeze(0).detach().cpu()
print("Pearson correlation (toy):", pearson_corr(preds, y.squeeze(0).cpu()))
