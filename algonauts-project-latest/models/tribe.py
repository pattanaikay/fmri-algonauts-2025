import torch
import torch.nn as nn
import random

class MultimodalTRIBE(nn.Module):
    def __init__(self, D_text, D_audio, D_video, proj_dim=64, n_subjects=5,
                 n_parcels=100, n_trs=4, transformer_layers=2, nheads=4,
                 ff_dim=512, dropout=0.1, modality_dropout_p=0.2, max_seq_len=300):
        super().__init__()
        d_model = 3 * proj_dim
        self.n_trs = n_trs
        self.txt_proj = nn.Sequential(nn.Linear(D_text, proj_dim), nn.LayerNorm(proj_dim))
        self.aud_proj = nn.Sequential(nn.Linear(D_audio, proj_dim), nn.LayerNorm(proj_dim))
        self.vid_proj = nn.Sequential(nn.Linear(D_video, proj_dim), nn.LayerNorm(proj_dim))
        self.pos_emb = nn.Parameter(torch.randn(1, max_seq_len, d_model) * 0.02)
        self.subj_emb = nn.Embedding(n_subjects, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nheads, dim_feedforward=ff_dim,
            dropout=dropout, activation="gelu", batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)
        self.pool = nn.AdaptiveAvgPool1d(n_trs)
        self.readout = nn.Linear(d_model, n_parcels)
        self.subj_bias = nn.Embedding(n_subjects, n_parcels)
        self.modality_dropout_p = modality_dropout_p

    def modality_dropout(self, x_txt, x_aud, x_vid):
        if not self.training or self.modality_dropout_p <= 0.0:
            return x_txt, x_aud, x_vid
        B = x_txt.shape[0]
        masks = [torch.bernoulli((1 - self.modality_dropout_p) * torch.ones(B, 1, 1, device=x_txt.device)) for _ in range(3)]
        for i in range(B):
            if sum(m[i].sum() for m in masks) == 0:
                masks[random.randint(0, 2)][i] = 1
        return x_txt * masks[0], x_aud * masks[1], x_vid * masks[2]

    def forward(self, x_txt, x_aud, x_vid, subject_ids):
        x_txt, x_aud, x_vid = self.modality_dropout(x_txt, x_aud, x_vid)
        x = torch.cat([self.txt_proj(x_txt), self.aud_proj(x_aud), self.vid_proj(x_vid)], dim=-1)
        x = x + self.pos_emb[:, :x.shape[1]] + self.subj_emb(subject_ids).unsqueeze(1)
        h = self.transformer(x)
        h_pooled = self.pool(h.transpose(1, 2)).transpose(1, 2)
        return self.readout(h_pooled) + self.subj_bias(subject_ids).unsqueeze(1)

    @torch.no_grad()
    def encode_only(self, x_txt, x_aud, x_vid, subject_ids):
        self.eval()
        x = torch.cat([self.txt_proj(x_txt), self.aud_proj(x_aud), self.vid_proj(x_vid)], dim=-1)
        x = x + self.pos_emb[:, :x.shape[1]] + self.subj_emb(subject_ids).unsqueeze(1)
        h = self.transformer(x)
        return self.pool(h.transpose(1, 2)).transpose(1, 2)
