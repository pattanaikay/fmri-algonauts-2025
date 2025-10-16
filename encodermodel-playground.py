"""
This script implements a multimodal transformer model (TRIBE) for fMRI prediction.
The model combines text, audio, and video features to predict brain activity patterns
across different subjects and brain parcels.

The implementation includes:
- MultimodalTRIBE: Main transformer-based model architecture
- ToyDataset: Synthetic dataset for testing and development
- Training loop with learning rate scheduling and TensorBoard logging
- Grid search functionality for hyperparameter optimization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
import math
import logging
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import LambdaLR

# Set up logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

import itertools
import time
import os
import json
from torch.utils.tensorboard import SummaryWriter

def make_run_name(cfg):
    """
    Generate a unique experiment name based on key hyperparameters and timestamp.
    
    Args:
        cfg (dict): Configuration dictionary containing model and training parameters
        
    Returns:
        str: A string combining key parameter values and timestamp for unique experiment identification
        
    Example:
        cfg = {"proj_dim": 128, "lr": 0.001}
        make_run_name(cfg) -> "p128_l0.001_1234"
    """
    parts = []
    for k in ["proj_dim","transformer_layers","nheads","ff_dim","dropout","modality_dropout_p","lr","warmup_steps"]:
        if k in cfg:
            parts.append(f"{k[:1]}{cfg[k]}")
    t = int(time.time()%10000)
    return "_".join(parts) + f"_{t}"

def log_experiment(cfg, description=""):
    """
    Log experiment configuration and details to a centralized log file.
    
    Args:
        cfg (dict): Configuration dictionary
        description (str): Description of the experiment
    """
    log_file = "experiment_tracking.jsonl"
    
    # Create log entry
    log_entry = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "run_name": cfg["run_name"],
        "description": description,
        "key_params": {
            "ff_dim": cfg.get("ff_dim"),
            "proj_dim": cfg.get("proj_dim"),
            "transformer_layers": cfg.get("transformer_layers"),
            "nheads": cfg.get("nheads"),
            "dropout": cfg.get("dropout"),
            "modality_dropout_p": cfg.get("modality_dropout_p"),
            "lr": cfg.get("lr"),
            "batch_size": cfg.get("batch_size"),
            "n_epochs": cfg.get("n_epochs")
        },
        "full_config": cfg
    }
    
    # Append to log file
    with open(log_file, "a") as f:
        f.write(json.dumps(log_entry) + "\n")

def run_grid_search(base_cfg, grid, run_single_experiment, seed=42, description=""):
    """
    Perform grid search over hyperparameters by running multiple experiments.
    
    Args:
        base_cfg (dict): Base configuration with default parameters
        grid (dict): Dictionary mapping parameter names to lists of values to try
        run_single_experiment (callable): Function that runs one experiment with given config
        seed (int): Random seed for reproducibility
        description (str): Description of this experiment series
        
    Example:
        grid = {
            "lr": [0.1, 0.01],
            "dropout": [0.1, 0.2]
        }
        run_grid_search(base_cfg, grid, run_experiment, description="Testing different learning rates")
    """
    keys, vals = zip(*grid.items())
    for v in itertools.product(*vals):
        cfg = base_cfg.copy()
        cfg.update(dict(zip(keys, v)))
        cfg["run_name"] = make_run_name(cfg)
        cfg["log_dir"] = os.path.join("runs", cfg["run_name"])
        os.makedirs(cfg["log_dir"], exist_ok=True)

        # write config for reproducibility
        with open(os.path.join(cfg["log_dir"], "config.json"), "w") as f:
            json.dump(cfg, f, indent=2)

        # Log experiment before running
        log_experiment(cfg, description)
        
        logger.info(f"Starting run: {cfg['run_name']}")
        logger.info(f"Configuration: {cfg}")
        
        run_single_experiment(cfg, seed=seed)
        
        logger.info(f"Finished run: {cfg['run_name']}")

# ---------- Model ----------
class MultimodalTRIBE(nn.Module):
    """
    Multimodal Transformer for Brain Encoding (TRIBE) model.
    
    This model combines text, audio, and video features through:
    1. Modality-specific projections to a common dimension
    2. Transformer-based fusion of multimodal features
    3. Temporal pooling and subject-specific readout
    
    Args:
        D_text (int): Dimension of text features
        D_audio (int): Dimension of audio features
        D_video (int): Dimension of video features
        proj_dim (int): Projection dimension for each modality
        n_subjects (int): Number of distinct subjects
        d_model (int, optional): Transformer model dimension (default: 3 * proj_dim)
        n_parcels (int): Number of brain parcels to predict
        n_trs (int): Number of time points in fMRI data
        max_seq_len (int): Maximum sequence length for positional embeddings
        transformer_layers (int): Number of transformer encoder layers
        nheads (int): Number of attention heads
        ff_dim (int): Feed-forward network dimension
        dropout (float): Dropout probability in transformer
        modality_dropout_p (float): Probability of dropping each modality during training
    """
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

        # Use batch_first=True so inputs are expected as (batch, seq, feature)
        # This also avoids a runtime warning and can improve inference performance.
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
        """
        Apply modality dropout during training to encourage robust multimodal fusion.
        
        Randomly zeros out entire modalities (text, audio, or video) with probability
        modality_dropout_p. Ensures at least one modality remains active.
        
        Args:
            x_txt (Tensor): Text features [B, T, D_text]
            x_aud (Tensor): Audio features [B, T, D_audio]
            x_vid (Tensor): Video features [B, T, D_video]
            
        Returns:
            tuple: Modified (text, audio, video) features with dropout applied
        """
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
        """
        Forward pass of the TRIBE model.
        
        Process steps:
        1. Apply modality dropout
        2. Project each modality to common dimension
        3. Concatenate modalities and add positional + subject embeddings
        4. Pass through transformer encoder
        5. Pool temporally to match fMRI timepoints
        6. Apply subject-specific readout to predict brain activity
        
        Args:
            x_txt (Tensor): Text features [B, T, D_text]
            x_aud (Tensor): Audio features [B, T, D_audio]
            x_vid (Tensor): Video features [B, T, D_video]
            subject_ids (Tensor): Subject identifiers [B]
            
        Returns:
            Tensor: Predicted brain activity [B, n_trs, n_parcels] 
        """
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
    """
    Synthetic dataset for testing the TRIBE model.
    
    Generates random features for text, audio, and video modalities,
    along with synthetic fMRI data for testing the model pipeline.
    
    Args:
        n_samples (int): Number of samples in dataset
        fT (int): Number of timepoints in feature sequences
        D_text (int): Dimension of text features
        D_audio (int): Dimension of audio features
        D_video (int): Dimension of video features
        n_trs (int): Number of fMRI timepoints
        n_parcels (int): Number of brain parcels
        n_subjects (int): Number of distinct subjects
    """
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

# ---------- Utility ----------
def pearson_corr(preds, targets, eps=1e-8):
    """
    Calculate Pearson correlation coefficient between predictions and targets.
    
    Computes correlation along first dimension (usually time),
    then averages across remaining dimensions (usually brain parcels).
    
    Args:
        preds (Tensor): Model predictions
        targets (Tensor): Ground truth values
        eps (float): Small constant for numerical stability
        
    Returns:
        float: Mean Pearson correlation coefficient
    """
    pred = preds - preds.mean(0)
    targ = targets - targets.mean(0)
    num = (pred * targ).sum(0)
    den = torch.sqrt((pred**2).sum(0) * (targ**2).sum(0)).clamp(min=eps)
    return (num / den).mean().item()

# ---------- Training Setup ----------
logger.info("Setting up training environment")
device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using device: {device}")

dataset = ToyDataset()
loader = DataLoader(dataset, batch_size=8, shuffle=True)
logger.info(f"Dataset size: {len(dataset)}, Batch size: 8")

model = MultimodalTRIBE(D_text=300, D_audio=64, D_video=128,
                        proj_dim=128, n_subjects=5,
                        n_parcels=50, n_trs=20).to(device)
logger.info("Model initialized and moved to device")

optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
logger.info("Optimizer configured")

# Learning rate scheduler with linear warmup followed by cosine decay
def lr_lambda(current_step, warmup_steps=50, total_steps=300):
    """
    Compute learning rate multiplier based on warmup and cosine decay schedule.
    
    The schedule has two phases:
    1. Linear warmup: LR increases linearly from 0 to base_lr
    2. Cosine decay: LR decreases following a cosine curve from base_lr to 0
    
    Args:
        current_step (int): Current training step
        warmup_steps (int): Number of warmup steps
        total_steps (int): Total number of training steps
        
    Returns:
        float: Learning rate multiplier between 0 and 1
    """
    # Linear warmup phase
    if current_step < warmup_steps:
        return float(current_step) / float(max(1, warmup_steps))
    
    # Cosine decay phase
    progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
scheduler = LambdaLR(optimizer, lr_lambda=lambda step: lr_lambda(step, 50, 300))

# TensorBoard writer
writer = SummaryWriter(log_dir="runs/toy_multimodal")
logger.info("TensorBoard writer initialized")

# ---------- Train + Validate ----------
def run_single_experiment(cfg, seed=42):
    """
    Run a single training experiment with given configuration.
    
    Handles:
    1. Setting random seeds for reproducibility
    2. Creating dataset and dataloader
    3. Initializing model, optimizer, and scheduler
    4. Training loop with validation
    5. Logging metrics to TensorBoard
    
    Args:
        cfg (dict): Configuration dictionary with model and training parameters
        seed (int): Random seed for reproducibility
        
    Returns:
        None (results are logged to TensorBoard)
    """
    import random, numpy as np, torch
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # build dataset / loader
    dataset = ToyDataset(n_samples=cfg["n_samples"], fT=cfg["fT"],
                         D_text=cfg["D_text"], D_audio=cfg["D_audio"], D_video=cfg["D_video"],
                         n_trs=cfg["n_trs"], n_parcels=cfg["n_parcels"], n_subjects=cfg["n_subjects"])
    loader = DataLoader(dataset, batch_size=cfg["batch_size"], shuffle=True, num_workers=2, pin_memory=True)

    # model, optimizer, scheduler
    model = MultimodalTRIBE(D_text=cfg["D_text"], D_audio=cfg["D_audio"], D_video=cfg["D_video"],
                            proj_dim=cfg["proj_dim"], n_subjects=cfg["n_subjects"],
                            n_parcels=cfg["n_parcels"], n_trs=cfg["n_trs"],
                            transformer_layers=cfg["transformer_layers"],
                            nheads=cfg["nheads"], ff_dim=cfg["ff_dim"],
                            dropout=cfg["dropout"], modality_dropout_p=cfg["modality_dropout_p"]).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg.get("weight_decay",1e-2))

    # LR schedule (warmup + cosine)
    total_steps = max(1, len(loader) * cfg["n_epochs"])
    def lr_lambda(step, warmup=cfg["warmup_steps"], total=total_steps):
        if step < warmup: return float(step)/float(max(1,warmup))
        progress = float(step - warmup) / float(max(1, total - warmup))
        return max(0.0, 0.5*(1.0 + math.cos(math.pi*progress)))
    scheduler = LambdaLR(optimizer, lr_lambda=lambda s: lr_lambda(s))

    writer = SummaryWriter(log_dir=cfg["log_dir"])

    global_step = 0
    for epoch in range(cfg["n_epochs"]):
        model.train()
        epoch_loss = 0.0
        for x_txt, x_aud, x_vid, subj, y in loader:
            x_txt,x_aud,x_vid,y = x_txt.to(device), x_aud.to(device), x_vid.to(device), y.to(device)
            subj = subj.to(device)
            preds = model(x_txt,x_aud,x_vid,subj)
            loss = F.mse_loss(preds, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            writer.add_scalar("Loss/train_step", loss.item(), global_step)
            writer.add_scalar("LR", scheduler.get_last_lr()[0], global_step)
            epoch_loss += loss.item()
            global_step += 1

        avg = epoch_loss / len(loader)
        writer.add_scalar("Loss/train_epoch", avg, epoch)

        # quick validation on small subset or entire dataset
        model.eval()
        with torch.no_grad():
            # here you can run on a validation loader; for simplicity run on first K samples
            vals = []
            for i in range(min(10, len(dataset))):
                x_txt,x_aud,x_vid,subj,y = dataset[i]
                x_txt,x_aud,x_vid,y = x_txt.unsqueeze(0).to(device), x_aud.unsqueeze(0).to(device), x_vid.unsqueeze(0).to(device), y.unsqueeze(0).to(device)
                subj = torch.tensor([subj], device=device)
                pred = model(x_txt,x_aud,x_vid,subj).squeeze(0).cpu()
                vals.append(pearson_corr(pred, y.squeeze(0).cpu()))
            val_mean = float(np.mean(vals))
        writer.add_scalar("Val/Pearson", val_mean, epoch)
        print(f"Run {cfg['run_name']} Epoch {epoch} avg_loss={avg:.4f} val_pearson={val_mean:.4f}")

    writer.close()

base_cfg = {
  "n_samples":200, "fT":60, "D_text":300, "D_audio":64, "D_video":128,
  "n_trs":20, "n_parcels":50, "n_subjects":5,
  "batch_size":8, "n_epochs":5,
  "proj_dim":128, "transformer_layers":2, "nheads":8, "ff_dim":2048,
  "dropout":0.3, "modality_dropout_p":0.2,
  "lr":1e-3, "warmup_steps":50
}
grid = {
  "proj_dim":[128,256],
  "transformer_layers":[2,4],
  "modality_dropout_p":[0.0,0.2]
}
if __name__ == '__main__':
    # On Windows the 'spawn' start method is used by default. Protect the
    # entry point of the program so child processes don't re-import and
    # re-execute top-level code. Calling freeze_support() is recommended
    # when freezing to an executable; it's safe to call regardless.
    try:
        # multiprocessing.freeze_support will be a no-op on non-frozen apps,
        # but is useful on Windows for some setups.
        import multiprocessing as _mp
        _mp.freeze_support()
    except Exception:
        pass

    run_grid_search(base_cfg, grid, run_single_experiment, description="Testing with ff_dim:2048, n_epochs:5")

    logger.info("Training completed")

    # Close any global writers if present
    try:
        writer.close()
    except Exception:
        pass
 
