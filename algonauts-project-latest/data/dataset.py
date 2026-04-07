import os
import glob
import h5py
import torch
import numpy as np
from torch.utils.data import Dataset
from ..utils.config import MOVIE10_SEQUENCE

def build_fmri_index(fmri_root):
    fmri_index = {}
    h5_files = glob.glob(os.path.join(fmri_root, "sub-*", "func", "*.h5"))
    for h5_path in h5_files:
        fname = os.path.basename(h5_path)
        subject = fname.split("_")[0]
        task_parts = fname.split("task-")
        if len(task_parts) > 1:
            task = task_parts[1].split("_")[0]
            fmri_index.setdefault(subject, {})[task] = h5_path
    return fmri_index

def episode_to_task(episode: str) -> str:
    try:
        # Expecting s01e01a format
        season = int(episode[1:3])
        return "friends" if season <= 5 else "movie10"
    except:
        return "movie10"

def episode_to_movie_clip(entry):
    idx = entry["movie10_index"]
    clip = MOVIE10_SEQUENCE[idx]
    if clip.startswith("bourne"): movie = "bourne"
    elif clip.startswith("wolf"): movie = "wolf"
    elif clip.startswith("life"): movie = "life"
    elif clip.startswith("figures"): movie = "figures"
    else: raise ValueError(f"Unknown movie clip: {clip}")
    return movie, clip

class RealFMRIDatasetPooled(Dataset):
    def __init__(self, aligned_data, subject_map, feature_root, fmri_index, n_trs=4, n_parcels_small=100):
        self.aligned_data = aligned_data
        self.subject_map = subject_map
        self.feature_root = feature_root
        self.fmri_index = fmri_index
        self.n_trs = n_trs
        self.n_parcels_small = n_parcels_small
        self.fmri_handles = {}

    def __len__(self): return len(self.aligned_data)

    def _pool(self, x, n_trs):
        if not hasattr(x, "shape") or len(x.shape) < 2:
            return np.zeros((n_trs, x.shape[-1] if hasattr(x, "shape") else 1), dtype=np.float32)
        T = x.shape[0]
        boundaries = np.linspace(0, T, n_trs + 1, dtype=int)
        pooled = [x[boundaries[i]:boundaries[i+1]].mean(axis=0) if boundaries[i+1] > boundaries[i]
                  else np.zeros(x.shape[1]) for i in range(n_trs)]
        return np.stack(pooled).astype(np.float32)

    def __getitem__(self, idx):
        entry = self.aligned_data[idx]
        subject, episode = entry["subject"], entry["episode"]
        task = episode_to_task(episode)
        h5_path = self.fmri_index[subject][task]
        if h5_path not in self.fmri_handles: self.fmri_handles[h5_path] = h5py.File(h5_path, "r")
        h5_file = self.fmri_handles[h5_path]
        
        # Matching key logic
        matching_keys = [k for k in h5_file.keys() if k.endswith(episode)]
        if not matching_keys and task == "movie10":
            _, clip = episode_to_movie_clip(entry)
            matching_keys = [k for k in h5_file.keys() if clip in k]
            
        if not matching_keys: raise KeyError(f"Episode {episode} not found in {h5_path}")
        
        y_raw = h5_file[matching_keys[0]][:]
        feat_path = os.path.join(self.feature_root, f"{episode}_features.npz")
        feat = np.load(feat_path, allow_pickle=True)
        
        def unwrap(f, k, dim):
            if k not in f: return np.zeros((y_raw.shape[0], dim))
            arr = f[k].item() if f[k].shape == () else f[k]
            return arr
            
        return (torch.from_numpy(self._pool(unwrap(feat, "language", 768), self.n_trs)),
                torch.from_numpy(self._pool(unwrap(feat, "audio", 20), self.n_trs)),
                torch.from_numpy(self._pool(unwrap(feat, "visual", 2048), self.n_trs)),
                torch.tensor(self.subject_map[subject], dtype=torch.long),
                torch.from_numpy(self._pool(y_raw[:, :self.n_parcels_small], self.n_trs)),
                torch.from_numpy(self._pool(y_raw, self.n_trs)))

def collate_fn_pad_sequences(batch):
    x_txts, x_auds, x_vids, subject_ids, y_smalls, y_alls = zip(*batch)
    max_seq_len = max(x.shape[0] for x in x_txts)

    def pad_to_len(x, target_len):
        if x.shape[0] == target_len: return x
        padding = target_len - x.shape[0]
        return torch.nn.functional.pad(x, (0, 0, 0, padding), mode='constant', value=0.0)

    return (torch.stack([pad_to_len(x, max_seq_len) for x in x_txts]),
            torch.stack([pad_to_len(x, max_seq_len) for x in x_auds]),
            torch.stack([pad_to_len(x, max_seq_len) for x in x_vids]),
            torch.stack(subject_ids),
            torch.stack([pad_to_len(y, max_seq_len) for y in y_smalls]),
            torch.stack([pad_to_len(y, max_seq_len) for y in y_alls]))

def load_fmri_for_subject_episode(subject, episode_info, fmri_dir, root_data_dir):
    episode = episode_info['episode']
    h5_files = glob.glob(os.path.join(fmri_dir, "*.h5"))
    if not h5_files: return None
    try:
        with h5py.File(h5_files[0], 'r') as f:
            matching_key = next((k for k in f.keys() if episode in k), None)
            return f[matching_key][()] if matching_key else None
    except: return None
