import numpy as np
from sklearn.linear_model import RidgeCV, Ridge
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
from scipy.linalg import svd
import torch
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader, Subset
from dask.distributed import Client, LocalCluster
import math
import os

# Utility: Pearson mean across columns
def pearson_mean_across_targets(Y_true, Y_pred):
    corrs = []
    for j in range(Y_true.shape[1]):
        a = Y_true[:, j]
        b = Y_pred[:, j]
        if np.std(a)== 0 or np.std(b)==0:
            corrs.append(0.0)
        else:
            corrs.append(pearsonr(a, b)[0])

    return np.nanmean(corrs)

# 1) VGG16 feature extraction (FC2)

def extract_vgg16_fc2_features(images_tensor, device='cpu', batch_size=32):
    # Load pre-trained VGG16 and remove final classification layer to get FC2 outputs
    vgg = models.vgg16(pretrained=True).to(device)
    vgg.eval()

    fc2_acts = []

    def hook_fn(module, input, output):
        # output shape(batch, 4096)
        fc2_acts.append(output.detach().cpu().numpy())

    handle = vgg.classifier[3].register_forward_hook(hook_fn)

    d1 = DataLoader(images_tensor, batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for xb in d1:
            xb = xb.to(device)
            _ = vgg(xb)
    
    handle.remove()
    feats = np.vstack(fc2_acts)
    return feats

# 2) Create toy dataset (use CIFAR10 small subset)
def make_toy_XY(n_samples=200, n_targets=50, concat_frames=1):
    
    # Use torchvision CIFAR10 dataset as image source (224x224 required -> we upsample)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    
    # Download small subset of CIFAR10
    cifar10 = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    subset = Subset(cifar10, list(range(min(len(cifar10), n_samples*concat_frames))))
    loader = DataLoader(subset, batch_size=len(subset))

    # Turn into one big tensor
    images_all, _=next(iter(loader)) # shape(N, 3, 224, 224)

    # If concat_frames>! we'd arrange sequences; for toy key concat_frames=1
    X_feats = extract_vgg16_fc2_features(images_all)

    # Optionally concatenate multiples frames 
    p = X_feats.shape[1]

    # Create synthetic linear targets: choose a random true B_true and make Y=X B_true + noise 
    rng = np.random.RandomState(0)
    B_true = rng.randn(p, n_targets) * 0.1
    Y = X_feats @ B_true + 0.05 * rng.randn(X_feats[0], n_targets)
    return X_feats, Y

# 3) SVD-based ridge (amortized over alphas)

def ridge_svd_fit_predict(X, Y, alphas, cv=5):
    n, p = X.shape
    _, s = Y.shape

    U, S, Vt = svd(X, full_matrices=False)
    V = Vt.T
    Ut = U.T
    k = s.shape[0]    

    # Prepare CV splits 
    kf = KFold(n_splits=cv, shuffle=True, random_state=0)

    mean_scores = []
    for lam in alphas:
        d = S / (S**2 + lam)
        cv_scores = []
        for train_ix, val_ix in kf.split(X):
            Utr = U[train_ix]
            Ytr = Y[train_ix]
            Xval = X[val_ix]

            M = V@(d[:, None]* Ut)
            B = M@Ytr
            Yhat_val = Xval @ B
            score = pearson_mean_across_targets(Y[val_ix], Yhat_val)
            cv_scores.append(score)
        
        mean_scores.append(np.mean(cv_scores))
    best_idx = int(np.argmax(mean_scores))
    best_alpha = alphas[best_idx]

    d = S / (S**2 + best_alpha)
    M = V @ (d[:, None] * Ut)
    B_final = M@Y
    Yhat = X @ B_final
    return best_alpha, B_final, Yhat

# 4) sklearn RidgeCV baseline

def sklearn_ridgecv_fit_predict(X, Y, alphas, cv=5):
    model = RidgeCV(alphas=alphas, cv=cv, store_cv_values=False)
    model.fit(X, Y)
    B = model.coef_.T
    Yhat = model.predict(X)
    return model.alpha_, B, Yhat