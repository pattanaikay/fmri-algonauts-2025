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
    vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).to(device)
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
        transforms.ToTensor(),  # Convert PIL Image to tensor first
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
    # Use X_feats.shape[0] to get number of samples for noise dimension
    Y = X_feats @ B_true + 0.05 * rng.randn(X_feats.shape[0], n_targets)
    return X_feats, Y

# 3) SVD-based ridge (amortized over alphas)

def ridge_svd_fit_predict(X, Y, alphas, cv=5):
    n, p = X.shape
    _, n_targets = Y.shape

    # Compute SVD of X once
    U, S, Vt = svd(X, full_matrices=False)
    V = Vt.T
    k = S.shape[0]

    # Prepare CV splits 
    kf = KFold(n_splits=cv, shuffle=True, random_state=0)
    mean_scores = []

    for lam in alphas:
        d = S / (S**2 + lam)
        cv_scores = []
        for train_ix, val_ix in kf.split(X):
            # Get training and validation sets
            Utr = U[train_ix]
            Ytr = Y[train_ix]
            Xval = X[val_ix]
            
            # Compute U.T @ Y for training data only
            UtYtr = Utr.T @ Ytr
            
            # Compute ridge solution: V @ diag(d) @ U.T @ Y
            B = V @ (d[:, None] * UtYtr)
            
            # Predict on validation set
            Yhat_val = Xval @ B
            score = pearson_mean_across_targets(Y[val_ix], Yhat_val)
            cv_scores.append(score)
        
        mean_scores.append(np.mean(cv_scores))
    
    # Use best alpha for final fit
    best_idx = int(np.argmax(mean_scores))
    best_alpha = alphas[best_idx]

    # Final solution with best alpha
    d = S / (S**2 + best_alpha)
    UtY = U.T @ Y
    B_final = V @ (d[:, None] * UtY)
    Yhat = X @ B_final
    
    return best_alpha, B_final, Yhat

# 4) sklearn RidgeCV baseline

def sklearn_ridgecv_fit_predict(X, Y, alphas, cv=5):
    # Initialize RidgeCV with alphas and cross-validation folds
    model = RidgeCV(alphas=alphas, cv=cv)
    model.fit(X, Y)
    # Get coefficients (transpose to match our convention)
    B = model.coef_.T
    Yhat = model.predict(X)
    return model.alpha_, B, Yhat

# 5) MOR vs B-MOR toy with local Dask
def mor_ridge_dask(X, Y, alphas, cv=3, n_workers=2, threads_per_worker=1):
    # Starting local cluster

    cluster = LocalCluster(n_workers=n_workers, threads_per_worker=threads_per_worker, processes=True)
    client = Client(cluster)

    import dask
    from dask import delayed, compute

    n_samples, p = X.shape
    _, s = Y.shape

    # MOR: submit one job per target (deplayed)
    def fit_target(x, y_col, alphas_local, cv_local):
        # small helper to fit RidgeCV on vector y_col
        model = RidgeCV(alphas=alphas, cv=cv_local, store_cv_values=False)
        model.fit(x, y_col)

        # sklearn RidgeCV with 1d y -> coef_ shape (n_features,)
        w = model.coef_.reshape(-1, 1)
        return float(model.alpha_), w

    mor_jobs = [delayed(fit_target)(X, Y[:,j], alphas, cv) for j in range(s)]
    mor_results = compute(*mor_jobs, schedule=client)[0:s]

    # assemble B and alphas
    B_mor = np.hstack([res[1] for res in mor_results])
    alphas_mor = [res[0] for res in mor_results]
    Yhat_mor = X @ B_mor

    # B-MOR: split Y into batches (c=n_workers)
    batch_size = math.cell(s / n_workers)
    def fit_batch(batch_Y):
        # fits RidgeCV to multioutput (batch) Y and returns alpha (single) and Bbatch
        model = RidgeCV(alphas=alphas, cv=cv, store_cv_values=False)
        model.fit(X, batch_Y)
        Bbatch = model.coef_.T
        return float(model.alpha_), Bbatch
    
    batched = []
    for i in range(n_workers):
        start = i * batch_size
        end = min((i+1)* batch_size, s)
        if start >= end:
            continue
        Yi = Y[:, start:end]
        batched.append(delayed(fit_batch)(Yi))

    bmor_results = compute(*batched, scheduler=client)
    # assemble B_bmor
    B_bmor = np.zeros((p, s))
    alphas_bmor = []
    for i, (alpha_i, Bbatch) in enumerate(bmor_results):
        start = i * batch_size
        end = min((i+1)*batch_size, s)
        B_bmor[:, start:end] = Bbatch
        alphas_bmor.append(alpha_i)
    Yhat_bmor = X @ B_bmor

    client.close()
    cluster.close()

    return {
        'mor':{'alpha_list':alphas_mor, 'B':B_mor, 'Yhat':Yhat_mor},
        'bmor':{'alpha_list':alphas_bmor, 'B':B_bmor, 'Yhat':Yhat_bmor}
    }

# Main demo

if __name__ == "__main__":
    print("Making toy data and extracting VGG16 features (this will download torchvision weights if needed)")
    X, Y = make_toy_XY(n_samples=200, n_targets=40)
    print("Shapes: X", X.shape, "Y", Y.shape)

    # train/test split (90/10 like the thesis)
    n = X.shape[0]
    tr = int(0.9*n)
    X_train, X_test = X[:tr], X[tr:]
    Y_train, Y_test = Y[:tr], Y[tr:]

    alphas = [0.1, 1.0, 10.0, 100.0, 1000.0]

    print("\n=== SVD-based ridge (amortized over alphas) ===")
    best_alpha_svd, B_svd, Yhat_train_svd = ridge_svd_fit_predict(X_train, Y_train, alphas, cv=3)
    Yhat_test_skd = X_test @ B_svd
    print("Best alpha (SVD ridge):", best_alpha_svd)
    print("Train Pearson (svd):", pearson_mean_across_targets(Y_train, Yhat_train_svd))
    print("Test Pearson (svd):", pearson_mean_across_targets(Y_test, Yhat_test_skd))

    print("\n=== sklearn RidgeCV baseline ===")
    best_alpha_skl, B_skl, Yhat_train_skl = sklearn_ridgecv_fit_predict(X_train, Y_train, alphas, cv=3)
    Yhat_test_skl = X_test @ B_skl
    print("Best alpha (sklearn RidgeCV):", best_alpha_skl)
    print("Train Pearson (sklearn):", pearson_mean_across_targets(Y_train, Yhat_train_skl))
    print("Test Pearson (sklearn):", pearson_mean_across_targets(Y_test, Yhat_test_skl))

    print("\n=== Dask MOR vs B-MOR ridge ===")
    res = mor_ridge_dask(X_train, Y_train, alphas, cv=3, n_workers=2, threads_per_worker=1)
    print("MOR mean alphas (first 5)", np.mean(res['mor']['alpha_list'][:5]))
    print("MOR train pearson:", pearson_mean_across_targets(Y_train, res["mor"]['Yhat']))
    print("B-MOR alphas per-batch:", res['bmore']['alpha_list'])
    print("B-MOR train pearson:", pearson_mean_across_targets(Y_train, res["bmor"]['Yhat']))

    print("\n Done. Note: This toy demo uses the VGG16 FC2 features and ridge regression in the way described in the thesis (VGG16->features->multi-target ridge)")