"""
This script implements a toy example of VGG16-based feature extraction followed by Ridge Regression
for multi-output prediction tasks. It demonstrates different approaches to Ridge Regression:
1. SVD-based implementation
2. scikit-learn's RidgeCV
3. Multi-output Ridge (MOR) and Batched Multi-output Ridge (B-MOR) using Dask

The code uses VGG16's FC2 layer features as input and implements various Ridge Regression
approaches to predict multiple target variables simultaneously.
"""

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

def pearson_mean_across_targets(Y_true, Y_pred):
    """
    Calculate the mean Pearson correlation coefficient across multiple target variables.
    
    Args:
        Y_true (np.ndarray): Ground truth values, shape (n_samples, n_targets)
        Y_pred (np.ndarray): Predicted values, shape (n_samples, n_targets)
    
    Returns:
        float: Mean Pearson correlation across all targets
    
    Note:
        Returns 0.0 for any target with zero standard deviation in either true or predicted values.
    """
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
    """
    Extract features from the FC2 (second fully connected) layer of a pre-trained VGG16 model.
    
    Args:
        images_tensor (torch.Tensor): Input images tensor of shape (N, C, H, W)
        device (str): Device to run the model on ('cpu' or 'cuda')
        batch_size (int): Batch size for processing images
    
    Returns:
        np.ndarray: Extracted FC2 features of shape (N, 4096)
    
    Note:
        Uses a forward hook to capture the FC2 layer activations before the final
        classification layer. The features are extracted using the pre-trained VGG16
        model from torchvision.
    """
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
    """
    Create a toy dataset using CIFAR10 images and synthetic target variables.
    
    Args:
        n_samples (int): Number of samples to generate
        n_targets (int): Number of target variables to generate
        concat_frames (int): Number of frames to concatenate (usually 1 for toy example)
    
    Returns:
        tuple: (X_feats, Y) where:
            - X_feats (np.ndarray): VGG16 FC2 features of shape (n_samples, 4096)
            - Y (np.ndarray): Synthetic target variables of shape (n_samples, n_targets)
    
    Note:
        1. Uses CIFAR10 images resized to 224x224 (VGG16 input size)
        2. Extracts VGG16 FC2 features as input features
        3. Creates synthetic targets using a linear model with noise:
           Y = X @ B_true + noise
    """
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

    # Extract VGG16 features
    X_feats = extract_vgg16_fc2_features(images_all)
    p = X_feats.shape[1]

    # Create synthetic linear targets: choose a random true B_true and make Y=X B_true + noise 
    rng = np.random.RandomState(0)
    B_true = rng.randn(p, n_targets) * 0.1
    Y = X_feats @ B_true + 0.05 * rng.randn(X_feats.shape[0], n_targets)
    return X_feats, Y

# 3) SVD-based ridge (amortized over alphas)

def ridge_svd_fit_predict(X, Y, alphas, cv=5):
    """
    Fit Ridge Regression using SVD decomposition and cross-validation.
    
    This implementation uses SVD for efficient computation of the Ridge solution
    across multiple regularization parameters (alphas). It's more computationally
    efficient than standard Ridge regression when trying multiple alpha values
    because it computes the SVD only once.
    
    Args:
        X (np.ndarray): Input features of shape (n_samples, n_features)
        Y (np.ndarray): Target variables of shape (n_samples, n_targets)
        alphas (list): List of alpha values to try for regularization
        cv (int): Number of cross-validation folds
    
    Returns:
        tuple: (best_alpha, B_final, Yhat) where:
            - best_alpha (float): Best regularization parameter
            - B_final (np.ndarray): Final coefficients matrix
            - Yhat (np.ndarray): Predicted values
    
    Note:
        Uses the formula B = V @ diag(s/(s^2 + alpha)) @ U.T @ Y
        where U, s, V = svd(X)
    """
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
    """
    Fit Ridge Regression using scikit-learn's RidgeCV implementation.
    
    This serves as a baseline implementation using scikit-learn's built-in
    RidgeCV which performs efficient cross-validation to select the best
    regularization parameter.
    
    Args:
        X (np.ndarray): Input features of shape (n_samples, n_features)
        Y (np.ndarray): Target variables of shape (n_samples, n_targets)
        alphas (list): List of alpha values to try for regularization
        cv (int): Number of cross-validation folds
    
    Returns:
        tuple: (alpha, B, Yhat) where:
            - alpha (float): Selected regularization parameter
            - B (np.ndarray): Coefficient matrix (transposed to match convention)
            - Yhat (np.ndarray): Predicted values
    """
    model = RidgeCV(alphas=alphas, cv=cv)
    model.fit(X, Y)
    # Get coefficients (transpose to match our convention)
    B = model.coef_.T
    Yhat = model.predict(X)
    return model.alpha_, B, Yhat

# 5) MOR vs B-MOR toy with local Dask
def mor_ridge_dask(X, Y, alphas, cv=3, n_workers=2, threads_per_worker=1):
    """
    Implement Multi-Output Ridge (MOR) and Batched Multi-Output Ridge (B-MOR) regression
    using Dask for parallel computation.
    
    This function implements two strategies for parallel Ridge regression:
    1. MOR: Fits a separate ridge model for each target variable
    2. B-MOR: Splits targets into batches and fits one model per batch
    
    Args:
        X (np.ndarray): Input features of shape (n_samples, n_features)
        Y (np.ndarray): Target variables of shape (n_samples, n_targets)
        alphas (list): List of alpha values to try for regularization
        cv (int): Number of cross-validation folds
        n_workers (int): Number of parallel workers for Dask
        threads_per_worker (int): Number of threads per worker
    
    Returns:
        dict: Dictionary containing results for both MOR and B-MOR:
            - mor: {'alpha_list': list of alphas, 'B': coefficients, 'Yhat': predictions}
            - bmor: {'alpha_list': list of alphas, 'B': coefficients, 'Yhat': predictions}
    
    Note:
        - MOR: One model per target (maximum parallelism)
        - B-MOR: One model per batch of targets (reduced communication overhead)
    """
    cluster = LocalCluster(n_workers=n_workers, threads_per_worker=threads_per_worker, processes=True)
    client = Client(cluster)

    import dask
    from dask import delayed, compute

    n_samples, p = X.shape
    _, s = Y.shape

    # MOR: submit one job per target (delayed)
    def fit_target(x, y_col, alphas_local, cv_local):
        # small helper to fit RidgeCV on vector y_col
        model = RidgeCV(alphas=alphas, cv=cv_local, store_cv_values=False)
        model.fit(x, y_col)
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
    """
    Main demonstration script that:
    1. Creates a toy dataset using VGG16 features from CIFAR10 images
    2. Compares three Ridge regression implementations:
       - SVD-based Ridge regression
       - scikit-learn's RidgeCV
       - Parallel MOR and B-MOR using Dask
    
    The demo follows these steps:
    1. Generate toy data with VGG16 FC2 features
    2. Split into train/test sets (90/10)
    3. Compare different Ridge regression implementations
    4. Report performance metrics (Pearson correlation)
    """
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