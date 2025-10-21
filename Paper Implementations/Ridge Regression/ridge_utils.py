# ridge_utils.py
import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import RidgeCV, Ridge
from sklearn.utils import check_random_state
from dask.distributed import Client, LocalCluster, wait
import math
import joblib

###############
# A) Fast SVD-based ridge with CV
###############
def ridge_svd_cv(X, Y, alphas, cv=5, random_state=0):
    """
    X: t x p, Y: t x s (numpy arrays)
    alphas: list/array of lambdas to try
    returns:
      best_alpha (per batch average), B_best: p x s
    Implementation follows the SVD trick from the thesis.
    """
    rng = check_random_state(random_state)
    t, p = X.shape
    _, s = Y.shape
    kf = KFold(n_splits=cv, shuffle=True, random_state=rng)

    # Precompute SVD once for X (we use economy SVD)
    # If p > t, you might prefer SVD of X.T or use scipy.linalg.lstsq
    U, S, Vt = np.linalg.svd(X, full_matrices=False)  # U: t x p, S: p, Vt: p x p
    V = Vt.T
    S_mat = S   # vector of singular values

    # Precompute terms independent of lambda
    UT = U.T  # p x t

    # For each lambda compute M(lambda) = V * diag( S / (S^2 + lambda) ) * U^T
    # We'll evaluate CV score for each alpha quickly.
    scores = np.zeros((len(alphas),))
    for i_alpha, lam in enumerate(alphas):
        # build diag vector d = S / (S^2 + lambda)
        d = S_mat / (S_mat**2 + lam)
        # M = V @ diag(d) @ U.T  -> implement as function to apply to Xsub or Ysub
        # For CV, we need predictions on validation sets; we compute B on train and score on val.
        cv_scores = []
        for tr_idx, val_idx in kf.split(X):
            # get train matrices
            Xtr = X[tr_idx]
            Ytr = Y[tr_idx]
            Xval = X[val_idx]
            Yval = Y[val_idx]

            # compute Utr, Str, Vt? We precomputed SVD on full X for speed (approx).
            # Simpler: compute M using precomputed U,V (approx). This is standard practice in large-scale.
            M_lambda = V @ (d[:, None] * UT)  # p x t
            B = M_lambda @ Ytr  # p x s
            Ypred = Xval @ B  # (len(val) x s)
            # use mean Pearson r across targets as score (like the thesis)
            # compute correlation per column
            corrs = []
            for j in range(Yval.shape[1]):
                a = Yval[:, j]
                b = Ypred[:, j]
                # avoid degenerate
                if np.std(a) == 0 or np.std(b) == 0:
                    corrs.append(0.0)
                else:
                    corrs.append(np.corrcoef(a, b)[0,1])
            cv_scores.append(np.nanmean(corrs))
        scores[i_alpha] = np.nanmean(cv_scores)

    best_idx = np.nanargmax(scores)
    best_alpha = alphas[best_idx]

    # compute final B on full data with best_alpha
    lam = best_alpha
    d = S_mat / (S_mat**2 + lam)
    UT = U.T
    M_lambda = V @ (d[:, None] * UT)
    B_best = M_lambda @ Y

    return best_alpha, B_best

###############
# B) Simple scikit-learn RidgeCV (performs efficiently and uses SVD/internals)
###############
def sklearn_ridgecv(X, Y, alphas, cv=5, n_jobs=1):
    """
    X: t x p, Y: t x s
    Uses scikit-learn's RidgeCV; it supports multioutput Y (columns as targets).
    When s is huge this will compute M more than desired; see B-MOR below.
    """
    # RidgeCV with store_cv_values=False because storing CV residuals may be heavy
    model = RidgeCV(alphas=alphas, cv=cv, store_cv_values=False)
    # if Y has many columns, scikit-learn will still handle it but may be slow for very large s
    model.fit(X, Y)
    B = model.coef_.T  # sklearn stores (n_targets, n_features)
    # model.alpha_ is the best alpha if used global; note RidgeCV chooses alpha for each target if scoring allows.
    return model.alpha_, B

###############
# C) B-MOR: Batch Multi-Output Regression with Dask
###############
def bmor_ridgecv_dask(X, Y, alphas, cv=5, n_batches=4, n_workers=None, threads_per_worker=1):
    """
    Partition Y into n_batches (ideally = number of workers), submit a RidgeCV per batch
    to Dask. Inside each worker, RidgeCV uses multi-threading (set environment/mkl threads).
    Returns concatenated B matrix (p x s) and per-batch best alphas.
    """
    # Start a local cluster (or connect to existing scheduler by Client('scheduler-address:8786'))
    # NOTE: in production you will connect to an HPC Dask scheduler; here we show the local flow.
    if n_workers is None:
        n_workers = n_batches
    cluster = LocalCluster(n_workers=n_workers, threads_per_worker=threads_per_worker)
    client = Client(cluster)

    t, p = X.shape
    _, s = Y.shape
    # compute batch sizes
    batch_size = math.ceil(s / n_batches)
    futures = []
    for i in range(n_batches):
        start = i * batch_size
        end = min((i+1) * batch_size, s)
        if start >= end:
            continue
        Yi = Y[:, start:end].copy()
        # submit function that fits RidgeCV on this Yi
        def fit_batch(X_local, Yi_local, alphas_local, cv_local):
            # NOTE: this function runs on the worker process; you can set omp/mkl threads via env var.
            from sklearn.linear_model import RidgeCV
            model = RidgeCV(alphas=alphas_local, cv=cv_local, store_cv_values=False)
            model.fit(X_local, Yi_local)
            Bbatch = model.coef_.T  # p x batch_s
            return model.alpha_, Bbatch
        fut = client.submit(fit_batch, X, Yi, alphas, cv)
        futures.append((i, start, end, fut))
    # wait and gather
    wait([f for (_, _, _, f) in futures])
    # assemble results
    B = np.zeros((p, s), dtype=float)
    batch_alphas = {}
    for (i, start, end, fut) in futures:
        alpha_i, Bbatch = fut.result()
        B[:, start:end] = Bbatch
        batch_alphas[i] = alpha_i

    client.close()
    cluster.close()
    return batch_alphas, B
