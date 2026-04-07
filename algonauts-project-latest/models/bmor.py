import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from joblib import Parallel, delayed

def fit_bmor_joblib(X, Y, n_jobs=4, alphas=None):
    if alphas is None:
        alphas = np.logspace(-4, 4, 9)
        
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    
    def fit_batch(yb):
        model = RidgeCV(alphas=alphas, cv=3)
        model.fit(Xs, yb)
        return model.coef_, model.intercept_
    
    # Split Y into batches for parallel processing
    n_batches = 16
    batches = np.array_split(Y, n_batches, axis=1)
    
    results = Parallel(n_jobs=n_jobs)(
        delayed(fit_batch)(b) for b in batches
    )
    
    return {
        'coefs': np.vstack([r[0] for r in results]),
        'intercepts': np.concatenate([r[1] for r in results]),
        'scaler': scaler
    }
