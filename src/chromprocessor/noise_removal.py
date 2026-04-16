#functions for automated noise removal
import numpy as np
from sklearn.neighbors import NearestNeighbors

def modoneNNestimator(Y, X, type_='median'):
    Y = np.asarray(Y).ravel()
    X = np.asarray(X)

    if X.ndim == 1:
        X = X.reshape(-1, 1)

    nbrs = NearestNeighbors(n_neighbors=3).fit(X)
    _, indices = nbrs.kneighbors(X)

    indices = indices[:, 1:]  # remove self

    Yn1 = Y[indices[:, 0]]
    Yn2 = Y[indices[:, 1]]

    mult_diff = np.abs(Y - Yn1) * np.abs(Y - Yn2)

    if type_ == 'median':
        return np.median(mult_diff)
    elif type_ == 'mean':
        return np.mean(mult_diff)
    else:
        raise ValueError("type_ must be 'median' or 'mean'")