# Functions for automated baseline correction

from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

from numpy.typing import ArrayLike, NDArray

import numpy as np
from scipy.signal import savgol_filter  # type: ignore
from scipy import sparse  # type: ignore
from scipy.sparse.linalg import spsolve  # type: ignore

def whittaker_smooth(x, w, lam, differences=2):
    """
    Weighted Whittaker Smoothing (equivalent to R's WhittakerSmooth)
    Args:
        x : signal (1D array)
        w : weights (1D array)
        lam : regularization parameter (lambda)
        differences : degree of difference (default 2)
    Returns:
        background : smoothed baseline
    """
    x = np.asarray(x, dtype=float).flatten()
    l = len(x)

    # Identity and difference operator
    e = sparse.eye(l, format='csc')
    d = e[1:] - e[:-1]
    for _ in range(differences - 1):
        d = d[1:] - d[:-1]

    # Weight matrix (diagonal)
    w = sparse.diags(w, 0, shape=(l, l), format='csc')

    # Solve (w + λ DᵀD) z = w x
    a = w + lam * (d.T @ d)
    b = w @ x
    background = spsolve(a, b)
    return np.array(background).flatten()


def airPLS(x, lam, differences=2, itermax=50):
    """
    Adaptive Iteratively Reweighted Penalized Least Squares (airPLS)
    Args:
        x : signal
        lam : regularization parameter
        differences : degree parameter
        itermax : maximum iterations
    Returns:
        z : estimated baseline
    """
    x = np.asarray(x, dtype=float).flatten()
    m = len(x)
    w = np.ones(m)
    i = 1
    control = True

    while control:
        z = whittaker_smooth(x, w, lam, differences)
        d = x - z
        sum_smaller = np.abs(np.sum(d[d < 0]))

        # Convergence criterion
        if sum_smaller < 0.001 * np.sum(np.abs(x)) or i == itermax:
            control = False
        else:
            w[d >= 0] = 0
            if np.any(d < 0):
                w[d < 0] = np.exp(i * np.abs(d[d < 0]) / sum_smaller)
                w[0] = np.exp(i * np.max(d[d < 0]) / sum_smaller)
                w[-1] = np.exp(i * np.max(d[d < 0]) / sum_smaller)
            i += 1

    return z


def erPLSrmse(y, lam, differences=2, itermax=50, plot=True):
    """
    Automated airPLS baseline correction (from paper:
    "An Automatic Baseline Correction Method Based on the Penalized Least Squares Method")
    Args:
        y : signal
        lam : regularization parameter
        differences : degree parameter
        itermax : number of iterations
        plot : whether to plot the extended signal
    Returns:
        err : RMSE of fitting
    """
    y = np.asarray(y, dtype=float).flatten()
    n = len(y)
    om = round(n / 20)
    w = round(n / 5)
    h = np.max(y)

    # Linear regression on last `om` points
    x = np.arange(1, om + 1).reshape(-1, 1)
    model = LinearRegression().fit(x, y[-om:])
    intercept, slope = model.intercept_, model.coef_[0]

    # Extrapolate end portion
    ye = intercept + slope * np.arange(1, om + w + 1)
    yext = ye[om:om + w]

    # Gaussian tail
    def gauss(t, mean, sd):
        return np.exp(-((t - mean) ** 2) / (2 * sd ** 2))

    yg = (h - yext[w - 1]) * gauss(np.arange(1, w + 1), w / 2, w / 20)
    ynew = yext + yg
    yadd = np.concatenate([y, ynew])

    if plot:
        plt.plot(yadd, label="Extended signal")
        plt.title("Extended signal for erPLS")
        plt.legend()
        plt.show()

    z = airPLS(yadd, lam, differences, itermax)

    # RMSE function
    def rmse(actual, predicted):
        return np.sqrt(np.mean((actual - predicted) ** 2))

    err = rmse(yext, z[-w:])
    return err

# try flatfit as alternative
# see Mocca implementation

def flatfit(data: ArrayLike, smoothness: float, p: float) -> NDArray:
    """
    FlatFit: least squares weighted by inverse scale of 1st and 2nd derivatives with smoothness penalty

    Parameters
    ----------

    data: ArrayLike
        1D data

    smoothness: float
        size of smoothness penalty

    p: float
        relative size of Savitzky-Golay filter

    Returns
    -------
    NDArray
        values that minimize the asymmetric squared error with smoothness penalty

    Description
    -----------

    This routine finds vector `z` that minimized:

    `(y-z).T @ W @ (y-z) + smoothness * z.T @ D.T @ D @ z`

    where `W` is determined by slope and curvature at given point

    """
    y = np.array(data)
    assert len(y.shape) == 1, "Incorrect data input shape for AsLS"
    assert len(y) > 3, "At least 4 data points muts be provided for AsLS"

    L = len(y)

    # scale lambda to maintain invariance for sampling frequency
    lamb = smoothness * L ** 4

    # calculate w
    filter_window = max(4, int(L * p))
    slope = savgol_filter(y, filter_window, 3, deriv=1)
    curvature = np.gradient(slope)
    slope = slope ** 2
    slope /= np.sum(slope)
    curvature = curvature ** 2
    curvature /= np.sum(curvature)

    w = 1 / (slope + curvature + 1e-10)

    # calculate baseline

    D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(L, L - 2))
    H = lamb * D.dot(D.transpose())

    W = sparse.spdiags(w, 0, L, L)
    Z = W + H
    z = spsolve(Z, w * y)

    return z