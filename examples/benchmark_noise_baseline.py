import numpy as np
from scipy.io import loadmat
from joblib import Parallel, delayed
import pandas as pd

import os

from chromprocessor.classes import Chromatogram

from scipy.integrate import cumulative_trapezoid

def rmse(y, yhat):
    """
    Root Mean Squared Error between two vectors
    """
    y = np.asarray(y)
    yhat = np.asarray(yhat)
    return np.sqrt(np.mean((y - yhat) ** 2))


def aucpr(chrom, z):
    """
    Compute relative error between AUCs of original and denoised chromatograms
    chrom = original chromatogram (without noise)
    z = denoised chromatogram
    err = relative error (percentage-like, but returned as fraction)
    """
    chrom = np.asarray(chrom)
    z = np.asarray(z)

    # Find peak regions
    k = np.where(chrom > 0.001)[0]

    # Cumulative trapezoidal integration
    Q1 = cumulative_trapezoid(chrom[k], initial=0)
    Q2 = cumulative_trapezoid(z[k], initial=0)

    err = np.abs(Q2[-1] - Q1[-1]) / Q1[-1]
    return err


def SNR(z):
    """
    Signal-to-noise ratio
    """
    z = np.asarray(z)

    Ph = np.max(z)
    Nmin = np.min(z[:25])
    Nmax = np.max(z[:25])

    snr = Ph / np.abs(Nmax - Nmin)
    return snr

os.chdir(r"C:\Users\emery\OneDrive - KU Leuven\Chromatograms\SimulatedData_Leon\Baseline 1")
# Preallocate arrays
RMSEVarBEADS = np.zeros((10, 10))
AUCVarBEADS = np.zeros((10, 10))
SNRVarBEADS = np.zeros((10, 10))

def process(i, j):
    filename = f"DataCoverageLevel{i}DataNoiseLevel{j}.mat"
    S = loadmat(filename)

    # Extract variables (adjust keys if needed depending on .mat structure)
    Signal = S["Signal"][0]
    added_noise = S["added_noise"][0]
    baseline = S["baseline"][0]

    y = Signal.astype(float)
    x = (Signal - added_noise - baseline).astype(float)
    xbaseline = (Signal - added_noise).astype(float)

    chrom = Chromatogram(y, None, None, None, np.linspace(0,10,len(y)))
    chrom.remove_blank()

    # processing of chromatogram
    #chrom.correct_baseline()
    chrom.remove_noise()

    rmse_val = rmse(xbaseline, chrom.chrom_1D)
    auc_val = aucpr(xbaseline, chrom.chrom_1D)
    snr_val = SNR(chrom.chrom_1D)

    return i - 1, j - 1, rmse_val, auc_val, snr_val


# Parallel loop (equivalent to MATLAB parfor)
results = Parallel(n_jobs=-1)(
    delayed(process)(i, j)
    for i in range(1, 11)
    for j in range(1, 11)
)

# Store results
for i, j, rmse_val, auc_val, snr_val in results:
    RMSEVarBEADS[i, j] = rmse_val
    AUCVarBEADS[i, j] = auc_val
    SNRVarBEADS[i, j] = snr_val


# Save to CSV (equivalent to writematrix)
pd.DataFrame(RMSEVarBEADS).to_csv(
    r"C:\Users\emery\OneDrive - KU Leuven\Own Work\ChromProcessor\Results\RMSEBaseline1.csv",
    index=False, header=False
)

pd.DataFrame(AUCVarBEADS).to_csv(
    r"C:\Users\emery\OneDrive - KU Leuven\Own Work\ChromProcessor\Results\AUCBaseline1.csv",
    index=False, header=False
)

pd.DataFrame(SNRVarBEADS).to_csv(
    r"C:\Users\emery\OneDrive - KU Leuven\Own Work\ChromProcessor\Results\SNRBaseline1.csv",
    index=False, header=False
)

