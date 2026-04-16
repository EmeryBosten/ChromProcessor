import numpy as np
from matplotlib import pyplot as plt
from scipy.io import loadmat
import os

from chromprocessor.classes import Chromatogram
from chromprocessor.peak_models import BiGaussianTailing

os.chdir(r"C:\Users\emery\OneDrive - KU Leuven\Matlab scripts\Automation")


filename = f"chrom1.mat"
S = loadmat(filename)

# Extract variables (adjust keys if needed depending on .mat structure)
Signal = S["myTable"].transpose()

y = Signal.astype(float)

chrom = Chromatogram(y, None, None, None, np.linspace(0,10,y.shape[1]),None)
chrom.remove_blank()
chrom.from_2D_to_1D()

# processing of chromatogram
chrom.correct_baseline()
chrom.remove_noise()
chrom.detect_peaks(height_ratio=15,width=5,remove_0_width=False)
chrom.deconvolve_peaks(BiGaussianTailing())

#plot spectra
#select compound group
group = 8
nb_cmps = len(chrom.peaks[group].all_maxima)
plt.figure()
for i in range(nb_cmps):
    plt.plot(chrom.peaks[group].spectra[i])
plt.show()

# plot concentration profiles
group = 8
nb_cmps = len(chrom.peaks[group].all_maxima)
plt.figure()
for i in range(nb_cmps):
    plt.plot(chrom.peaks[group].concentrations[i])

plt.show()

