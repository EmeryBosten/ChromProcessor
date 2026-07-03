import numpy as np
from matplotlib import pyplot as plt
from scipy.io import loadmat
import os

from chromprocessor.classes import Chromatogram, Chromatograms
from chromprocessor.peak_models import BiGaussianTailing, Bemg, FraserSuzuki

os.chdir(r"C:\Users\emery\OneDrive - KU Leuven\Matlab scripts\Automation")


filename = f"chrom1.mat"
S = loadmat(filename)

# Extract variables (adjust keys if needed depending on .mat structure)
Signal = S["myTable"].transpose()

y = Signal.astype(float)

chrom1 = Chromatogram(y, None, None, None, np.linspace(0,10,y.shape[1]),None)
chrom1.remove_blank()
chrom1.from_2D_to_1D()

# processing of chromatogram
chrom1.correct_baseline(method="flatfit")
chrom1.remove_noise()
chrom1.detect_peaks(height_ratio=10,width=0,remove_0_width=True)

# Plot
plt.rcParams.update({'font.size': 11.5})
plt.figure()
plt.plot(chrom1.time,chrom1.chrom_1D, linewidth=2)
for peak in chrom1.peaks:
    plt.plot(chrom1.time[peak.all_maxima],chrom1.chrom_1D[peak.all_maxima], marker='*', linestyle='None')


chrom1.deconvolve_peaks(FraserSuzuki())

filename = f"chrom3.mat"
S = loadmat(filename)

# Extract variables (adjust keys if needed depending on .mat structure)
Signal = S["myTable"].transpose()

y = Signal.astype(float)

chrom2 = Chromatogram(y, None, None, None, np.linspace(0,10,y.shape[1]),None)
chrom2.remove_blank()
chrom2.from_2D_to_1D()

# processing of chromatogram
chrom2.correct_baseline(method="flatfit")
chrom2.remove_noise()
chrom2.detect_peaks(height_ratio=10,width=0,remove_0_width=True)

# Plot
plt.rcParams.update({'font.size': 11.5})
plt.figure()
plt.plot(chrom2.time,chrom2.chrom_1D, linewidth=2)
for peak in chrom2.peaks:
    plt.plot(chrom2.time[peak.all_maxima],chrom2.chrom_1D[peak.all_maxima], marker='*', linestyle='None')


chrom2.deconvolve_peaks(FraserSuzuki())

#plot spectra of selected compounds
#select compound group
for g in [1, 2, 3, 5 ]:
    group = g
    plt.rcParams.update({'font.size': 11.5})
    plt.figure()
    plt.plot(np.arange(200,320,2),chrom1.peaks[group].spectra[0], linewidth=2, color = "blue")
    plt.plot(np.arange(200,320,2),chrom2.peaks[group].spectra[0], linewidth=2, color = "red")
    plt.show()


chroms = Chromatograms([chrom1, chrom2],False)
[retentionTimeTable, multi_peaks] = chroms.track_peaks(preprocessing=True, height_ratio=10, width = 0, remove_0_width=True)
print(retentionTimeTable)