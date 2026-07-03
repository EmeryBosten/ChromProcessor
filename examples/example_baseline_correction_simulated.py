import numpy as np
from matplotlib import pyplot as plt
from scipy.io import loadmat
import os

from chromprocessor.classes import Chromatogram

os.chdir(r"C:\Users\emery\OneDrive - KU Leuven\Chromatograms\SimulatedData_Leon\Baseline 1")


filename = f"DataCoverageLevel{5}DataNoiseLevel{1}.mat"
S = loadmat(filename)

# Extract variables (adjust keys if needed depending on .mat structure)
Signal = S["Signal"][0]
added_noise = S["added_noise"][0]
baseline = S["baseline"][0]

y = Signal.astype(float)
x = (Signal - added_noise - baseline).astype(float)
xbaseline = (Signal - added_noise).astype(float)

chrom = Chromatogram(y, None, None, None, np.linspace(0,10,len(y)),None)
chrom.remove_blank()

# processing of chromatogram
chrom.correct_baseline(method = "flatfit")

plt.figure()
plt.rcParams.update({'font.size': 11.5})
plt.plot(np.linspace(0,10,len(y)),y, color = "blue",linewidth=2.0)
plt.plot(np.linspace(0,10,len(y)),baseline, color = "green",linewidth=2.0)
plt.plot(np.linspace(0,10,len(y)),chrom.baseline, color = "red",linewidth=2.5, linestyle="--")

plt.ylim(-0.5, 3)
plt.show()


chrom = Chromatogram(y, None, None, None, np.linspace(0,10,len(y)),None)
chrom.remove_blank()
chrom.correct_baseline(method = "erpls")

plt.figure()
plt.rcParams.update({'font.size': 11.5})
plt.plot(np.linspace(0,10,len(y)),y, color = "blue",linewidth=2.0)
plt.plot(np.linspace(0,10,len(y)),baseline, color = "green",linewidth=2.5)
plt.plot(np.linspace(0,10,len(y)),chrom.baseline, color = "red",linewidth=2.0, linestyle="--")

plt.ylim(-0.5, 3)
plt.show()
