import numpy as np
from matplotlib import pyplot as plt
from scipy.io import loadmat
import os

from chromprocessor.classes import Chromatogram

os.chdir(r"C:\Users\emery\OneDrive - KU Leuven\Chromatograms\SimulatedData_Leon\Baseline 3")


filename = f"DataCoverageLevel{3}DataNoiseLevel{2}.mat"
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
chrom.correct_baseline()

plt.figure()
plt.plot(y)
plt.plot(chrom.chrom_1D)
plt.plot(chrom.baseline)
#plt.ylim(-10, 200)
plt.show()
chrom.remove_noise()
plt.figure()
plt.plot(y-baseline)
plt.plot(chrom.chrom_1D)
#plt.ylim(-10, 200)
plt.show()
