import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from chromprocessor.classes import Chromatogram

# Load chromatogram
base_path = Path(r"C:\Users\emery\OneDrive - KU Leuven\Chromatograms\Autoencoder - raw data\GRA")

chrom = pd.read_csv(base_path / "MIX_F10.CSV", sep="\t", encoding="utf-16").to_numpy()

#time
x = chrom[:,0]
#signal
y = chrom[:,1]

#add some noise
noise = 0.25 * (np.random.rand(len(y)) - 0.5)
y = y + noise

#Initialize chromatogram
chrom = Chromatogram(y, None, None, None, x, None)
chrom.remove_blank()
chrom.from_2D_to_1D()

# processing of chromatogram
chrom.correct_baseline()
chrom.remove_noise()
chrom.detect_peaks()

plt.rcParams.update({'font.size': 11.5})
fig, axs = plt.subplots(3,1)
axs[0].plot(chrom.time,chrom.chrom_1D,linewidth=2)
#plt.xlabel("Time", fontsize=12.5)
#plt.ylabel("Absorbance", fontsize=12.5)
for peak in chrom.peaks:
    axs[0].plot(chrom.time[peak.all_maxima],chrom.chrom_1D[peak.all_maxima], marker='*', linestyle='None')


filenames = [
"12.CSV", "25.CSV", "26.CSV", "30.CSV", "31.CSV", "33.CSV", "35.CSV", "37.CSV", "38.CSV", "39.CSV", "51.CSV", "55.CSV", "58.CSV", "59.CSV", "60.CSV", "62.CSV", "63.CSV", "67.CSV", "68.CSV", "69.CSV", "76.CSV", "78.CSV", "81.CSV", "93.CSV"]
cmps = [pd.read_csv(base_path / f, sep="\t", encoding="utf-16").to_numpy() for f in filenames]

for cmp in cmps:
    axs[2].plot(cmp[:,0], cmp[:,1], linewidth=2)

for peak in chrom.peaks:
    axs[2].plot(chrom.time[peak.all_maxima],0, marker='|', linestyle='None', color = "red", linewidth=2)

axs[1].plot(chrom.time[:-1],np.diff(y), color="blue",linewidth=2)
axs[1].plot(chrom.time[:-1],np.diff(chrom.chrom_1D), color="black",linewidth=1)


#plt.plot(x[apex_loc], -5*np.ones_like(apex_loc), '|r', linewidth=1)
#plt.xlabel("Time", fontsize=12.5)
#plt.ylabel("Absorbance", fontsize=12.5)
plt.show()

plt.figure()
plt.plot(chrom.time,chrom.signal_raw_UV,linewidth=3)
plt.show()

plt.figure()
plt.plot(chrom.time,chrom.chrom_1D,linewidth=3)
plt.show()