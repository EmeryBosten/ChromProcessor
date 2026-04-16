import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from chromprocessor.classes import Chromatogram

base_path = Path(r"C:\Users\emery\OneDrive - KU Leuven\Chromatograms\CNN\GRA")

chrom = pd.read_csv(base_path / "MIX_F10.CSV", sep="\t", encoding="utf-16").to_numpy()
baseline = pd.read_csv(base_path / "BASELINE.CSV", sep="\t",  encoding="utf-16").to_numpy()

x = chrom[:,0]
y = chrom[:,1]
base = baseline[:,1]

noise = 0.00 * (np.random.rand(len(y)) - 0.5)
y = y - base + noise

#Initialize chromatogram
chrom = Chromatogram(y, None, None, None, x, None)
chrom.remove_blank()
chrom.from_2D_to_1D()

# processing of chromatogram
chrom.correct_baseline()
chrom.remove_noise()
chrom.detect_peaks()

fig, axs = plt.subplots(2,1)
axs[0].plot(chrom.time,chrom.chrom_1D)
plt.xlabel("Time", fontsize=12.5)
plt.ylabel("Absorbance", fontsize=12.5)
for peak in chrom.peaks:
    axs[0].plot(chrom.time[peak.all_maxima],chrom.chrom_1D[peak.all_maxima], marker='*', linestyle='None')


filenames = [
"02.CSV","04.CSV","05.CSV","12.CSV","27.CSV","30.CSV","34.CSV","38.CSV","39.CSV",
"55.CSV","59.CSV","67.CSV","68.CSV","70.CSV","73.CSV","74.CSV","76.CSV","81.CSV",
"85.CSV","90.CSV","LII.CSV","LIV.CSV","LV.CSV","LVI.CSV","LX.CSV","XI.CSV","XL.CSV",
"XLI.CSV","XLV.CSV","XLVI.CSV","XLVIII.CSV","XV.CSV","XXI.CSV","XXIII.CSV","XXIX.CSV",
"XXVII.CSV","XXVIII.CSV","XXXIII.CSV","XXXIX.CSV","XXXV.CSV","XXXVII.CSV","XXXVIII.CSV"
]
cmps = [pd.read_csv(base_path / f, sep="\t", encoding="utf-16").to_numpy() for f in filenames]

for cmp in cmps:
    plt.plot(cmp[:,0], cmp[:,1] - base, linewidth=1.25)

for cmp in cmps:
    axs[1].plot(cmp[:,0], cmp[:,1], linewidth=1.25)

#plt.plot(x[apex_loc], -5*np.ones_like(apex_loc), '|r', linewidth=1)
plt.xlabel("Time", fontsize=12.5)
plt.ylabel("Absorbance", fontsize=12.5)
plt.show()
