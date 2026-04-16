import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from chromprocessor.classes import Chromatogram

base_path = Path(r"C:\Users\emery\OneDrive - KU Leuven\Chromatograms\Autoencoder - raw data\ISO")

chrom = pd.read_csv(base_path / "MIX_F10.CSV", sep="\t", encoding="utf-16").to_numpy()

x = chrom[:,0]
y = chrom[:,1]

noise = 0.00 * (np.random.rand(len(y)) - 0.5)
y = y + noise

#Initialize chromatogram
chrom = Chromatogram(y, None, x)
chrom.remove_blank()
chrom.from_2D_to_1D()

# processing of chromatogram
chrom.correct_baseline()
chrom.remove_noise()
chrom.detect_peaks()

filenames = [
"26.CSV", "30.CSV", "31.CSV", "35.CSV", "37.CSV", "38.CSV", "39.CSV", "51.CSV", "55.CSV", "58.CSV", "59.CSV", "60.CSV", "62.CSV", "69.CSV", "76.CSV", "78.CSV"
]
cmps = [pd.read_csv(base_path / f, sep="\t", encoding="utf-16").to_numpy() for f in filenames]

for cmp in cmps:
    plt.plot(cmp[:,0], cmp[:,1], linewidth=1.25)

#plt.plot(x[apex_loc], -5*np.ones_like(apex_loc), '|r', linewidth=1)
plt.xlabel("Time", fontsize=17.5)
plt.ylabel("Absorbance", fontsize=17.5)
plt.xlim(0,14)
plt.show()