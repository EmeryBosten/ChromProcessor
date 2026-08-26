import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from chromprocessor.classes import Chromatogram
from chromprocessor.deconvolution import fit_peak_model, spectra_from_concentrations
from chromprocessor.peak_models import FraserSuzuki

path = "C:/Users/emery/OneDrive - KU Leuven/Chromatograms/Agilent JnJ"


def partitioning(chromatogram, start, end):
    if chromatogram.ndim == 1:

        chromatogram = chromatogram[int(start * len(chromatogram)): int(end * len(chromatogram))]

    else:

        chromatogram = chromatogram[:, int(start * np.shape(chromatogram)[1]): int(end * np.shape(chromatogram)[1])]

    return chromatogram


# Chromatogram 1


data = pd.read_csv(path + "/Batch Processing//UV/UV3.csv")

timeUV = partitioning(data["time"].to_numpy(), 1 / 10, 1 / 2)

chromUV = partitioning(data["ordinate_values"].to_numpy(), 1 / 10, 1 / 2)

chromMS = partitioning(np.load(path + "/Batch Processing/MS/MS3.npy"), 1 / 10, 1 / 2)

timeMS = partitioning(np.load(path + "/Batch Processing/MS/time_MS3.npy"), 1 / 10, 1 / 2)

timeUVnew = np.linspace(timeUV[0], timeUV[-1], 2000)

chromUV = np.interp(timeUVnew, timeUV, chromUV)

# Initialize chromatogram

chrom1 = Chromatogram(chromUV, None, chromMS, None, timeUVnew, timeMS)
chrom1.remove_blank()
chrom1.from_2D_to_1D()

# processing of chromatogram
chrom1.correct_baseline()
chrom1.remove_noise()
chrom1.detect_peaks(height_ratio=10, width=0, distance_ratio=400,remove_0_width=True)
chrom1.getMSspec()

# Chromatogram 3

data = pd.read_csv(path + "/Batch Processing/UV/UV8.csv")

timeUV = partitioning(data["time"].to_numpy(), 1 / 10, 1 / 2)

chromUV = partitioning(data["ordinate_values"].to_numpy(), 1 / 10, 1 / 2)

chromMS = partitioning(np.load(path + "/Batch Processing/MS/MS8.npy"), 1 / 10, 1 / 2)

timeMS = partitioning(np.load(path + "/Batch Processing/MS/time_MS8.npy"), 1 / 10, 1 / 2)

# Initialize chromatogram

chrom3 = Chromatogram(chromUV, None, chromMS, None, timeUV, timeMS)
chrom3.remove_blank()
chrom3.from_2D_to_1D()

# processing of chromatogram
chrom3.correct_baseline()
chrom3.remove_noise()
chrom3.detect_peaks(height_ratio=10, width=0, distance_ratio=400,remove_0_width=True)
chrom3.getMSspec()

# Chromatogram 5

data = pd.read_csv(path + "/Batch Processing/UV/UV10.csv")

timeUV = partitioning(data["time"].to_numpy(), 1 / 10, 1 / 2)

chromUV = partitioning(data["ordinate_values"].to_numpy(), 1 / 10, 1 / 2)

chromMS = partitioning(np.load(path + "/Batch Processing/MS/MS10.npy"), 1 / 10, 1 / 2)

timeMS = partitioning(np.load(path + "/Batch Processing/MS/time_MS10.npy"), 1 / 10, 1 / 2)

# Initialize chromatogram

chrom5 = Chromatogram(chromUV, None, chromMS, None, timeUV, timeMS)
chrom5.remove_blank()
chrom5.from_2D_to_1D()

# processing of chromatogram
chrom5.correct_baseline()
chrom5.remove_noise()
chrom5.detect_peaks(height_ratio=10, width=0, distance_ratio=400,remove_0_width=True)
chrom5.getMSspec()

chroms = [chrom1, chrom3, chrom5]

chrom3peak0,chrom3start0,chrom3end0 = chrom3.peaks[0].data_MS(chrom3.chrom_2D_MS,chrom3.time,chrom3.time_MS,chrom3.shift)
chrom5peak0,chrom5start0,chrom5end0 = chrom5.peaks[0].data_MS(chrom5.chrom_2D_MS,chrom5.time,chrom5.time_MS,chrom5.shift)

plt.figure()
plt.plot(np.max(chrom3peak0,axis=0))
plt.show()

plt.figure()
plt.plot(chrom1.peaks[0].spectra_MS[0],color='red')
plt.plot(chrom1.peaks[1].spectra_MS[0],color='blue')
plt.show()

peak_data = chrom3peak0
model = FraserSuzuki()
peak = chrom3.peaks[0]
spectra = np.asarray([chrom1.peaks[0].spectra_MS[0], chrom1.peaks[1].spectra_MS[0]])
concs, mse, _ = fit_peak_model(peak_data, model, peak,spectra,n_comps=2)
spectra, mse = spectra_from_concentrations(peak_data, concs)

norm_factors = np.max(spectra, axis=1)
spectra = (spectra.T / norm_factors).T
concs = (concs.T * norm_factors).T

plt.figure()
plt.plot(concs[0])
plt.plot(concs[1])
plt.plot(concs[0]+concs[1])
plt.plot(np.max(chrom3.chrom_2D_MS[:, chrom3start0: chrom3end0],axis=0))
plt.show()

plt.figure()
plt.plot(spectra[0],color='red')
plt.plot(spectra[1],color='blue')
plt.show()