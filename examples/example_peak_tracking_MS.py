import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.lines import lineStyles

from chromprocessor.classes import Chromatogram, Chromatograms
from chromprocessor.peak_models import FraserSuzuki
from examples.example_deconvolution_mocca import chromatogram

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

chroms = Chromatograms(chroms, True)

[retentionTimeTable, multi_peaks] = chroms.track_peaks(preprocessing=True, height_ratio=10, width=0, distance_ratio=400,
                                                       remove_0_width=True)

print(retentionTimeTable)

# Plot
plt.rcParams.update({'font.size': 11.5})
fig, axs = plt.subplots(3,1)
axs[0].plot(chrom1.time,chrom1.chrom_1D, linewidth=2)
colors = ["orange", "green", "blue","red", "magenta"]
i = 0
for peak in chrom1.peaks:
    axs[0].plot(chrom1.time[peak.all_maxima],chrom1.chrom_1D[peak.all_maxima], marker='*', linestyle='None', color=colors[i])
    i = i + 1

axs[1].plot(chrom3.time,chrom3.chrom_1D, linewidth=2)
colors = ["orange", "blue","red", "magenta"]
i = 0
for peak in chrom3.peaks:
    axs[1].plot(chrom3.time[peak.all_maxima],chrom3.chrom_1D[peak.all_maxima], marker='*', linestyle='None', color = colors[i])
    i = i + 1

axs[2].plot(chrom5.time,chrom5.chrom_1D, linewidth=2)
colors = ["green", "orange", "blue","red", "magenta"]
i = 0
for peak in chrom5.peaks:
    axs[2].plot(chrom5.time[peak.all_maxima],chrom5.chrom_1D[peak.all_maxima], marker='*', linestyle='None', color = colors[i])
    i = i + 1

plt.show()


#plot spectra
colors = ["orange", "green", "blue","red", "magenta"]
plt.rcParams.update({'font.size': 11.5})
fig, axs = plt.subplots(3,1)
j = 0
for g in range(len(chrom1.peaks)):
    group = g
    nb_cmps = len(chrom1.peaks[group].all_maxima)
    for i in range(nb_cmps):
        axs[0].plot(chrom1.peaks[group].spectra_MS[i], linewidth=1, color = colors[j])
        axs[0].tick_params(axis='x', labelbottom=False)

        j = j + 1

colors = ["orange", "blue","red", "magenta"]
j = 0
for g in range(len(chrom3.peaks)):
    group = g
    nb_cmps = len(chrom3.peaks[group].all_maxima)
    for i in range(nb_cmps):
        axs[1].plot(chrom3.peaks[group].spectra_MS[i], linewidth=1,color = colors[j])
        axs[1].tick_params(axis='x', labelbottom=False)
        j = j + 1

colors = ["green", "orange", "blue","red", "magenta"]
j = 0
for g in range(len(chrom5.peaks)):
    group = g
    nb_cmps = len(chrom5.peaks[group].all_maxima)
    for i in range(nb_cmps):
        axs[2].plot(chrom5.peaks[group].spectra_MS[i], linewidth=1,color = colors[j])
        axs[2].tick_params(axis='x', labelbottom=False)
        j = j + 1

plt.show()

chroms.deconvolute_missing_peaks()
print(chroms.compounds[1].retention_times)

plt.figure()
plt.plot(chroms.compounds[0].spectra_MS[1])
plt.plot(chroms.compounds[1].spectra_MS[1])
plt.show()