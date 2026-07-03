from matplotlib import pyplot as plt
from mocca2 import example_data
from chromprocessor.classes import Chromatogram, Chromatograms
from chromprocessor.peak_models import BiGaussianTailing, FraserSuzuki,Bemg
import numpy as np

# Load example chromatogram
chromatogram = example_data.diterpene_esters()["chromatogram"].to_numpy()[-1]
chrom = Chromatogram(chromatogram.data[:,2500:-1], None, None, None, chromatogram.time[2500:-1],None)
#chromatogram = example_data.example_2()
#chrom = Chromatogram(chromatogram.data[:,500:-1], None, None, None, chromatogram.time[500:-1],None)
chrom.remove_blank()
chrom.from_2D_to_1D()

# processing of chromatogram
chrom.correct_baseline()
chrom.remove_noise()
plt.rcParams.update({'font.size': 11.5})
plt.plot(chrom.time,np.mean(chromatogram.data[:,2500:-1],axis = 0), linewidth=2, color = "blue")
plt.plot(chrom.time,chrom.chrom_1D, linewidth=2, color = "black")
plt.plot(chrom.time,chrom.baseline, linewidth=2, color = "red")


chrom.detect_peaks(remove_0_width=True)

# Plot
plt.rcParams.update({'font.size': 11.5})
plt.figure()
plt.plot(chrom.time,chrom.chrom_1D, linewidth=2)
for peak in chrom.peaks:
    plt.plot(chrom.time[peak.all_maxima],chrom.chrom_1D[peak.all_maxima], marker='*', linestyle='None')

chrom.deconvolve_peaks(FraserSuzuki())
colors = ["orange", "green", "blue","red"]
plt.rcParams.update({'font.size': 11.5})
plt.figure()
plt.plot(chrom.time,chrom.chrom_1D, linewidth=2)
for idx, peak in enumerate(chrom.peaks):
    for j in range(chrom.peaks[idx].concentrations.shape[0]):
        plt.plot(chrom.peaks[idx].time(chrom.time), chrom.peaks[idx].concentrations[j], linewidth=2,color = colors[j])

#plot spectra
colors = ["orange", "green", "blue","red"]
plt.figure()
for g in range(len(chrom.peaks)):
    group = g
    nb_cmps = len(chrom.peaks[group].all_maxima)
    for i in range(nb_cmps):
        plt.plot(np.arange(200,401,1),chrom.peaks[group].spectra[i], linewidth=2, color = colors[i])
plt.show()
