from matplotlib import pyplot as plt
from parse import parse_empower
from classes import Chromatogram
from peak_models import BiGaussianTailing

#Load data
#chromatogram Menin project

path = "C:/Users/emery/OneDrive - KU Leuven/Chromatograms/Menin/experiment 4 s1.5 short"
time,chrom = parse_empower(path + "/Mix.arw")
time_blank, blank = parse_empower(path + "/Blank.arw")

#Initialize chromatogram
chrom = Chromatogram.Chromatogram(chrom, blank, time)
chrom.remove_blank()
chrom.from_2D_to_1D()

# processing of chromatogram
chrom.correct_baseline()
chrom.remove_noise()
chrom.detect_peaks()
chrom.deconvolve_peaks(BiGaussianTailing)

#plot spectra
#select compound group
group = 2
nb_cmps = len(chrom.peaks[group].all_maxima)
plt.figure()
for i in range(nb_cmps):
    plt.plot(chrom.peaks[group].spectra[i])
plt.show()

# plot concentration profiles
group = 2
nb_cmps = len(chrom.peaks[group].all_maxima)
plt.figure()
for i in range(nb_cmps):
    plt.plot(chrom.peaks[group].concentrations[i])

plt.show()