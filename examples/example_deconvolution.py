import numpy as np
from matplotlib import pyplot as plt

from chromprocessor.classes import Chromatogram, Chromatograms
from chromprocessor.peak_models import BiGaussianTailing, FraserSuzuki

from sklearn.metrics import r2_score
from numpy.linalg import norm


#functions to generate simulated chromatogram
#peak shape
def compute_gaussian_peak(x, A, loc, scale, asymmetry=None, eps=1e-7):
    if asymmetry is None:
        return A * np.exp((-(x - loc)**2) / (2 * scale**2))
    return A * np.exp(-1/2*((x - loc)/(eps + scale + asymmetry * (x - loc)))**2)

#noise
def apply_white_noise(x, A, snr, random_seed):
    '''
    Random sampling from a normal distribution to generate white noise.
    '''
    np.random.seed(random_seed)
    stddev = A.mean() / snr / 4 # approximately noise levels that match snr
    noise = np.random.normal(0, stddev, len(x))
    return x + noise

#simulator class
class Simulator:

    def __init__(self,
                 time,
                 resolution,
                 num_peaks,
                 snr_range,
                 A_range,
                 loc,
                 scale,
                 asymmetry_range,
                 ):

        self.time = time
        self.resolution = resolution
        self.x = np.linspace(0, time, resolution)
        self.num_peaks = num_peaks
        self.snr_range = snr_range
        self.A_range = A_range
        self.loc = loc
        self.scale = scale
        self.asymmetry_range = asymmetry_range


    def run(self, random_seed):

        np.random.seed(random_seed)

        # Randomly obtain parameters of the peaks
        self.A = self.A_range

        # Obtain asymmetry if not None (i.e. peak fronting/tailing)
        if self.asymmetry_range is not None:
            self.asymmetry = np.random.uniform(*self.asymmetry_range, size=(self.num_peaks,))
        else:
            self.asymmetry = [None] * self.num_peaks

        # Generate (noise-free) chromatogram
        chromatogram = np.zeros_like(self.x)
        peaks = []
        for a, m, s, t in zip(self.A, self.loc, self.scale, self.asymmetry):
            peaks.append(compute_gaussian_peak(self.x, a, m, s, t))
            chromatogram += compute_gaussian_peak(self.x, a, m, s, t)

        # Obtain random signal-to-noise ratio (SNR)
        self.snr = np.random.uniform(*self.snr_range)
        chromatogram_noisy = apply_white_noise(
                chromatogram, self.A, self.snr, random_seed)

        return (
            chromatogram,
            chromatogram_noisy,
            peaks,
            self.loc,
            self.scale,
            self.A,
            self.x
        )

# Case 1 : UV deconvolution and peak tracking
# same height and width of peaks
# 1st chrom -> baseline separated

def generateChrom(time,resolution,num_peaks, snr_range, A_range, loc, scale, asymmetry_range ):
    #initiate simulator
    simulator = Simulator(
        time=time, #add t min for highly retained peaks
        resolution=resolution,
        num_peaks=num_peaks,
        snr_range=snr_range,
        A_range=A_range,
        loc=loc,
        scale=scale,
        asymmetry_range=asymmetry_range  # Include asymmetry
    )

    # generate chromatograms

    # initialize lists
    chroms = []
    chroms_noisy = []

    # generate chromatograms with simulator
    c, c_noisy, peaks, p_loc, p_scale, p_A, p_x = simulator.run(1)
    chroms.append(c)
    chroms_noisy.append(c_noisy)

    # plot chromatogram
    #fig, ax = plt.subplots()

    #ax.plot(p_x, chroms[0], linewidth=1.0)
    #ax.plot(p_x, chroms_noisy[0] + 150, linewidth=2.0)

    #ax.set(xlim=(0, time),
    #       ylim=(-5, 50))

    #plt.show()

    return c, c_noisy, peaks, p_loc, p_scale, p_A, p_x

# load DAD spectra
from jcamp import jcamp_readfile
jcamp_dict_testosterone = jcamp_readfile("C:/Users/emery/OneDrive - KU Leuven/Python scripts/testosterone_UVVis.jdx")
jcamp_dict_paracetamol = jcamp_readfile("C:/Users/emery/OneDrive - KU Leuven/Python scripts/paracetamol_UVVis.jdx")
jcamp_dict_carbamazepine = jcamp_readfile("C:/Users/emery/OneDrive - KU Leuven/Python scripts/carbamazepine_UVVis.jdx")

#plt.figure()
#plt.plot(jcamp_dict_testosterone["x"], jcamp_dict_testosterone["y"])
#plt.plot(jcamp_dict_paracetamol["x"], jcamp_dict_paracetamol["y"])
#plt.plot(jcamp_dict_carbamazepine["x"], jcamp_dict_carbamazepine["y"])
#plt.show()

# interpolation of DAD chromatograms
xvals = np.linspace(210, 310, 100)
yinterp_testosterone = np.interp(xvals, jcamp_dict_testosterone["x"], jcamp_dict_testosterone["y"])
yinterp_paracetamol = np.interp(xvals, jcamp_dict_paracetamol["x"], jcamp_dict_paracetamol["y"])
yinterp_carbamazepine = np.interp(xvals, jcamp_dict_carbamazepine["x"], jcamp_dict_carbamazepine["y"])

#plt.figure()
#plt.plot(xvals, yinterp_testosterone)
#plt.plot(xvals, yinterp_paracetamol)
#plt.plot(xvals, yinterp_carbamazepine)
#plt.show()

#load MS spectra
spec_MS_cmpA = np.loadtxt('C:/Users/emery/OneDrive - KU Leuven/Python scripts/Testosterone_MS.txt')
spec_MS_cmpB = np.loadtxt('C:/Users/emery/OneDrive - KU Leuven/Python scripts/Paracetamol_MS.txt')
spec_MS_cmpC = np.loadtxt('C:/Users/emery/OneDrive - KU Leuven/Python scripts/Carbamazepine_MS.txt')

# homogenize
# 50 -> 300
spec_MS_arr = np.zeros((3,len(np.arange(50,300,1))))
for i in range(np.shape(spec_MS_arr)[1]):
    for j in range(np.shape(spec_MS_cmpA)[0]):
        if np.arange(50,300,1)[i] == round(spec_MS_cmpA[j][0],0):
            spec_MS_arr[0,i] = spec_MS_cmpA[j][2]/1000
    for k in range(np.shape(spec_MS_cmpB)[0]):
        if np.arange(50,300,1)[i] == round(spec_MS_cmpB[k][0],0):
            spec_MS_arr[1,i] = spec_MS_cmpB[k][2]/1000
    for l in range(np.shape(spec_MS_cmpC)[0]):
        if np.arange(50,300,1)[i] == round(spec_MS_cmpC[l][0],0):
            spec_MS_arr[2,i] = spec_MS_cmpC[l][2]/1000

#plt.figure()
#plt.plot(spec_MS_arr[0,:])
#plt.plot(spec_MS_arr[1,:])
#plt.plot(spec_MS_arr[2,:])
#plt.show()

# 1st chrom -> partial overlap
c, c_noisy, peaks_chrom1, p_loc, p_scale, p_A1, p_x1 = generateChrom(time = 5,
resolution = 1000,
num_peaks = 3,
snr_range = (30, 30),
A_range = np.asarray([15,15,20]),
loc = [2.8, 3.0, 3.2],
scale = [0.05, 0.05, 0.05],
asymmetry_range = None)

res1 = 2*0.2/4*(0.1)
print(res1)

# 2nd chrom -> complete overlap
c, c_noisy, peaks_chrom2, p_loc, p_scale, p_A2, p_x2 = generateChrom(time = 5,
resolution = 1000,
num_peaks = 3,
snr_range = (30, 30),
A_range = np.asarray([15,15,20]),
loc = [2.9, 3.0, 3.1],
scale = [0.05, 0.05, 0.05],
asymmetry_range = None)

res1 = 2*0.1/4*(0.1)
print(res1)


# Generate DAD matrices
#load NK peaks
conc_arr1 = np.asarray(peaks_chrom1) + np.random.normal(0, 0.005, np.asarray(peaks_chrom1).shape)
conc_arr2 = np.asarray(peaks_chrom2) + np.random.normal(0, 0.005, np.asarray(peaks_chrom2).shape)
spec_UV_cmpA = yinterp_testosterone
spec_UV_cmpB = yinterp_paracetamol
spec_UV_cmpC = yinterp_carbamazepine
spec_UV_arr = np.array([spec_UV_cmpA, spec_UV_cmpB, spec_UV_cmpC])

mixture_dad1 = np.dot(np.transpose(conc_arr1), spec_UV_arr)
mixture_dad2 = np.dot(np.transpose(conc_arr2), spec_UV_arr)

mixture_MS2 = np.dot(np.transpose(conc_arr2), spec_MS_arr)
mixture_MS1 = np.dot(np.transpose(conc_arr1), spec_MS_arr)

# Deconvolution chrom1
chrom = Chromatogram(mixture_dad1.T, None, mixture_MS1.T, None, np.linspace(0,5,1000),np.linspace(0,5,1000))
chrom.remove_blank()
chrom.from_2D_to_1D()

# processing of chromatogram
chrom.detect_peaks()
colors = ["orange", "green", "red"]
plt.rcParams.update({'font.size': 11.5})
plt.figure()
plt.plot(chrom.time,chrom.chrom_1D, linewidth=2)
i = 0
for peak in chrom.peaks:
    plt.plot(chrom.time[peak.all_maxima],chrom.chrom_1D[peak.all_maxima], marker='*', linestyle='None', color = colors[i])
    i = i + 1

chrom.deconvolve_peaks(FraserSuzuki())
colors = ["orange", "red", "green"]
plt.rcParams.update({'font.size': 11.5})
plt.figure()
plt.plot(chrom.time,chrom.chrom_1D, linewidth=2)
for idx, peak in enumerate(chrom.peaks):
    for j in range(chrom.peaks[idx].concentrations.shape[0]):
        plt.plot(chrom.peaks[idx].time(chrom.time), chrom.peaks[idx].concentrations[j], linewidth=2, color = colors[j])
chrom.getMSspec()

#plot spectra
#select compound group
plt.rcParams.update({'font.size': 11.5})
fig, axs = plt.subplots(2,1)
axs[0].plot(np.arange(50,300,1),spec_MS_arr[0,:]/np.linalg.norm(spec_MS_arr[0,:]), linewidth = 2, color = "orange")
axs[0].plot(np.arange(50,300,1),spec_MS_arr[1,:]/np.linalg.norm(spec_MS_arr[1,:]), linewidth = 2, color = "green")
axs[0].plot(np.arange(50,300,1),spec_MS_arr[2,:]/np.linalg.norm(spec_MS_arr[2,:]), linewidth = 2, color = "red")
group = 0
colors = ["green", "red", "orange"]
nb_cmps = len(chrom.peaks[group].all_maxima)
for i in range(nb_cmps):
    axs[1].plot(np.arange(50,300,1),chrom.peaks[group].spectra_MS[i]/np.linalg.norm(chrom.peaks[group].spectra_MS[i]), linewidth = 2, color = colors[i])
plt.show()

plt.rcParams.update({'font.size': 11.5})
fig, axs = plt.subplots(2,1)
axs[0].plot(xvals,spec_UV_arr[0,:]/np.linalg.norm(spec_UV_arr[0,:]), linewidth = 2, color = "orange")
axs[0].plot(xvals,spec_UV_arr[1,:]/np.linalg.norm(spec_UV_arr[1,:]), linewidth = 2, color = "green")
axs[0].plot(xvals,spec_UV_arr[2,:]/np.linalg.norm(spec_UV_arr[2,:]), linewidth = 2, color = "red")
group = 0
colors = ["orange", "red", "green"]
nb_cmps = len(chrom.peaks[group].all_maxima)
for i in range(nb_cmps):
    axs[1].plot(xvals,chrom.peaks[group].spectra[i]/np.linalg.norm(chrom.peaks[group].spectra[i]), linewidth = 2, color = colors[i])
plt.show()

print(r2_score(peaks_chrom1[0][chrom.peaks[0].left:chrom.peaks[0].right]*np.mean(spec_UV_arr[0]), chrom.peaks[0].concentrations[0]))
print(r2_score(peaks_chrom1[1][chrom.peaks[0].left:chrom.peaks[0].right]*np.mean(spec_UV_arr[1]), chrom.peaks[0].concentrations[2]))
print(r2_score(peaks_chrom1[2][chrom.peaks[0].left:chrom.peaks[0].right]*np.mean(spec_UV_arr[2]), chrom.peaks[0].concentrations[1]))

def cosine_similarity(A,B):
    # compute cosine similarity
    cosine = np.dot(A, B) / (norm(A) * norm(B))
    return cosine

print(cosine_similarity(spec_UV_arr[0],chrom.peaks[0].spectra[0]))
print(cosine_similarity(spec_UV_arr[1],chrom.peaks[0].spectra[2]))
print(cosine_similarity(spec_UV_arr[2],chrom.peaks[0].spectra[1]))

print(cosine_similarity(spec_MS_arr[0],chrom.peaks[0].spectra_MS[2]))
print(cosine_similarity(spec_MS_arr[1],chrom.peaks[0].spectra_MS[0]))
print(cosine_similarity(spec_MS_arr[2],chrom.peaks[0].spectra_MS[1]))

plt.figure()
plt.plot(peaks_chrom1[0][chrom.peaks[0].left:chrom.peaks[0].right]*np.mean(spec_UV_arr[0]))
plt.plot(chrom.peaks[0].concentrations[0])
plt.plot(chrom.chrom_1D[chrom.peaks[0].left:chrom.peaks[0].right])
plt.show()


# Deconvolution chrom2
chrom = Chromatogram(mixture_dad2.T, None, mixture_MS2.T, None, np.linspace(0,5,1000),np.linspace(0,5,1000))
chrom.remove_blank()
chrom.from_2D_to_1D()

# processing of chromatogram
chrom.detect_peaks()
colors = ["orange", "green", "red"]
plt.rcParams.update({'font.size': 11.5})
plt.figure()
plt.plot(chrom.time,chrom.chrom_1D, linewidth=2)
i = 0
for peak in chrom.peaks:
    plt.plot(chrom.time[peak.all_maxima],chrom.chrom_1D[peak.all_maxima], marker='*', linestyle='None', color = colors[i])
    i = i + 1

chrom.deconvolve_peaks(FraserSuzuki())
colors = ["red", "orange", "green"]
plt.rcParams.update({'font.size': 11.5})
plt.figure()
plt.plot(chrom.time,chrom.chrom_1D, linewidth=2)
for idx, peak in enumerate(chrom.peaks):
    for j in range(chrom.peaks[idx].concentrations.shape[0]):
        plt.plot(chrom.peaks[idx].time(chrom.time), chrom.peaks[idx].concentrations[j], linewidth=2, color = colors[j])
chrom.getMSspec()

#plot spectra
#select compound group
plt.rcParams.update({'font.size': 11.5})
fig, axs = plt.subplots(2,1)
axs[0].plot(np.arange(50,300,1),spec_MS_arr[0,:]/np.linalg.norm(spec_MS_arr[0,:]), linewidth = 2, color = "orange")
axs[0].plot(np.arange(50,300,1),spec_MS_arr[1,:]/np.linalg.norm(spec_MS_arr[1,:]), linewidth = 2, color = "green")
axs[0].plot(np.arange(50,300,1),spec_MS_arr[2,:]/np.linalg.norm(spec_MS_arr[2,:]), linewidth = 2, color = "red")
group = 0
colors = ["red", "orange", "green"]
nb_cmps = len(chrom.peaks[group].all_maxima)
for i in range(nb_cmps):
    axs[1].plot(np.arange(50,300,1),chrom.peaks[group].spectra_MS[i]/np.linalg.norm(chrom.peaks[group].spectra_MS[i]), linewidth = 2, color = colors[i])
plt.show()

plt.rcParams.update({'font.size': 11.5})
fig, axs = plt.subplots(2,1)
axs[0].plot(xvals,spec_UV_arr[0,:]/np.linalg.norm(spec_UV_arr[0,:]), linewidth = 2, color = "orange")
axs[0].plot(xvals,spec_UV_arr[1,:]/np.linalg.norm(spec_UV_arr[1,:]), linewidth = 2, color = "green")
axs[0].plot(xvals,spec_UV_arr[2,:]/np.linalg.norm(spec_UV_arr[2,:]), linewidth = 2, color = "red")
group = 0
colors = ["red", "orange", "green"]
nb_cmps = len(chrom.peaks[group].all_maxima)
for i in range(nb_cmps):
    axs[1].plot(xvals,chrom.peaks[group].spectra[i]/np.linalg.norm(chrom.peaks[group].spectra[i]), linewidth = 2, color = colors[i])
plt.show()

print(r2_score(peaks_chrom2[0][chrom.peaks[0].left:chrom.peaks[0].right]*np.mean(spec_UV_arr[0]), chrom.peaks[0].concentrations[1]))
print(r2_score(peaks_chrom2[1][chrom.peaks[0].left:chrom.peaks[0].right]*np.mean(spec_UV_arr[1]), chrom.peaks[0].concentrations[2]))
print(r2_score(peaks_chrom2[2][chrom.peaks[0].left:chrom.peaks[0].right]*np.mean(spec_UV_arr[2]), chrom.peaks[0].concentrations[0]))

print(cosine_similarity(spec_UV_arr[0],chrom.peaks[0].spectra[1]))
print(cosine_similarity(spec_UV_arr[1],chrom.peaks[0].spectra[2]))
print(cosine_similarity(spec_UV_arr[2],chrom.peaks[0].spectra[0]))

print(cosine_similarity(spec_MS_arr[0],chrom.peaks[0].spectra_MS[1]))
print(cosine_similarity(spec_MS_arr[1],chrom.peaks[0].spectra_MS[2]))
print(cosine_similarity(spec_MS_arr[2],chrom.peaks[0].spectra_MS[0]))

plt.figure()
plt.plot(peaks_chrom2[0][chrom.peaks[0].left:chrom.peaks[0].right]*np.mean(spec_UV_arr[0]))
plt.plot(chrom.peaks[0].concentrations[1])
plt.plot(chrom.chrom_1D[chrom.peaks[0].left:chrom.peaks[0].right])
plt.show()
