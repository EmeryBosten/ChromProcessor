import numpy as np
from matplotlib import pyplot as plt

from chromprocessor.classes import Chromatogram, Chromatograms
from chromprocessor.peak_models import BiGaussianTailing


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
        self.A = np.random.uniform(*self.A_range, size=(self.num_peaks,))

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
    fig, ax = plt.subplots()

    ax.plot(p_x, chroms[0], linewidth=1.0)
    #ax.plot(p_x, chroms_noisy[0] + 150, linewidth=2.0)

    ax.set(xlim=(0, time),
           ylim=(-5, 50))

    plt.show()

    return c, c_noisy, peaks, p_loc, p_scale, p_A, p_x

# load DAD spectra
from jcamp import jcamp_readfile
jcamp_dict_testosterone = jcamp_readfile("C:/Users/emery/OneDrive - KU Leuven/Python scripts/testosterone_UVVis.jdx")
jcamp_dict_paracetamol = jcamp_readfile("C:/Users/emery/OneDrive - KU Leuven/Python scripts/paracetamol_UVVis.jdx")
jcamp_dict_carbamazepine = jcamp_readfile("C:/Users/emery/OneDrive - KU Leuven/Python scripts/carbamazepine_UVVis.jdx")

plt.figure()
plt.plot(jcamp_dict_testosterone["x"], jcamp_dict_testosterone["y"])
plt.plot(jcamp_dict_paracetamol["x"], jcamp_dict_paracetamol["y"])
plt.plot(jcamp_dict_carbamazepine["x"], jcamp_dict_carbamazepine["y"])
plt.show()

# interpolation of DAD chromatograms
xvals = np.linspace(210, 310, 100)
yinterp_testosterone = np.interp(xvals, jcamp_dict_testosterone["x"], jcamp_dict_testosterone["y"])
yinterp_paracetamol = np.interp(xvals, jcamp_dict_paracetamol["x"], jcamp_dict_paracetamol["y"])
yinterp_carbamazepine = np.interp(xvals, jcamp_dict_carbamazepine["x"], jcamp_dict_carbamazepine["y"])

plt.figure()
plt.plot(xvals, yinterp_testosterone)
plt.plot(xvals, yinterp_paracetamol)
plt.plot(xvals, yinterp_carbamazepine)
plt.show()

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

plt.figure()
plt.plot(spec_MS_arr[0,:])
plt.plot(spec_MS_arr[1,:])
plt.plot(spec_MS_arr[2,:])
plt.show()

# Case 1 : UV deconvolution and peak tracking
# same height and width of peaks
# 1st chrom -> baseline separated
c, c_noisy, peaks_chrom1, p_loc, p_scale, p_A, p_x = generateChrom(time = 5,
resolution = 1000,
num_peaks = 3,
snr_range = (30, 30),
A_range = (15,15),
loc = [2, 3, 4],
scale = [0.05, 0.05, 0.05],
asymmetry_range = None)

# 2nd chrom -> partial overlap
c, c_noisy, peaks_chrom2, p_loc, p_scale, p_A, p_x = generateChrom(time = 5,
resolution = 1000,
num_peaks = 3,
snr_range = (30, 30),
A_range = (15,15),
loc = [3.1, 3, 4],
scale = [0.05, 0.05, 0.05],
asymmetry_range = None)

# 3rd chrom -> change in retention order

c, c_noisy, peaks_chrom3, p_loc, p_scale, p_A, p_x = generateChrom(time = 5,
resolution = 1000,
num_peaks = 3,
snr_range = (30, 30),
A_range = (15,15),
loc = [3.5, 3, 4],
scale = [0.033, 0.033, 0.033],
asymmetry_range = None)

# Generate DAD matrices
#load NK peaks
conc_arr1 = np.asarray(peaks_chrom1)
conc_arr2 = np.asarray(peaks_chrom2)
conc_arr3 = np.asarray(peaks_chrom3)
spec_UV_cmpA = yinterp_testosterone
spec_UV_cmpB = yinterp_paracetamol
spec_UV_cmpC = yinterp_carbamazepine
spec_UV_arr = np.array([spec_UV_cmpA, spec_UV_cmpB, spec_UV_cmpC])

mixture_dad1 = np.dot(np.transpose(conc_arr1), spec_UV_arr)
mixture_dad2 = np.dot(np.transpose(conc_arr2), spec_UV_arr)
mixture_dad3 = np.dot(np.transpose(conc_arr3), spec_UV_arr)

mixture_MS1 = np.dot(np.transpose(conc_arr1), spec_MS_arr)
mixture_MS2 = np.dot(np.transpose(conc_arr2), spec_MS_arr)
mixture_MS3 = np.dot(np.transpose(conc_arr3), spec_MS_arr)

# Deconvolution chrom2
chrom = Chromatogram(mixture_dad2.T, None, mixture_MS2.T, None, np.linspace(0,5,1000),np.linspace(0,5,1000))
chrom.remove_blank()
chrom.from_2D_to_1D()

# processing of chromatogram
chrom.detect_peaks()
chrom.deconvolve_peaks(BiGaussianTailing())
chrom.getMSspec()

#plot spectra
#select compound group

fig, axs = plt.subplots(2,1)
axs[0].plot(spec_MS_arr[0,:])
axs[0].plot(spec_MS_arr[1,:])
group = 0
nb_cmps = len(chrom.peaks[group].all_maxima)
for i in range(nb_cmps):
    axs[1].plot(chrom.peaks[group].spectra_MS[i])
plt.show()

fig, axs = plt.subplots(2,1)
axs[0].plot(spec_MS_arr[2,:])
group = 1
nb_cmps = len(chrom.peaks[group].all_maxima)
for i in range(nb_cmps):
    axs[1].plot(chrom.peaks[group].spectra_MS[i])
plt.show()

# Peak tracking

chrom1 = Chromatogram(mixture_dad1.T, None, mixture_MS1.T, None, np.linspace(0,5,1000),np.linspace(0,5,1000))
chrom2 = Chromatogram(mixture_dad2.T, None, mixture_MS2.T, None, np.linspace(0,5,1000),np.linspace(0,5,1000))
chrom3 = Chromatogram(mixture_dad3.T, None, mixture_MS3.T, None, np.linspace(0,5,1000),np.linspace(0,5,1000))


chroms = [chrom1, chrom2, chrom3]
chroms = Chromatograms(chroms,True)
[retentionTimeTable, multi_peaks] = chroms.track_peaks(preprocessing=False)

print(retentionTimeTable)
