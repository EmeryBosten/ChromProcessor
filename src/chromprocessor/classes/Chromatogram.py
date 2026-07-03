from typing import List, Literal

import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt

from . import Peak
from . import PeakModel

# functions for baseline correction
from ..baseline_correction import erPLSrmse, airPLS, autoFlatFit, flatfit
from ..noise_removal import modoneNNestimator

# libraries for noise removal
try:
    import pybeads as be
except ModuleNotFoundError:
    import sys
    from pathlib import Path

    # Fallback to local vendored dependency when pybeads is not installed globally.
    _vendor_path = Path(__file__).resolve().parents[3] / "vendor" / "pybeads"
    if _vendor_path.exists():
        sys.path.insert(0, str(_vendor_path))
        import pybeads as be
    else:
        raise
# functions for peak detection
from ..peak_detection import findMin, peakStartEnd
# libraries for peak detection
from scipy.signal import peak_widths
from scipy.signal import peak_prominences
#functions for deconvolution
from ..deconvolution import merge_overlapping_peaks, fit_peak_model, spectra_from_concentrations, simplisma, \
    estimate_background_noise, expand_peaks
from ..peak_models import BiGaussianTailing


class Chromatogram:
    """  """
    peaks: List[Peak]
    """Peaks in the chromatogram"""
    signal_raw_UV: NDArray
    """Unprocessed UV signal"""
    blank_UV: NDArray
    """Blank UV signal"""
    signal_raw_MS: NDArray
    """Unprocessed MS signal"""
    blank_MS: NDArray
    """Blank MS signal"""
    time: NDArray
    """Timeframe of chromatogram"""
    time_MS: NDArray
    """Timeframe of MS chromatogram"""
    chrom_2D: NDArray
    """2 dimensional chromatogram"""
    chrom_2D_MS: NDArray
    """2 dimensional MS chromatogram"""
    chrom_1D: NDArray
    """1 dimensional chromatogram"""
    baseline: NDArray
    """Baseline of 1 dimensional chromatogram"""
    shift : float
    """shift from UV to MS data"""

    def __init__(self, signal_raw_UV: NDArray, blank_UV: NDArray,signal_raw_MS: NDArray, blank_MS: NDArray, time: NDArray, time_MS: NDArray):
        self.signal_raw_UV = signal_raw_UV
        self.blank_UV = blank_UV
        self.signal_raw_MS = signal_raw_MS
        self.blank_MS = blank_MS
        self.time = time
        self.time_MS = time_MS
        if signal_raw_UV.ndim == 1:
            self.chrom_1D = signal_raw_UV
        else :
            self.chrom_2D = signal_raw_UV

    def remove_blank(self):
        """Remove blank of unprocessed signal(s)"""
        if self.signal_raw_UV.ndim == 2:
            if self.blank_UV is None:
                self.chrom_2D = self.signal_raw_UV
            else :
                self.chrom_2D = self.signal_raw_UV - self.blank_UV
        if self.signal_raw_UV.ndim == 1:
            if self.blank_UV is None:
                self.chrom_1D = self.signal_raw_UV
            else :
                self.chrom_1D = self.signal_raw_UV - self.blank_UV
        if self.signal_raw_MS is not None and self.blank_MS is not None:
            self.chrom_2D_MS = self.signal_raw_MS - self.blank_MS
        if self.signal_raw_MS is not None and self.blank_MS is None:
            self.chrom_2D_MS = self.signal_raw_MS

    def from_2D_to_1D(self):
        """Transform 2 dimensional to 1 dimensional chromatogram"""
        if self.signal_raw_UV.ndim == 2:
            self.chrom_1D = np.mean(self.chrom_2D, axis=0)

    def correct_baseline(self, base_lambdas: NDArray = np.logspace(1, 12, 10), d: int = 2, n: int = 20, method: Literal[ "erpls", "flatfit"] = "erpls", p = 0.03):
        """
        Correct baseline of chromatogram

        Parameters

        base_lambdas: NDArray
            hyperparameter values to test out

        d: int
            difference order
        n: int
            number of iterations

        """

        err = np.zeros(len(base_lambdas))  # preallocate error array

        if method == "erpls":

            # Compute errors for each lambda
            for i, lam in enumerate(base_lambdas):
                err[i] = erPLSrmse(self.chrom_1D, lam, d, n, plot=False)

            # Find minimum error and corresponding index
            #m = np.min(err)
            ind = np.argmin(err)

            # Perform baseline correction using the optimal lambda
            z = airPLS(self.chrom_1D, base_lambdas[ind], d, n)

        else:

            # Compute errors for each lambda
            for i, lam in enumerate(base_lambdas):
                err[i] = autoFlatFit(self.chrom_1D, lam, p, plot=False)

            # Find minimum error and corresponding index
            # m = np.min(err)
            ind = np.argmin(err)

            # Perform baseline correction using the optimal lambda
            z = flatfit(self.chrom_1D, base_lambdas[ind], p = 0.03)

        # Subtract the estimated baseline from the signal
        y_baseline_corrected = self.chrom_1D - z

        # Plot the baseline-corrected signal
        plt.figure()
        plt.plot(self.time, self.chrom_1D, label="signal")
        plt.title("Baseline correction")
        plt.xlabel("Time")
        plt.ylabel("Intensity")
        plt.plot(self.time, z, label="estimated baseline")
        plt.plot(self.time, y_baseline_corrected, label="corrected signal")
        plt.legend()
        plt.show()

        self.chrom_1D = y_baseline_corrected
        self.baseline = z

    def remove_noise(self, fc: float = 0.006, d: int = 1, r: int = 6, alphas: NDArray = np.logspace(-6, 1, 20),
                     nit: int = 20, pen: str = 'L1_v2'):
        """
        Remove noise of chromatogram

        Parameters

        fc = 0.005
        d = 1
        r = 1
        alphas = np.logspace(-6,6,20)
        nit = 20
        pen = 'L1_v1'
        """

        # noise estimate of chrom
        # Median absolute deviation
        #differences_y = np.diff(self.chrom_1D)
        #k = 1.4826
        #v = k * np.median(np.absolute(differences_y - np.median(differences_y)))

        #noise estimate of chrom
        # residual variance by modified k nearest neighbor

        v = modoneNNestimator(self.chrom_1D, np.linspace(0,len(self.chrom_1D),len(self.chrom_1D)), type_='median')
        err = np.zeros(len(alphas))  # preallocate error array

        # Compute MAD of residuals for each hyperparameter value
        i = 0
        for alpha in alphas:
            lam0 = 1 * alpha
            lam1 = 10 * alpha
            lam2 = 8 * alpha
            f, b, cost = be.beads(self.chrom_1D, d, fc, r, nit, lam0, lam1, lam2, pen, conv=3)
            #err[i] = (np.median(np.absolute((self.chrom_1D - (f + b)) - np.median((self.chrom_1D - (f + b))))))
            err[i] = (sum((self.chrom_1D-(f+b))**2))/len(self.chrom_1D)
            i = i + 1

        # Choose lambda
        ind = np.argmin(np.abs(err - v))
        alpha = alphas[ind]
        lam0 = 1 * alpha
        lam1 = 10* alpha
        lam2 = 8 * alpha
        z, b, cost = be.beads(self.chrom_1D, d, fc, r, nit, lam0, lam1, lam2, pen, conv=3)

        # Plot
        plt.figure()
        plt.plot(self.time, self.chrom_1D)
        #plt.title("Noise removal")
        #plt.xlabel("Time")
        #plt.ylabel("Intensity")
        plt.plot(self.time, z + b)
        #plt.legend()
        plt.show()

        self.chrom_1D = z + b

    def detect_peaks(self, noise_thresh: int = 2, height_ratio: int = 25, distance_ratio = 400, width = 3, wlen: int = 250, remove_0_width = False):

        """
        detect peaks in chromatogram

        Parameters

        noise_thresh = 5
        height_ratio = 50
        distance_ratio = 300
        width = 5
        wlen = 250

        """
        dz2 = np.diff(self.chrom_1D, 2)

        plt.figure()
        plt.plot(dz2)
        plt.show()

        # locate peaks
        distance = len(self.chrom_1D)/distance_ratio

        apex_loc = findMin(dz2, self.chrom_1D, noise_thresh, height_ratio, distance, width)

        left, right = peakStartEnd(self.chrom_1D, apex_loc)
        res = (self.chrom_1D - self.chrom_1D.min()) / (self.chrom_1D.max() - self.chrom_1D.min())

        # peak widths
        diff = np.asarray(right) - np.asarray(left)
        wlen = max(diff)
        prominences = peak_prominences(res, apex_loc, wlen)
        _, _, left, right = peak_widths(res, apex_loc, rel_height=1, prominence_data=prominences)

        colors = [
            "#1f77b4",
            "#ff7f0e",
            "#2ca02c",
            "#d62728",
            "#9467bd",
            "#8c564b",
            "#e377c2",
            "#7f7f7f",
            "#bcbd22",
            "#17becf",
        ]

        # Find way of precise peak width estimation
        # too narrow !
        # using derivatives ?

        # Create instances of Peak
        peaks = []
        loc = []
        for idx in range(len(apex_loc)):
            if remove_0_width :
                if prominences[0][idx] != 0: # remove peaks with prominence = 0 (too stringent ?)
                    peak = Peak(left=int(left[idx]),
                                right=int(right[idx]),
                                maximum=apex_loc[idx],
                                height=self.chrom_1D[apex_loc][idx],
                                prominence=prominences[0][idx])
                    loc.append(apex_loc[idx])
                    peaks.append(peak)
            else :
                peak = Peak(left=int(left[idx]),
                            right=int(right[idx]),
                            maximum=apex_loc[idx],
                            height=self.chrom_1D[apex_loc][idx],
                            prominence=prominences[0][idx])
                loc.append(apex_loc[idx])
                peaks.append(peak)

        self.peaks = peaks

        plt.figure()
        plt.clf()
        plt.plot(self.time,self.chrom_1D)
        plt.plot(self.time[loc], self.chrom_1D[loc], marker='*', linestyle='None')
        #plt.xlabel('Time')
        #plt.ylabel('Absorbance')
        #plt.title('Peak Detection')

        for i in range(len(loc)):
            plt.hlines(y=0, xmin=self.time[int(peaks[i].left)], xmax=self.time[int(peaks[i].right)], colors=colors[i % len(colors)])

        plt.show()

        return len(loc)

    def deconvolve_peaks(  self,model: PeakModel):
        """
        Deconvolves peaks with increasingly more components until MSE limit is reached. See `deconvolve_adaptive()` for details.

        Parameters
        ----------
        model: PeakModel | Literal['BiGaussian', 'BiGaussianTailing', 'FraserSuzuki']
            mathematical model used for fitting shapes of components of peaks

        Returns
        -------
        Chromatogram
            Returns self
        """
        # merge peaks into groups
        merged_peaks = merge_overlapping_peaks(self.chrom_1D, self.peaks)
        self.peaks = merged_peaks

        # extend boundaries of merged peaks
        # needs a bit of fine-tuning
        background_std = estimate_background_noise(self.chrom_1D, self.peaks)
        self.peaks = expand_peaks(self.chrom_1D, self.peaks, background_std)

        plt.figure()
        plt.plot(self.time, self.chrom_1D)

        for idx, peak in enumerate(self.peaks):
            # prepare input data
            peak_data = peak.data(self.chrom_2D)

            #max_comps = len(peak.all_maxima)

            # Deconvolve peak with fixed number of components
            concs, mse, _ = fit_peak_model(peak_data, model, peak)
            spectra, mse = spectra_from_concentrations(peak_data, concs)

            # Normalize the spectra and scale concentrations accordingly
            norm_factors = np.mean(spectra, axis=1) + 1e-7
            spectra = (spectra.T / norm_factors).T
            concs = (concs.T * norm_factors).T

            # save deconvolved peak
            self.peaks[idx].concentrations = concs
            self.peaks[idx].spectra = spectra

            for j in range(self.peaks[idx].concentrations.shape[0]):
                plt.plot(self.peaks[idx].time(self.time), self.peaks[idx].concentrations[j])
        plt.show()

    def UVtoMSshift(self):
        loc_main_peak_MS = self.time_MS[np.argmax(np.max(self.chrom_2D_MS, axis=0))]
        loc_main_peak_UV = self.time[np.argmax(self.chrom_1D)]
        self.shift = loc_main_peak_MS - loc_main_peak_UV

    def getMSspec(self):

        def find_nearest_index(array, values, return_each=True):

            """

            Find nearest index(es) in `array` to `values`.



            Parameters

            - array: 1-D array-like of numeric values

            - values: scalar or iterable of target values

            - return_each: if True, return a numpy array of indices (one per value).

                           if False, return a single index: the array index

                           whose element is closest to any of the given values.



            Returns

            - int (when return_each=False) or numpy.ndarray of ints (when return_each=True)

            """

            array = np.asarray(array)

            # handle empty array

            if array.size == 0:
                raise ValueError("`array` must contain at least one element")

            # Normalize values to 1-D array

            if np.isscalar(values):

                vals = np.atleast_1d(values)

            else:

                vals = np.asarray(values)

                if vals.ndim == 0:

                    vals = np.atleast_1d(vals)

                elif vals.ndim > 1:

                    vals = vals.ravel()

            # Compute absolute differences between array elements and each value

            # shape will be (array.size, vals.size)

            diffs = np.abs(array[:, None] - vals[None, :])

            if return_each:

                # For each value, find the index of the closest array element

                # axis=0 over array elements, so result length = vals.size

                indices = diffs.argmin(axis=0)

                return indices

            else:

                # For each array element, find the minimal distance to any value

                # then pick the array index with the smallest of those distances

                min_dist_per_array_elem = diffs.min(axis=1)

                closest_index = int(min_dist_per_array_elem.argmin())

                return closest_index

        self.UVtoMSshift()

        for idx, peak in enumerate(self.peaks):

            # method 1 : spectra at peak maxima
            #maxima_UV = self.time[np.asarray(peak.all_maxima)] + self.shift
            #maxima_MS = find_nearest_index(self.time_MS, maxima_UV)
            #spectra = self.chrom_2D_MS[:, maxima_MS].T

            #method 2 : simplisma
            start_time_UV = self.time[peak.left] + self.shift
            end_time_UV = self.time[peak.right] + self.shift
            start_loc_MS = find_nearest_index(self.time_MS, start_time_UV)
            end_loc_MS = find_nearest_index(self.time_MS, end_time_UV)
            peak_data = self.chrom_2D_MS[:, start_loc_MS[0]: end_loc_MS[0]]
            spectra, initial_concs = simplisma(peak_data, len(peak.all_maxima), 5)

            # Normalize the spectra and scale concentrations accordingly
            #norm_factors = np.mean(spectra, axis=1) + 1e-7
            #spectra = (spectra.T / norm_factors).T

            # save spectra peak
            self.peaks[idx].spectra_MS = spectra

