from __future__ import annotations
from typing import List, Literal, Dict, Callable, Any

import matplotlib.axes
import numpy as np
import matplotlib

from ChromProcessor.classes import Data2D, Peak, DeconvolvedPeak, Component, Compound
from mocca2 import parsers
from mocca2.baseline import estimate_baseline
from mocca2.peaks import find_peaks
from mocca2.deconvolution.deconvolve import deconvolve_adaptive
from mocca2.deconvolution.fit_peak_model import fit_peak_model
from mocca2.deconvolution.nonnegative_lstsq import concentrations_from_spectra
from mocca2.deconvolution.peak_models import (
    PeakModel,
    Bemg,
    FraserSuzuki,
    BiGaussian,
    BiGaussianTailing,
)
from mocca2.serializing import dict_encoder


# libraries for noise removal
import pybeads as be

# libraries for peak detection
from scipy.signal import peak_widths
from scipy.signal import peak_prominences

class Chromatogram():
    """  """
    peaks: List[Peak]
    """Peaks in the chromatogram"""
    signal_raw: NDArray
    """Unprocessed signal"""
    blank: NDArray
    """Blank signal"""
    time: NDArray
    """Timeframe of chromatogram"""
    chrom_2D: NDArray
    """2 dimensional chromatogram"""
    chrom_1D: NDArray
    """1 dimensional chromatogram"""
    baseline: NDArray
    """Baseline of 1 dimensional chromatogram"""

    def __init__(self, signal_raw: NDArray, blank: NDArray, time: NDArray):
        self.signal_raw = signal_raw
        self.blank = blank
        self.time = time

    def remove_blank(self):
        """Remove blank of unprocessed signal"""
        self.chrom_2D = self.signal_raw - self.blank

    def from_2D_to_1D(self):
        """Transform 2 dimensional to 1 dimensional chromatogram"""
        if self.chrom_2D.shape[1] == 1:
            self.chrom_1D = self.chrom_2D
        else:
            self.chrom_1D = np.mean(self.chrom_2D, axis=0)

    def correct_baseline(self, base_lambdas: NDArray = np.logspace(1, 12, 10), d: int = 2, n: int = 20):
        """
        Correct baseline of chromatogram

        Parameters

        base_lambdas: NDArray
            hyperpameter values to test out

        d: int
            difference order
        n: int
            number of iterations

        """

        err = np.zeros(len(base_lambdas))  # preallocate error array

        # Compute errors for each lambda
        for i, lam in enumerate(base_lambdas):
            err[i] = erPLSrmse(self.chrom_1D, lam, d, n, plot=False)

        # Find minimum error and corresponding index
        m = np.min(err)
        ind = np.argmin(err)

        # Perform baseline correction using the optimal lambda
        z = airPLS(self.chrom_1D, base_lambdas[ind], d, n)

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

    def remove_noise(self, fc: float = 0.005, d: int = 1, r: int = 1, alphas: NDArray = np.logspace(-5, 5, 10),
                     Nit: int = 20, pen: str = 'L1_v1'):
        """
        Remove noise of chromatogram

        Parameters

        fc = 0.005
        d = 1
        r = 1
        alphas = np.logspace(-3,3,10)
        Nit = 20
        pen = 'L1_v1'
        """

        # noise estimate of chrom
        # Median absolute deviation
        differences_y = np.diff(self.chrom_1D)
        V = np.median(np.absolute(differences_y - np.median(differences_y)))

        err = np.zeros(len(alphas))  # preallocate error array

        # Compute MAD of residuals for each hyperparameter value
        i = 0
        for alpha in alphas:
            lam0 = 0.1 * alpha;
            lam1 = 5 * alpha;
            lam2 = 5 * alpha;
            f, b, cost = be.beads(self.chrom_1D, d, fc, r, Nit, lam0, lam1, lam2, pen, conv=None)
            err[i] = np.median(np.absolute((self.chrom_1D - (f + b)) - np.median((self.chrom_1D - (f + b)))))
            i = i + 1
            print(i / len(alphas))

        # Choose lambda
        ind = np.argmin(abs(err - V))
        alpha = alphas[ind]
        lam0 = 0.1 * alpha;
        lam1 = 5 * alpha;
        lam2 = 5 * alpha;
        z, b, cost = be.beads(self.chrom_1D, d, fc, r, Nit, lam0, lam1, lam2, pen, conv=None)

        self.chrom_1D = z

        # Plot
        plt.figure()
        plt.plot(self.time, np.mean(self.signal_raw, axis=0), label="signal")
        plt.title("Noise removal")
        plt.xlabel("Time")
        plt.ylabel("Intensity")
        plt.plot(self.time, self.chrom_1D, label="corrected signal")
        plt.legend()
        plt.show()

    def detect_peaks(self, noise_thresh: int = 2, height_ratio: int = 50, minimum_depth: float = 0.01, wlen: int = 250):

        """
        detect peaks in chromatogram

        Parameters

        noise_thresh = 2
        height_ratio = 50
        minimum_depth = 0.01
        wlen = 250

        """
        dz2 = np.diff(self.chrom_1D, 2);

        # locate peaks
        apex_loc = findMin(dz2, self.chrom_1D, noise_thresh, height_ratio, minimum_depth)
        startpoint, endpoint = peakStartEnd(self.chrom_1D, apex_loc)
        res = (self.chrom_1D - self.chrom_1D.min()) / (self.chrom_1D.max() - self.chrom_1D.min())

        # peak widths
        # wlen needs to be dependent on number of datapoints
        diff = np.asarray(endpoint) - np.asarray(startpoint)
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

        # Create instances of Peak
        peaks = []
        loc = []
        for idx in range(len(apex_loc)):
            # if prominences[0][idx] != 0: # remove peaks with prominence = 0 (too stringent ?)
            peak = Peak(
                left=int(left[idx]),
                right=int(right[idx]),
                maximum=apex_loc[idx],
                height=self.chrom_1D[apex_loc][idx],
                prominence=prominences[0][idx]
            )
            loc.append(apex_loc[idx])
            peaks.append(peak)

        self.peaks = peaks

        plt.figure()
        plt.clf()
        plt.plot(self.chrom_1D)
        plt.plot(loc, self.chrom_1D[loc], marker='*', linestyle='None')
        plt.xlabel('Time')
        plt.ylabel('Absorbance')
        plt.title('Peak Detection')

        for i in range(len(loc)):
            plt.hlines(y=0, xmin=left[i], xmax=right[i], colors=colors[i % len(colors)])

        plt.show()

        print(len(loc))

    def deconvolve_peaks(
            self,
            model: (
                    PeakModel
            )
    ):
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
        # merge peaks into groupq
        merged_peaks = merge_overlapping_peaks(self.chrom_1D, self.peaks)
        self.peaks = merged_peaks
        print(len(self.peaks))

        plt.figure()
        plt.plot(self.time, self.chrom_1D)

        for idx, peak in enumerate(self.peaks):
            # prepare input data
            peak_data = peak.data(self.chrom_2D)

            max_comps = len(peak.all_maxima)

            # Deconvolve peak with fixed umber of components
            concs, mse, _ = fit_peak_model(peak_data, model, peak, n_compounds=max_comps)
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
