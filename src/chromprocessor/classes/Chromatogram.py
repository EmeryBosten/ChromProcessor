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
    """ Container and processing interface for a single chromatographic run.

    A `Chromatogram` stores raw UV/DAD and optional MS data, applies common
    preprocessing steps, detects chromatographic peaks, and stores peak-level
    deconvolution and spectral information.

    The class supports both:

    - 1D UV chromatograms, where the signal is already a single intensity trace.
    - 2D UV/DAD chromatograms, where rows typically represent spectral channels
      and columns represent time points.

    Typical processing workflow
    ---------------------------
    1. Subtract blanks with :meth:`remove_blank`.
    2. Convert 2D UV/DAD data to a representative 1D trace with
       :meth:`from_2D_to_1D`.
    3. Optionally correct the baseline with :meth:`correct_baseline`.
    4. Optionally remove noise with :meth:`remove_noise`.
    5. Detect peaks with :meth:`detect_peaks`.
    6. Deconvolve overlapping UV peaks with :meth:`deconvolve_peaks`.
    7. Extract corresponding MS spectra with :meth:`getMSspec`.

    Notes
    -----
    The class mutates its internal state in-place. For example,
    `correct_baseline()` replaces `self.chrom_1D` by the baseline-corrected
    chromatogram and stores the estimated baseline in `self.baseline`.
    """
    peaks: List[Peak]
    """Detected or merged Peaks in the chromatogram"""
    signal_raw_UV: NDArray
    """Raw UV or DAD signal before blank substraction"""
    blank_UV: NDArray
    """UV or DAD blank signal to subtract from signal_raw_UV"""
    signal_raw_MS: NDArray
    """URaw MS signal before blank subtraction"""
    blank_MS: NDArray
    """MS blank signal to subtract from signal_raw_MS"""
    time: NDArray
    """Time axis corresponding to the UV/DAD chromatogram"""
    time_MS: NDArray
    """Time axis corresponding to the MS chromatogram"""
    chrom_2D: NDArray
    """Blank-corrected 2D UV/DAD chromatogram with shape (channels, time)"""
    chrom_2D_MS: NDArray
    """Blank-corrected 2D MS chromatogram with shape (m/z channels, time)"""
    chrom_1D: NDArray
    """One-dimensional chromatographic intensity trace"""
    baseline: NDArray
    """Estimated baseline of chrom_1D"""
    shift: float
    """Estimated retention-time shift from UV/DAD data to MS data"""

    def __init__(self, signal_raw_UV: NDArray, blank_UV: NDArray,signal_raw_MS: NDArray, blank_MS: NDArray, time: NDArray, time_MS: NDArray):
        """
        Initialize a chromatogram from raw UV/DAD and optional MS data.

        Parameters
        ----------
        signal_raw_UV : NDArray
            Raw UV or DAD signal. A 1D array is interpreted as an already
            integrated chromatogram. A 2D array is interpreted as spectral data
            over time, usually with shape `(channels, time)`.
        blank_UV : NDArray or None
            UV/DAD blank signal. If provided, it must be broadcast-compatible
            with `signal_raw_UV`.
        signal_raw_MS : NDArray or None
            Raw MS signal, usually with shape `(m/z channels, time)`.
        blank_MS : NDArray or None
            MS blank signal. If provided, it must be broadcast-compatible with
            `signal_raw_MS`.
        time : NDArray
            Time axis for the UV/DAD data.
        time_MS : NDArray
            Time axis for the MS data.

        Notes
        -----
        Blank subtraction is not performed during initialization. Call
        :meth:`remove_blank` before downstream processing when blank correction
        is needed.
        """
        self.signal_raw_UV = signal_raw_UV
        self.blank_UV = blank_UV
        self.signal_raw_MS = signal_raw_MS
        self.blank_MS = blank_MS
        self.time = time
        self.time_MS = time_MS
        # Store the raw UV signal in the appropriate representation.
        # For 2D data, `chrom_2D` will later be overwritten by the
        # blank-corrected signal in `remove_blank()`.
        if signal_raw_UV.ndim == 1:
            self.chrom_1D = signal_raw_UV
        else :
            self.chrom_2D = signal_raw_UV
        # Processed 2D data are initialized as unavailable until
        # blank correction has been applied.
        self.chrom_2D = None
        self.chrom_2D_MS = None

    def remove_blank(self):
        """
        Subtract blank measurements from raw UV/DAD and MS signals.

        If no blank is available for a signal type, the raw signal is copied to
        the corresponding processed chromatogram attribute.

        Updates
        -------
        self.chrom_1D : NDArray
            Blank-corrected 1D UV trace when `signal_raw_UV` is 1D.
        self.chrom_2D : NDArray
            Blank-corrected 2D UV/DAD matrix when `signal_raw_UV` is 2D.
        self.chrom_2D_MS : NDArray
            Blank-corrected 2D MS matrix when MS data are available.
        """
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
        """
        Collapse a 2D UV/DAD chromatogram into a 1D chromatographic trace.

        The current implementation averages all spectral channels at each time
        point. This creates a representative total UV/DAD intensity trace that
        can be used for baseline correction, noise removal, and peak detection.

        Updates
        -------
        self.chrom_1D : NDArray
            Mean chromatographic signal across the spectral/channel axis.
        """
        if self.signal_raw_UV.ndim == 2:
            self.chrom_1D = np.mean(self.chrom_2D, axis=0)

    def correct_baseline(self, base_lambdas: NDArray = np.logspace(1, 12, 10), d: int = 2, n: int = 20, method: Literal[ "erpls", "flatfit"] = "erpls", p = 0.03):
        """
        Estimate and subtract the chromatographic baseline from `chrom_1D`.

        Several smoothing/regularization strengths are evaluated, and the one
        with the lowest error according to the selected method is used for the
        final baseline estimate.

        Parameters
        ----------
        base_lambdas : NDArray, default=np.logspace(1, 12, 10)
            Candidate smoothness parameters. Larger values generally produce
            smoother baseline estimates.
        d : int, default=2
            Difference order used by the baseline correction algorithm.
        n : int, default=20
            Number of iterations used by the iterative baseline correction
            routine.
        method : {"erpls", "flatfit"}, default="erpls"
            Baseline correction strategy.

            - `"erpls"` evaluates candidates with `erPLSrmse` and estimates the
              final baseline with `airPLS`.
            - `"flatfit"` evaluates candidates with `autoFlatFit` and estimates
              the final baseline with `flatfit`.
        p : float, default=0.03
            Asymmetry or weighting parameter used by the FlatFit method.

        Updates
        -------
        self.chrom_1D : NDArray
            Baseline-corrected chromatographic signal.
        self.baseline : NDArray
            Estimated baseline that was subtracted from the signal.

        Notes
        -----
        This method displays a diagnostic plot showing the original signal,
        estimated baseline, and corrected chromatogram.
        """

        err = np.zeros(len(base_lambdas))  # preallocate error array

        if method == "erpls":

            # Evaluate each lambda by measuring how well it explains baseline
            # variation without overfitting chromatographic peaks.
            for i, lam in enumerate(base_lambdas):
                err[i] = erPLSrmse(self.chrom_1D, lam, d, n, plot=False)

            # Select the smoothing parameter with the lowest estimated error.
            ind = np.argmin(err)

            # Compute the final baseline with the selected parameter.
            z = airPLS(self.chrom_1D, base_lambdas[ind], d, n)

        else:

            # Evaluate each lambda using the flatness-based criterion.
            for i, lam in enumerate(base_lambdas):
                err[i] = autoFlatFit(self.chrom_1D, lam, p, plot=False)

            # Select the smoothing parameter with the lowest estimated error.
            ind = np.argmin(err)

            # Compute the final baseline with the selected parameter.
            z = flatfit(self.chrom_1D, base_lambdas[ind], p = 0.03)

        # Subtract the estimated baseline from the observed chromatogram.
        y_baseline_corrected = self.chrom_1D - z

        # Plot the baseline-correction result for visual inspection.
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
        Remove high-frequency noise from the 1D chromatogram using BEADS.

        The method first estimates the residual noise level of the signal.
        It then evaluates several BEADS regularization strengths and selects
        the one whose residual variance is closest to the estimated noise.

        Parameters
        ----------
        fc : float, default=0.006
            Cut-off frequency used by BEADS. Lower values enforce smoother
            signals.
        d : int, default=1
            Difference order used in the BEADS penalty.
        r : int, default=6
            Asymmetry ratio parameter used by BEADS.
        alphas : NDArray, default=np.logspace(-6, 1, 20)
            Candidate regularization strengths.
        nit : int, default=20
            Maximum number of BEADS iterations.
        pen : str, default="L1_v2"
            Penalty type passed to BEADS.

        Updates
        -------
        self.chrom_1D : NDArray
            Denoised chromatographic signal.

        Notes
        -----
        This method displays a diagnostic plot comparing the original signal
        with the denoised result.
        """

        # Median absolute deviation method (currently not applied)
        #differences_y = np.diff(self.chrom_1D)
        #k = 1.4826
        #v = k * np.median(np.absolute(differences_y - np.median(differences_y)))

        # Estimate the noise level using a modified one-nearest-neighbor
        # estimator. This gives a target residual variance for model selection.
        v = modoneNNestimator(self.chrom_1D, np.linspace(0,len(self.chrom_1D),len(self.chrom_1D)), type_='median')
        err = np.zeros(len(alphas))  # Residual variance for each candidate alpha.

        # Evaluate BEADS for each regularization strength. The lambdas control
        # signal sparsity, derivative sparsity, and baseline smoothness.
        i = 0
        for alpha in alphas:
            lam0 = 1 * alpha
            lam1 = 10 * alpha
            lam2 = 8 * alpha
            f, b, cost = be.beads(self.chrom_1D, d, fc, r, nit, lam0, lam1, lam2, pen, conv=3)
            #err[i] = (np.median(np.absolute((self.chrom_1D - (f + b)) - np.median((self.chrom_1D - (f + b))))))

            # Compare each reconstruction to the original signal using the mean
            # squared residual. The selected alpha should leave residuals close
            # to the independently estimated noise level.
            err[i] = (sum((self.chrom_1D-(f+b))**2))/len(self.chrom_1D)
            i = i + 1

        # Select the alpha whose residual variance best matches the estimated
        # chromatographic noise.
        ind = np.argmin(np.abs(err - v))
        alpha = alphas[ind]
        lam0 = 1 * alpha
        lam1 = 10* alpha
        lam2 = 8 * alpha
        z, b, cost = be.beads(self.chrom_1D, d, fc, r, nit, lam0, lam1, lam2, pen, conv=3)

        # Plot the original and denoised chromatograms for visual inspection.
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
        Detect chromatographic peaks in the processed 1D chromatogram.

        Peak detection is performed on the second derivative of the 1D signal.
        Candidate apex positions are first identified with `findMin`, then peak
        boundaries and prominences are refined using SciPy peak-width utilities.

        The detected peaks are stored in `self.peaks` as `Peak` objects.

        Parameters
        ----------
        noise_thresh : int, default=2
            Noise threshold used by `findMin` to reject small derivative
            fluctuations that are unlikely to correspond to real peaks.

        height_ratio : int, default=25
            Relative height filter used by `findMin`. Larger values make peak
            detection more selective.

        distance_ratio : int, default=400
            Controls the minimum spacing between detected peak apices.
            The actual distance is calculated as
            `len(self.chrom_1D) / distance_ratio`.

        width : int, default=3
            Minimum peak width parameter passed to `findMin`.

        wlen : int, default=250
            Initial window length for prominence estimation. This value is
            recalculated from the preliminary peak boundaries before calling
            `peak_prominences`.

        remove_0_width : bool, default=False
            If True, peaks with zero prominence are discarded.

        Returns
        -------
        int
            Number of detected peaks retained after optional filtering.

        Notes
        -----
        This method assumes that `self.chrom_1D` already exists. For 2D UV data,
        call `from_2D_to_1D()` before peak detection.
        """

        # The second derivative highlights changes in curvature.
        # Chromatographic peak apices often correspond to strong curvature changes.
        dz2 = np.diff(self.chrom_1D, 2)

        # Diagnostic plot of the second derivative used for candidate detection.
        plt.figure()
        plt.plot(dz2)
        plt.show()

        # Convert the user-friendly distance ratio into an absolute minimum
        # sample distance between peak candidates.
        distance = len(self.chrom_1D)/distance_ratio

        # Locate candidate peak apices from the second derivative and original signal.
        apex_loc = findMin(dz2, self.chrom_1D, noise_thresh, height_ratio, distance, width)

        # Estimate preliminary peak boundaries around each detected apex.
        left, right = peakStartEnd(self.chrom_1D, apex_loc)

        # Normalize signal before prominence and width estimation.
        # This makes prominence values independent of absolute signal scale
        res = (self.chrom_1D - self.chrom_1D.min()) / (self.chrom_1D.max() - self.chrom_1D.min())

        # Use preliminary boundaries to derive a suitable prominence window.
        # The widest preliminary peak defines the local search window.
        diff = np.asarray(right) - np.asarray(left)
        wlen = max(diff)
        # Estimate peak prominence and refine left/right boundaries using
        # SciPy's peak-width calculation at full prominence height.
        prominences = peak_prominences(res, apex_loc, wlen)
        _, _, left, right = peak_widths(res, apex_loc, rel_height=1, prominence_data=prominences)

        # Reusable color palette for plotting detected peak regions.
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


        # Convert each accepted detection into a Peak object.
        peaks = []
        loc = []

        for idx in range(len(apex_loc)):
            prominence = prominences[0][idx]
            # Optionally discard detections with zero prominence.
            # These are usually poorly defined or degenerate peaks.
            if remove_0_width and prominence ==0:
                continue

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

    def deconvolve_peaks(self, model: PeakModel):
        """
        Perform deconvolution on overlapping chromatographic peaks using a specified peak model.

        This method first merges nearby peaks into groups, then extends the boundaries
        of these merged peaks based on estimated background noise. For each resulting
        peak, two-dimensional UV data is extracted and deconvolved into individual components.
        The deconvolution process computes both the concentration profiles and corresponding spectra,
        which are then normalized to ensure consistent scaling.

        Parameters
        ----------
        model : PeakModel or Literal['BiGaussian', 'BiGaussianTailing', 'FraserSuzuki']
            The mathematical model that describes the expected shapes of the peak components.

        Returns
        -------
        Chromatogram
            Returns self with updated `peaks` that now include deconvolved concentration
            profiles and corresponding spectral data.
        """
        # Step 1: Merge individual peaks into groups if their boundaries overlap.
        merged_peaks = merge_overlapping_peaks(self.chrom_1D, self.peaks)
        self.peaks = merged_peaks

        # Step 2: Extend the boundaries of merged peaks.
        # The extension is computed based on the background standard deviation,
        # allowing a more comprehensive inclusion of the peak's tail region.
        background_std = estimate_background_noise(self.chrom_1D, self.peaks)
        self.peaks = expand_peaks(self.chrom_1D, self.peaks, background_std)

        # Initialize a plot to visualize the chromatogram and deconvolution progress.
        plt.figure()
        plt.plot(self.time, self.chrom_1D)

        # Step 3: Loop over each peak to perform deconvolution separately.
        for idx, peak in enumerate(self.peaks):
            # Extract 2D UV data specific to the current peak.
            # This data provides the spectral intensity profiles across wavelengths.
            peak_data = peak.data(self.chrom_2D)

            # Step 4: Deconvolve the 2D peak data using the provided model.
            # fit_peak_model returns concentration profiles (concs), mean squared error (mse),
            # and additional intermediate results (discarded here with '_').
            concs, mse, _ = fit_peak_model(peak_data, model, peak)

            # Step 5: Extract corresponding spectra from the computed concentrations.
            # This step transforms the concentration profiles into spectral information.
            spectra, mse = spectra_from_concentrations(peak_data, concs)

            # Step 6: Normalize the spectra along the wavelength axis to stabilize scaling.
            # A small constant (1e-7) is added to avoid division by zero.
            norm_factors = np.mean(spectra, axis=1) + 1e-7
            spectra = (spectra.T / norm_factors).T
            concs = (concs.T * norm_factors).T

            # Step 7: Save the deconvolved concentration and spectral data back to the peak.
            self.peaks[idx].concentrations = concs
            self.peaks[idx].spectra = spectra

            # Plot each concentration component for visual assessment.
            for j in range(self.peaks[idx].concentrations.shape[0]):
                plt.plot(self.peaks[idx].time(self.time), self.peaks[idx].concentrations[j])

        # Display the final plot showing the original chromatogram along with the decomposed peak components.
        plt.show()

    def UVtoMSshift(self):
        """
        Calculate the retention time shift between the MS and UV chromatograms.

        The method computes the difference between the retention time corresponding to
        the main peak in the MS chromatogram and the main peak in the UV chromatogram.
        It uses the maximum intensities from the chromatograms as proxies for the major peaks.

        Process:
        1. For the MS data: the function takes the maximum across all m/z channels (axis=0)
           of the 2D MS chromatogram and finds the corresponding time from `time_MS` for the
           maximum intensity.
        2. For the UV data: it finds the time corresponding to the maximum value in the
           1D UV chromatogram.
        3. The difference between these two times is stored in `self.shift`, which is then
           used to align UV and MS peaks.
        """
        # Find the retention time corresponding to the highest intensity in the MS chromatogram.
        # np.max over axis=0 collapses m/z channels to yield the maximal intensity for each time point.
        loc_main_peak_MS = self.time_MS[np.argmax(np.max(self.chrom_2D_MS, axis=0))]

        # Identify the retention time for the main peak in the UV chromatogram.
        loc_main_peak_UV = self.time[np.argmax(self.chrom_1D)]

        # Compute the retention time shift (MS minus UV) to align the two datasets.
        self.shift = loc_main_peak_MS - loc_main_peak_UV

    def getMSspec(self):
        """
        Extract MS spectra for each detected peak using the Simplisma algorithm.

        This method first aligns the UV and MS chromatograms by calling UVtoMSshift(),
        which computes a shift based on the main peaks. Then, for each detected peak,
        a region in the MS chromatogram is defined by shifting the UV peak boundaries.
        In that region, the Simplisma algorithm is applied on the MS data to extract
        the corresponding spectra. These extracted MS spectra are stored in the respective
        Peak instance for later use.

        The method uses an internal helper function, find_nearest_index, to find the index
        in the MS time axis (time_MS) corresponding to a given target value. This function
        is versatile and handles both scalar and iterable targets.

        Returns
        -------
        None
            The extracted MS spectra for each peak are stored in each peak’s 'spectra_MS'
            attribute.
        """

        def find_nearest_index(array, values, return_each=True):
            """
            Find the nearest index(es) in `array` for one or more target `values`.

            Parameters
            ----------
            array : array-like
                One-dimensional array of numeric values.
            values : scalar or array-like
                Target value(s) for which the closest index will be found.
            return_each : bool, optional, default True
                If True, return an array of indices (one per value). Otherwise, return the index
                corresponding to the single closest element across all target values.

            Returns
            -------
            int or numpy.ndarray of ints
                The index (or array of indices) of the closest element(s) in `array` for the given value(s).
            """
            array = np.asarray(array)
            # Ensure the array is not empty.
            if array.size == 0:
                raise ValueError("`array` must contain at least one element")
            # Normalize values to a 1-D array.
            if np.isscalar(values):
                vals = np.atleast_1d(values)
            else:
                vals = np.asarray(values)
                if vals.ndim == 0:
                    vals = np.atleast_1d(vals)
                elif vals.ndim > 1:
                    vals = vals.ravel()
            # Compute absolute differences between each element of array and each target value.
            diffs = np.abs(array[:, None] - vals[None, :])
            if return_each:
                # For each target value, get the index of the array element with minimum difference.
                indices = diffs.argmin(axis=0)
                return indices
            else:
                # Alternatively, return the index corresponding to the smallest difference overall.
                min_dist_per_array_elem = diffs.min(axis=1)
                closest_index = int(min_dist_per_array_elem.argmin())
                return closest_index

        # Align the MS and UV chromatograms by computing the retention time shift.
        self.UVtoMSshift()

        # Iterate over each detected peak and extract corresponding MS spectra.
        for idx, peak in enumerate(self.peaks):
            # Method 1 (commented out): Extract spectra at peak maxima.
            # maxima_UV = self.time[np.asarray(peak.all_maxima)] + self.shift
            # maxima_MS = find_nearest_index(self.time_MS, maxima_UV)
            # spectra = self.chrom_2D_MS[:, maxima_MS].T

            # Method 2 (active): Use peak boundaries with the computed shift.
            # Determine the MS time boundaries corresponding to the UV peak boundaries.
            start_time_UV = self.time[peak.left] + self.shift
            end_time_UV = self.time[peak.right] + self.shift
            # Find the nearest indices on the MS time axis to these shifted boundaries.
            start_loc_MS = find_nearest_index(self.time_MS, start_time_UV)
            end_loc_MS = find_nearest_index(self.time_MS, end_time_UV)
            # Extract the segment of the 2D MS data corresponding to the peak region.
            peak_data = self.chrom_2D_MS[:, start_loc_MS[0]: end_loc_MS[0]]
            # Apply the Simplisma algorithm to the extracted MS data.
            # The number of components is determined by the number of maxima in the peak.
            spectra, initial_concs = simplisma(peak_data, len(peak.all_maxima), 5)

            # (Optional normalization commented out)
            # norm_factors = np.mean(spectra, axis=1) + 1e-7
            # spectra = (spectra.T / norm_factors).T

            # Save the extracted MS spectra to the current peak.
            self.peaks[idx].spectra_MS = spectra
            # Optionally, store initial concentrations:
            # self.peaks[idx].concentrations = initial_concs
