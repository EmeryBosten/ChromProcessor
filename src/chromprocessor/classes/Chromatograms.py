
import numpy as np
import pandas as pd
from typing import List, Any
from matplotlib import pyplot as plt

from . import Chromatogram
from .Compound import Compound
from ..deconvolution import spectra_from_concentrations, fit_peak_model
from ..peak_models import FraserSuzuki

#functions for peak tracking
from ..peak_tracking import SimMeasurementMS, SimMeasurementUV, chosenPairMS, chosenPairUV

class Chromatograms:
    """
    Container for handling and processing multiple chromatograms.

    This class aggregates a collection of Chromatogram objects and provides methods
    for peak detection, peak deconvolution, and peak tracking across chromatograms.
    It supports data from both UV/DAD (ultraviolet/diode array) and MS (mass spectrometry)
    sources, and can optionally preprocess the chromatograms before processing.

    Attributes
    ----------
    chromatograms : List[Chromatogram]
        A list of Chromatogram objects to be processed.
    MS : Any
        A flag or indicator specifying if the data includes MS information.
        Typically, this could be a Boolean flag or other type defining whether to process MS data.
    preprocessing : Any
        An indicator (or flag) specifying whether the chromatograms have been preprocessed.
    compounds : List[Compound]
        A list of Compound objects that tracks the compound-specific information extracted
        during peak tracking across chromatograms; each Compound aggregates chromatographic
        metrics (e.g., retention times, spectra, peak profiles).
    """

    chromatograms: List[Chromatogram]
    """Chromatograms to process."""
    MS: Any
    """Indicates whether MS data are present (e.g., True for MS, False for UV-only)."""
    preprocessing: Any
    """Indicator for whether preprocessing has been applied to the chromatograms."""
    compounds: List[Compound]
    """List of compounds identified by aligning and tracking peaks across chromatograms."""

    def __init__(self, chromatograms: List, MS: Any):
        """
        Initialize a Chromatograms container with a list of chromatogram objects.

        Parameters
        ----------
        chromatograms : List
            A list of Chromatogram objects.
        MS : Any
            Indicator specifying whether the data includes MS; for example, a boolean value.
        """
        self.chromatograms = chromatograms
        self.MS = MS

    def track_peaks(self, preprocessing=False, height_ratio=25, distance_ratio=400, width=3, remove_0_width=True):
        """
        Detect and track peaks across multiple chromatograms, returning a retention time table.

        This method optionally preprocesses chromatograms (blank removal, conversion to 1D,
        baseline correction, and noise removal), then performs peak detection and deconvolution.
        It subsequently extracts peak metrics from each chromatogram and aligns peaks across
        chromatograms using peak tracking algorithms. Finally, it builds a retention time table
        and updates compound information.

        Parameters
        ----------
        preprocessing : bool, optional
            If True, preprocess each chromatogram before peak detection.
        height_ratio : int, optional
            Relative height filter for peak detection (default is 25).
        distance_ratio : int, optional
            Parameter to compute the minimum sample distance between detected peaks (default is 400).
        width : int, optional
            Minimum peak width parameter passed to the peak detection algorithm (default is 3).
        remove_0_width : bool, optional
            If True, peaks with zero prominence are discarded (default is True).

        Returns
        -------
        pd.DataFrame
            A DataFrame summarizing retention times of tracked peaks across chromatograms.
        multi_peaks : list
            A list containing peak lists for each chromatogram.
        """

        # Helper to preprocess chromatograms if required.
        def preprocess_all(chroms):
            preprocessed = []
            for chrom in chroms:
                chrom.remove_blank()
                chrom.from_2D_to_1D()
                chrom.correct_baseline(method="flatfit")
                chrom.remove_noise()
                preprocessed.append(chrom)
            return preprocessed

        # Optional preprocessing step
        if preprocessing:
            self.chromatograms = preprocess_all(self.chromatograms)

        # Detect peaks and perform deconvolution/extraction for each chromatogram.
        multi_peaks = []
        for chrom in self.chromatograms:
            # Detect peaks using given parameters.
            chrom.detect_peaks(height_ratio=height_ratio, width=width, distance_ratio=distance_ratio,
                               remove_0_width=remove_0_width)
            # For 2D UV chromatograms: deconvolve peaks using a FraserSuzuki model.
            if chrom.chrom_2D is not None:
                chrom.deconvolve_peaks(FraserSuzuki())
            # For MS data: extract MS spectra using Simplisma.
            if chrom.chrom_2D_MS is not None:
                chrom.getMSspec()
            multi_peaks.append(chrom.peaks)

        # Helper to extract metrics from peaks for each chromatogram.
        def extract_metrics(multi_peaks):
            Specs_UV, Specs_MS = [], []
            Concs = []
            Peak_heights = []
            relative_rts = []
            rts = []
            peak_data = []

            for chrom_idx, peaks in enumerate(multi_peaks):
                chrom_Specs_UV, chrom_Concs = [], []
                chrom_Specs_MS = []  # Only used if MS data is present
                chrom_Peak_heights = []
                chrom_relative_rts = []
                chrom_rts = []
                chrom_peak_data = []
                for peak in peaks:
                    # Retrieve 2D data depending on whether MS data is available.
                    if self.MS:
                        # For MS: use MS-specific extraction.
                        chrom_peak_data.append(
                            peak.data_MS(self.chromatograms[chrom_idx].chrom_2D_MS,
                                         self.chromatograms[chrom_idx].time,
                                         self.chromatograms[chrom_idx].time_MS,
                                         self.chromatograms[chrom_idx].shift)[0]
                        )
                    else:
                        chrom_peak_data.append(peak.data(self.chromatograms[chrom_idx].chrom_2D))
                    # Loop through all detected maxima for the peak.
                    for j in range(len(peak.all_maxima)):
                        if self.MS:
                            chrom_Specs_MS.append(peak.spectra_MS[j])
                        else:
                            chrom_Specs_UV.append(peak.spectra[j])
                            chrom_Concs.append(peak.concentrations[j])
                        chrom_Peak_heights.append(peak.all_maxima[j])
                        # Normalize retention time relative to the end of the time axis.
                        chrom_relative_rts.append(
                            self.chromatograms[chrom_idx].time[peak.all_maxima[j]] /
                            self.chromatograms[chrom_idx].time[-1]
                        )
                        chrom_rts.append(self.chromatograms[chrom_idx].time[peak.all_maxima[j]])
                Specs_UV.append(chrom_Specs_UV)
                if self.MS:
                    Specs_MS.append(chrom_Specs_MS)
                Concs.append(chrom_Concs)
                Peak_heights.append(chrom_Peak_heights)
                relative_rts.append(chrom_relative_rts)
                rts.append(chrom_rts)
                peak_data.append(chrom_peak_data)
            return Specs_UV, Specs_MS, Concs, Peak_heights, relative_rts, rts, peak_data

        # Extract metrics for each chromatogram.
        Specs_UV, Specs_MS, Concs, Peak_heights, relative_rts, rts, peak_data = extract_metrics(multi_peaks)

        # Determine reference chromatogram based on number of peaks.
        numPeaks = [len(rt_list) for rt_list in relative_rts]
        MaxNumPeaks = np.max(numPeaks)
        indMaxNumPeaks = np.argmax(numPeaks)
        num_chroms = len(numPeaks)

        # Align peaks across chromatograms using provided peak tracking functions.
        Tracks = []
        i = 0
        while i < num_chroms:
            if i == indMaxNumPeaks:
                i += 1
                continue
            if self.MS:
                heightSim, countMS, posSim = SimMeasurementMS(
                    MaxNumPeaks, numPeaks[i],
                    Peak_heights[indMaxNumPeaks], Peak_heights[i],
                    Specs_MS[indMaxNumPeaks], Specs_MS[i],
                    relative_rts[indMaxNumPeaks], relative_rts[i]
                )
                pairs, _ = chosenPairMS(heightSim, countMS, posSim)
                Tracks.append(pairs)
            else:
                simUV, heightSim, posSim = SimMeasurementUV(
                    MaxNumPeaks, numPeaks[i],
                    Specs_UV[indMaxNumPeaks], Specs_UV[i],
                    Peak_heights[indMaxNumPeaks], Peak_heights[i],
                    relative_rts[indMaxNumPeaks], relative_rts[i]
                )
                pairs, _ = chosenPairUV(simUV, heightSim, posSim)
                Tracks.append(pairs)
            i += 1

        # Build the retention time table and compound metrics.
        def build_summary_table(rts, Specs_UV, Specs_MS, Concs, peak_data, Tracks):
            numPeaks_arr = np.asarray(numPeaks)
            max_num_peaks = np.max(numPeaks_arr)
            n_cols = len(numPeaks_arr)
            # Initialize a table for retention times.
            tab = np.zeros((max_num_peaks, n_cols), dtype=int)
            tab[:, 0] = np.arange(1, max_num_peaks + 1)

            # Create summary lists for each peak across chromatograms.
            summaryPeaks = [[None for _ in range(n_cols)] for _ in range(max_num_peaks)]
            spectra_UV = [[None for _ in range(n_cols)] for _ in range(max_num_peaks)] if len(
                Specs_UV[0]) != 0 else None
            concentrations = [[None for _ in range(n_cols)] for _ in range(max_num_peaks)] if (
                        not self.MS and len(Specs_UV[0]) != 0) else None
            spectra_MS = [[None for _ in range(n_cols)] for _ in range(max_num_peaks)] if (
                        self.MS and len(Specs_MS[0]) != 0) else None
            PeakTable = [[None for _ in range(n_cols)] for _ in range(max_num_peaks)]

            indRef = np.argmax(numPeaks_arr)
            # First column: take values from the reference chromatogram.
            for i in range(max_num_peaks):
                summaryPeaks[i][0] = rts[indRef][i]
                if spectra_UV is not None:
                    spectra_UV[i][0] = Specs_UV[indRef][i]
                    concentrations[i][0] = Concs[indRef][i]
                if spectra_MS is not None:
                    spectra_MS[i][0] = Specs_MS[indRef][i]
                PeakTable[i][0] = peak_data[indRef][i]

            # Fill subsequent columns using aligned peak orders in Tracks.
            for col in range(1, n_cols):
                n_rows = Tracks[col - 1].shape[0]
                for j in range(n_rows):
                    k = int(Tracks[col - 1][j, 0])
                    tab[k, col] = int(Tracks[col - 1][j, 1])
                    summaryPeaks[k][col] = rts[col][tab[k, col]]
                    if spectra_UV is not None:
                        spectra_UV[k][col] = Specs_UV[col][tab[k, col]]
                        concentrations[k][col] = Concs[col][tab[k, col]]
                    if spectra_MS is not None:
                        spectra_MS[k][col] = Specs_MS[col][tab[k, col]]
                    PeakTable[k][col] = peak_data[col][tab[k, col]]
            return summaryPeaks, spectra_UV, concentrations, spectra_MS, PeakTable

        summaryPeaks, spectra_UV, concentrations, spectra_MS, PeakTable = build_summary_table(rts, Specs_UV, Specs_MS,
                                                                                              Concs, peak_data, Tracks)

        # Create Compound instances from the summary peak table.
        compounds = []
        for i in range(MaxNumPeaks):
            compound = Compound(retention_times=summaryPeaks[i][:])
            if spectra_UV is not None:
                compound.spectra_UV = spectra_UV[i][:]
            if spectra_MS is not None:
                compound.spectra_MS = spectra_MS[i][:]
            compound.peak_data_2D = PeakTable[i][:]
            compounds.append(compound)
        self.compounds = compounds

        # Return a DataFrame of retention times and the list of peak lists.
        return pd.DataFrame(summaryPeaks), multi_peaks

    def deconvolute_missing_peaks(self):
        """
        For each compound missing a peak (i.e. where a None is found in its retention times),
        perform deconvolution on an adjacent chromatogram's peak data in order to fill in the gap.

        For the compound at the first chromatogram (index 0), the next chromatogram on the right
        is used; otherwise, the closest neighboring compound is selected based on retention times.
        The deconvolution is performed using the FraserSuzuki model and the extracted spectra are normalized.
        Compound information is updated accordingly.
        """

        # Helper to perform deconvolution on missing peak data and update the compound.
        def process_missing_peak(compound, ref_comp_index, missing_index, peak_data_extractor):
            """
            Process the missing peak for a given compound using peak data from the reference compound.

            Parameters
            ----------
            compound : Compound
                The compound with a missing peak.
            ref_comp_index : int
                The compound index whose peak data is used as reference.
            missing_index : int
                The position (index) in the retention time list where the missing peak was found.
            peak_data_extractor : function
                A function that returns the 2D peak data given a compound index and a missing index.

            Returns
            -------
            None
            """
            # Extract the missing peak's 2D data from the reference compound.
            missing_peak_data = peak_data_extractor(ref_comp_index, missing_index)
            model = FraserSuzuki()

            # Depending on data type, choose MS or UV extraction.
            if self.MS:
                spectra = np.asarray([
                    compound.spectra_MS[missing_index + 1],
                    self.compounds[ref_comp_index].spectra_MS[missing_index + 1]
                ])
            else:
                spectra = np.asarray([
                    compound.spectra_UV[missing_index + 1],
                    self.compounds[ref_comp_index].spectra_UV[missing_index + 1]
                ])

            # Perform deconvolution with a fixed number (n_comps=2) of components.
            concs, mse, _ = fit_peak_model(missing_peak_data, model, None, spectra, n_comps=2)
            # Extract spectral information from concentrations.
            extracted_spectra, mse = spectra_from_concentrations(missing_peak_data, concs)
            # Normalize the spectra and scale the concentrations.
            norm_factors = np.max(extracted_spectra, axis=1)
            normalized_spectra = (extracted_spectra.T / norm_factors).T
            scaled_concs = (concs.T * norm_factors).T

            # Update the compound's information based on the data type.
            if self.MS:
                compound.spectra_MS[missing_index] = normalized_spectra[0]
            else:
                compound.spectra_UV[missing_index] = normalized_spectra[0]
                compound.concentrations[missing_index] = scaled_concs[0]
            # Update the 2D peak data.
            compound.peak_data_2D[missing_index] = missing_peak_data

            # (In the MS case, retention time is calculated from MS data;
            # in UV, it is adjusted by the left boundary of the given peak.)
            if self.MS:
                # Use the reference compound's peak data (using ref_comp_index) for retention time adjustment.
                adjusted_rt = self.chromatograms[missing_index].time_MS[
                                  np.argmax(normalized_spectra[0]) +
                                  self.chromatograms[missing_index].peaks[ref_comp_index].data_MS(
                                      self.chromatograms[missing_index].chrom_2D_MS,
                                      self.chromatograms[missing_index].time,
                                      self.chromatograms[missing_index].time_MS,
                                      self.chromatograms[missing_index].shift
                                  )[1]
                                  ] - self.chromatograms[missing_index].shift
                compound.retention_times[missing_index] = adjusted_rt
            else:
                adjusted_rt = self.chromatograms[missing_index].time[
                    np.argmax(normalized_spectra[0]) + self.chromatograms[missing_index].peaks[ref_comp_index].left
                    ]
                compound.retention_times[missing_index] = adjusted_rt

            # Plot for diagnostic purposes.
            plt.figure()
            plt.plot(scaled_concs[0])
            plt.plot(scaled_concs[1])
            plt.plot(scaled_concs[0] + scaled_concs[1])
            plt.plot(np.max(missing_peak_data, axis=0))
            plt.show()

        # Main loop: iterate over compounds and fix missing peaks.
        for comp_index, compound in enumerate(self.compounds):
            try:
                # Attempt to locate a missing value in the compound's retention times.
                missing_index = compound.retention_times.index(None)
                missing_found = True
            except ValueError:
                missing_found = False

            if missing_found:
                # Determine the reference compound:
                # If it is the first compound, use the next one (comp_index + 1).
                if comp_index == 0:
                    ref_index = comp_index + 1
                else:
                    # For compounds after the first, choose the closer neighboring compound.
                    # Calculate absolute differences between the retention times directly before the missing index.
                    diff_prev = abs(
                        compound.retention_times[missing_index - 1] - self.compounds[comp_index - 1].retention_times[
                            missing_index - 1])
                    diff_next = abs(
                        compound.retention_times[missing_index - 1] - self.compounds[comp_index + 1].retention_times[
                            missing_index - 1])
                    ref_index = comp_index - 1 if diff_prev < diff_next else comp_index + 1

                # Define a simple lambda to extract the 2D peak data from the chosen reference compound.
                extract_peak_data = lambda idx, m_idx: self.compounds[idx].peak_data_2D[m_idx]
                # Process the missing peak with the chosen reference and update the compound.
                process_missing_peak(compound, ref_index, missing_index, extract_peak_data)
