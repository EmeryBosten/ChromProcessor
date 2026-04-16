import numpy as np
import pandas as pd
from typing import List, Any

from . import Chromatogram

from ..peak_models import BiGaussian, BiGaussianTailing

#functions for peak tracking
from ..peak_tracking import SimMeasurementMS, SimMeasurementUV, chosenPairMS, chosenPairUV

class Chromatograms:
    """  """
    chromatograms: List[Chromatogram]
    """chromatograms to process"""
    MS: Any
    """MS or UV data"""
    preprocessing: Any
    """indicate if preprocess chroms """

    def __init__(self, chromatograms: List, MS: Any):
        self.chromatograms = chromatograms
        self.MS = MS

    def track_peaks(self, preprocessing=False, height_ratio=25, distance_ratio=400, width=3, remove_0_width = True):

        """
        Track peaks across chromatograms
        :param preprocessing:
        :return: retention time table
        """

        if preprocessing:
            preprocessed_chromatograms = []
            for chromatogram in self.chromatograms:
                chromatogram.remove_blank()
                chromatogram.correct_baseline()
                chromatogram.remove_noise()
                preprocessed_chromatograms.append(chromatogram)

            self.chromatograms = preprocessed_chromatograms

        # peak detection and deconvolution
        multi_peaks = []
        for chromatogram in self.chromatograms:
            chromatogram.remove_blank()
            chromatogram.from_2D_to_1D()
            chromatogram.detect_peaks(height_ratio=height_ratio, width=width, distance_ratio=distance_ratio, remove_0_width = remove_0_width)
            if self.MS:
                chromatogram.getMSspec()
            else :
                chromatogram.deconvolve_peaks(BiGaussianTailing())
            multi_peaks.append(chromatogram.peaks)

        # peak tracking
        # Reshape chromatogram metrics
        Specs_UV = []
        if self.MS:
            Specs_MS = []
        Peak_heights = []
        relative_rts = []
        rts = []

        for chrom_idx in range(len(multi_peaks)):
            chrom_Specs_UV = []
            if self.MS:
                chrom_Specs_MS = []
            chrom_Peak_heights = []
            chrom_relative_rts = []
            chrom_rts = []
            for idx, peak in enumerate(multi_peaks[chrom_idx]):
                for j in range(len(peak.all_maxima)):
                    if self.MS:
                        chrom_Specs_MS.append(peak.spectra_MS[j])
                    else :
                        chrom_Specs_UV.append(peak.spectra[j])
                    chrom_Peak_heights.append(
                        self.chromatograms[chrom_idx].chrom_1D[peak.all_maxima[j]])
                    chrom_relative_rts.append(
                        (self.chromatograms[chrom_idx].time[peak.all_maxima[j]]) /
                        self.chromatograms[chrom_idx].time[-1])
                    chrom_rts.append(
                        (self.chromatograms[chrom_idx].time[peak.all_maxima[j]]))

            Specs_UV.append(chrom_Specs_UV)
            if self.MS:
                Specs_MS.append(chrom_Specs_MS)
            Peak_heights.append(chrom_Peak_heights)
            relative_rts.append(chrom_relative_rts)
            rts.append(chrom_rts)

        # set reference chromatogram
        numPeaks = [len(sublist) for sublist in relative_rts]
        MaxNumPeaks = np.max(numPeaks)
        indMaxNumPeaks = np.argmax(numPeaks)
        num_chroms = len(numPeaks)

        # Track peaks compared to reference chromatogram
        Tracks = []
        i = 0
        while i <= num_chroms - 1:
            if i == indMaxNumPeaks:
                i = i + 1
            if self.MS:
                [heightSim, countMS, posSim] = SimMeasurementMS(MaxNumPeaks, numPeaks[i],
                                                                Peak_heights[indMaxNumPeaks], Peak_heights[i],
                                                                Specs_MS[indMaxNumPeaks], Specs_MS[i],
                                                                relative_rts[indMaxNumPeaks], relative_rts[i])
                [pairs, _] = chosenPairMS(heightSim, countMS, posSim)
                Tracks.append(pairs)
                i = i + 1
            else:
                [simUV, heightSim, posSim] = SimMeasurementUV(MaxNumPeaks, numPeaks[i],
                                                              Specs_UV[indMaxNumPeaks], Specs_UV[i],
                                                              Peak_heights[indMaxNumPeaks],Peak_heights[i],
                                                              relative_rts[indMaxNumPeaks], relative_rts[i])
                [pairs, _] = chosenPairUV(simUV, heightSim, posSim)
                Tracks.append(pairs)
                i = i + 1
        # Create Retention Time Table
        # Convert numPeaks to numpy array
        numPeaks = np.asarray(numPeaks)
        max_num_peaks = np.max(numPeaks)
        n_cols = len(numPeaks)

        # Create retention timetable
        tab = np.zeros((max_num_peaks, n_cols), dtype=int)

        tab[:, 0] = np.arange(1, max_num_peaks + 1)

        # Retention Time Table -> list of lists
        summaryPeaks = [[None for _ in range(n_cols)]
                        for _ in range(max_num_peaks)]

        indMaxNumPeaks = np.argmax(numPeaks)

        # Fill first column of summaryPeaks
        for i in range(max_num_peaks):
            summaryPeaks[i][0] = rts[indMaxNumPeaks][i]
        # Main loop
        for i in range(1, n_cols):
            n_rows = Tracks[i-1].shape[0]
            # Tracks[i-1] is assumed to be a 2D array
            for j in range(n_rows):
                k = int(Tracks[i - 1][j, 0])
                tab[k, i] = int(Tracks[i - 1][j, 1])
                summaryPeaks[k][i] = rts[i][tab[k, i]]
        return pd.DataFrame(summaryPeaks), multi_peaks
