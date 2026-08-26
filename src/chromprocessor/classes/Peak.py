from typing import List, Optional
import numpy as np
from numpy.typing import NDArray


class Peak:
    """Represents a single chromatographic peak (or a merged group of peaks).

    This class stores the basic positional and quantitative descriptors of a peak
    detected in a chromatogram. It is designed to be a lightweight container used
    by peak-picking and alignment routines; downstream processing can attach
    spectra and concentration arrays as attributes.

    Attribute conventions and expected shapes:
    - left, right, maximum: integer indices into a 1D time/intensity trace.
      left is inclusive, right is exclusive (slice semantics are used by data()).
    - height: absolute intensity at the peak maximum (float).
    - prominence: measure of peak prominence relative to local baseline (float).
    - all_maxima: list of integer indices for sub-peaks when peaks were merged.
    - concentrations: Optional NDArray typically shaped (N,) or (N, k) where N is
      number of samples/runs and k is number of concentration channels.
    - spectra, spectra_MS: Optional spectral arrays aligned with chromatogram
      indices; the first axis should correspond to time/run index when present.

    Note: the class intentionally does not enforce strict shape validation to
    remain flexible during intermediate processing steps. Callers should ensure
    arrays are consistent with these conventions.
    """

    left: int
    """Index of peak start (inclusive)."""

    right: int
    """Index of peak end (exclusive)."""

    maximum: int
    """Index of the peak's maximum intensity."""

    height: float
    """Absolute height (intensity) of the peak at `maximum`."""

    prominence: float
    """Prominence of the peak relative to its baseline."""

    all_maxima: List[int]
    """Indices of all maxima for merged peaks; includes `maximum` by default."""

    concentrations: Optional[NDArray] = None
    """Optional concentration profile for this peak across samples/runs."""

    spectra: Optional[NDArray] = None
    """Optional UV or other spectra associated with this peak."""

    spectra_MS: Optional[NDArray] = None
    """Optional MS spectra associated with this peak."""

    def __init__(
            self,
            left: int,
            right: int,
            maximum: int,
            height: float,
            prominence: float,
            all_maxima: List[int] | None = None,
    ) -> None:
        """Initialize a Peak instance.

        Parameters:
        - left, right, maximum: indices describing the peak boundaries and apex.
        - height: intensity at `maximum`.
        - prominence: peak prominence (baseline-corrected height measure).
        - all_maxima: optional list of maxima indices when the peak represents a
          merged group. If omitted, defaults to [maximum].
        """
        self.left = left
        self.right = right
        self.maximum = maximum
        self.height = height
        self.prominence = prominence
        # Ensure all_maxima is a list and contains at least the main maximum
        self.all_maxima = all_maxima if all_maxima is not None else [maximum]

    def __call__(
            self,
            left: int,
            right: int,
            maximum: int,
            height: float,
            prominence: float,
            all_maxima: List[int] | None = None,
    ) -> None:
        """Update this Peak in-place using the same parameters as the constructor.

        This callable API is a convenience used by some processing pipelines to
        overwrite peak attributes without creating a new object.
        """
        self.left = left
        self.right = right
        self.maximum = maximum
        self.height = height
        self.prominence = prominence
        self.all_maxima = all_maxima if all_maxima is not None else [maximum]

    def data(self, data: NDArray) -> NDArray:
        """Return the slice of 2D intensity data that covers this peak.

        Parameters:
        - data: expected to be an array where axis 0 indexes channels/wavelengths
          and axis 1 is the time/intensity axis. The method returns:
            data[:, left:right]

        Returns the sub-array corresponding to the peak window. No shape
        validation is performed; callers must ensure the provided array is
        compatible with the stored indices.
        """
        # Use Python slice semantics: left inclusive, right exclusive
        return data[:, self.left: self.right]

    def time(self, data: NDArray) -> NDArray:
        """Return the time (1D) slice that contains this peak.

        The input `data` is expected to be a 1D time axis array (e.g., retention
        time values). The returned slice corresponds to indices [left:right].
        """
        return data[self.left: self.right]

    def data_MS(self, data: NDArray, time_UV: NDArray, time_MS: NDArray, shift: float):
        """Map a UV peak window to the corresponding MS data window and return it.

        Parameters:
        - data: 2D MS data array where axis 1 is the MS time axis.
        - time_UV: 1D array of retention times for the UV trace (aligned with left/right).
        - time_MS: 1D array of retention times for the MS trace (may have different sampling).
        - shift: time offset to apply when mapping UV times to MS timebase.

        Returns a tuple (ms_slice, start_index, end_index) where ms_slice is the
        sub-array data[:, start_index:end_index] and the indices are the located
        MS time indices. Indices are chosen by nearest-neighbor mapping; callers
        may want to expand the window slightly to account for sampling differences.
        """

        def find_nearest_index(array, value):
            # Convert array to a NumPy array if it isn't already
            array = np.asarray(array)
            # Compute absolute differences and find the index with minimum distance
            differences = np.abs(array - value)
            closest_index = differences.argmin()
            return int(closest_index)

        # Map the UV window into MS timebase by applying the shift
        start_time_UV = time_UV[self.left] + shift
        end_time_UV = time_UV[self.right] + shift
        # Find nearest indices in the MS time array
        start_loc_MS = find_nearest_index(time_MS, start_time_UV)
        end_loc_MS = find_nearest_index(time_MS, end_time_UV)
        # Return the MS data slice plus the located index bounds
        return data[:, start_loc_MS: end_loc_MS], start_loc_MS, end_loc_MS
