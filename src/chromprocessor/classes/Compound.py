from numpy.typing import NDArray
from typing import Optional


class Compound:
    """Represents a single chemical compound across multiple chromatographic runs.

    This lightweight container groups arrays that describe the compound's measured
    properties across N chromatograms (runs, injections, or timepoints). All array
    attributes should be aligned along the first axis: index 0 corresponds to the
    same run for every attribute.

    Typical attribute shapes (convention):
    - retention_times: 1D, shape (N,) giving the compound's retention time per run.
    - peak_data_2D: 2D or higher, with first axis N (e.g., (N, T) for time series,
      (N, M) for multi-wavelength data).
    - concentrations: 1D or 2D, often (N,) or (N, C) for multiple measured concentrations.
    - spectra_UV, spectra_MS: arrays where first axis is N and remaining axes are
      the spectral dimensions.

    Notes:
    - NDArray is used for static typing. At runtime, any array-like structure
      compatible with numpy arrays is acceptable.
    - This class is intentionally minimal: it only stores data and provides a
      callable shorthand to update retention_times in-place.
    """

    retention_times: NDArray
    """1D array of retention times; first axis length N (one value per chromatogram)."""

    peak_data_2D: Optional[NDArray] = None
    """Optional 2D (or higher) array holding UV/MS peak shapes or intensity maps.
    First axis must correspond to chromatogram/run index."""

    concentrations: Optional[NDArray] = None
    """Optional concentration profiles for the compound across chromatograms.
    Often shape (N,) or (N, n_components)."""

    spectra_UV: Optional[NDArray] = None
    """Optional UV spectra aligned with chromatograms. First axis is chromatogram index."""

    spectra_MS: Optional[NDArray] = None
    """Optional mass-spectra aligned with chromatograms. First axis is chromatogram index."""

    def __init__(self, retention_times: NDArray[float]) -> None:
        """Create a Compound with required retention times.

        Parameters:
        - retention_times: 1D numpy array-like of floats with length N (number of runs).

        The other arrays are left as None and may be assigned later by processing
        stages that extract peak shapes, spectra and concentration profiles.
        """
        # Store retention times; keep the array as provided to avoid unexpected copies.
        self.retention_times = retention_times

    def __call__(self, retention_times: NDArray[float]) -> None:
        """Update the stored retention_times in-place.

        This provides a concise API to overwrite retention times after alignment
        or reprocessing steps. The method intentionally does not validate shapes;
        callers should ensure the provided array matches the expected length N.
        """
        self.retention_times = retention_times
