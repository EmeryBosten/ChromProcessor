from typing import List
from numpy.typing import NDArray

class Peak:
    """Information about single peak"""

    left: int
    """Index of peak start"""
    right: int
    """Index of peak end"""
    maximum: int
    """Index of peak maximum"""
    height: float
    """Absolute height of the peak"""
    prominence: float
    """Height of the peak from the base"""
    all_maxima: List[int]
    """Indices of all maxima of the merged peaks"""
    concentrations: NDArray
    """concentration profile of peak or merged peaks"""
    spectra: NDArray
    """spectra of peak or merged peaks"""

    def __init__(
            self,
            left: int,
            right: int,
            maximum: int,
            height: float,
            prominence: float,
            all_maxima: List[int] | None = None,
    ):
        self.left = left
        self.right = right
        self.maximum = maximum
        self.height = height
        self.prominence = prominence
        self.all_maxima = all_maxima if all_maxima is not None else [maximum]

    def data(self, data: NDArray) -> NDArray:
        """Returns part of the 2D data that contains this peak"""
        return data[:, self.left: self.right]

    def time(self, data: NDArray) -> NDArray:
        """Returns part of the timescale that contains this peak"""
        return data[self.left: self.right]