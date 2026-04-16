from typing import List
import numpy as np
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
    spectra_MS: NDArray
    """MS spectra of peak or merged peaks"""

    def __init__(
            self,
            left: int,
            right: int,
            maximum: int,
            height: float,
            prominence: float,
            all_maxima: List[int] | None = None,
    ) -> None:
        self.left = left
        self.right = right
        self.maximum = maximum
        self.height = height
        self.prominence = prominence
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

    def data_MS(self, data: NDArray, time_UV: NDArray, time_MS: NDArray, shift: float) -> NDArray:
        """Returns part of the 2D MS data that contains this peak"""

        def find_nearest_index(array, value):
            # Convert array to a NumPy array if it isn't already
            array = np.asarray(array)
            # Calculate the absolute difference between each element and the value
            differences = np.abs(array - value)
            # Find the index of the minimum difference
            closest_index = differences.argmin()
            return closest_index
        start_time_UV = time_UV[self.left] + shift
        end_time_UV = time_UV[self.right] + shift
        start_loc_MS = find_nearest_index(time_MS, start_time_UV)
        end_loc_MS = find_nearest_index(time_MS, end_time_UV)
        return data[:, start_loc_MS: end_loc_MS]
