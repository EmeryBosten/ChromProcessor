# function for automated peak detection

from scipy.signal import find_peaks


import numpy as np
from scipy.signal import find_peaks as scipy_find_peaks  # type: ignore



def findMin(deriv2, y, noise_thresh, height_ratio, distance, width):
    """
    Function to find peak apex based on minima in second derivative

    Parameters
    ----------
    deriv2 : array_like
        Second derivative of the signal
    y : array_like
        Original signal
    noise_thresh : float
        Noise threshold multiplier
    height_ratio : float
        Height ratio constraint
    distance : int
        minimum spacing between minima
    width : int
        minimum width of minima

    Returns
    -------
    apex_loc : np.ndarray
        Indices of detected minima

    """
    deriv2 = np.asarray(deriv2, dtype=float)
    y = np.asarray(y, dtype=float)

    # normalization (between -1 and 1)
    max_deriv2 = np.max(np.abs(deriv2))
    deriv2 = deriv2 / max_deriv2

    k = 1.4826
    max_y_noise = k * np.median(np.abs(y - np.median(y)))  # noise estimate original signal
    max_y = np.max(y)  # highest peak in original signal
    max_deriv2_noise = k * np.median(np.abs(deriv2 - np.median(deriv2)))  # noise estimate second derivative

    #distance and width parameters might be dependent on acquisition rate
    candidate_apex_loc, props = find_peaks(-deriv2, distance=distance, width= width)  # identifying minimas
    final_apex_loc = []

    for apex_loc in candidate_apex_loc:

        # --- filtering conditions ---
        cond1 = deriv2[apex_loc] < -abs(max_deriv2_noise)

        cond2 = (
                y[apex_loc] > noise_thresh * max_y_noise and
                y[apex_loc] > max_y / height_ratio
        )

        if cond1 and cond2:

            # local search for true apex
            idx = np.argmax(y[apex_loc-10:apex_loc+11]) + (apex_loc-10)
            final_apex_loc.append(idx)



    final_apex_loc = np.unique(np.array(final_apex_loc))

    return np.array(final_apex_loc)


def peakStartEnd(y, apex):
    """
    Function to find start and end of peaks
    :param y:  1D chromatogram
    :param apex: apex locations of peaks
    :return:
    startpoint = start point of peaks
    endpoint = end point of peaks

    """

    e = 0.05 # very arbitrary threshold

    startpoint = []
    endpoint = []

    max_y = np.max(np.abs(y))
    y = y / max_y

    I = np.extract(y < min(y) + e, np.arange(len(y)))

    for i in range(len(apex)):
        S = np.extract(I < apex[i], np.arange(len(I)))
        s = I[S[-1]]
        startpoint.append(s)
        G = np.extract(I > apex[i], np.arange(len(I)))
        g = I[G[1]]
        endpoint.append(g)

    return startpoint, endpoint


from scipy.signal import savgol_filter
import numpy as np

def smooth_signal(y, window=11, poly=3):
    return savgol_filter(y, window_length=window, polyorder=poly)


def compute_derivatives(y, window=11, poly=3):
    y_smooth = savgol_filter(y, window, poly)
    dy = savgol_filter(y, window, poly, deriv=1)
    d2y = savgol_filter(y, window, poly, deriv=2)
    return y_smooth, dy, d2y

def peak_start_end_simple(y, apex):
    y = np.asarray(y)
    dy = np.gradient(y)

    starts = []
    ends = []

    for a in apex:
        # left
        i = a
        while i > 1 and dy[i] > 0:
            i -= 1
        starts.append(i)

        # right
        i = a
        while i < len(y)-2 and dy[i] < 0:
            i += 1
        ends.append(i)

    return starts, ends
