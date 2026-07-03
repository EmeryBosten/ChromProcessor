import os

from typing import List, Sequence, Literal

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

from chromprocessor.classes import Chromatogram, Chromatograms
from chromprocessor.peak_models import Bemg


def read_file(path: str) -> pd.DataFrame:
    """

    Read whitespace-separated file skipping first 4 lines.

    """

    return pd.read_table(path, sep=r"\s+", header=None, skiprows=4, engine="python")


def combineChroms(

        path: str,

        filenames: Sequence[str],

        time_col: int = 0,

        align: Literal["truncate", "pad", "interpolate"] = "truncate",

        pad_value: float = 0.0,

        interpolate_method: str = "linear",

) -> pd.DataFrame:
    """

    Combine multidimensional chromatograms (multiple channels per file).



    Parameters

    - path: directory containing the files

    - filenames: list/sequence of file names (not full paths)

    - time_col: index of the time column (default 0)

    - align: alignment strategy:

        - "truncate": truncate to the shortest length (R behavior)

        - "pad": pad shorter files with pad_value to match the longest length

        - "interpolate": resample every file's channel columns onto the time axis of the first file

    - pad_value: numeric value used when align="pad" (default 0.0)

    - interpolate_method: currently only "linear" supported (kept for API clarity)



    Returns

    - DataFrame with the combined time axis (from the first file, maybe trimmed/padded)

      and combined channel columns named abs_1, abs_2, ... (one per channel/column after time).

    """

    filenames = list(filenames)

    if not filenames:
        raise ValueError("filenames must contain at least one file")

    # Identify blanks and move them to the end (preserve order)

    is_blank = [("blank" in fn.lower()) for fn in filenames]

    if any(is_blank):

        non_blanks = [fn for fn, b in zip(filenames, is_blank) if not b]

        blanks = [fn for fn, b in zip(filenames, is_blank) if b]

        filenames = non_blanks + blanks

        has_blank = True

    else:

        has_blank = False

    # Read all files (we will process slicing/resampling later)

    data_frames = [read_file(os.path.join(path, fn)) for fn in filenames]

    # Validate that each file has at least two columns (time + >=1 channel)

    for i, df in enumerate(data_frames):

        if df.shape[1] <= time_col:
            raise ValueError(f"File {filenames[i]} does not have a time column at index {time_col}")

        if df.shape[1] < time_col + 2:
            raise ValueError(f"File {filenames[i]} must have at least one channel column after time")

    # Extract time axes and channel arrays

    times = [df.iloc[:, time_col].to_numpy(dtype=float) for df in data_frames]

    channels = [df.drop(df.columns[time_col], axis=1).to_numpy(dtype=float) for df in data_frames]

    # After drop, column order is consistent: relative channel columns for each file

    # Ensure channel counts are the same across files

    n_channels_list = [c.shape[1] for c in channels]

    if len(set(n_channels_list)) != 1:
        raise ValueError(f"Inconsistent number of channels across files: {n_channels_list}")

    n_channels = n_channels_list[0]

    # Helper: produce column names for channels

    channel_names = [f"abs_{i + 1}" for i in range(n_channels)]

    def sum_arrays(arr_list: List[np.ndarray]) -> np.ndarray:

        """Elementwise sum of list of 2D arrays with same shape (rows, channels)."""

        return np.sum(np.stack(arr_list, axis=0), axis=0)

    # Determine how to align / resample

    if align == "truncate":

        # R-like: operate on top rows only. Use min rows across involved sample files (excluding blank)

        sample_count = len(filenames) - 1 if has_blank else len(filenames)

        sample_dfs = data_frames[:sample_count] if has_blank else data_frames

        min_rows = min(df.shape[0] for df in sample_dfs)

        # Trim channels and times

        trimmed_channels = [c[:min_rows, :] for c in channels[:sample_count]]

        t_ref = times[0][:min_rows]

        combined = sum_arrays(trimmed_channels)

        # handle blank: last file is blank if has_blank

        if has_blank:
            blank_arr = channels[-1][:min_rows, :]

            combined = combined - (sample_count) * blank_arr

        final_time = t_ref

        final_channels = combined



    elif align == "pad":

        # Pad all sample files to max length (zero/ pad_value for channels; time padded with NaN)

        sample_count = len(filenames) - 1 if has_blank else len(filenames)

        sample_dfs = data_frames[:sample_count] if has_blank else data_frames

        max_rows = max(df.shape[0] for df in sample_dfs)

        padded_channels = []

        for c in channels[:sample_count]:

            rows = c.shape[0]

            if rows < max_rows:

                pad_shape = (max_rows - rows, n_channels)

                pad_block = np.full(pad_shape, pad_value, dtype=float)

                padded = np.vstack([c, pad_block])

            else:

                padded = c[:max_rows, :]

            padded_channels.append(padded)

        combined = sum_arrays(padded_channels)

        if has_blank:

            # pad blank similarly to max_rows

            b = channels[-1]

            if b.shape[0] < max_rows:

                pad_block = np.full((max_rows - b.shape[0], n_channels), pad_value, dtype=float)

                b = np.vstack([b, pad_block])

            else:

                b = b[:max_rows, :]

            combined = combined - (sample_count) * b

        # time axis: take first file time and pad with NaN if necessary

        t0 = times[0]

        if t0.shape[0] < max_rows:

            t_pad = np.full(max_rows - t0.shape[0], np.nan, dtype=float)

            final_time = np.concatenate([t0, t_pad])

        else:

            final_time = t0[:max_rows]

        final_channels = combined



    elif align == "interpolate":

        # Use time axis of the first file as reference and resample all channels to it (linear)

        t_ref = times[0]

        sample_count = len(filenames) - 1 if has_blank else len(filenames)

        resampled_channels = []

        for i in range(sample_count):

            t_i = times[i]

            c_i = channels[i]

            # For each channel, interpolate onto t_ref

            # Use numpy.interp for fast linear interp; assumes t_i monotonic increasing

            if not np.all(np.diff(t_i) >= 0):
                # enforce monotonic by sorting (rarely needed), but warn would be good

                order = np.argsort(t_i)

                t_i = t_i[order]

                c_i = c_i[order, :]

            # For each column channel j:

            cols_res = np.vstack([np.interp(t_ref, t_i, c_i[:, j], left=np.nan, right=np.nan)

                                  for j in range(n_channels)]).T

            # replace NaNs outside original range by pad_value (mimic trunc/pad choices)

            # here we treat NaN as pad_value to allow summation

            cols_res = np.where(np.isnan(cols_res), pad_value, cols_res)

            resampled_channels.append(cols_res)

        combined = sum_arrays(resampled_channels)

        if has_blank:

            # resample blank onto t_ref similarly

            b_i = channels[-1]

            t_b = times[-1]

            if not np.all(np.diff(t_b) >= 0):
                order = np.argsort(t_b)

                t_b = t_b[order]

                b_i = b_i[order, :]

            b_res = np.vstack([np.interp(t_ref, t_b, b_i[:, j], left=np.nan, right=np.nan)

                               for j in range(n_channels)]).T

            b_res = np.where(np.isnan(b_res), pad_value, b_res)

            combined = combined - (sample_count) * b_res

        final_time = t_ref

        final_channels = combined



    else:

        raise ValueError("align must be one of 'truncate', 'pad', or 'interpolate'")

    # Build result DataFrame: first column time, then abs_1 .. abs_n

    result_df = pd.DataFrame(final_channels, columns=channel_names)

    result_df.insert(0, "t", final_time)

    return result_df

# Load chromatograms
#PDA data
#Example1 PDA
filenames = ["Ref_2781.arw", "Mix1_2796.arw", "Mix2_2811.arw", "Ribps_2826.arw", "Stress_2841.arw", "blank_2766.arw"]
chrom_PDA1 = combineChroms("C:/Users/emery/OneDrive - KU Leuven/Chromatograms/PULSAR/Example_PDA/Example_PDA/Example1",filenames,align = "truncate")
#Example1 MS
filenames = ["Ref_2779.arw", "Mix1_2794.arw", "Mix2_2809.arw", "Ribps_2824.arw", "Stress_2839.arw", "blank_2764.arw"]
chrom_MS1 = combineChroms("C:/Users/emery/OneDrive - KU Leuven/Chromatograms/PULSAR/Example_MS/Example_MS/Example1",filenames,align = "truncate")
chrom1 = Chromatogram(chrom_PDA1.iloc[:,1:-1].to_numpy().T,None,chrom_MS1.iloc[:,1:-1].to_numpy().T,None,chrom_PDA1.iloc[:,0].to_numpy(),chrom_MS1.iloc[:,0].to_numpy())

#Example2 PDA
filenames = ["Ref_2904.arw", "Mix1_2919.arw", "Mix2_2934.arw", "Ribps_2949.arw", "Stress_2964.arw", "blank_2889.arw"]
chrom_PDA2 = combineChroms("C:/Users/emery/OneDrive - KU Leuven/Chromatograms/PULSAR/Example_PDA/Example_PDA/Example2",filenames,align = "truncate")
#Example2 MS
filenames = ["Ref_2902.arw", "Mix1_2917.arw", "Mix2_2932.arw", "Ribps_2947.arw", "Stress_2962.arw", "blank_2887.arw"]
chrom_MS2 = combineChroms("C:/Users/emery/OneDrive - KU Leuven/Chromatograms/PULSAR/Example_MS/Example_MS/Example2",filenames,align = "truncate")
chrom2 = Chromatogram(chrom_PDA2.iloc[:,1:-1].to_numpy().T,None,chrom_MS2.iloc[:,1:-1].to_numpy().T,None,chrom_PDA2["t"].to_numpy(),chrom_MS2["t"].to_numpy())

#Example3 PDA
filenames = ["Ref_3265.arw", "Mix1_3280.arw", "Mix2_3295.arw", "Ribps_3310.arw", "Stress_3325.arw", "blank_3250.arw"]
chrom_PDA3 = combineChroms("C:/Users/emery/OneDrive - KU Leuven/Chromatograms/PULSAR/Example_PDA/Example_PDA/Example3",filenames,align = "truncate")
#Example3 MS
filenames = ["Ref_3263.arw", "Mix1_3278.arw", "Mix2_3293.arw", "Ribps_3308.arw", "Stress_3323.arw", "blank_3248.arw"]
chrom_MS3 = combineChroms("C:/Users/emery/OneDrive - KU Leuven/Chromatograms/PULSAR/Example_MS/Example_MS/Example3",filenames,align = "truncate")
chrom3 = Chromatogram(chrom_PDA3.iloc[:,1:-1].to_numpy().T,None,chrom_MS3.iloc[:,1:-1].to_numpy().T,None,chrom_PDA3["t"].to_numpy(),chrom_MS3["t"].to_numpy())

#Example4 PDA
filenames = ["Ref_3385.arw", "Mix1_3400.arw", "Mix2_3415.arw", "Ribps_3430.arw", "Stress_3445.arw", "blank_3370.arw"]
chrom_PDA4 = combineChroms("C:/Users/emery/OneDrive - KU Leuven/Chromatograms/PULSAR/Example_PDA/Example_PDA/Example4",filenames, align = "truncate")
#Example4 MS
filenames = ["Ref_3383.arw", "Mix1_3398.arw", "Mix2_3413.arw", "Ribps_3428.arw", "Stress_3443.arw", "blank_3368.arw"]
chrom_MS4 = combineChroms("C:/Users/emery/OneDrive - KU Leuven/Chromatograms/PULSAR/Example_MS/Example_MS/Example4",filenames, align = "truncate")
chrom4 = Chromatogram(chrom_PDA4.iloc[:,1:-1].to_numpy().T,None,chrom_MS4.iloc[:,1:-1].to_numpy().T,None,chrom_PDA4["t"].to_numpy(),chrom_MS4["t"].to_numpy())

chrom1.remove_blank()
chrom1.from_2D_to_1D()
chrom1.correct_baseline()
chrom1.remove_noise()
chrom1.detect_peaks(height_ratio=125)
chrom1.deconvolve_peaks(Bemg())
chrom1.getMSspec()

#plot spectra
plt.figure()
for g in range(len(chrom1.peaks)):
    group = g
    nb_cmps = len(chrom1.peaks[group].all_maxima)
    for i in range(nb_cmps):
        plt.plot(chrom1.peaks[group].spectra[i])
plt.show()

plt.figure()
for g in range(len(chrom1.peaks)):
    group = g
    nb_cmps = len(chrom1.peaks[group].all_maxima)
    for i in range(nb_cmps):
        plt.plot(chrom1.peaks[group].spectra_MS[i])
plt.show()

chrom2.remove_blank()
chrom2.from_2D_to_1D()
chrom2.correct_baseline()
chrom2.remove_noise()
chrom2.detect_peaks(height_ratio=125)
chrom2.deconvolve_peaks(Bemg())
chrom2.getMSspec()

#plot spectra
plt.figure()
for g in range(len(chrom2.peaks)):
    group = g
    nb_cmps = len(chrom2.peaks[group].all_maxima)
    for i in range(nb_cmps):
        plt.plot(chrom2.peaks[group].spectra[i])
plt.show()

plt.figure()
for g in range(len(chrom2.peaks)):
    group = g
    nb_cmps = len(chrom2.peaks[group].all_maxima)
    for i in range(nb_cmps):
        plt.plot(chrom2.peaks[group].spectra_MS[i])
plt.show()

chrom3.remove_blank()
chrom3.from_2D_to_1D()
chrom3.correct_baseline()
chrom3.remove_noise()
chrom3.detect_peaks(height_ratio=125)
chrom3.deconvolve_peaks(Bemg())
chrom3.getMSspec()

#plot spectra
plt.figure()
for g in range(len(chrom3.peaks)):
    group = g
    nb_cmps = len(chrom3.peaks[group].all_maxima)
    for i in range(nb_cmps):
        plt.plot(chrom3.peaks[group].spectra[i])
plt.show()

plt.figure()
for g in range(len(chrom3.peaks)):
    group = g
    nb_cmps = len(chrom3.peaks[group].all_maxima)
    for i in range(nb_cmps):
        plt.plot(chrom3.peaks[group].spectra_MS[i])
plt.show()

chrom4.remove_blank()
chrom4.from_2D_to_1D()
chrom4.correct_baseline()
chrom4.remove_noise()
chrom4.detect_peaks(height_ratio=125)
chrom4.deconvolve_peaks(Bemg())
chrom4.getMSspec()

#plot spectra
plt.figure()
for g in range(len(chrom4.peaks)):
    group = g
    nb_cmps = len(chrom4.peaks[group].all_maxima)
    for i in range(nb_cmps):
        plt.plot(chrom4.peaks[group].spectra[i])
plt.show()

plt.figure()
for g in range(len(chrom4.peaks)):
    group = g
    nb_cmps = len(chrom4.peaks[group].all_maxima)
    for i in range(nb_cmps):
        plt.plot(chrom4.peaks[group].spectra_MS[i])
plt.show()



chroms = Chromatograms([chrom1, chrom2, chrom3, chrom4],True)
[retentionTimeTable, multi_peaks] = chroms.track_peaks(preprocessing=True, height_ratio=125, distance_ratio=400, remove_0_width=False)
print(retentionTimeTable)