<img width="533" height="385" alt="Logo" src="https://github.com/user-attachments/assets/42b8c838-b7df-4a1a-82d9-99a4771b5d4d" />

# Chromprocessor

Python tools for chromatogram parsing, preprocessing, peak detection, tracking, and deconvolution.

## Install

git clone https://github.com/EmeryBosten/ChromProcessor.git

## Quick Start

The example below demonstrates a complete chromatographic processing workflow:

Load a chromatogram from a CSV file
Remove blank contributions
Correct the baseline
Remove noise
Detect chromatographic peaks
Visualize the results

## 1. Load and initialize chromatogram

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from chromprocessor.classes import Chromatogram

# Load chromatogram
base_path = Path(r"./data")

chrom = pd.read_csv(base_path / "MIX_F10.CSV", sep="\t", encoding="utf-16").to_numpy()

#time
x = chrom[:,0]
#signal
y = chrom[:,1]

#add some noise
noise = 0.00 * (np.random.rand(len(y)) - 0.5)
y = y + noise

#Initialize chromatogram
chrom = Chromatogram(y, None, None, None, x, None)
chrom.remove_blank()
chrom.from_2D_to_1D()
```

## 2. Process chromatogram

```python
chrom.correct_baseline()
```
<img width="640" height="480" alt="baseline_correction" src="https://github.com/user-attachments/assets/e1e61569-5b6a-43eb-93e6-bcd089530a66" />
```python
chrom.remove_noise()
```
<img width="640" height="480" alt="noise_removal" src="https://github.com/user-attachments/assets/110c4c4a-147a-4201-a3ee-6e591e9ada72" />
```python
chrom.detect_peaks()
```
<img width="640" height="480" alt="peak_detection" src="https://github.com/user-attachments/assets/e4c2b94c-bd98-46f9-8d2c-6afd1b709c29" />

```

## 3. Generate additional Figures

```python
plt.rcParams.update({'font.size': 11.5})
fig, axs = plt.subplots(3,1)
axs[0].plot(chrom.time,chrom.chrom_1D,linewidth=2)
#plt.xlabel("Time", fontsize=12.5)
#plt.ylabel("Absorbance", fontsize=12.5)
for peak in chrom.peaks:
    axs[0].plot(chrom.time[peak.all_maxima],chrom.chrom_1D[peak.all_maxima], marker='*', linestyle='None')


filenames = [
"26.CSV", "30.CSV", "31.CSV", "35.CSV", "37.CSV", "38.CSV", "39.CSV", "51.CSV", "55.CSV", "58.CSV", "59.CSV", "60.CSV", "62.CSV", "69.CSV", "76.CSV", "78.CSV"
]
cmps = [pd.read_csv(base_path / f, sep="\t", encoding="utf-16").to_numpy() for f in filenames]

for cmp in cmps:
    axs[2].plot(cmp[:,0], cmp[:,1], linewidth=2)

for peak in chrom.peaks:
    axs[2].plot(chrom.time[peak.all_maxima],0, marker='|', linestyle='None', color = "red", linewidth=2)

axs[1].plot(chrom.time[:-1],np.diff(y), color="blue",linewidth=2)
axs[1].plot(chrom.time[:-1],np.diff(chrom.chrom_1D), color="black",linewidth=1)


#plt.plot(x[apex_loc], -5*np.ones_like(apex_loc), '|r', linewidth=1)
#plt.xlabel("Time", fontsize=12.5)
#plt.ylabel("Absorbance", fontsize=12.5)
plt.show()
```

<img width="640" height="480" alt="peak_detection_results" src="https://github.com/user-attachments/assets/4018f2f7-a728-44bf-bf4b-96da56a37d21" />

## Citation
publication in process

## Licence 
See License "PolyForm NonCommercial 1.0.0.txt" for details



