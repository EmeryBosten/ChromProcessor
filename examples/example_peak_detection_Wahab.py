import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


from chromprocessor.classes import Chromatogram


base_path = Path(r"C:/Users/emery/OneDrive - KU Leuven/Chromatograms")

chrom = pd.read_excel(base_path / "ReplicationData_Wahabetal2024.xlsx", sheet_name = "Figure2").to_numpy()

x = chrom[:,0]
y = chrom[:,1]

#Initialize chromatogram
chrom = Chromatogram(y, None, None, None, x, None)

# processing of chromatogram
chrom.correct_baseline()
chrom.remove_noise()
chrom.detect_peaks()

for peak in chrom.peaks:
    print(x[peak.all_maxima])