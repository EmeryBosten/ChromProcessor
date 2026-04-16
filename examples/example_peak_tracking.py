import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from chromprocessor.classes import Chromatogram
from chromprocessor.classes import Chromatograms
from chromprocessor.peak_models import BiGaussianTailing

path = "C:/Users/emery/OneDrive - KU Leuven/Chromatograms/Agilent JnJ/Batch Processing"

def partitioning(chromatogram,start,end):
    if chromatogram.ndim == 1:
        chromatogram = chromatogram[int(start*len(chromatogram)): int(end*len(chromatogram))]
    else :
        chromatogram = chromatogram[:, int(start*np.shape(chromatogram)[1]): int(end*np.shape(chromatogram)[1])]
    return chromatogram


#Chromatogram 1

data = pd.read_csv(path +  "/UV/UV3.CSV")

timeUV = partitioning(data["time"].to_numpy(),1/10,1/2)
chromUV = partitioning(data["ordinate_values"].to_numpy(),1/10,1/2)
chromMS = partitioning(np.load(path + "/MS/MS3.npy"),1/10,1/2)
timeMS = partitioning(np.load(path + "/MS/time_MS3.npy"),1/10,1/2)
timeUVnew = np.linspace(timeUV[0],timeUV[-1], 2000)
chromUV = np.interp(timeUVnew, timeUV, chromUV)

# Initialize chromatogram
chrom1 = Chromatogram(chromUV,None,chromMS,None,timeUVnew,timeMS)
chrom1.remove_blank()
chrom1.from_2D_to_1D()

#Chromatogram 2
data = pd.read_csv(path +  "/UV/UV5.CSV")

timeUV = partitioning(data["time"].to_numpy(),1/10,1/2)
chromUV = partitioning(data["ordinate_values"].to_numpy(),1/10,1/2)
chromMS = partitioning(np.load(path + "/MS/MS5.npy"),1/10,1/2)
timeMS = partitioning(np.load(path + "/MS/time_MS5.npy"),1/10,1/2)
timeUVnew = np.linspace(timeUV[0],timeUV[-1], 2000)
chromUV = np.interp(timeUVnew, timeUV, chromUV)

# Initialize chromatogram
chrom2 = Chromatogram(chromUV,None,chromMS,None,timeUVnew,timeMS)
chrom2.remove_blank()
chrom2.from_2D_to_1D()

#Chromatogram 3
data = pd.read_csv(path +  "/UV/UV8.CSV")

timeUV = partitioning(data["time"].to_numpy(),1/10,1/2)
chromUV = partitioning(data["ordinate_values"].to_numpy(),1/10,1/2)
chromMS = partitioning(np.load(path + "/MS/MS8.npy"),1/10,1/2)
timeMS = partitioning(np.load(path + "/MS/time_MS8.npy"),1/10,1/2)

# Initialize chromatogram
chrom3 = Chromatogram(chromUV,None,chromMS,None,timeUV,timeMS)
chrom3.remove_blank()
chrom3.from_2D_to_1D()

#Chromatogram 4
data = pd.read_csv(path +  "/UV/UV10.CSV")

timeUV = partitioning(data["time"].to_numpy(),1/10,1/2)
chromUV = partitioning(data["ordinate_values"].to_numpy(),1/10,1/2)
chromMS = partitioning(np.load(path + "/MS/MS10.npy"),1/10,1/2)
timeMS = partitioning(np.load(path + "/MS/time_MS10.npy"),1/10,1/2)

# Initialize chromatogram
chrom4 = Chromatogram(chromUV,None,chromMS,None,timeUV,timeMS)
chrom4.remove_blank()
chrom4.from_2D_to_1D()

#Chromatogram 5
data = pd.read_csv(path +  "/UV/UV11.CSV")

timeUV = partitioning(data["time"].to_numpy(),1/10,1/2)
chromUV = partitioning(data["ordinate_values"].to_numpy(),1/10,1/2)
chromMS = partitioning(np.load(path + "/MS/MS11.npy"),1/10,1/2)
timeMS = partitioning(np.load(path + "/MS/time_MS11.npy"),1/10,1/2)

# Initialize chromatogram
chrom5 = Chromatogram(chromUV,None,chromMS,None,timeUV,timeMS)
chrom5.remove_blank()
chrom5.from_2D_to_1D()

chroms = [chrom1, chrom2, chrom3, chrom4, chrom5]
chroms = Chromatograms(chroms,True)
[retentionTimeTable, multi_peaks] = chroms.track_peaks(preprocessing=True, height_ratio=10, width=0, distance_ratio=400, remove_0_width = True)

print(retentionTimeTable)

plt.figure()
for chrom in multi_peaks:
    plt.plot(chrom[0].spectra_MS[0])

plt.show()

for chrom in multi_peaks:
    plt.figure()
    for peak in chrom:
        for cmp in peak.spectra_MS:
            plt.plot(cmp)
    plt.show()