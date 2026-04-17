import numpy as np
import matplotlib.pyplot as plt

from chromprocessor.classes import Chromatogram

filename = "C:/Users/emery/OneDrive - KU Leuven/Chromatograms/"
# Eight chromatograms with different background levels look like this
data = np.genfromtxt(filename + 'chromatograms_and_noise.csv', skip_header=4, delimiter=',')
fig, axes = plt.subplots(1, 2, figsize=(15, 4))
for i in range(8):
    axes[0].plot(data[:, i], label=i)
    axes[1].plot(data[:, i], '.-', label=i)
axes[1].set_ylim(0, 100)
axes[1].set_xlim(1500, 3500)
axes[1].legend(ncol=4)
plt.show()

# We are going to use forth data + noise
#y = data[:, 5] + data[:, 8]
y = data[:, 5]
fig, axes = plt.subplots(1, 2, figsize=(15, 3))
axes[0].plot(y)
axes[1].plot(y)
axes[1].set_ylim(-10, 200)
plt.show()

chrom = Chromatogram(y, None, None, None, np.linspace(0,10,len(y)),None)
chrom.remove_blank()

# processing of chromatogram
chrom.correct_baseline(method = "flatfit")

plt.figure()
plt.plot(y)
plt.plot(chrom.chrom_1D)
plt.plot(chrom.baseline)
plt.ylim(-10, 200)
plt.show()
chrom.remove_noise()
plt.figure()
plt.plot(y)
plt.plot(chrom.chrom_1D)
plt.ylim(-10, 200)
plt.show()
