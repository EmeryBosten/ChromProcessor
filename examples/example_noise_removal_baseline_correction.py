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
y = data[:, 4] + data[:, 8]
#y = data[:, 5]
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
plt.rcParams.update({'font.size': 11.5})
plt.plot( np.linspace(0,10,len(y)),y)
plt.plot( np.linspace(0,10,len(y)),chrom.chrom_1D)
plt.plot( np.linspace(0,10,len(y)),chrom.baseline)
plt.ylim(-10, 200)
plt.show()
chrom.remove_noise()
plt.figure()
plt.plot(np.linspace(0,10,len(y)),y)
plt.plot(np.linspace(0,10,len(y)),chrom.chrom_1D)
plt.ylim(-10, 200)
plt.show()

ax1 = plt.subplot(2,1,1)
ax1.plot(np.linspace(0,10,len(y)),y, color = "blue")
ax1.set_ylim(-10, 300)
#ax2 = plt.subplot(3,2,2)
#ax2.plot(np.linspace(0,10,len(y)),y, color = "blue")
#ax2.set_ylim(-10, 200)
ax3 = plt.subplot(4,1,3)
ax3.plot(np.linspace(0,10,len(y)),y, linewidth=1.5, color = "blue")
ax3.plot(np.linspace(0,10,len(y)),chrom.chrom_1D, color = "green", linewidth = 1.5)
ax3.plot(np.linspace(0,10,len(y)),chrom.baseline, linestyle='--', color='red', linewidth=1)
ax4 = plt.subplot(4,2,7)
ax4.set_xlim(1.25, 2.5)
ax4.set_ylim(-20, 300)
ax4.plot(np.linspace(0,10,len(y)),y, color = "blue")
ax4.plot(np.linspace(0,10,len(y)),chrom.chrom_1D+chrom.baseline, color = "black")
ax4.plot(np.linspace(0,10,len(y)),chrom.chrom_1D, color = "green")

ax5 = plt.subplot(4,2,8)
ax5.set_xlim(3.125, 4.37)
ax5.set_ylim(-20, 550)
ax5.plot(np.linspace(0,10,len(y)),y, color = "blue")
ax5.plot(np.linspace(0,10,len(y)),chrom.chrom_1D + chrom.baseline, color = "black")
ax5.plot(np.linspace(0,10,len(y)),chrom.chrom_1D, color = "green")

plt.subplots_adjust(hspace=0.3)
plt.show()