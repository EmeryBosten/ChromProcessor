import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from chromprocessor.classes import Chromatogram


def generate_gaussians_general(overlap, sigma1=1.0, sigma2=1.5, amp1=1.0, amp2=0.8,

                               x_range=(-10, 10), num_points=1000):
    """

    Generate two Gaussian peaks with given normalized overlap,

    allowing different amplitudes and widths.



    overlap: desired normalized overlap (0 < overlap <= possible_max)

    sigma1, sigma2: standard deviations for Gaussian 1 and 2

    amp1, amp2: amplitudes for Gaussian 1 and 2

    """

    # Check maximum achievable overlap (when peaks coincide)

    max_overlap = np.sqrt((2 * sigma1 * sigma2) / (sigma1 ** 2 + sigma2 ** 2))

    if overlap > max_overlap:
        raise ValueError(f"Overlap cannot be greater than {max_overlap:.3f} for given σ values.")

    if overlap <= 0:
        raise ValueError("Overlap must be > 0.")

    # Compute separation Δμ from overlap formula

    delta_mu = np.sqrt(

        -2 * (sigma1 ** 2 + sigma2 ** 2) *

        np.log(overlap / max_overlap)

    )

    # Place peaks symmetrically around 0

    mu1 = -delta_mu / 2

    mu2 = +delta_mu / 2

    # Create x values and Gaussians

    x = np.linspace(x_range[0], x_range[1], num_points)

    g1 = amp1 * np.exp(- (x - mu1) ** 2 / (2 * sigma1 ** 2))

    g2 = amp2 * np.exp(- (x - mu2) ** 2 / (2 * sigma2 ** 2))

    return x, g1, g2, mu1, mu2, max_overlap


# Example usage

x, g1, g2, mu1, mu2, max_ov = generate_gaussians_general(

    overlap=0.6, sigma1=1.0, sigma2=1.5, amp1=1.0, amp2=0.5

)

plt.figure(figsize=(8, 4))

plt.plot(x, g1, label=f'Gaussian 1 (A={1.0}, σ={1.0})')

plt.plot(x, g2, label=f'Gaussian 2 (A={0.5}, σ={1.5})')

plt.plot(x, g1 + g2, '--', label='Sum')

plt.axvline(mu1, color='gray', linestyle=':')

plt.axvline(mu2, color='gray', linestyle=':')

plt.legend()

plt.title("Two Gaussian Peaks with Controlled Overlap, Different Amplitudes & Widths")

plt.show()

overlaps = [0.001, 0.01, 0.05, 0.1, 0.25, 0.33, 0.5, 0.75, 0.85, 0.9]
amplitudes = [1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
detectedPeaks = np.empty((len(overlaps), len(amplitudes)))
i = 0
for o in overlaps:
    j = 0
    for a in amplitudes:
        x, g1, g2, mu1, mu2, max_ov = generate_gaussians_general(

            overlap=o, sigma1=0.5, sigma2=0.5, amp1=1.0, amp2=a

        )
        chrom = Chromatogram(g1+g2, None, None, None, x, None)
        chrom.remove_blank()
        chrom.from_2D_to_1D()
        detectedPeaks[i,j] = chrom.detect_peaks()
        j = j + 1
    i = i + 1

print(pd.DataFrame(detectedPeaks))


