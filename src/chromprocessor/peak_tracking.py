import numpy as np
from matplotlib import pyplot as plt


# functions for peak tracking
def angle_btv(yA, yB):
    """
    Compute normalized cosine-angle similarity between two vectors.

    Parameters
    ----------
    yA, yB : array-like
        Input vectors.

    Returns
    -------
    scos : float
        Normalized similarity score.
    """

    # Ensure input vectors are column vectors (1D arrays)
    yA = np.asarray(yA).reshape(-1)
    yB = np.asarray(yB).reshape(-1)

    # Compute cosine similarity
    dot_product = np.dot(yA, yB)
    norm_product = np.linalg.norm(yA) * np.linalg.norm(yB)

    if norm_product == 0:
        raise ValueError("One of the input vectors is zero, SCOS undefined.")

    cos_theta = dot_product / norm_product

    # Clip to avoid domain errors due to numerical precision
    cos_theta = np.clip(cos_theta, -1.0, 1.0)

    # Compute angle (in radians) between vectors
    theta = np.arccos(cos_theta)

    # Normalize by max angle (90 degrees = pi/2 radians)
    theta_max = np.pi / 2
    scos = 1 - (theta / theta_max)

    return scos


def near_si(v1, v2):
    """
    Nearness index (NEAR) based on normalized Euclidean distance,
    converted into a 0–1 normalized similarity index.

    Parameters
    ----------
    v1, v2 : array-like
        Input vectors.

    Returns
    -------
    si : float
        Similarity index in [0, 1]
    """

    # Convert inputs to NumPy arrays
    v1 = np.asarray(v1)
    v2 = np.asarray(v2)

    # Ensure row vectors (1D arrays)
    if v1.ndim > 1:
        v1 = v1.reshape(-1)
    if v2.ndim > 1:
        v2 = v2.reshape(-1)

    # Compute numerator and denominator
    numerator = np.dot(v1 - v2, v1 - v2)
    denominator = np.dot(v1 + v2, v1 + v2)

    if denominator == 0:
        raise ValueError("Denominator is zero; similarity index undefined.")

    si = 1 - np.sqrt(numerator / denominator)

    return si


def countHits(finalIndices1, finalIndices2):
    """
    Counts number of common MS peak hits.

    finalIndices1 : indices of top MS peaks of spectrum 1
    finalIndices2 : indices of top MS peaks of spectrum 2
    """
    finalIndices1 = np.sort(np.asarray(finalIndices1))
    finalIndices2 = np.sort(np.asarray(finalIndices2))

    hitCount = 0

    for idx in finalIndices1:
        if np.any(np.abs(finalIndices2 - idx) <= 3):
            hitCount += 1

    count = hitCount / max(len(finalIndices1), len(finalIndices2))
    return count


def SimMeasurementUV(numPeaks1, numPeaks2,
                     UVspec1, UVspec2,
                     heights1, heights2,
                     rrts1, rrts2):
    """
    Computes different similarity measurements between peak pairs
    of two chromatograms.

    UVspec1, UVspec2 : UV spectra of peaks (2D arrays, shape [n_wavelengths, n_peaks])
    heights1, heights2 : peak heights
    rrts1, rrts2 : relative retention times
    """

    simUV = np.zeros((numPeaks1, numPeaks2))
    heightSim = np.zeros((numPeaks1, numPeaks2))
    posSim = np.zeros((numPeaks1, numPeaks2))

    # Normalize heights
    heights1 = heights1 / np.max(heights1)
    heights2 = heights2 / np.max(heights2)

    for i in np.arange(numPeaks1):
        for j in np.arange(numPeaks2):
            # UV similarity
            ang = angle_btv(UVspec1[i], UVspec2[j])
            si = near_si(UVspec1[i], UVspec2[j])
            simUV[i, j] = (ang + si) / 2.0

            # Height similarity
            heightSim[i, j] = 1.0 - abs(heights1[i] - heights2[j])

            # Position similarity
            posSim[i, j] = 1.0 - abs(rrts1[i] - rrts2[j])

    return simUV, heightSim, posSim


def SimMeasurementMS(numPeaks1, numPeaks2,
                     heights1, heights2,
                     MSspec1, MSspec2,
                     rrts1, rrts2):
    """
    Computes different similarity measurements between peak pairs
    of two chromatograms.

    heights1, heights2 : peak heights
    MSspec1, MSspec2 : MS spectra
    rrts1, rrts2 : relative retention times
    """

    heightSim = np.zeros((numPeaks1, numPeaks2))
    posSim = np.zeros((numPeaks1, numPeaks2))
    countMS = np.zeros((numPeaks1, numPeaks2))

    # Normalize heights
    heights1 = heights1 / np.max(heights1)
    heights2 = heights2 / np.max(heights2)

    for i in range(numPeaks1):
        for j in range(numPeaks2):
            # Height similarity
            heightSim[i, j] = 1.0 - abs(heights1[i] - heights2[j])

            # MS similarity
            countMS[i, j] = countHits(
                MSpeaks(MSspec1[i]),
                MSpeaks(MSspec2[j])
            )

            # Position similarity
            posSim[i, j] = 1.0 - abs(rrts1[i] - rrts2[j])

    return heightSim, countMS, posSim


def chosenPairUV(simUV, heightSim, posSim):
    """
    Matches peaks across two different chromatograms.

    simUV      : similarity coefficients of UV spectra
    heightSim  : similarity measurement of peak height
    posSim     : position similarity
    """

    row = simUV.shape[0]

    # Compute the score for each pair
    Scores = ScoreFunctionUV(simUV, heightSim, posSim)

    # Find maximum score per row
    pairscore = np.max(Scores, axis=1)
    c1 = np.arange(row)  # MATLAB (1:row)' → Python 0-based indices

    # Handle mismatched matrix sizes
    if Scores.shape[0] != Scores.shape[1]:
        mismatch = Scores.shape[0] - Scores.shape[1]

        # Sort rows by score (ascending)
        idx = np.argsort(pairscore)[:mismatch]

        Scores = np.delete(Scores, idx, axis=0)
        c1 = np.delete(c1, idx)

    # Find best matching column for each row
    c2 = np.argmax(Scores, axis=1)

    pairs = np.column_stack((c1, c2))

    # ---- Check for duplicate matches ----
    if len(np.unique(pairs[:, 1])) != len(pairs[:, 1]):

        _, w = np.unique(pairs[:, 1], return_index=True)
        w = np.sort(w)

        duplicate_indices = np.setdiff1d(np.arange(len(pairs)), w)
        list_duplicates = np.unique(c2[duplicate_indices])

        for dup in list_duplicates:
            # Indices of duplicate matches
            inter_ind = np.where(pairs[:, 1] == dup)[0]

            # Find best pair among duplicates
            row_scores = np.max(Scores[inter_ind, :], axis=1)
            interMaxind = inter_ind[np.argmax(row_scores)]

            # Indices to reassign
            pairsToChangeInd = inter_ind[inter_ind != interMaxind]

            for idx in pairsToChangeInd:
                availPeaks = np.arange(Scores.shape[1])
                used = np.unique(pairs[:, 1])
                availPeaks = np.setdiff1d(availPeaks, used)

                if availPeaks.size == 0:
                    break

                secind = availPeaks[np.argmax(Scores[idx, availPeaks])]
                pairs[idx, 1] = secind

    return pairs, Scores


def ScoreFunctionUV(simUV, heightSim, posSim):
    """
    Computes similarity score of a peak pair.

    simUV     : similarity coefficients of UV spectra
    heightSim : similarity measurement of peak heights
    posSim    : position similarity
    """

    a = 1/3
    b = 1/3
    c = 1/3

    Score = (a * simUV + b * heightSim + c * posSim) / (a + b + c)

    return Score


def ScoreFunctionMS(heightSim, countMS, posSim):
    """
    Computes similarity score of a peak pair.

    heightSim : similarity measurement of peak heights
    countMS   : MS similarity measure
    posSim    : position similarity
    """

    a = 0.2
    b = 0.6
    c = 0.2

    Score = (a * heightSim + b * countMS + c * posSim) / (a + b + c)

    return Score


def chosenPairMS(heightSim, countMS, posSim):
    """
    Matches peaks across two different chromatograms.

    heightSim  : similarity measurement of peak height
    countMS    : MS similarity
    posSim     : position similarity
    """

    row = countMS.shape[0]

    # Compute the score for each pair
    Scores = ScoreFunctionMS(heightSim, countMS, posSim)

    # Find maximum score per row
    pairscore = np.max(Scores, axis=1)
    c1 = np.arange(row)  # MATLAB (1:row)' → Python 0-based indices

    # Handle mismatched matrix sizes
    if Scores.shape[0] != Scores.shape[1]:
        mismatch = Scores.shape[0] - Scores.shape[1]

        # Sort rows by score (ascending)
        idx = np.argsort(pairscore)[:mismatch]

        Scores = np.delete(Scores, idx, axis=0)
        c1 = np.delete(c1, idx)

    # Find best matching column for each row
    c2 = np.argmax(Scores, axis=1)

    pairs = np.column_stack((c1, c2))

    # ---- Check for duplicate matches ----
    if len(np.unique(pairs[:, 1])) != len(pairs[:, 1]):

        _, w = np.unique(pairs[:, 1], return_index=True)
        w = np.sort(w)

        duplicate_indices = np.setdiff1d(np.arange(len(pairs)), w)
        list_duplicates = np.unique(c2[duplicate_indices])

        for dup in list_duplicates:
            # Indices of duplicate matches
            inter_ind = np.where(pairs[:, 1] == dup)[0]

            # Find best pair among duplicates
            row_scores = np.max(Scores[inter_ind, :], axis=1)
            interMaxind = inter_ind[np.argmax(row_scores)]

            # Indices to reassign
            pairsToChangeInd = inter_ind[inter_ind != interMaxind]

            for idx in pairsToChangeInd:
                availPeaks = np.arange(Scores.shape[1])
                used = np.unique(pairs[:, 1])
                availPeaks = np.setdiff1d(availPeaks, used)

                if availPeaks.size == 0:
                    break

                secind = availPeaks[np.argmax(Scores[idx, availPeaks])]
                pairs[idx, 1] = secind

    return pairs, Scores


def MSpeaks(msspec):
    """
    Finds most prominent MS peaks

    msspec  : MS spectrum
    """

    msspec = np.asarray(msspec)

    # Step 1: Get top 15 max values and their indices
    indices = np.argpartition(msspec, -5)[-5:]
    stored = msspec[indices]

    # Step 2: Sort indices for easier grouping
    sort_order = np.argsort(indices)
    sorted_indices = indices[sort_order]
    sorted_values = stored[sort_order]

    # Step 3: Initialize grouping
    groups = []
    current_group = [sorted_indices[0]]
    current_values = [sorted_values[0]]

    for i in range(1, len(sorted_indices)):
        if sorted_indices[i] - sorted_indices[i - 1] <= 5:
            # Same group
            current_group.append(sorted_indices[i])
            current_values.append(sorted_values[i])
        else:
            # New group
            groups.append({
                "indices": current_group,
                "values": current_values
            })
            current_group = [sorted_indices[i]]
            current_values = [sorted_values[i]]

    # Add the last group
    groups.append({
        "indices": current_group,
        "values": current_values
    })

    # Step 4: From each group, select the index with the highest value
    final_indices = []
    for g in groups:
        max_idx = np.argmax(g["values"])
        final_indices.append(g["indices"][max_idx])

    # Step 5: Sort the final indices
    final_indices = np.sort(final_indices)

    return final_indices

