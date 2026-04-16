# functions two deconvolute peaks
from matplotlib import pyplot as plt
from numpy.typing import NDArray
from typing import List, Tuple

from .classes import Peak
from .classes import PeakModel

def simplisma(d, nr, error):
    """
    Simplisma algorithm [1]. Finds 'purest' spectra based on the explained variance by spectra.

    Parameters
    ----------
    d : np.ndarray
        The input data.
    nr : int
        The number of purest spectra to find.
    error : float
        A small fraction of the maximmum mean in data to avoid division by zero.

    Returns
    -------
    np.ndarray
        The nb_purest purest spectra.
    """
    def wmat(c, imp, irank, jvar):
        dm = np.zeros((irank + 1, irank + 1))
        dm[0, 0] = c[jvar, jvar]

        for k in range(irank):
            kvar = int(imp[k])

            dm[0, k + 1] = c[jvar, kvar]
            dm[k + 1, 0] = c[kvar, jvar]

            for kk in range(irank):
                kkvar = int(imp[kk])
                dm[k + 1, kk + 1] = c[kvar, kkvar]

        return dm

    nrow, ncol = d.shape

    dl = np.zeros((nrow, ncol))
    imp = np.zeros(nr)
    mp = np.zeros(nr)

    w = np.zeros((nr, ncol))
    p = np.zeros((nr, ncol))
    s = np.zeros((nr, ncol))

    error = error / 100
    mean = np.mean(d, axis=0)
    error = np.max(mean) * error

    s[0, :] = np.std(d, axis=0)
    w[0, :] = (s[0, :] ** 2) + (mean ** 2)
    p[0, :] = s[0, :] / (mean + error)

    imp[0] = int(np.argmax(p[0, :]))
    mp[0] = p[0, :][int(imp[0])]

    l = np.sqrt((s[0, :] ** 2) + ((mean + error) ** 2))

    for j in range(ncol):
        dl[:, j] = d[:, j] / l[j]

    c = np.dot(dl.T, dl) / nrow

    w[0, :] = w[0, :] / (l ** 2)
    p[0, :] = w[0, :] * p[0, :]
    s[0, :] = w[0, :] * s[0, :]

    for i in range(nr - 1):
        for j in range(ncol):
            dm = wmat(c, imp, i + 1, j)
            w[i + 1, j] = np.linalg.det(dm)
            p[i + 1, j] = w[i + 1, j] * p[0, j]
            s[i + 1, j] = w[i + 1, j] * s[0, j]

        imp[i + 1] = int(np.argmax(p[i + 1, :]))
        mp[i + 1] = p[i + 1, int(imp[i + 1])]

    sp = np.zeros((nr,nrow))

    for i in range(nr):
        sp[i, 0:nrow] = d[0:nrow, int(imp[i])]

    concs = np.dot(d.T,np.linalg.pinv(sp))
    concs = concs.reshape(nr,ncol)

    return sp, concs

def merge_overlapping_peaks(data: NDArray, peaks: List[Peak]):
    """
    Merges any overlapping peaks

    Parameters
    ----------

    data: NDArray
        1D chromatogram data

    peaks: List[Peak]
        peaks that are checked for overlap and merged

    Returns
    -------

    List[Peak]
        List of new peaks. Non-overlapping peaks are copied as-is, and overlapping peaks are replaced by merged peaks

    """

    if len(peaks) < 2:
        return peaks

    sorted_peaks = sorted(peaks, key=lambda p: p.left)

    merged_peaks = [peaks[0]]
    for peak in sorted_peaks[1:]:
        if _overlap(merged_peaks[-1], peak) >= 0:
            merged_peaks[-1] = _merge_two_peaks(data, merged_peaks[-1], peak)
        else:
            merged_peaks.append(peak)

    return merged_peaks


def _overlap(p1: Peak, p2: Peak) -> int:
    """Returns overlap (or negative distance) of two peaks"""
    left = max(p1.left, p2.left)
    right = min(p1.right, p2.right)
    return right - left


def _merge_two_peaks(data: NDArray, p1: Peak, p2: Peak) -> Peak:
    """Merges two peaks"""
    larger = p1 if data[p1.maximum] > data[p2.maximum] else p2

    return Peak(
        left=min(p1.left, p2.left),
        right=max(p1.right, p2.right),
        maximum=larger.maximum,
        height=larger.height,
        prominence=larger.prominence,
        all_maxima=p1.all_maxima + p2.all_maxima
    )


# functions for deconvoltion

from numpy.typing import NDArray
from typing import List
from scipy.optimize import minimize
from scipy.signal import savgol_filter
from scipy.signal import find_peaks


def fit_peak_model(
        data: NDArray,
        model: PeakModel,
        peaks: Peak,
        spectra: NDArray | None = None,
        adjust_spectra: bool = True,
        initial_params: None | NDArray = None,
        initial_concs: None | NDArray = None,
) -> Tuple[NDArray, float, NDArray]:
    """
    Fits peak shapes to data. Spectra can be fixed or fitted too.

    Parameters
    ----------
    data: NDArray
        2D chromatogram data, [wavelength, time]

    model: PeakModel
        mathematical model of the peak shape

    peaks: Peak
            list of peaks

    spectra: NDArray | None
        absorption spectra of individual components, [components, wavelength]. If not provided, guess_spectra() is used.

    adjust_spectra: bool
        whether the spectra should be also fitted. When False, the spectra are kept constant

    initial_params: None | NDArray
        initial parameters for the models, flattened

    initial_concs: None | NDArray
        initial guess of the concentrations

    Returns
    -------
    NDArray:
        Concentrations [component, time]

    float
        Mean Squared Error

    NDArray:
        The optimized parameters, flattened

    """

    n_comps = len(peaks.all_maxima)

    # get spectra
    #maxima = np.asarray(peaks.all_maxima) - peaks.left
    #spectra = data[:, maxima].T
    # using simplisma
    spectra, initial_concs = simplisma(data, n_comps, 5)

    # Some generally useful variables
    t = np.arange(data.shape[1], dtype=float)
    n_params = model.n_params()

    bounds = model.get_bounds(t[-1])

    # Get initial guess of the model parameters
    if initial_params is None:
        p0 = []

        if initial_concs is None:
            # adjust spectra guess to mean=1
            spectra = (spectra.T / np.mean(spectra, axis=1)).T
            # add peaks and remove their contribution one by one
            residual_data = data.copy()
            for peak_index in range(spectra.shape[0]):
                # get concentrations guess
                concentrations, _ = concentrations_from_spectra(
                    residual_data, spectra[peak_index:]
                )
                # smooth out the concentrations
                for idx in range(concentrations.shape[0]):
                    concentrations[idx] = savgol_filter(
                        concentrations[idx], max(data.shape[1] // 20, 5), 2
                    )

                # find highest peak
                component, _ = np.unravel_index(
                    np.argmax(concentrations), concentrations.shape
                )

                conc = concentrations[component]

                peaks, info = find_peaks(conc, width=2, height=0, rel_height=0.3)
                if len(peaks) > 0:
                    idx = np.argmax(info["peak_heights"])
                    height = info["peak_heights"][idx]
                    mu = peaks[idx]
                    s1 = peaks[idx] - info["left_ips"][idx]
                    s2 = info["right_ips"][idx] - peaks[idx]

                #if len(peaks.all_maxima) > 0:
                #   idx = np.argmax(data[:,np.asarray(peaks.all_maxima) - peaks.left])
                #    height = np.mean(data[:,idx])
                #    mu = idx
                #    s1 = idx
                #    s2 = peaks.right - peaks.left - idx

                # otherwise use moments
                else:
                    height = np.max(conc)
                    mu = np.sum(conc * t) / (np.sum(conc) + 1e-7)
                    s1 = np.sqrt(
                        np.sum(conc * (t - mu) ** 2 * (t < mu))
                        / (np.sum(conc * (t < mu)) + 1e-7)
                    )
                    s2 = np.sqrt(
                        np.sum(conc * (t - mu) ** 2 * (t < mu))
                        / (np.sum(conc * (t < mu)) + 1e-7)
                    )

                # get the peak parameters
                params = model.init_guess(height, mu, s1 * 1.4, s2 * 1.4)
                p0.append(params)

                # substract data accounted for by the peak
                residual_data -= np.outer(
                    spectra[component + peak_index], model.val(t, *params)
                )

                # swap remove the spectrum guess
                spectra[[peak_index, component + peak_index]] = spectra[
                    [component + peak_index, peak_index]
                ]

        else:
            concentrations = initial_concs

            for conc in concentrations:
                # if peak is present use it to get initial parameters
                peaks, info = find_peaks(conc, width=2, height=0.0, rel_height=0.3)
                if len(peaks) > 0:
                    idx = np.argmax(info["peak_heights"])
                    height = info["peak_heights"][idx]
                    mu = peaks[idx]
                    s1 = peaks[idx] - info["left_ips"][idx]
                    s2 = info["right_ips"][idx] - peaks[idx]
                # otherwise use moments
                else:
                    height = np.max(conc)
                    mu = np.sum(conc * t) / (np.sum(conc) + 1e-7)
                    s1 = np.sqrt(
                        np.sum(conc * (t - mu) ** 2 * (t < mu))
                        / (np.sum(conc * (t < mu)) + 1e-7)
                    )
                    s2 = np.sqrt(
                        np.sum(conc * (t - mu) ** 2 * (t < mu))
                        / (np.sum(conc * (t < mu)) + 1e-7)
                    )

                # get initial guess
                params = model.init_guess(height, mu, s1 * 1.4, s2 * 1.4)

                # make sure the initial guess is within bounds
                for i, (low, high) in enumerate(bounds):
                    params[i] = np.clip(params[i], low, high)

                p0.append(params)

        initial_params = np.array(p0).flatten()

    def loss_and_grad(
            params: NDArray, data: NDArray, spectra: NDArray, t: NDArray
    ) -> Tuple[float, NDArray]:
        """Returns MSE and gradient. Would be good to test that this is correct."""
        # Reshape params to [component, param]
        if not adjust_spectra:
            params = params.reshape([n_comps, n_params])
        else:
            params = params.reshape([n_comps, n_params - 1])
            params = np.concatenate(
                [np.ones([n_comps, 1], dtype=float), params], axis=1
            )

        # Get concentrations [component, time]
        concs = np.array([model.val(t, *p) for p in params])

        # Get spectra
        if adjust_spectra:
            spectra = spectra_from_concentrations(data, concs)[0]

        # Get residues and mse
        residues = spectra.T @ concs - data
        mse = np.mean(residues ** 2)

        # Get gradients of reconstructed data [param, wl, time]
        if not adjust_spectra:
            sTc_grad = np.zeros([n_comps * n_params, data.shape[0], data.shape[1]])
            for comp_idx, param in enumerate(params):
                sTc_grad[comp_idx * n_params: (comp_idx + 1) * n_params] = np.einsum(
                    "w,pt->pwt", spectra[comp_idx], model.grad(t, *param)
                )
        else:
            sTc_grad = np.zeros(
                [n_comps * (n_params - 1), data.shape[0], data.shape[1]]
            )
            for comp_idx, param in enumerate(params):
                sTc_grad[
                    comp_idx * (n_params - 1): (comp_idx + 1) * (n_params - 1)
                ] = np.einsum("w,pt->pwt", spectra[comp_idx], model.grad(t, *param)[1:])

        # Get gradients fo square residues [param, wavele, time]
        res_grad = 2 * sTc_grad * residues

        # Average over wavelength and time axis
        grad = np.mean(res_grad, axis=(1, 2))

        # Return MSE
        return mse, grad

    # Prepare for the optimization
    if not adjust_spectra:
        bounds = bounds * n_comps
    else:
        bounds = bounds[1:] * n_comps

    # remove height from params if spectra can be adjusted
    if adjust_spectra:
        initial_params = initial_params.reshape([n_comps, n_params])
        initial_params = initial_params[:, 1:]
        initial_params = initial_params.flatten()

    # These methods seem to work well: L-BFGS-B (with increased maxls), TNC, SLSQP
    result = minimize(
        lambda params: loss_and_grad(params, data, spectra, t),
        x0=initial_params,
        jac=True,
        bounds=bounds,
        method="L-BFGS-B",
        options={"maxls": 100},
        tol=1e-10,
    )

    # it is not a big issue if the optimization does not converged, if this function is called iteratively
    if not result.success:
        warnings.warn(
            "fit_peak_model(): Optimization failed to converge: " + result.message
        )

    params = result.x
    # Reshape params to [component, param]
    if not adjust_spectra:
        params = params.reshape([len(params) // n_params, n_params])
    else:
        n_peaks = len(params) // (n_params - 1)
        params = params.reshape([n_peaks, n_params - 1])
        params = np.concatenate([np.ones([n_peaks, 1], dtype=float), params], axis=1)
    # Get concentrations [component, time]
    concs = np.array([model.val(t, *p) for p in params])

    return concs, result.fun, params.flatten()


def deconvolve_fixed(data: NDArray, model: PeakModel, relaxe_concs: bool) -> Tuple[
    NDArray, NDArray, float]:
    """
    Deconvolves data with given number of components. Returns concentration, spectra and MSE

    Parameters
    ----------
    data: NDArray
        2D data [wavelength, data]

    model: PeakModel
        mathematical model used for fitting shapes of components of peaks

    relaxe_concs: bool
        If False, the fitted bigaussian functions are returned

        Otherwise, the concentrations are refined with restricted least squares

    Returns
    -------
    NDArray
        concentrations [compound, time]

    NDArray
        spectra [compound, wavelength], normalized such that mean = 1

    float
        MSE

    """

    # Fit peak model
    concs, mse, _ = fit_peak_model(data, model)

    # Get spectra
    spectra, mse = spectra_from_concentrations(data, concs)

    # Normalize the spectra and scale concentrations accordingly
    norm_factors = np.mean(spectra, axis=1) + 1e-7
    spectra = (spectra.T / norm_factors).T
    concs = (concs.T * norm_factors).T

    return concs, spectra, mse


def concentrations_from_spectra(
        data: NDArray, spectra: NDArray
) -> Tuple[NDArray, float]:
    """
    Calculates positive concentrations such that the mean squared error is minimized. Returns concentrations and MSE

    Parameters
    ----------
    data: NDArray
        Absorbance data [wavelength, time]

    spectra: NDArray
        Spectra of the individual compounds [compound, wavelength]

    Returns
    -------
    NDArray
        Positive concentrations that minimize the MSE [compound, time]
    float
        MSE

    """

    assert (
            data.shape[0] == spectra.shape[1]
    ), "The shapes do not match in concentrations_from_spectra()"

    concs = []
    error = 0
    for t in range(data.shape[1]):
        try:
            conc, loss = nnls(spectra.T, data[:, t])
        except:
            # if failed to converge, solve with l2 norm
            conc, loss = nnls(spectra.T, data[:, t], l2=1e-3)

        concs.append(conc)
        error += loss ** 2

    concentrations = np.array(concs).T
    mse = error / data.shape[0] / data.shape[1]

    return concentrations, mse


def spectra_from_concentrations(
        data: NDArray, concentrations: NDArray
) -> Tuple[NDArray, float]:
    """
    Calculates positive spectra such that the mean squared error is minimized. Returns spectra and MSE

    Parameters
    ----------
    data: NDArray
        Absorbance data [wavelength, time]

    concentrations: NDArray
        Concentrations of the individual compounds [compound, time]

    Returns
    -------
    NDArray
        Positive spectra that minimize the MSE [compound, wavelength]
    float
        MSE
    """

    assert (
            data.shape[1] == concentrations.shape[1]
    ), "The shapes do not match in spectra_from_concentrations()"

    # Mathematically, this is identical problem, so reuse the other function
    return concentrations_from_spectra(data.T, concentrations)


import numpy as np
from scipy.linalg import lstsq, LinAlgWarning
import warnings


def nnls(A, b, maxiter=None, tol=None, l2=0.0):
    """
    Non-negative least squares solver, adapted from SciPy's nnls
    """
    m, n = A.shape

    AtA = A.T @ A + np.eye(n) * l2
    Atb = b @ A  # Result is 1D - let NumPy figure it out

    if not maxiter:
        maxiter = 3 * n
    if tol is None:
        tol = 10 * max(m, n) * np.spacing(1.0)

    # Initialize vars
    x = np.zeros(n, dtype=np.float64)
    s = np.zeros(n, dtype=np.float64)
    # Inactive constraint switches
    P = np.zeros(n, dtype=bool)

    # Projected residual
    w = Atb.copy().astype(np.float64)  # x=0. Skip (-AtA @ x) term

    # Overall iteration counter
    # Outer loop is not counted, inner iter is counted across outer spins
    iter = 0

    while (not P.all()) and (w[~P] > tol).any():  # B
        # Get the "most" active coeff index and move to inactive set
        k = np.argmax(w * (~P))  # B.2
        P[k] = True  # B.3

        # Iteration solution
        s[:] = 0.0
        # B.4
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message="Ill-conditioned matrix", category=LinAlgWarning
            )
            s[P] = lstsq(AtA[np.ix_(P, P)], Atb[P])[0]

        # Inner loop
        while (iter < maxiter) and (s[P].min() < 0):  # C.1
            iter += 1
            inds = P * (s < 0)
            alpha = (x[inds] / (x[inds] - s[inds])).min()  # C.2
            x *= 1 - alpha
            x += alpha * s
            P[x <= tol] = False
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore", message="Ill-conditioned matrix", category=LinAlgWarning
                )
                s[P] = lstsq(AtA[np.ix_(P, P)], Atb[P])[0]
            s[~P] = 0  # C.6

        x[:] = s[:]
        w[:] = Atb - AtA @ x

        if iter == maxiter:
            raise RuntimeError("NNLS: Iteration limit reached")

    return x, np.linalg.norm(A @ x - b)

