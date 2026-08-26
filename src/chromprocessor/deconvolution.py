# functions to deconvolute peaks
from numpy.typing import NDArray
from typing import List, Tuple

import numpy as np
import warnings
from scipy.ndimage import gaussian_filter
from scipy.linalg import lstsq, LinAlgWarning

from .classes import Peak
from .classes import PeakModel

from scipy.optimize import minimize
from scipy.signal import savgol_filter
from scipy.signal import find_peaks


def simplisma(d, nr, error):
    """Simplisma algorithm: find 'pure' component spectra from multichannel data.

    This implementation follows the Simplisma approach: it identifies the most
    spectrally 'pure' columns (time points) and uses them to build an initial
    estimate of component spectra and concentrations.

    Parameters
    ----------
    d : array-like, shape (n_wavelengths, n_timepoints)
        Input absorbance (or intensity) matrix where rows correspond to spectral
        channels (wavelength or m/z) and columns to timepoints.
    nr : int
        Number of pure component spectra to extract.
    error : float
        Small positive percentage used to avoid division by zero in scoring.
        Typically a small value like 1-5 (percent).

    Returns
    -------
    sp : ndarray, shape (nr, n_wavelengths)
        Extracted 'purest' spectra (each row is a spectrum).
    concs : ndarray, shape (nr, n_timepoints)
        Estimated concentrations obtained by projecting the data onto the
        extracted spectra (non-optimized initial estimate).
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

    d = np.asarray(d)
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

    sp = np.zeros((nr, nrow))

    for i in range(nr):
        sp[i, 0:nrow] = d[0:nrow, int(imp[i])]

    concs = np.dot(d.T, np.linalg.pinv(sp))
    concs = concs.reshape(nr, ncol)

    return sp, concs


def merge_overlapping_peaks(data: NDArray, peaks: List[Peak]) -> List[Peak]:
    """Merge overlapping Peak objects into combined Peak instances.

    Parameters
    ----------
    data : NDArray
        1D chromatogram array used to decide which peak apex is dominant when
        merging two overlapping peaks.
    peaks : list of Peak
        Input peaks sorted/unsorted; overlapping entries will be merged.

    Returns
    -------
    merged_peaks : list of Peak
        New list of Peak instances with overlaps resolved.
    """

    if len(peaks) < 2:
        return peaks

    sorted_peaks = sorted(peaks, key=lambda p: p.left)

    merged_peaks = [sorted_peaks[0]]
    for peak in sorted_peaks[1:]:
        if _overlap(merged_peaks[-1], peak) >= 0:
            merged_peaks[-1] = _merge_two_peaks(data, merged_peaks[-1], peak)
        else:
            merged_peaks.append(peak)

    return merged_peaks


def _overlap(p1: Peak, p2: Peak) -> int:
    """Return overlap length (or negative distance) between two peaks.

    Positive result indicates overlapping samples count; negative means gap length.
    """
    left = max(p1.left, p2.left)
    right = min(p1.right, p2.right)
    return right - left


def _merge_two_peaks(data: NDArray, p1: Peak, p2: Peak) -> Peak:
    """Merge two overlapping peaks into a single Peak object.

    The resulting Peak spans the combined left/right bounds and uses the
    peak apex (maximum, height, prominence) of the higher-intensity peak.
    """
    larger = p1 if data[p1.maximum] > data[p2.maximum] else p2

    return Peak(
        left=min(p1.left, p2.left),
        right=max(p1.right, p2.right),
        maximum=larger.maximum,
        height=larger.height,
        prominence=larger.prominence,
        all_maxima=p1.all_maxima + p2.all_maxima,
    )


def estimate_background_noise(data: NDArray, peaks: List[Peak], mult: float = 4) -> float:
    """Estimate background noise level using regions outside detected peaks.

    The function iteratively computes the standard deviation of signal values
    outside peak windows and clips extreme values until convergence. The final
    value is used as a robust estimate of background noise (std).

    Parameters
    ----------
    data : NDArray
        1D chromatogram signal.
    peaks : list of Peak
        Detected peaks whose intervals will be excluded from noise estimation.
    mult : float, optional
        Clipping multiplier used to remove extreme values during iteration
        (default: 4).

    Returns
    -------
    float
        Estimated background standard deviation.
    """

    y = np.array(data)

    keep = np.ones(y.shape) == 1.0
    for peak in peaks:
        keep[peak.left:peak.right] = False

    y = y[keep]

    # avoid empty
    if len(y) == 0:
        y = np.array(data)
    min_std = np.std(y) * 0.01

    old_std = 0
    while True:
        std = np.std(y)
        if std < min_std:
            return min_std
        if np.abs(old_std - std) / std < 1e-5:
            return std
        y = np.clip(y, -mult * std, mult * std)
        old_std = std


def expand_peaks(data: NDArray, peaks: List[Peak], max_cutoff: float) -> List[Peak]:
    """Expand peak boundaries outward while signal remains significant.

    For each Peak instance, the left boundary is decreased while the smoothed
    signal or its derivative indicates continued peak contribution; similarly
    the right boundary is increased. This helps ensure tails of broad peaks are
    included before deconvolution.

    Parameters
    ----------
    data : NDArray
        1D chromatogram signal.
    peaks : list of Peak
        Peaks whose boundaries will be expanded in-place.
    max_cutoff : float
        Threshold derived from background noise (see estimate_background_noise).

    Returns
    -------
    peaks : list of Peak
        The modified list of Peak objects with updated left/right indices.
    """

    smooth = gaussian_filter(data, 1.0)
    deriv = gaussian_filter(data, 1.0, order=1)

    for p in peaks:
        while p.left > 0 and (smooth[p.left] > max_cutoff or deriv[p.left] > 0):
            p.left -= 1

        while p.right < smooth.shape[0] - 1 and (
            smooth[p.right] > max_cutoff or deriv[p.right] < 0
        ):
            p.right += 1

    return peaks


def fit_peak_model(
    data: NDArray,
    model: PeakModel,
    peaks: Peak | None = None,
    spectra: NDArray | None = None,
    n_comps: int | None = None,
    adjust_spectra: bool = True,
    initial_params: None | NDArray = None,
    initial_concs: None | NDArray = None,
) -> Tuple[NDArray, float, NDArray]:
    """Fit parametric peak shapes to 2D chromatographic data and optionally fit spectra.

    This routine fits a parametric model (provided by `model`) to explain the
    observed 2D data (wavelength x time) as a linear combination of component
    spectra and parametric concentration/time profiles. It supports either
    fixing spectra and fitting only concentration shapes, or jointly fitting
    both shapes and spectra (controlled by ``adjust_spectra``).

    Parameters
    ----------
    data : NDArray, shape (n_wavelengths, n_timepoints)
        Observed 2D chromatographic data (e.g., DAD: wavelengths x time).
    model : PeakModel
        An instance implementing the parametric peak shape (val, grad, init_guess, bounds).
    peaks : Peak, optional
        A Peak instance representing a (possibly merged) multi-component peak.
        The number of components defaults to ``len(peaks.all_maxima)`` when
        ``n_comps`` is not provided.
    spectra : NDArray, optional, shape (n_components, n_wavelengths)
        Initial spectral estimates for each component. If None, spectra are
        guessed from the data at reported maxima.
    n_comps : int, optional
        Number of parametric components to fit. If omitted, inferred from the
        provided Peak object.
    adjust_spectra : bool, default True
        If True, spectra are re-estimated inside the fitting loop; otherwise
        spectra are fixed and only the parametric concentration shapes are fitted.
    initial_params : ndarray, optional
        Flattened initial parameter vector for the optimizer. If None, sensible
        guesses are constructed from peak moments or detected local maxima.
    initial_concs : ndarray, optional
        Initial concentration guesses (n_components x n_timepoints). If provided
        these are used to initialise peak parameter guesses.

    Returns
    -------
    concentrations : ndarray, shape (n_components, n_timepoints)
        The fitted concentration/time profiles (one row per component).
    mse : float
        Mean squared error of the reconstruction: mean((S.T @ C - data)**2).
    params : ndarray
        Flattened optimized model parameters for all components.
    """

    if n_comps is None:
        n_comps = len(peaks.all_maxima)

    # get spectra
    if spectra is None:
        maxima = np.asarray(peaks.all_maxima) - peaks.left
        spectra = data[:, maxima].T
        # using simplisma
        # spectra, initial_concs = simplisma(data, n_comps, 5)

    # Some generally useful variables
    t = np.arange(data.shape[1], dtype=float)
    n_params = model.n_params()

    bounds = model.get_bounds(t[-1])

    # Get initial guess of the model parameters
    if initial_params is None:
        p0 = []

        if initial_concs is None:
            # Build initial guesses by iterative deflation of the data
            residual_data = data.copy()
            for peak_index in range(spectra.shape[0]):
                # get concentrations guess
                concentrations, _ = concentrations_from_spectra(residual_data, spectra[peak_index:])
                # smooth out the concentrations
                for idx in range(concentrations.shape[0]):
                    concentrations[idx] = savgol_filter(
                        concentrations[idx], max(data.shape[1] // 20, 5), 2
                    )

                # find highest peak
                component, _ = np.unravel_index(np.argmax(concentrations), concentrations.shape)

                conc = concentrations[component]

                peaks_loc, info = find_peaks(conc, width=2, height=0, rel_height=0.3)
                if len(peaks_loc) > 0:
                    idx = np.argmax(info["peak_heights"])
                    height = info["peak_heights"][idx]
                    mu = peaks_loc[idx]
                    s1 = peaks_loc[idx] - info["left_ips"][idx]
                    s2 = info["right_ips"][idx] - peaks_loc[idx]

                # otherwise use moments
                else:
                    height = np.max(conc)
                    mu = np.sum(conc * t) / (np.sum(conc) + 1e-7)
                    s1 = np.sqrt(
                        np.sum(conc * (t - mu) ** 2 * (t < mu)) / (np.sum(conc * (t < mu)) + 1e-7)
                    )
                    s2 = np.sqrt(
                        np.sum(conc * (t - mu) ** 2 * (t < mu)) / (np.sum(conc * (t < mu)) + 1e-7)
                    )

                # get the peak parameters
                params = model.init_guess(height, mu, s1 * 1.4, s2 * 1.4)
                p0.append(params)

                # subtract data accounted for by the peak
                residual_data -= np.outer(spectra[component + peak_index], model.val(t, *params))

                # swap/remove the spectrum guess
                spectra[[peak_index, component + peak_index]] = spectra[
                    [component + peak_index, peak_index]
                ]

        else:
            concentrations = initial_concs

            for conc in concentrations:
                # if peak is present use it to get initial parameters
                peaks_loc, info = find_peaks(conc, width=2, height=0.0, rel_height=0.3)
                if len(peaks_loc) > 0:
                    idx = np.argmax(info["peak_heights"])
                    height = info["peak_heights"][idx]
                    mu = peaks_loc[idx]
                    s1 = peaks_loc[idx] - info["left_ips"][idx]
                    s2 = info["right_ips"][idx] - peaks_loc[idx]
                # otherwise use moments
                else:
                    height = np.max(conc)
                    mu = np.sum(conc * t) / (np.sum(conc) + 1e-7)
                    s1 = np.sqrt(
                        np.sum(conc * (t - mu) ** 2 * (t < mu)) / (np.sum(conc * (t < mu)) + 1e-7)
                    )
                    s2 = np.sqrt(
                        np.sum(conc * (t - mu) ** 2 * (t < mu)) / (np.sum(conc * (t < mu)) + 1e-7)
                    )

                # get initial guess
                params = model.init_guess(height, mu, s1 * 1.4, s2 * 1.4)

                # make sure the initial guess is within bounds
                for i, (low, high) in enumerate(bounds):
                    params[i] = np.clip(params[i], low, high)

                p0.append(params)

        initial_params = np.array(p0).flatten()

    def loss_and_grad(params: NDArray, data: NDArray, spectra: NDArray, t: NDArray) -> Tuple[float, NDArray]:
        """
    Compute mean squared error and its gradient with respect to flattened parameters.

    Parameters
    ----------
    params : NDArray
        A 1D array of flattened parameters, each representing a parameter in the model.
    data : NDArray
        A 1D array of data values corresponding to the model's predictions.
    spectra : NDArray
        A 2D array of spectral data points.
    t : NDArray
        A 1D array of time points.

    Returns
    -------
    mse : float
        Mean squared error between the model's predictions and the actual data.
    grad : NDArray
        Gradient of the mean squared error with respect to each flattened parameter.
    """
        # Reshape params to [component, param]
        if not adjust_spectra:
            params = params.reshape([n_comps, n_params])
        else:
            params = params.reshape([n_comps, n_params - 1])
            params = np.concatenate([np.ones([n_comps, 1], dtype=float), params], axis=1)

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
            sTc_grad = np.zeros([n_comps * (n_params - 1), data.shape[0], data.shape[1]])
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
    Deconvolves data with given number of components. Returns concentration, spectra, and MSE

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


def nnls(A, b, maxiter=None, tol=None, l2=0.0):
    """
    Non-negative least squares solver with Tikhonov regularization

    Solves the optimization problem: minimize ||Ax - b||² subject to x ≥ 0
    with optional L2 regularization for improved numerical stability.

    Parameters:
        A (np.ndarray): Design matrix of shape (m, n) where m is the number
                        of observations and n is the number of variables
        b (np.ndarray): Target vector of shape (m,) containing observed values
        maxiter (int, optional): Maximum number of iterations for the solver
                               Defaults to 1000 if not provided
        tol (float, optional): Tolerance for convergence criteria
                           Defaults to 1e-6 if not provided
        l2 (float, optional): L2 regularization parameter (lambda)
                          Adds λI to the normal equations for regularization

    Returns:
        np.ndarray: Non-negative solution vector x of shape (n,)

    Notes:
        - This implementation is adapted from SciPy's nnls with modifications
          for improved numerical stability and regularization support
        - The solver uses a modified Gram-Schmidt approach with iterative refinement
        - L2 regularization helps prevent overfitting in ill-conditioned systems
        - The solution is guaranteed to be non-negative by construction
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

