import numpy as np
import numba

#@numba.njit
def l1_norm(eigenvals):
    """
    Applies l1 norm to a set of eigenvalues.
    """
    # Because of the norm fn below, we can't use numba
    # Possible to redo the data, but not sure the benefit would be worth it
    #norm = np.linalg.norm(z, ord=1, axis=1, keepdims=True)
    norm = np.sum(np.abs(eigenvals))
    return eigenvals / norm


#@numba.njit
def dist_sq_from_pcs(v1, v2, xvec, yvec):
    """
    Given two matrices, calculates the distance between them.
    """
    Xt = np.dot(xvec, yvec.T)
    
    return (
        np.square(v1).sum()
        + np.square(v2).sum()
        - 2 * np.sum(v1 * np.sum(Xt * (v2 * Xt), axis=1))
    )

#@numba.njit
def construct_vi(X: np.ndarray, rank: int = 2):

    #cov = np.ma.cov(np.ma.array(X, mask=np.isnan(X)), rowvar=False)
    cov = np.cov(X, rowvar=False)
    # get the data covariance matrix
    U, S, V = np.linalg.svd(cov)

    #vals, vectors = np.linalg.eigh(cov)

    #vals = vals[:rank].real.astype(np.float64)
    #vectors = vectors[:, :rank].T.real.astype(np.float64)

    vals = S ** 2

    return vals[:rank], U[:, :rank].T


#@numba.njit
def row_wise_geom_mean(arr: np.ndarray):
    n_rows, n_cols = arr.shape
    geom_mean = np.zeros(n_rows)
    for row_i in np.arange(n_rows):
        mean = np.power(np.prod(arr[row_i]), 1 / n_cols)
        geom_mean[row_i] = mean
    return geom_mean


#@numba.njit
def clr(X: np.ndarray) -> np.ndarray:
    """
    perform a centered log-ratio transform
    """
    # the geometric mean acts as the center of the composition
    geom_mean = row_wise_geom_mean(X)
    clr = np.log(X / geom_mean.reshape(-1, 1))
    clr = replace_nan(clr)
    return clr


#@numba.njit
def replace_nan(arr: np.ndarray):
    shape = arr.shape
    arr_unraveled = arr.ravel()
    arr_unraveled[(np.isnan(arr_unraveled)) | (np.isinf(arr_unraveled))] = 0
    arr = arr_unraveled.reshape(shape)
    return arr

#@numba.njit
def convert_to_fraction(spectrum: np.ndarray):
    """CLR-transform the mutation spectrum in a window,
    treating the spectrum as the fraction of mutations in
    each k-mer context.

    Args:
        windowed_spectra (np.ndarray): _description_

    Returns:
        _type_: _description_
    """
    # calculate the sum of mutations in each sample in each window
    sample_totals = np.sum(spectrum, axis=1)
    # calculate the mutation fractions
    sample_fractions = spectrum / sample_totals.reshape(-1, 1)
    sample_fractions = replace_nan(sample_fractions)
    return sample_fractions


#@numba.njit
def compare_windows(a_spectra: np.ndarray, b_spectra: np.ndarray):

    # convert to fractions
    a_frac, b_frac = (
        convert_to_fraction(a_spectra),
        convert_to_fraction(b_spectra),
    )

    # get eigenvalues and eigenvectors of cov matrix
    va, Va = construct_vi(a_frac)
    vb, Vb = construct_vi(b_frac)

    # norm the eigenvalues
    va_norm, vb_norm = l1_norm(va), l1_norm(vb)

    distance = dist_sq_from_pcs(va_norm, vb_norm, Va, Vb)
    #distance = dist_sq_from_pcs(va, vb, Va, Vb)

    return distance


#@numba.njit(parallel=True)
def take_poisson_draws(lambdas: np.ndarray):
    n_rows, n_cols = lambdas.shape
    out_arr = np.zeros((n_rows, n_cols))
    for row_i in numba.prange(n_rows):
        for col_i in numba.prange(n_cols):
            draw = np.random.poisson(lambdas[row_i, col_i])
            out_arr[row_i, col_i] = draw
    return out_arr


#@numba.njit(parallel=True)
def take_binomial_draws(probabilities: np.ndarray, totals: np.ndarray):
    n_rows, n_cols = probabilities.shape
    out_arr = np.zeros((n_rows, n_cols))
    for row_i in numba.prange(n_rows):
        total = totals[row_i]
        for col_i in numba.prange(n_cols):
            draw = np.random.binomial(total, p=probabilities[row_i, col_i])
            out_arr[row_i, col_i] = draw
    return out_arr


#@numba.njit
def get_min_mut_count(a_spectra: np.ndarray, b_spectra: np.ndarray) -> int:
    # figure out the minimum total mutation count across
    # all samples in either window
    a_sample_min, b_sample_min = (
        min(np.sum(a_spectra, axis=1)),
        min(np.sum(b_spectra, axis=1)),
    )

    min_sample_total = min([
        a_sample_min, b_sample_min
    ])

    return min_sample_total

#@numba.njit
def rescale_window_pair(a_spectra: np.ndarray, b_spectra: np.ndarray):

    n_samples, n_kmers = a_spectra.shape

    # get total counts of mutations in each sample in either window
    a_totals = np.sum(a_spectra, axis=1)
    b_totals = np.sum(b_spectra, axis=1)

    a_totals[a_totals == 0] = 1
    b_totals[b_totals == 0] = 1

    # calculate the ratio of mutations in window a vs b
    a_vs_b_totals = a_totals / b_totals
    
    # figure out how many mutations to grab per sample
    a_muts, b_muts = np.zeros(n_samples), np.zeros(n_samples)
    for i in np.arange(n_samples):
        # if sample i has fewer mutations in window A than in B,
        # downsample that sample's mutation in window B
        if a_vs_b_totals[i] < 1:
            a_muts[i] = a_totals[i]
            b_muts[i] = b_totals[i] * a_vs_b_totals[i]
        # if it's the other way around
        elif a_vs_b_totals[i] >= 1:
            a_muts[i] = a_totals[i] * a_vs_b_totals[i]
            b_muts[i] = b_totals[i]

    # resample the mutations in each sample from a Poisson
    # distribution

    # first, convert mutation counts in each sample to fractions
    a_sample_fracs = convert_to_fraction(a_spectra)
    b_sample_fracs = convert_to_fraction(b_spectra)

    # multiply mutation fractions by the downweighted total
    a_sample_lambdas = a_sample_fracs * a_muts.reshape(-1, 1)
    b_sample_lambdas = b_sample_fracs * b_muts.reshape(-1, 1)

    # take a single draw from a Poisson distribution for each sample
    a_sample_counts = take_poisson_draws(a_sample_lambdas)
    b_sample_counts = take_poisson_draws(b_sample_lambdas)

    return a_sample_counts, b_sample_counts