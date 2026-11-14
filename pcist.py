"""
PCIst (Perturbational Complexity Index, state transitions variant), JAX
reimplementation. Canonical PCIst implementation for this project, used by
inference_utils.py (SBI) and directly in the notebooks for the empirical/
sliding window PCIst.

This stays close to the original pcist_jax.py on purpose: same function
names, same overall structure, same algorithm inside quantify_transitions.
Everything else here is just comments, docstrings and formatting.

Sections in this file:
    main PCIst pipeline      compute_pci and the steps it calls in order
    signal prep helpers       resample, reference, baseline correction
    batched / chunked PCI     the jit/vmap wrapped public entry points

"""
from functools import partial

import jax
import jax.numpy as jnp
import jax.scipy.fft
from jax.numpy import linalg
from jax import vmap, jit, lax
import numpy as np
from tqdm.auto import tqdm


cpu_device = jax.devices('cpu')[0]
if 'gpu' in [d.platform for d in jax.devices()]:
    gpu_device = jax.devices('gpu')[0]
else:
    gpu_device = None


##############################################################################
# MAIN PCIST PIPELINE
##############################################################################

def compute_pci(signal_evk, times, full_return=False, n_components=20,
                baseline_window=(-50, 0), response_window=(0, 500), k=1.2,
                min_snr=1.1, max_var=99, embed=False, n_steps=100):
    """
    Full PCIst pipeline for one trial: preprocess, SVD-reduce to n_components, run
    quantify_transitions, mask out low variance/SNR components, and sum the remaining dNST
    into one PCI value. full_return=True also returns every intermediate (SVD, SNR, masks,
    transition quantities) for inspection/plotting.
    """

    signal_evk, times = preprocess(signal_evk, times)

    signal_svd, var_exp, eigenvalues, snrs, component_mask = reduce_dimensions(
        signal_evk, times, response_window, n_components, max_var, min_snr,
        baseline_window=baseline_window)

    STQ = quantify_transitions(signal_svd, times, k, baseline_window, response_window,
                              embed, None, None, n_steps, 1.0)

    dNST_masked = STQ['dNST'] * component_mask
    dNST_final = jnp.maximum(dNST_masked, 0)
    PCI = jnp.sum(dNST_final)

    if full_return:
        STQ['dNST'] = dNST_final
        return {'PCI': PCI, **STQ, 'signal_evk': signal_evk, 'times': times,
                'signal_svd': signal_svd, 'eigenvalues': eigenvalues,
                'var_exp': var_exp, 'snrs': snrs, 'component_mask': component_mask}
    return PCI


def reduce_dimensions(signal, times, response_window, n_components, max_var, min_snr, baseline_window):
    """
    SVD-project signal onto its top n_components response window components, and build the
    mask (variance + SNR) marking which of those components actually get kept.
    """
    Vk, eigenvalues = get_svd(signal, times, response_window, n_components)
    var_exp_all = 100 * eigenvalues**2 / (jnp.sum(eigenvalues**2) + 1e-12)
    signal_svd = apply_svd(signal, Vk)

    var_mask = get_variance_mask(eigenvalues, max_var)
    snrs = compute_snr(signal_svd, times, baseline_window, response_window)
    snr_mask = snrs > min_snr
    component_mask = var_mask & snr_mask

    return signal_svd, var_exp_all, eigenvalues, snrs, component_mask


def compute_snr(signal_svd, times, baseline_window, response_window):
    """Response window power over baseline window power, per component, sqrt of the ratio."""
    base_mask = (times >= baseline_window[0]) & (times < baseline_window[1])
    resp_mask = (times >= response_window[0]) & (times < response_window[1])

    base_signal_sq = jnp.square(signal_svd)
    resp_signal_sq = jnp.square(signal_svd)

    base_mask_bc = base_mask[jnp.newaxis, :]
    resp_mask_bc = resp_mask[jnp.newaxis, :]

    base_power_sum = jnp.sum(jnp.where(base_mask_bc, base_signal_sq, 0.0), axis=1)
    base_count = jnp.sum(base_mask)
    base_power = base_power_sum / lax.cond(base_count > 0,
                                        lambda: base_count.astype(base_power_sum.dtype),
                                        lambda: 1.0)

    resp_power_sum = jnp.sum(jnp.where(resp_mask_bc, resp_signal_sq, 0.0), axis=1)
    resp_count = jnp.sum(resp_mask)
    resp_power = resp_power_sum / lax.cond(resp_count > 0,
                                         lambda: resp_count.astype(resp_power_sum.dtype),
                                         lambda: 1.0)

    snrs = jnp.sqrt(resp_power / base_power)
    snrs = jnp.nan_to_num(snrs, nan=0.0, posinf=0.0, neginf=0.0)
    return snrs


def get_svd(signal_evk, times, response_window, n_components):
    """
    SVD of the response window, right singular vectors and singular values, padded or
    truncated to exactly n_components.
    """
    ini_t, end_t = response_window
    resp_mask = (times >= ini_t) & (times < end_t)
    signal_resp = jnp.where(resp_mask[None, :], signal_evk, 0.0).T

    U, S, V = jnp.linalg.svd(signal_resp, full_matrices=False)
    V = V.T
    K = V.shape[1]

    def no_pad(_):
        return V[:, :n_components], S[:n_components]

    def do_pad(_):
        pad_v = n_components - K
        Vp = jnp.concatenate([V, jnp.zeros((V.shape[0], pad_v))], axis=1)
        Sp = jnp.concatenate([S, jnp.zeros(pad_v)], axis=0)
        return Vp[:, :n_components], Sp[:n_components]

    Vk, eigs = lax.cond(K >= n_components, no_pad, do_pad, operand=None)
    return Vk, eigs


def apply_svd(signal, V):
    """Project signal onto the SVD basis V (signal.T @ V, transposed back)."""
    return signal.T.dot(V).T


def get_variance_mask(eigenvalues, max_var):
    """
    Keep components up to max_var percent of cumulative explained variance (all of them if
    max_var >= 100).
    """
    if max_var >= 100:
        return jnp.ones(len(eigenvalues), dtype=bool)

    sorted_indices = jnp.argsort(eigenvalues)[::-1]
    eigenvalues_sorted = eigenvalues[sorted_indices]

    var_explained = eigenvalues_sorted ** 2
    total_var = jnp.sum(var_explained)
    cum_var = jnp.cumsum(var_explained) / total_var * 100

    exceeds_threshold = cum_var >= max_var
    n_keep = jnp.argmax(exceeds_threshold) + 1
    n_keep = jnp.where(jnp.any(exceeds_threshold), n_keep, len(eigenvalues))

    rank = jnp.argsort(sorted_indices)
    mask = rank < n_keep

    return mask


def quantify_transitions(signal, times, k, baseline_window, response_window,
                        embed, L, tau, n_steps, max_thr_p):
    """
    Core state-transition quantification: for each SVD component, sweep a range of
    recurrence thresholds and, at each one, count normalized state transitions (NST) in the
    baseline and response windows. Keeps the threshold that maximizes NST_response minus
    k*NST_baseline per component, dNST is that best difference scaled by the response window
    length. This is the memory heavy part: it builds and keeps the full (n_dims, N_prep,
    N_prep) transition matrix stack for every threshold step at once, since that turned out
    to be the fastest option on GPU.
    """
    n_dims = signal.shape[0]
    if n_dims == 0:
        return {'dNST': jnp.array([]), 'n_dims': 0}

    if embed and L is not None and tau is not None:
        cut = (L-1) * tau
        times = times[cut:]
        signal = vmap(partial(embed_signal, L=L, tau=tau))(signal)
    else:
        signal = signal[:, jnp.newaxis, :]

    N_prep = times.shape[0]
    base_mask_1d = (times >= baseline_window[0]) & (times < baseline_window[1])
    resp_mask_1d = (times >= response_window[0]) & (times < response_window[1])
    n_baseline = jnp.sum(base_mask_1d)
    n_response = jnp.sum(resp_mask_1d)
    base_mask_2d = base_mask_1d[:, jnp.newaxis] & base_mask_1d[jnp.newaxis, :]
    resp_mask_2d = resp_mask_1d[:, jnp.newaxis] & resp_mask_1d[jnp.newaxis, :]

    def compute_distances_for_dim(i):
        return compute_recurrence_matrix(signal[i], mode='distance')

    D_full = vmap(compute_distances_for_dim)(jnp.arange(n_dims))

    def get_thresholds_for_dim(i):
        d_full = D_full[i]
        db_masked = jnp.where(base_mask_2d, d_full, jnp.nan)
        min_thr = jnp.nanmedian(db_masked)
        dr_masked = jnp.where(resp_mask_2d, d_full, -jnp.inf)
        max_thr = jnp.nanmax(dr_masked) * max_thr_p
        max_thr = jnp.where(jnp.isneginf(max_thr), min_thr, max_thr)
        return jnp.linspace(min_thr, max_thr, n_steps)

    thresholds = vmap(get_thresholds_for_dim)(jnp.arange(n_dims))  # (n_dims, n_steps)

    # sweep every threshold and, for every single one, build the full
    # (n_dims, N_prep, N_prep) transition matrix and keep the whole stack
    # around. this is the memory heavy part (n_steps of those matrices), but
    # it is also what turned out to actually be fastest on GPU, see the module
    # docstring for why the "only keep two numbers per step" version got
    # reverted.
    def compute_for_threshold(step_idx):
        thr_vals = thresholds[:, step_idx]

        def compute_for_dim(j):
            transition_full = distance_to_transition(D_full[j], thr_vals[j])
            tb = jnp.where(base_mask_2d, transition_full, 0.0)
            tr = jnp.where(resp_mask_2d, transition_full, 0.0)
            nst_base = jnp.sum(tb) / jnp.maximum(1.0, n_baseline ** 2)
            nst_resp = jnp.sum(tr) / jnp.maximum(1.0, n_response ** 2)
            return nst_base, nst_resp, tb, tr

        nst_base, nst_resp, T_base, T_resp = vmap(compute_for_dim)(jnp.arange(n_dims))
        return nst_base, nst_resp, T_base, T_resp

    def body_fn(step_idx, carry):
        nst_base_all, nst_resp_all, T_base_all, T_resp_all = carry
        nst_base, nst_resp, T_base, T_resp = compute_for_threshold(step_idx)
        nst_base_all = nst_base_all.at[step_idx].set(nst_base)
        nst_resp_all = nst_resp_all.at[step_idx].set(nst_resp)
        T_base_all = T_base_all.at[step_idx].set(T_base)
        T_resp_all = T_resp_all.at[step_idx].set(T_resp)
        return (nst_base_all, nst_resp_all, T_base_all, T_resp_all)

    init_carry = (
        jnp.zeros((n_steps, n_dims)),
        jnp.zeros((n_steps, n_dims)),
        jnp.zeros((n_steps, n_dims, N_prep, N_prep)),
        jnp.zeros((n_steps, n_dims, N_prep, N_prep)),
    )
    NST_base, NST_resp, T_base, T_resp = lax.fori_loop(0, n_steps, body_fn, init_carry)

    NST_diff = NST_resp - k * NST_base
    ixs = jnp.argmax(NST_diff, axis=0)          # best step per dim, shape (n_dims,)
    dim_indices = jnp.arange(n_dims)
    max_thresholds = thresholds[dim_indices, ixs]   # thresholds is (dim, step), index dim then step (bugfix, see module docstring)
    dNST = NST_diff[ixs, dim_indices] * n_response  # NST_diff is (step, dim), index step then dim
    T_resp_final = T_resp[ixs, dim_indices]
    T_base_final = T_base[ixs, dim_indices]
    D_base_masked = jnp.where(base_mask_2d, D_full, 0.0)
    D_resp_masked = jnp.where(resp_mask_2d, D_full, 0.0)

    return {'dNST': dNST, 'n_dims': n_dims, 'D_base': D_base_masked, 'D_resp': D_resp_masked,
            'T_base': T_base_final, 'T_resp': T_resp_final, 'thresholds': thresholds,
            'NST_diff': NST_diff, 'NST_resp': NST_resp, 'NST_base': NST_base, 'max_thresholds': max_thresholds}


def compute_recurrence_matrix(signal, mode, thr=None):
    """
    Pairwise distance/recurrence/transition matrix for one (possibly embedded) signal
    dimension, mode picks which of the three.
    """
    if len(signal.shape) == 1:
        signal = signal[jnp.newaxis, :]
    diffs = signal[:, :, jnp.newaxis] - signal[:, jnp.newaxis, :]
    R = jnp.linalg.norm(diffs, ord=2, axis=0)
    mask = (R <= thr) if thr is not None else jnp.zeros(R.shape, dtype=bool)

    if mode == 'distance':
        return jnp.where(mask, 0.0, R)
    if mode == 'recurrence':
        return mask.astype(jnp.int32)
    if mode == 'transition':
        return diff_matrix(mask.astype(jnp.int32), symmetric=False)
    return jnp.zeros_like(R)


def distance_to_transition(dist_R, thr):
    """
    Threshold a distance matrix at thr into a binary recurrence matrix, then take its row-
    wise transition matrix.
    """
    mask = dist_R <= thr
    return diff_matrix(mask.astype(jnp.int32), symmetric=False)


def diff_matrix(A, symmetric=False):
    """
    Row wise abs diff of a binary matrix (marks where consecutive entries flip), optionally
    symmetrized.
    """
    B = jnp.abs(jnp.diff(A, axis=1))
    B = jnp.pad(B, ((0, 0), (0, 1)))
    if symmetric:
        B = (B + B.T)
        B = jnp.where(B > 0, 1, 0)
    return B


def embed_signal(x, L, tau):
    """Time delay embedding of a 1D signal: L delayed copies spaced tau samples apart."""
    n_times = x.shape[0]
    n_embedded = n_times - (L-1) * tau
    offsets = jnp.arange(L-1, -1, -1) * tau
    base_indices = jnp.arange(n_embedded)
    all_indices = offsets[:, jnp.newaxis] + base_indices[jnp.newaxis, :]
    return x[all_indices]


##############################################################################
# SIGNAL PREP HELPERS
##############################################################################

def resample_fourier(x, num, t=None, axis=0):
    """
    FFT based resampling of x (and optionally its time axis t) to num samples, real or
    complex input.
    """
    x = jnp.asarray(x)
    n_x = x.shape[axis]
    if n_x == num:
        if t is not None:
            return x, t
        return x

    if x.ndim > 1:
        x = jnp.moveaxis(x, axis, -1)

    s_fac = n_x / num
    m = min(num, n_x)
    m2 = m // 2 + 1

    if not jnp.iscomplexobj(x):
        X = jax.scipy.fft.rfft(x)
        X_r = jnp.zeros(X.shape[:-1] + (m2,), dtype=X.dtype)
        X_r = X_r.at[..., :m2].set(X[..., :m2])
        if m % 2 == 0 and num != n_x:
            scale = 2.0 if num < n_x else 0.5
            X_r = X_r.at[..., m//2].multiply(scale)
        x_r = jax.scipy.fft.irfft(X_r / s_fac, n=num)
    else:
        X = jax.scipy.fft.fft(x)
        Y = jnp.zeros(X.shape[:-1] + (num,), dtype=X.dtype)
        Y = Y.at[..., :m2].set(X[..., :m2])
        if m2 < m:
            Y = Y.at[..., m2-m:].set(X[..., m2-m:])
        if m % 2 == 0 and num < n_x:
            Y = Y.at[..., -m//2].add(X[..., -m//2])
        elif m % 2 == 0 and num > n_x:
            val = Y[..., m//2] / 2.0
            Y = Y.at[..., m//2].set(val)
            Y = Y.at[..., num-m//2].set(val)
        x_r = jax.scipy.fft.ifft(Y / s_fac)
        if not jnp.iscomplexobj(x):
            x_r = jnp.real(x_r)

    if x.ndim > 1:
        x_r = jnp.moveaxis(x_r, -1, axis)

    if t is not None:
        t = jnp.asarray(t)
        t_r = t[0] + (t[1] - t[0]) * s_fac * jnp.arange(num, dtype=t.dtype)
        return x_r, t_r
    return x_r


def preprocess(signal_evk, times):
    """
    No-op hook for optional preprocessing, kept for interface compatibility with the
    original pcist_jax.py.
    """
    return signal_evk, times


def average_reference(signal):
    """Subtract the across-channel mean at every timepoint (average reference montage)."""
    channels_mean = jnp.mean(signal, axis=0)[jnp.newaxis, :]
    return signal - channels_mean


def baseline_correct(Y, times, delta=0):
    """Subtract each channel's own mean over times < delta."""
    mask = times < delta
    mask_bc = mask[jnp.newaxis, :]
    baseline_sum = jnp.sum(jnp.where(mask_bc, Y, 0.0), axis=1)
    baseline_count = jnp.sum(mask)
    baseline_mean = baseline_sum / jnp.maximum(1.0, baseline_count.astype(baseline_sum.dtype))
    return Y - baseline_mean[:, jnp.newaxis]


##############################################################################
# BATCHED / CHUNKED PCI (PUBLIC ENTRY POINTS)
##############################################################################

def compute_pci_single(signal_evk, times, full_return=False, n_components=16, baseline_window=(-50, 0), response_window=(0, 500), k=1.2, min_snr=1.1, max_var=99, embed=False, n_steps=100):
    """
    jit wrapped compute_pci for one trial, static over every keyword argument. Public entry
    point used everywhere else in the project.
    """
    return compute_pci(signal_evk, times, full_return=full_return, n_components=n_components, baseline_window=baseline_window, response_window=response_window, k=k, min_snr=min_snr, max_var=max_var, embed=embed, n_steps=n_steps)
compute_pci_single = partial(jax.jit, static_argnames=('full_return', 'n_components', 'baseline_window', 'response_window', 'k', 'min_snr', 'max_var', 'embed', 'n_steps'), device=gpu_device)(compute_pci_single)


def compute_pci_batch(batch_signals, times, full_return=False, n_components=16, baseline_window=(-50, 0), response_window=(0, 500), k=1.2, min_snr=1.1, max_var=99, embed=False, n_steps=100):
    """
    compute_pci_single vmapped over a batch of trials (batch_signals: (n_trials, n_channels,
    n_times)).
    """
    return vmap(partial(compute_pci_single, times=times, full_return=full_return, n_components=n_components, baseline_window=baseline_window,
                        response_window=response_window, k=k, min_snr=min_snr, max_var=max_var, embed=embed, n_steps=n_steps))(batch_signals)
compute_pci_batch = partial(jax.jit, static_argnames=('full_return', 'n_components', 'baseline_window', 'response_window', 'k', 'min_snr', 'max_var', 'embed', 'n_steps'), device=gpu_device)(compute_pci_batch)


def compute_pci_batch_chunked(Y_batch, time_vec, batch_size=50, **pci_params):
    """
    Same as compute_pci_batch but processed in chunks so memory stays bounded instead of
    vmapping the whole batch at once. This is the algorithm's memory knob now that
    quantify_transitions builds the full per-threshold stack (see its docstring): lower
    batch_size if a run hits RESOURCE_EXHAUSTED, raise it if there's headroom to spare.
    """
    batch_size_total = Y_batch.shape[0]
    num_chunks = (batch_size_total + batch_size - 1) // batch_size
    pci_chunks = []

    for i in range(num_chunks):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, batch_size_total)
        chunk = Y_batch[start_idx:end_idx]

        pci_chunk = compute_pci_batch(chunk, time_vec, **pci_params)
        pci_chunks.append(pci_chunk)

    return jnp.concatenate(pci_chunks, axis=0)


def sliding_pci(eeg, time_ms, crop_window=(-450, 600), baseline_window=(-400, -50),
                 win_size=100, step_size=50, resp_start=0, resp_end=550,
                 show_progress=True, desc=None):
    """
    PCIst computed in sliding response windows (win_size, step_size) instead of one fixed
    window, per trial and channel.
    """
    crop_mask = (time_ms >= crop_window[0]) & (time_ms <= crop_window[1])
    times_cropped = jnp.array(time_ms[crop_mask])
    batch_signals = np.transpose(eeg[crop_mask], (2, 1, 0))  # (trials, channels, times)
    n_trials, n_channels = batch_signals.shape[0], batch_signals.shape[1]
    windows = [(t, t + win_size) for t in range(resp_start, resp_end - win_size + 1, step_size)]
    window_centers = np.array([np.mean(w) for w in windows])
    pci_per_trial = np.zeros((n_trials, len(windows)))
    window_iter = tqdm(list(enumerate(windows)), desc=desc or 'sliding PCIst', disable=not show_progress, leave=False)
    for w_idx, (t0, t1) in window_iter:
        for tr in range(n_trials):
            val = compute_pci_single(
                jnp.array(batch_signals[tr]), times_cropped, n_components=n_channels,
                baseline_window=baseline_window, response_window=(t0, t1), k=1.2, min_snr=1.1, max_var=99,
                embed=False, n_steps=100)
            pci_per_trial[tr, w_idx] = float(val)
    return window_centers, pci_per_trial
