"""
Gain matrices, SBI dataset construction, SNPE training and posterior
sampling, and the pipeline (0 to 4) branching logic that ties them together.

This is the code behind 03_sbi.ipynb. The five pipelines all follow the same
pattern (compute gain matrix or matrices -> project single channel
simulations to sensor space -> compute PCIst -> encode the condition/subject/
trial as extra feature columns -> train SNPE -> sample the posterior given
the empirical PCIst), they just differ in how many gain matrices get used and
how much gets encoded alongside the PCIst value. See the notebook for the
narrative explanation of each pipeline, this module is just the mechanics.
"""
import os
import errno
import gc
import pickle
from copy import deepcopy

import numpy as np
import pandas as pd
import xarray as xr
import sklearn as skl
import jax
import jax.numpy as jnp
import torch
import sbi
import sbi.utils
import sbi.inference
from tqdm import trange
from tqdm.auto import tqdm
from scipy.stats import gaussian_kde

import pcist
import jansen_rit


cpu_device = jax.devices('cpu')[0]
gpu_device = jax.devices('gpu')[0] if 'gpu' in [d.platform for d in jax.devices()] else None

# The four condition labels present across the whole dataset. Used for the
# one hot encodings in pipeline 2 and up.
CONDITIONS = ['wake', 'keta', 'prop', 'sevo']


# ---------------------------------------------------------------------------
# Gain matrix and forward projection
# ---------------------------------------------------------------------------

def compute_gain_matrix_jax(sensor_erps, n_comp=4, weight_by_variance=True):
    """
    Compute a gain matrix from a (time, channel) empirical ERP array: SVD it
    and keep the top n_comp right singular vectors (the spatial/channel-space
    principal components). The result maps n_comp single-channel "modes"
    onto the real sensor channels, shape (n_channels, n_comp).

    weight_by_variance=False (the default of this function, kept for
    backwards compatibility and for the "baseline" demonstration in
    03_sbi.ipynb) normalizes every column to unit norm, so the resulting G
    always has orthonormal columns no matter which empirical data it was
    built from. That has a real consequence: for the forward projection +
    PCIst pipeline used throughout this project (single source, fixed sample
    delays, then an SVD-based summary), the final PCIst value turns out to
    be exactly invariant to which orthonormal G you use, see the "Gain
    matrix invariance" section in 03_sbi.ipynb for the proof and a numerical
    demonstration. weight_by_variance=True instead scales each column by its
    share of the singular value energy (S / sum(S)) before returning, which
    breaks that invariance by letting the relative weight of each of the
    n_comp modes reflect how much of the empirical data's variance that mode
    actually explained. 03_sbi.ipynb sets this to True everywhere via
    compute_shared_gain's own weight_by_variance argument (below), precisely
    so Pipelines 1 to 4 have a real (if still limited) spatial signal to
    condition on instead of an exactly invariant one.
    """
    U, s, Vt = jnp.linalg.svd(sensor_erps, full_matrices=False)
    G = Vt[:n_comp].T
    S = s[:n_comp]
    G_unit = G / jnp.linalg.norm(G, axis=0)
    if weight_by_variance:
        weights = S / jnp.sum(S)
        G_out = G_unit * weights[None, :] * n_comp
    else:
        G_out = G_unit
    return G_out, S
compute_gain_matrix_jax = jax.jit(compute_gain_matrix_jax, static_argnames=('n_comp', 'weight_by_variance'))


def compute_forward_solution_multimode_jax(source_evoked, gain_matrix, dmin=0, dmax=330):
    """
    Project one single channel simulated ERP to sensor space using a gain
    matrix. Since the model only produces one time series but the gain
    matrix has n_comp modes, this gives each mode a slightly different
    integer sample delay (linearly spaced between dmin and dmax) before
    projecting through its gain matrix column and summing. The delays are
    what keep the projected sensor data from being perfectly rank 1, without
    them every channel would just be a scaled copy of the same trace and
    PCIst's SVD step would have nothing to work with. This is a modeling
    convenience, not a claim that these delays exist physiologically, see the
    "Gain matrix: concept" section in the notebook.

    dmax=330 is the delay range decision from 03_sbi.ipynb's "Choosing the
    delay range" section, picked in the case of weighted gain matrix.
    330 is kept as a default here rather than passed explicitly at every call site so
    Pipelines 0 to 4 all use it consistently without having to repeat it everywhere.
    
    For unweigthed we use another dmax value and in the unweigthed subsection we manually set dmax to the alternative value.
    """
    n_ch, n_comp = gain_matrix.shape
    delays = jnp.linspace(dmin, dmax, n_comp).astype(jnp.int32)
    n_t = len(source_evoked)

    def single_mode_contribution(accumulator, ind):
        d = jnp.squeeze(jnp.take(delays, ind))
        t = jnp.arange(n_t)

        src_shifted = jnp.roll(source_evoked, d)
        mask_pos = (d > 0) & (t < d)
        mask_neg = (d < 0) & (t >= n_t + d)
        zero_mask = mask_pos | mask_neg
        src_shifted = jnp.where(zero_mask, 0.0, src_shifted)

        Y_contrib = jnp.outer(src_shifted, gain_matrix[:, ind])
        return None, Y_contrib

    _, Y_contributions = jax.lax.scan(single_mode_contribution, None, jnp.arange(n_comp))
    Y = jnp.sum(Y_contributions, axis=0)
    return Y.T
compute_forward_solution_multimode_jax = jax.jit(compute_forward_solution_multimode_jax, device=cpu_device)


def compute_forward_solution_multimode_jax_batch(sim_array_batch, gain_matrix, dmin=0, dmax=330, batch_size=100):
    """Same as compute_forward_solution_multimode_jax but vmapped over a batch of simulations, processed in chunks to keep memory bounded."""
    batch_size_total, n_t = sim_array_batch.shape
    num_chunks = (batch_size_total + batch_size - 1) // batch_size
    Y_chunks = []

    for i in range(num_chunks):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, batch_size_total)
        chunk = sim_array_batch[start_idx:end_idx]

        def single_simulation_forward(sim_signal):
            return compute_forward_solution_multimode_jax(sim_signal, gain_matrix, dmin, dmax)

        batch_forward = jax.vmap(single_simulation_forward)
        Y_chunks.append(batch_forward(chunk))

    return jnp.concatenate(Y_chunks, axis=0)


def get_anes_cond_label(erp_data):
    """Return the non wake condition label present in an ERP DataArray, e.g. 'keta'."""
    return [c for c in np.unique(erp_data.condition.values) if c != 'wake'][0]


def compute_gains_subject(erp_data, subject, num_components=4, weight_by_variance=False):
    """Wake and anesthesia gain matrix for one subject. Not used by the pipeline 0 to 4 flow (which computes gains per pipeline directly via compute_shared_gain), kept as a small standalone convenience for ad hoc per-subject checks."""
    gains = {}
    anes_cond = get_anes_cond_label(erp_data)
    for cond in ['wake', anes_cond]:
        erp = erp_data.sel(subject=subject, condition=cond).dropna('region').to_numpy().copy()
        G, _ = compute_gain_matrix_jax(erp, n_comp=num_components, weight_by_variance=weight_by_variance)
        gains[cond] = G
    return gains


def _pool_wake_erp_per_subject(erp_datasets, anes_keys):
    """
    Average a subject's wake ERP across every anesthesia session they appear
    in. Wake is duplicated once per anesthesia subfolder on disk (see
    data.py's module docstring), but it is the same underlying condition
    every time, recorded alongside a different anesthetic on a different
    day. Used by compute_shared_gain (pipeline 3) so wake gets one gain
    matrix per subject instead of one per subject per session, the same way
    pipeline 2 already pools wake across subjects before computing its
    single wake gain matrix.

    Crops to the smallest common channel count per subject first, in case
    one session happened to have a channel or two fewer than another
    (differing bad channels between recordings).
    """
    per_subject = {}
    for anes_name in anes_keys:
        erp_data = erp_datasets[anes_name]
        for subj in erp_data.subject.values:
            wake_erp = erp_data.sel(condition='wake', subject=subj).dropna('region').to_numpy().copy()
            per_subject.setdefault(subj, []).append(wake_erp)

    pooled = {}
    for subj, erps in per_subject.items():
        n_ch_common = min(e.shape[-1] for e in erps)
        pooled[subj] = np.mean(np.stack([e[..., :n_ch_common] for e in erps]), axis=0)
    return pooled


def _pool_wake_erp_per_subject_trial_resolved(trial_datasets, anes_keys):
    """
    Same idea as _pool_wake_erp_per_subject, but starting from trial-resolved
    data: average a subject's wake trials within each session first, then
    average those session-level means across sessions. Used by
    compute_shared_gain (pipeline 4) to give wake a single, stable
    per-subject gain matrix.

    This is deliberately not trial-resolved on the wake side, unlike
    anesthesia: wake trials and anesthesia trials are recorded independently
    (no shared stimulus timing), so "wake trial 0" and "anesthesia trial 0"
    aren't paired observations the way a genuine single anesthesia trial is
    its own real response. Per-trial personalization is meaningful for
    anesthesia (each trial is a real, distinct observation), not for wake,
    so wake gets one stable per-subject reference instead of being forced
    into an arbitrary trial-to-trial pairing.

    Crops to the smallest common channel count per subject first, same as
    _pool_wake_erp_per_subject.
    """
    per_subject = {}
    for anes_name in anes_keys:
        trial_data = trial_datasets[anes_name]
        for subj in trial_data.subject.values:
            wake_trials = trial_data.sel(condition='wake', subject=subj).dropna('region').to_numpy().copy()
            session_mean = wake_trials.mean(axis=0)  # (trial, time, region) -> (time, region)
            per_subject.setdefault(subj, []).append(session_mean)

    pooled = {}
    for subj, erps in per_subject.items():
        n_ch_common = min(e.shape[-1] for e in erps)
        pooled[subj] = np.mean(np.stack([e[..., :n_ch_common] for e in erps]), axis=0)
    return pooled


def compute_shared_gain(erp_datasets, anes_keys=None, pipeline=0, num_components=4, pool_wake_across_sessions=True,
                         weight_by_variance=False):
    """
    Compute the gain matrix (or matrices) for a given pipeline:

    pipeline 0: one matrix, from the grand average ERP across every subject and condition.
    pipeline 1: two matrices, one from the pooled wake average, one from the pooled anesthesia average (all three anesthetics combined).
    pipeline 2: four matrices, one per condition (wake, keta, prop, sevo), each averaged across subjects.
    pipeline 3: one matrix per subject per condition (wake and anesthesia), fully personalized. Returns gains_dict[anes_name][subject][condition].
    pipeline 4: same as pipeline 3 but also per trial. Returns gains_dict[anes_name][subject][trial][condition].

    erp_datasets is a single combined DataArray for pipeline 0, and a dict
    anes_name -> DataArray for pipelines 1 to 4.

    pool_wake_across_sessions (pipeline 3 and 4 only, default True): wake is
    recorded once per anesthesia session (see data.py), so a subject who
    took part in all three anesthesia sessions has three separate wake
    recordings. With pooling on, pipeline 3 averages those into a single
    per-subject wake gain matrix, and pipeline 4 does the same but starting
    from the trial-resolved data (average over trials within a session,
    then over sessions), still one wake gain matrix per subject, reused for
    every one of that subject's anesthesia trials, since wake trials have
    no natural 1-to-1 pairing with anesthesia trials the way per-trial
    personalization is meaningful on the anesthesia side. With pooling off,
    you get the original behaviour: one wake gain matrix per subject per
    session (pipeline 3) or per subject per trial per session (pipeline 4),
    computed only from that session's own wake recording (keeps the
    recording montage matched to the anesthesia recording from the same
    day, at the cost of treating wake as several different conditions).

    weight_by_variance: passed straight through to every compute_gain_matrix_jax
    call below. False (default here) gives the original unit-norm gain matrices,
    which are exactly PCIst-invariant, see the "Gain matrix invariance" section
    in 03_sbi.ipynb. The notebook itself sets this to True everywhere, so the
    gain matrices actually used carry a bit of real, condition/subject/trial
    dependent information instead of being mathematically interchangeable.
    """
    gains_dict = {}

    if pipeline == 0:
        avg_erp = erp_datasets.mean(dim=['subject']).mean(dim=['condition']).dropna('region').to_numpy().copy()
        G_shared, _ = compute_gain_matrix_jax(avg_erp, n_comp=num_components, weight_by_variance=weight_by_variance)
        gains_dict = {'wake': G_shared, 'anes': G_shared}

    elif pipeline == 1:
        if anes_keys is None:
            raise ValueError('anes_keys must be provided for pipeline 1')

        all_wake_erps, all_anes_erps = [], []
        for anes_name in anes_keys:
            erp_data = erp_datasets[anes_name]
            anes_cond = get_anes_cond_label(erp_data)
            all_wake_erps.append(erp_data.sel(condition='wake').mean(dim='subject').dropna('region').to_numpy().copy())
            all_anes_erps.append(erp_data.sel(condition=anes_cond).mean(dim='subject').dropna('region').to_numpy().copy())

        global_wake = np.mean(np.stack(all_wake_erps), axis=0)
        global_anes = np.mean(np.stack(all_anes_erps), axis=0)

        G_wake, _ = compute_gain_matrix_jax(global_wake, n_comp=num_components, weight_by_variance=weight_by_variance)
        G_anes, _ = compute_gain_matrix_jax(global_anes, n_comp=num_components, weight_by_variance=weight_by_variance)
        gains_dict = {'wake': G_wake, 'anes': G_anes}

    elif pipeline == 2:
        if anes_keys is None:
            raise ValueError('anes_keys must be provided for pipeline 2')

        all_wake_erps = []
        for anes_name in anes_keys:
            erp_data = erp_datasets[anes_name]
            all_wake_erps.append(erp_data.sel(condition='wake').mean(dim='subject').dropna('region').to_numpy().copy())

        global_wake = np.mean(np.stack(all_wake_erps), axis=0)
        G_wake, _ = compute_gain_matrix_jax(global_wake, n_comp=num_components, weight_by_variance=weight_by_variance)
        gains_dict['wake'] = G_wake

        for anes_name in anes_keys:
            erp_data = erp_datasets[anes_name]
            anes_cond = get_anes_cond_label(erp_data)
            anes_erp = erp_data.sel(condition=anes_cond).mean(dim='subject').dropna('region').to_numpy().copy()
            G_anes, _ = compute_gain_matrix_jax(anes_erp, n_comp=num_components, weight_by_variance=weight_by_variance)
            gains_dict[anes_name] = G_anes

    elif pipeline == 3:
        if anes_keys is None:
            raise ValueError('anes_keys must be provided for pipeline 3')

        pooled_wake = _pool_wake_erp_per_subject(erp_datasets, anes_keys) if pool_wake_across_sessions else None

        for anes_name in anes_keys:
            erp_data = erp_datasets[anes_name]
            anes_cond = get_anes_cond_label(erp_data)
            gains_dict[anes_name] = {}

            for subj in erp_data.subject.values:
                gains_dict[anes_name][subj] = {}

                if pool_wake_across_sessions:
                    wake_erp = pooled_wake[subj]
                else:
                    wake_erp = erp_data.sel(condition='wake', subject=subj).dropna('region').to_numpy().copy()
                G_wake, _ = compute_gain_matrix_jax(wake_erp, n_comp=num_components, weight_by_variance=weight_by_variance)
                gains_dict[anes_name][subj]['wake'] = G_wake

                anes_erp = erp_data.sel(condition=anes_cond, subject=subj).dropna('region').to_numpy().copy()
                G_anes, _ = compute_gain_matrix_jax(anes_erp, n_comp=num_components, weight_by_variance=weight_by_variance)
                gains_dict[anes_name][subj][anes_cond] = G_anes

    elif pipeline == 4:
        if anes_keys is None:
            raise ValueError('anes_keys must be provided for pipeline 4')

        pooled_wake = _pool_wake_erp_per_subject_trial_resolved(erp_datasets, anes_keys) if pool_wake_across_sessions else None

        for anes_name in anes_keys:
            erp_data = erp_datasets[anes_name]
            anes_cond = get_anes_cond_label(erp_data)
            gains_dict[anes_name] = {}

            for subj in erp_data.subject.values:
                gains_dict[anes_name][subj] = {}
                subj_data = erp_data.sel(subject=subj)

                if pool_wake_across_sessions:
                    # one stable wake gain matrix per subject, computed once and
                    # reused for every trial below, see the docstring above and
                    # _pool_wake_erp_per_subject_trial_resolved for why wake isn't
                    # trial-resolved here the way anesthesia is
                    G_wake, _ = compute_gain_matrix_jax(pooled_wake[subj], n_comp=num_components, weight_by_variance=weight_by_variance)

                for tr in subj_data.trial.values:
                    gains_dict[anes_name][subj][tr] = {}

                    if pool_wake_across_sessions:
                        gains_dict[anes_name][subj][tr]['wake'] = G_wake
                    else:
                        wake_erp_tr = subj_data.sel(condition='wake', trial=tr).dropna('region').to_numpy().copy()
                        G_wake_tr, _ = compute_gain_matrix_jax(wake_erp_tr, n_comp=num_components, weight_by_variance=weight_by_variance)
                        gains_dict[anes_name][subj][tr]['wake'] = G_wake_tr

                    anes_erp_tr = subj_data.sel(condition=anes_cond, trial=tr).dropna('region').to_numpy().copy()
                    G_anes, _ = compute_gain_matrix_jax(anes_erp_tr, n_comp=num_components, weight_by_variance=weight_by_variance)
                    gains_dict[anes_name][subj][tr][anes_cond] = G_anes

    else:
        raise ValueError(f'Invalid pipeline {pipeline}, must be 0, 1, 2, 3 or 4')

    return gains_dict


# ---------------------------------------------------------------------------
# Condition encoding
# ---------------------------------------------------------------------------

def one_hot(index, size):
    vec = np.zeros(size, dtype=np.float32)
    vec[index] = 1.0
    return vec


def encode_condition(cond_name, pipeline):
    """
    Extra feature columns appended after the PCIst value, encoding which
    condition a row belongs to. Pipeline 0 needs nothing (one shared gain
    matrix, condition is not part of the conditioning). Pipeline 1 needs one
    binary wake/anesthesia flag. Pipeline 2 and up need the full one hot over
    the four conditions.
    """
    cond_name = str(cond_name).lower()

    if pipeline == 0:
        return np.array([], dtype=np.float32)

    if pipeline == 1:
        return np.array([0.0 if cond_name == 'wake' else 1.0], dtype=np.float32)

    if pipeline >= 2:
        if cond_name not in CONDITIONS:
            raise ValueError(f'Unknown condition: {cond_name}')
        return one_hot(CONDITIONS.index(cond_name), len(CONDITIONS))

    raise ValueError(f'Invalid pipeline {pipeline}')


# ---------------------------------------------------------------------------
# Building the SBI training set (simulations) and the empirical feature arrays
# ---------------------------------------------------------------------------

def _unpack_pci_pars(pci_pars):
    """
    Pull out the PCIst settings pcist.py actually needs, by key rather than
    by position. pci_pars historically also carried a 'baseline_corr' entry
    that pcist.py never used (it doesn't baseline correct), some pci_pars
    dicts include it and some don't, so unpacking by key means both are fine.
    """
    return (pci_pars['baseline_window'], pci_pars['response_window'], pci_pars['k'],
            pci_pars['min_snr'], pci_pars['max_var'], pci_pars['embed'], pci_pars['n_steps'])


def make_sbi_dataset(sim_array, theta, gains_dict, time_vec, pci_pars, pipeline, forward_chunks=100, pci_chunks=50):
    """
    Turn a set of single channel simulations into an SBI training set for the
    given pipeline: project every simulation through the relevant gain
    matrix (or matrices), compute PCIst, and attach the condition/subject/
    trial encoding as extra feature columns. Returns (theta_tensor,
    x_features_tensor), both torch tensors.
    """
    baseline_window, response_window, k, min_snr, max_var, embed, n_steps = _unpack_pci_pars(pci_pars)

    if pipeline == 0:
        G = list(gains_dict.values())[0]
        Y_all = compute_forward_solution_multimode_jax_batch(jnp.array(sim_array), G, batch_size=forward_chunks)
        pci_all = pcist.compute_pci_batch_chunked(
            Y_all, time_vec, batch_size=pci_chunks, full_return=False, n_components=Y_all.shape[1],
            baseline_window=baseline_window, response_window=response_window, k=k, min_snr=min_snr,
            max_var=max_var, embed=embed, n_steps=n_steps)

        all_feature_rows = [[pci_val] for pci_val in pci_all]
        theta_arr = theta

    elif pipeline in (1, 2):
        # gains_dict[condition] = one gain matrix per condition
        rows, theta_rows = [], []
        num_sims = sim_array.shape[0]

        for cond_name, G in tqdm(gains_dict.items(), total=len(gains_dict), desc=f'Pipeline {pipeline}'):
            cond_encoding = encode_condition(cond_name, pipeline)
            Y_all = compute_forward_solution_multimode_jax_batch(jnp.array(sim_array), G, batch_size=forward_chunks)
            pci_all = pcist.compute_pci_batch_chunked(
                Y_all, time_vec, batch_size=pci_chunks, full_return=False, n_components=Y_all.shape[1],
                baseline_window=baseline_window, response_window=response_window, k=k, min_snr=min_snr,
                max_var=max_var, embed=embed, n_steps=n_steps)

            for sim_idx in range(num_sims):
                rows.append([pci_all[sim_idx], *cond_encoding])
                theta_rows.append(theta[sim_idx])

        all_feature_rows = rows
        theta_arr = np.array(theta_rows, dtype=np.float32)

    elif pipeline == 3:
        # gains_dict[anes_name][subject]['wake' or anes] = one gain matrix per subject/condition
        all_subject_ids = sorted({subj for subj_dict in gains_dict.values() for subj in subj_dict.keys()})
        n_subjects = len(all_subject_ids)
        num_sims = sim_array.shape[0]
        total_steps = sum(len(subj_dict) * 2 for subj_dict in gains_dict.values())

        rows, theta_rows = [], []
        with tqdm(total=total_steps, desc='Pipeline 3') as pbar:
            for anes_name, subj_dict in gains_dict.items():
                for subj_key, state_dict in subj_dict.items():
                    subj_idx = all_subject_ids.index(subj_key)
                    subject_encoding = one_hot(subj_idx, n_subjects)

                    for state_name, G in state_dict.items():
                        pbar.update(1)
                        cond_encoding = encode_condition(state_name, pipeline)
                        Y_all = compute_forward_solution_multimode_jax_batch(jnp.array(sim_array), G, batch_size=forward_chunks)
                        pci_all = pcist.compute_pci_batch_chunked(
                            Y_all, time_vec, batch_size=pci_chunks, full_return=False, n_components=Y_all.shape[1],
                            baseline_window=baseline_window, response_window=response_window, k=k, min_snr=min_snr,
                            max_var=max_var, embed=embed, n_steps=n_steps)

                        for sim_idx in range(num_sims):
                            rows.append([pci_all[sim_idx], *cond_encoding, *subject_encoding])
                            theta_rows.append(theta[sim_idx])

        all_feature_rows = rows
        theta_arr = np.array(theta_rows, dtype=np.float32)

    elif pipeline == 4:
        # gains_dict[anes_name][subject][trial]['wake' or anes] = one gain matrix per subject/trial/condition
        all_subject_ids = sorted({subj for subj_dict in gains_dict.values() for subj in subj_dict.keys()})
        n_subjects = len(all_subject_ids)
        all_trial_ids = sorted({trial for subj_dict in gains_dict.values()
                                 for trial_dict in subj_dict.values() for trial in trial_dict.keys()})
        n_trials = len(all_trial_ids)
        num_sims = sim_array.shape[0]
        total_steps = sum(len(trial_dict) * 2 for subj_dict in gains_dict.values() for trial_dict in subj_dict.values())

        rows, theta_rows = [], []
        with tqdm(total=total_steps, desc='Pipeline 4') as pbar:
            for anes_name, subj_dict in gains_dict.items():
                for subj_key, trial_dict in subj_dict.items():
                    subj_idx = all_subject_ids.index(subj_key)
                    subject_encoding = one_hot(subj_idx, n_subjects)

                    for trial_key, state_dict in trial_dict.items():
                        trial_idx = all_trial_ids.index(trial_key)
                        trial_encoding = one_hot(trial_idx, n_trials)

                        for state_name, G in state_dict.items():
                            pbar.update(1)
                            cond_encoding = encode_condition(state_name, pipeline)
                            Y_all = compute_forward_solution_multimode_jax_batch(jnp.array(sim_array), G, batch_size=forward_chunks)
                            pci_all = pcist.compute_pci_batch_chunked(
                                Y_all, time_vec, batch_size=pci_chunks, full_return=False, n_components=Y_all.shape[1],
                                baseline_window=baseline_window, response_window=response_window, k=k, min_snr=min_snr,
                                max_var=max_var, embed=embed, n_steps=n_steps)

                            for sim_idx in range(num_sims):
                                rows.append([pci_all[sim_idx], *cond_encoding, *subject_encoding, *trial_encoding])
                                theta_rows.append(theta[sim_idx])

        all_feature_rows = rows
        theta_arr = np.array(theta_rows, dtype=np.float32)

    else:
        raise ValueError(f'Invalid pipeline {pipeline}, must be 0, 1, 2, 3 or 4')

    x_features_tensor = torch.as_tensor(np.array(all_feature_rows, dtype=np.float32))
    theta_tensor = torch.as_tensor(np.asarray(theta_arr), dtype=torch.float32)
    return theta_tensor, x_features_tensor


def build_empirical_features(emp_features, pipeline, subject_index=None, trial_index=None):
    """
    Encode an empirical PCIst DataArray (dims subject, condition, and trial
    for pipeline 4) the same way make_sbi_dataset encoded the simulations, so
    it can be fed into posterior.sample_batched as the conditioning variable.

    subject_index / trial_index let you pass in a fixed global subject/trial
    ordering (as used during training) instead of whatever order this
    particular DataArray happens to have, this matters for pipeline 3 and 4
    where the one hot position has to match what the network was trained on.
    """
    cond_vals = emp_features.condition.values
    subj_vals = emp_features.subject.values

    all_subject_ids = subject_index if subject_index is not None else sorted(subj_vals)
    n_subjects = len(all_subject_ids)

    has_trials = 'trial' in emp_features.dims
    if has_trials:
        trial_vals = emp_features.trial.values
        all_trial_ids = trial_index if trial_index is not None else sorted(trial_vals)
        n_trials = len(all_trial_ids)

    rows = []
    if has_trials and pipeline == 4:
        for cond_name in cond_vals:
            for subj in subj_vals:
                subj_idx = all_subject_ids.index(subj)
                for trial_val in trial_vals:
                    pci_val = float(emp_features.sel(condition=cond_name, subject=subj, trial=trial_val).values)
                    trial_idx = all_trial_ids.index(trial_val)
                    row = [pci_val, *encode_condition(cond_name, pipeline),
                           *one_hot(subj_idx, n_subjects), *one_hot(trial_idx, n_trials)]
                    rows.append(row)
    else:
        for cond_name in cond_vals:
            for subj_idx, subj in enumerate(subj_vals):
                pci_val = float(emp_features.sel(condition=cond_name, subject=subj).values)
                row = [pci_val, *encode_condition(cond_name, pipeline)]
                if pipeline >= 3:
                    row.extend(one_hot(all_subject_ids.index(subj), n_subjects))
                rows.append(row)

    return np.array(rows, dtype=np.float32)


def sample_from_posterior(posterior, summary_obs, pipeline, n_samples=1000, cond_label=None, device='cpu', show_pbars=False):
    """Sample the posterior for one PCIst value, optionally with a condition encoding attached."""
    if cond_label is not None:
        cond_ind = encode_condition(cond_label, pipeline)
        x_np = np.concatenate([np.atleast_1d(summary_obs), np.atleast_1d(cond_ind)])[None, :]
    else:
        x_np = np.atleast_1d(summary_obs)[None, :]

    x_t = torch.tensor(x_np, dtype=torch.float32, device=device)
    samples = posterior.sample((n_samples,), x=x_t, show_progress_bars=show_pbars)
    return samples.cpu().numpy()


# ---------------------------------------------------------------------------
# Gain matrix subspace similarity (used at the end of the notebook)
# ---------------------------------------------------------------------------

def subspace_similarity(G1, G2):
    """
    How much two gain matrices' column spaces overlap: project one onto the
    other and look at the singular values of that projection. 1 means the
    subspaces coincide, 0 means they are orthogonal. Returns nan if the two
    matrices don't even have the same number of channels (can happen for
    subject/trial gain matrices when channels were dropped).

    Columns get normalized to unit length before the projection. Without this,
    weight_by_variance_gain=True (used everywhere in this notebook) means each
    column is scaled by its own singular value share, the same scaling that
    makes PCIst sensitive to which gain matrix was used ("Gain matrix
    invariance" above). That's real and intentional for PCIst, but it has
    nothing to do with which direction a gain matrix's subspace points in, and
    letting it leak in here breaks the [0, 1] bound the singular values are
    supposed to have. Normalizing isolates direction from magnitude, which is
    what this function is actually meant to measure.
    """
    G1, G2 = jnp.asarray(G1), jnp.asarray(G2)
    if G1.shape[0] != G2.shape[0]:
        return np.nan, None

    G1 = G1 / jnp.linalg.norm(G1, axis=0, keepdims=True)
    G2 = G2 / jnp.linalg.norm(G2, axis=0, keepdims=True)

    M = G1.T @ G2
    _, s, _ = jnp.linalg.svd(M, full_matrices=False)
    return float(jnp.mean(s ** 2)), np.array(s)


def compute_similarity_matrix(gain_list):
    """Pairwise subspace_similarity for every gain matrix in gain_list."""
    n = len(gain_list)
    sim_mat = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            sim_mat[i, j], _ = subspace_similarity(gain_list[i], gain_list[j])
    return sim_mat


def compute_cross_pipeline_similarity(gain_list_A, gain_list_B):
    """subspace_similarity between every gain matrix of pipeline A and every gain matrix of pipeline B."""
    sim_mat = np.zeros((len(gain_list_A), len(gain_list_B)))
    for i, G1 in enumerate(gain_list_A):
        for j, G2 in enumerate(gain_list_B):
            sim_mat[i, j], _ = subspace_similarity(G1, G2)
    return sim_mat


def flatten_gains(gains_dict, pipeline):
    """Flatten a pipeline's (possibly nested) gains_dict into a flat list of matrices plus matching labels, for the similarity comparisons."""
    gain_list, labels = [], []

    def add_gain(G, label):
        if len(G) == 0:
            return
        gain_list.append(G)
        labels.append(label)

    if pipeline == 0:
        for key, G in gains_dict.items():
            add_gain(G, f'p0_{key}')
    elif pipeline in (1, 2):
        for key, G in gains_dict.items():
            add_gain(G, f'p{pipeline}_{key}')
    elif pipeline == 3:
        for anes_name, subj_dict in gains_dict.items():
            for subj, cond_dict in subj_dict.items():
                for cond, G in cond_dict.items():
                    add_gain(G, f'p3_{anes_name}_{subj}_{cond}')
    elif pipeline == 4:
        for anes_name, subj_dict in gains_dict.items():
            for subj, trial_dict in subj_dict.items():
                for tr, cond_dict in trial_dict.items():
                    for cond, G in cond_dict.items():
                        add_gain(G, f'p4_{anes_name}_{subj}_tr{tr}_{cond}')
    else:
        raise ValueError(f'Invalid pipeline {pipeline}')

    # gain matrices can have slightly different channel counts (a subject/
    # trial can be missing a few channels), crop to the smallest so the
    # similarity comparison has a common space to work in.
    common_shape = min(G.shape[0] for G in gain_list)
    gain_list = [G[:common_shape, :] for G in gain_list]
    return gain_list, labels


def summarize_similarity(sim_mat):
    """Mean and std of the off diagonal entries of a within pipeline similarity matrix."""
    n = sim_mat.shape[0]
    off_diag = sim_mat[~np.eye(n, dtype=bool)]
    return np.mean(off_diag), np.std(off_diag)


def summarize_cross_similarity(cross_mat):
    return np.mean(cross_mat), np.std(cross_mat)


# ---------------------------------------------------------------------------
# Priors and prior/posterior sampling
# ---------------------------------------------------------------------------

def uniform_from_mean_std(theta_means, theta_stds):
    lows = theta_means - theta_stds * np.sqrt(3)
    highs = theta_means + theta_stds * np.sqrt(3)
    return lows, highs


def gamma_from_mean_std(means, stds):
    shapes = (means / stds) ** 2
    rates = shapes / means
    return shapes, rates


def make_prior(theta_means, theta_stds, prm_names, dist_type='gamma', sim_path=None, overwrite=True, bounds=None):
    """
    Build an independent prior over prm_names. dist_type='gamma' keeps
    everything positive (used for the early attempts), 'uniform' is a plain
    linear-scale uniform prior, 'loguniform' is an older attempt (see the
    prior section for why it got dropped), 'normal' is what the g parameters
    use now. 'gamma', 'uniform' and 'normal' derive their distribution
    directly from theta_means/theta_stds (mean and std per parameter, same
    order as prm_names). 'loguniform' instead needs explicit (low, high)
    bounds per parameter passed via bounds, since "uniform in log space
    between these two numbers" isn't naturally described by a mean/std,
    theta_means and theta_stds are ignored for this dist_type.
    """
    if dist_type.lower() == 'gamma':
        shapes, rates = gamma_from_mean_std(theta_means, theta_stds)
        param_dists = torch.distributions.Gamma(concentration=torch.tensor(shapes), rate=torch.tensor(rates))
        prior_dist = torch.distributions.Independent(param_dists, 1)
    elif dist_type.lower() == 'uniform':
        lows, highs = uniform_from_mean_std(theta_means, theta_stds)
        param_dists = torch.distributions.Uniform(low=torch.tensor(lows), high=torch.tensor(highs))
        prior_dist = torch.distributions.Independent(param_dists, 1)
    elif dist_type.lower() == 'normal':
        # plain independent normal, mean/std per parameter as given. Switched to from
        # loguniform because prior predictive checks kept finding parameter combinations
        # (high excitatory gain together with low inhibitory gain) that push the model into
        # a sustained oscillation instead of a decaying evoked response, a normal centered
        # on the DCM defaults puts much less mass on that combination than a wide log-uniform did.
        param_dists = torch.distributions.Normal(loc=torch.tensor(theta_means, dtype=torch.float32),
                                                  scale=torch.tensor(theta_stds, dtype=torch.float32))
        prior_dist = torch.distributions.Independent(param_dists, 1)
    elif dist_type.lower() == 'loguniform':
        if bounds is None:
            raise ValueError("dist_type='loguniform' needs explicit (low, high) bounds per parameter, "
                              "e.g. bounds=[(1e-2, 1e2)] * len(prm_names)")
        # uniform density on log(g), then exponentiate: keeps g positive and gives
        # small and large values comparable prior mass instead of a linear-scale
        # uniform, which would put almost all its mass on the large end.
        log_lows = torch.log(torch.tensor([b[0] for b in bounds], dtype=torch.float32))
        log_highs = torch.log(torch.tensor([b[1] for b in bounds], dtype=torch.float32))
        base_dist = torch.distributions.Independent(torch.distributions.Uniform(low=log_lows, high=log_highs), 1)
        prior_dist = torch.distributions.TransformedDistribution(base_dist, [torch.distributions.ExpTransform()])
    else:
        raise ValueError(f"Unknown dist_type '{dist_type}', expected 'gamma', 'uniform', 'normal' or 'loguniform'")

    prior, _, _ = sbi.utils.user_input_checks.process_prior(prior_dist)

    if sim_path is not None:
        if overwrite:
            torch.save(prior_dist, f=sim_path + '/prior.pt')
        else:
            # weights_only=False: this file holds a torch.distributions object, not
            # just plain tensors, so the restricted unpickler (default since torch
            # 2.6) can't load it. Fine here.
            loaded_dist = torch.load(sim_path + '/prior.pt', weights_only=False)
            prior, _, _ = sbi.utils.user_input_checks.process_prior(loaded_dist)

    return prior


def sample_from_prior(prior, num_sim, sim_path=None, overwrite=True):
    theta = prior.sample((num_sim,))
    if sim_path is not None:
        if overwrite:
            torch.save(theta, f=sim_path + '/theta.pt')
        else:
            theta = torch.load(sim_path + '/theta.pt', weights_only=True)
    return theta


def init_sim_dirs(parent_preprocess_dir, sim_name):
    """Create (if needed) input/<sim_name> and output/<sim_name> next to the notebook, and return their paths."""
    input_dir = os.path.join(parent_preprocess_dir, 'input/')
    output_dir = os.path.join(parent_preprocess_dir, 'output/')

    for d in (input_dir, output_dir, input_dir + sim_name, output_dir + sim_name):
        try:
            os.makedirs(d)
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise

    return input_dir + sim_name, output_dir + sim_name


def simulate_from_theta(theta, sim_path, input_params, ts, nt, ds_sim=10, noise_params=(0., 0., 1.),
                         alpha=None, free_param_names=None, num_batches=1, save_sims=True):
    """
    Run the Jansen-Rit model (temporal/pulse input variant) once per row of
    theta, downsample by ds_sim, and save the batch to
    sim_path/simulations_<batch_ind>.npz.

    free_param_names controls which JR parameter each column of theta maps
    to, in order, defaults to jansen_rit.FREE_PARAM_NAMES so this stays in
    sync with the "g-only for now" scope: extending which parameters get
    inferred is a matter of changing FREE_PARAM_NAMES (and the prior), not
    this function.
    """
    free_param_names = free_param_names or jansen_rit.FREE_PARAM_NAMES
    alpha = jansen_rit.ALPHA_DEFAULT if alpha is None else alpha

    num_sim = theta.shape[0]
    batch_size = int(num_sim / num_batches)
    master_key = jax.random.PRNGKey(0)

    if not os.path.exists(sim_path):
        os.makedirs(sim_path)

    for batch_ind in trange(num_batches):
        theta_batch = theta[batch_size * batch_ind: batch_size * (batch_ind + 1), :]
        sim_batch = np.zeros((batch_size, nt // ds_sim))

        for s_ind in trange(batch_size, leave=False):
            master_key, sub_key = jax.random.split(master_key)
            theta_row = np.asarray(theta_batch[s_ind, :])
            free_params = dict(zip(free_param_names, theta_row))

            trace = jansen_rit.simulate_erp_temporal(
                ts, input_params, free_params=free_params, alpha=alpha,
                key=sub_key, noise_params=noise_params)
            sim_batch[s_ind, :] = np.asarray(trace)[::ds_sim]

        if save_sims:
            np.savez(sim_path + '/simulations_' + str(batch_ind) + '.npz', x=sim_batch, theta=np.asarray(theta_batch))

        del sim_batch
        gc.collect()


def load_all_simulations(sim_path):
    """Reload every simulations_*.npz batch saved by simulate_from_theta and concatenate them."""
    files = sorted(f for f in os.listdir(sim_path) if f.startswith('simulations_') and f.endswith('.npz'))
    X = [np.load(os.path.join(sim_path, f), allow_pickle=True)['x'] for f in files]
    Theta = [torch.tensor(np.load(os.path.join(sim_path, f), allow_pickle=True)['theta']) for f in files]
    return np.concatenate(X, axis=0), torch.cat(Theta, dim=0)


# ---------------------------------------------------------------------------
# Small helpers for the plotting side
# ---------------------------------------------------------------------------

def pool_posteriors(pdict_to_pool):
    """Stack a dict of pipeline posteriors (dataset name -> xr.DataArray) into one long dataframe for the boxplot helpers in plotting.py."""
    all_df_list = []
    for dataset_name, pthetas in pdict_to_pool.items():
        df_long = pthetas.to_series().reset_index(name='value')
        df_long = df_long.rename(columns={'theta': 'theta_label'})
        df_long['dataset'] = dataset_name
        all_df_list.append(df_long)
    return pd.concat(all_df_list, ignore_index=True)


def compute_kde_overlap(sim_pci, emp_pci, grid=np.linspace(0, 200, 400)):
    """
    A continuous alternative to a fixed threshold fraction outside range
    metric. Fits a Gaussian KDE to each batch, evaluates both on the same
    grid, and returns the cosine similarity between the two density curves,
    basically their inner product as functions, normalized by their own
    norms so the result sits between 0 (no overlap at all) and 1 (identical
    shape and location). Reacts to the whole distribution instead of just
    counting how many points fall outside a hard cutoff.
    """
    f_sim = gaussian_kde(sim_pci)(grid)
    f_emp = gaussian_kde(emp_pci)(grid)
    inner = np.trapz(f_sim * f_emp, grid)
    norm_sim = np.sqrt(np.trapz(f_sim ** 2, grid))
    norm_emp = np.sqrt(np.trapz(f_emp ** 2, grid))
    return float(inner / (norm_sim * norm_emp))


def fit_pcist_rescale(sim_pci, emp_pci, margin_lo=0.1, margin_hi=0.1):
    """
    Fit a linear rescale a * sim_pci + b so the rescaled simulated batch's
    lowest value sits margin_lo below the empirical minimum, and its highest
    value sits margin_hi above the empirical maximum (both margins as a
    fraction of the empirical range, and independent of each other). Meant
    to be fit once on a representative simulated batch, then reused on every
    later PCIst value coming out of the same gain matrix.
    """
    sim_pci = np.asarray(sim_pci)
    emp_pci = np.asarray(emp_pci)
    emp_lo, emp_hi = emp_pci.min(), emp_pci.max()
    emp_range = emp_hi - emp_lo
    target_lo = emp_lo - margin_lo * emp_range
    target_hi = emp_hi + margin_hi * emp_range
    sim_lo, sim_hi = sim_pci.min(), sim_pci.max()
    a = (target_hi - target_lo) / (sim_hi - sim_lo)
    b = target_lo - a * sim_lo
    return a, b


def apply_pcist_rescale(pci, a, b):
    """Apply a previously fitted linear rescale. Works on numpy arrays and torch tensors alike."""
    return a * pci + b