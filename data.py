"""
Loading and preprocessing of the raw rat ERP `.mat` files.

Raw data layout on disk (see data/ folder, kept local only):

    data/PCI-like_<anes>1/PCI-like_<subject>_<condition>-r<run>.mat

Each anesthesia subfolder (keta, prop, sevo) contains one wake recording and
one anesthesia recording per subject that was recorded in that session. Not
every subject has all three anesthesia types.

This module is the only place raw .mat files get read. Notebook 1 walks through
what these functions do step by step and then calls them to build the final
preprocessed datasets; notebooks 2 and 3 only ever load the resulting .nc files.

Sections in this file:
    discovering raw files      scanning data/ for what subjects/conditions exist
    preprocessing               averaging, normalizing, smoothing into .nc datasets
    downstream helpers          used later by 02_mcmc.ipynb on the preprocessed data

"""
import os
import re
import glob
from collections import defaultdict

import numpy as np
import xarray as xr
import scipy as scp


##############################################################################
# DISCOVERING RAW FILES
##############################################################################

def list_subject_files(folder, pattern=r'PCI-like_(j\d+)_(\w+)-r\d+\.mat'):
    """
    Scan a raw-data folder (e.g. data/PCI-like_keta1/) and group the .mat files
    by subject and condition.

    Returns:
        subject_files: dict subject -> {condition: filepath}
        conditions: sorted list of condition names found (e.g. ['keta', 'wake'])
        subjects: subject names, sorted numerically (j12, j13, j23, ...)
    """
    files = sorted(glob.glob(os.path.join(folder, '*.mat')))
    subject_files = defaultdict(dict)
    all_conditions, all_subjects = set(), set()

    for f in files:
        fname = os.path.basename(f)
        match = re.match(pattern, fname)
        if match:
            subj, cond = match.groups()
            cond_norm = re.sub(r'\d+$', '', cond).lower()
            subject_files[subj][cond_norm] = f
            all_conditions.add(cond_norm)
            all_subjects.add(subj)

    conditions = sorted(all_conditions)
    subjects = sorted(all_subjects, key=lambda x: int(re.findall(r'\d+', x)[0]))

    return subject_files, conditions, subjects


def discover_available_conditions(data_root, anes_types=('keta', 'prop', 'sevo')):
    """
    Scan every anesthesia subfolder under data_root and figure out, per subject,
    which anesthesia types they have recordings for. A subject can appear under
    1, 2, or all 3 anesthesia types depending on which sessions they were
    recorded in.

    Returns: dict subject -> sorted list of anesthesia types available for them,
    e.g. {'j12': ['keta', 'prop', 'sevo'], 'j13': ['keta', 'sevo'], ...}
    """
    subject_to_anes = defaultdict(list)
    for anes in anes_types:
        folder = os.path.join(data_root, f'PCI-like_{anes}1')
        if not os.path.isdir(folder):
            continue
        _, _, subjects = list_subject_files(folder)
        for subj in subjects:
            subject_to_anes[subj].append(anes)

    return {subj: sorted(anes_list) for subj, anes_list in sorted(
        subject_to_anes.items(), key=lambda kv: int(re.findall(r'\d+', kv[0])[0]))}


def load_raw_recording(mat_path):
    """
    Load a single raw .mat recording as-is (no time selection, no normalization).
    Returns a dict with time in ms, sampling frequency in Hz, the full
    (samples, channels, trials) eeg array, and the bad-channel indices.
    """
    raw = scp.io.loadmat(mat_path)
    return {
        'time_ms': raw['time'].squeeze() * 1000,
        'fs': float(np.squeeze(raw['fs'])),
        'eeg': raw['eeg'],
        'bad_channels': np.atleast_1d(raw['badCh'].squeeze()),
    }


def load_raw_pair(data_root, subject, anes, cond_wake='wake'):
    """
    Load the raw wake + anesthesia recordings for one subject/anesthesia
    session, e.g. load_raw_pair('data', 'j12', 'keta') loads
    PCI-like_j12_wake-rX.mat and PCI-like_j12_keta1-rX.mat from
    data/PCI-like_keta1/. Used for the exploratory (pre-preprocessing) plots.
    """
    folder = os.path.join(data_root, f'PCI-like_{anes}1')
    subject_files, _, _ = list_subject_files(folder)
    if subject not in subject_files:
        raise KeyError(f"Subject '{subject}' not found under {folder}")
    wake = load_raw_recording(subject_files[subject][cond_wake])
    anes_rec = load_raw_recording(subject_files[subject][anes])
    return wake, anes_rec


##############################################################################
# PREPROCESSING
##############################################################################

def smooth_erp(erp, tmin_orig, tmax_orig, sfreq=500, freq_scale=2, order=3, lp_freq=20, axis=0, apply_lowpass=False):
    """
    Cubic-interpolate an ERP onto a finer time grid (freq_scale x more samples),
    optionally low-pass filtering first. Returns the smoothed array plus the
    attrs dict (tmin, tmax, sfreq) describing its new time axis.
    """
    n_samples_orig = erp.shape[0]
    n_samples_fine = freq_scale * (n_samples_orig - 1) + 1
    time_orig = np.linspace(tmin_orig, tmax_orig, n_samples_orig)
    time_fine = np.linspace(tmin_orig, tmax_orig, n_samples_fine)

    if apply_lowpass:
        lowpass = scp.signal.butter(order, lp_freq, 'lp', fs=sfreq, output='sos')
        erp_processed = scp.signal.sosfiltfilt(lowpass, erp.copy(), axis=axis)
    else:
        erp_processed = erp.copy()

    interp_fn = scp.interpolate.interp1d(time_orig, erp_processed, kind='cubic', axis=axis)
    erp_smoothed = interp_fn(time_fine)

    attrs = dict(tmin=float(tmin_orig), tmax=float(tmax_orig), sfreq=sfreq * freq_scale)
    return erp_smoothed, attrs


def preprocess_trial_averaged(folder, cond_wake='wake', cond_anes='keta', tbeg=2490, tend=2665,
                               sfreq=500, amp_scaler=6, axis=0, save_output=True, data_dir=None):
    """
    Build the trial-averaged empirical dataset for one anesthesia folder:
    average over trials -> normalize amplitude (per subject, using the joint
    wake/anesthesia max so relative amplitude is preserved) -> smooth/interpolate.

    Saves (if save_output) to <data_dir>/<cond_anes>_all_erps.nc and also
    returns the resulting xr.DataArray, dims (subject, condition, time, region).
    """
    subject_files, conditions, subjects = list_subject_files(folder)
    sample_data = scp.io.loadmat(list(subject_files.values())[0][cond_wake])
    data_time = sample_data['time'].squeeze()
    tbeg_sec, tend_sec = data_time[tbeg], data_time[tend - 1]

    all_subject_erps = []
    for subj in subjects:
        erp_wake = scp.io.loadmat(subject_files[subj][cond_wake])['eeg'].mean(-1)[tbeg:tend, :]
        erp_anes = scp.io.loadmat(subject_files[subj][cond_anes])['eeg'].mean(-1)[tbeg:tend, :]

        erp_max = np.max(np.abs([erp_wake, erp_anes]))
        erp_wake = erp_wake / erp_max * amp_scaler
        erp_anes = erp_anes / erp_max * amp_scaler

        erp_wake_sm, attrs = smooth_erp(erp_wake, tbeg_sec, tend_sec, sfreq=sfreq, axis=axis)
        erp_anes_sm, _ = smooth_erp(erp_anes, tbeg_sec, tend_sec, sfreq=sfreq, axis=axis)
        n_regions = erp_wake_sm.shape[-1]

        cond_erps = xr.DataArray(
            [erp_wake_sm, erp_anes_sm], dims=('condition', 'time', 'region'),
            coords={'condition': [cond_wake, cond_anes], 'region': np.arange(n_regions)}, attrs=attrs)
        all_subject_erps.append(cond_erps)

    all_subject_erps = xr.concat(all_subject_erps, dim='subject', join='outer')
    erp_coords = {'subject': subjects, 'condition': [cond_wake, cond_anes],
                  'time': np.linspace(attrs['tmin'], attrs['tmax'], all_subject_erps.sizes['time'])}
    all_subject_erps.coords.update(erp_coords)
    all_subject_erps.attrs = attrs

    if save_output:
        all_subject_erps.to_netcdf(os.path.join(data_dir, f'{cond_anes}_all_erps.nc'))

    return all_subject_erps


def preprocess_trial_resolved(folder, cond_wake='wake', cond_anes='keta', tbeg=2490, tend=2665, sfreq=500,
                               amp_scaler=6, axis=0, n_trials=10, save_output=True, data_dir=None, random_state=None):
    """
    Same idea as preprocess_trial_averaged, but keeps n_trials individual
    (randomly selected) trials per subject/condition instead of averaging them.
    Used for the trial-resolved SBI pipelines and for the per-trial exploratory
    plots. Dims (subject, condition, trial, time, region).
    """
    rng = np.random.default_rng(random_state)

    subject_files, conditions, subjects = list_subject_files(folder)
    sample_data = scp.io.loadmat(list(subject_files.values())[0][cond_wake])
    data_time = sample_data['time'].squeeze()
    tbeg_sec, tend_sec = data_time[tbeg], data_time[tend - 1]

    all_subject_trials = []
    for subj in subjects:
        eeg_wake = scp.io.loadmat(subject_files[subj][cond_wake])['eeg'][tbeg:tend, :, :]
        eeg_anes = scp.io.loadmat(subject_files[subj][cond_anes])['eeg'][tbeg:tend, :, :]

        sel_wake = rng.choice(eeg_wake.shape[-1], size=min(n_trials, eeg_wake.shape[-1]), replace=False)
        sel_anes = rng.choice(eeg_anes.shape[-1], size=min(n_trials, eeg_anes.shape[-1]), replace=False)

        eeg_wake = eeg_wake[:, :, sel_wake]
        eeg_anes = eeg_anes[:, :, sel_anes]

        erp_max = max(np.abs(eeg_wake).max(), np.abs(eeg_anes).max())
        eeg_wake = eeg_wake / erp_max * amp_scaler
        eeg_anes = eeg_anes / erp_max * amp_scaler

        def smooth_all_trials(eeg_data):
            smoothed = [smooth_erp(eeg_data[:, :, tr], tbeg_sec, tend_sec, sfreq=sfreq, axis=axis)[0]
                        for tr in range(eeg_data.shape[-1])]
            return np.stack(smoothed, axis=0)

        wake_sm = smooth_all_trials(eeg_wake)
        anes_sm = smooth_all_trials(eeg_anes)
        _, attrs = smooth_erp(eeg_wake[:, :, 0], tbeg_sec, tend_sec, sfreq=sfreq, axis=axis)

        cond_trials = np.stack([wake_sm, anes_sm], axis=0)
        cond_da = xr.DataArray(
            cond_trials, dims=('condition', 'trial', 'time', 'region'),
            coords={'condition': [cond_wake, cond_anes], 'trial': np.arange(cond_trials.shape[1]),
                    'region': np.arange(cond_trials.shape[-1])}, attrs=attrs)
        cond_da = cond_da.expand_dims(subject=[subj])
        all_subject_trials.append(cond_da)

    all_subject_trials = xr.concat(all_subject_trials, dim='subject', join='outer')
    erp_coords = {'subject': subjects, 'condition': [cond_wake, cond_anes],
                  'time': np.linspace(attrs['tmin'], attrs['tmax'], all_subject_trials.sizes['time'])}
    all_subject_trials.coords.update(erp_coords)
    all_subject_trials.attrs = attrs

    if save_output:
        all_subject_trials.to_netcdf(os.path.join(data_dir, f'{cond_anes}_single_trials.nc'))

    return all_subject_trials


##############################################################################
# DOWNSTREAM HELPERS
##############################################################################

def combine_across_subjects(erp_datasets, pairs, channel):
    """
    Grand-average trace at one channel, pooled equally across every
    individual subject recording named in `pairs`, a list of
    (anes_key, condition) tuples into erp_datasets. Each matching subject
    recording counts once, regardless of which anesthesia session it came
    from, so e.g. pairs=[('keta', 'wake'), ('prop', 'wake'), ('sevo', 'wake')]
    pools every wake recording across all three sessions into one combined
    wake average, and pairs=[('keta', 'keta'), ('prop', 'prop'), ('sevo', 'sevo')]
    pools every anesthesia recording regardless of drug into one combined
    anesthesia average.

    Used for 02_mcmc.ipynb's condition-combined fits, where "the average
    signal of all keta data combined" means averaged across every keta
    subject, as opposed to the single example_subject trace used for the
    rest of that notebook's fits.

    Returns (time_ms, mean_trace, n_traces_pooled).
    """
    traces = []
    time_ms = None
    for anes_key, condition in pairs:
        da = erp_datasets[anes_key].sel(condition=condition, region=channel)
        if time_ms is None:
            time_ms = da['time'].to_numpy() * 1000  # .nc time coord is in seconds
        for subj in da['subject'].to_numpy():
            trace = da.sel(subject=subj).to_numpy()
            if np.all(np.isnan(trace)):
                continue  # this subject doesn't actually have this condition
            traces.append(trace)

    if not traces:
        raise ValueError(f'No traces found for pairs={pairs}, channel={channel}')

    mean_trace = np.mean(np.stack(traces, axis=0), axis=0)
    return time_ms, mean_trace, len(traces)


def wasserstein_distance(sample1, sample2, n_permutations=100):
    """Thin wrapper around scipy's Wasserstein distance, used to annotate posterior comparison plots."""
    return scp.stats.wasserstein_distance(sample1, sample2)
