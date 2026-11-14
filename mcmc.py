"""
NumPyro/NUTS model, priors, and run/diagnostic helpers for 02_mcmc.ipynb.

Unlike the SBI notebook, this fits the Jansen-Rit model directly to a single
subject/channel's mean ERP waveform (full time series likelihood, no summary
statistics), which is why it works with one subject/channel at a time. It
wraps jansen_rit.simulate_erp_temporal (the pulse input variant), the same
simulator Notebooks 1 and 3 use.

Like jansen_rit.py, this is built around jansen_rit.FREE_PARAM_NAMES rather
than a hardcoded parameter count, so going from g-only to more parameters
is a matter of extending that list, not rewriting the model.

Sections in this file:
    priors                  analytic prior densities and prior spec lookup
    NUTS kernel presets      named target_accept/tree_depth configurations
    the NumPyro model        build_model, the actual fit target
    data / signal prep       turning a raw waveform into the model's input
    running NUTS             run_nuts and prior/posterior predictive sampling
    posterior diagnostics    winsorizing, MAP, tail sampling, ELPD stacking

"""
import time

import numpy as np
from scipy.signal import butter, sosfiltfilt
from scipy.special import erf
import jax
import jax.numpy as jnp

import numpyro as npr
from numpyro import plate
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, Predictive, init_to_sample
import arviz as az

import jansen_rit


# Gaussian input pulse (amplitude, width, center, ms), fixed and not inferred,
# same as Notebooks 1 and 3. Exposed at module level so any cell can reuse it
# outside of a NUTS run (e.g. a quick forward simulation).
DEFAULT_INPUT_PARAMS = (20.0, 6.0, 15.0)

# Normal prior for every parameter in jansen_rit.PARAM_NAMES_TEMPORAL: mean is
# its DCM default, std is PRIOR_CV of that value. Same CV everywhere so adding
# a free parameter later doesn't need a hand picked number. Replaces an earlier
# (1e-2, 1e2) log-uniform prior that put too much mass on sustained-oscillation
# gain combinations, see Notebook 3's prior predictive checks.
PRIOR_CV = 0.3
DEFAULT_PRIOR_SPECS = {name: (val, PRIOR_CV * val) for name, val in jansen_rit.DEFAULT_PARAMS_TEMPORAL.items()}


##############################################################################
# PRIORS
##############################################################################

def loguniform_pdf(x, low, high):
    """
    Analytic density of a loguniform(low, high), for overlaying on a plot. Unused since the
    switch to normal priors, kept for reference.
    """
    x = np.asarray(x, dtype=float)
    width = np.log(high) - np.log(low)
    pdf = np.zeros_like(x)
    mask = (x > low) & (x < high)
    pdf[mask] = 1.0 / (x[mask] * width)
    return pdf


def normal_pdf(x, mean, std):
    """Analytic density of a normal(mean, std), for overlaying on a plot."""
    x = np.asarray(x, dtype=float)
    return np.exp(-0.5 * ((x - mean) / std) ** 2) / (std * np.sqrt(2 * np.pi))


def truncated_normal_pdf(x, mean, std, low=0.0):
    """
    Analytic density of a normal(mean, std) truncated to [low, inf), for overlaying on a
    plot. Matches build_model's actual prior (dist.TruncatedNormal), unlike normal_pdf which
    would show density below low that the real prior doesn't have.
    """
    x = np.asarray(x, dtype=float)
    z = (x - mean) / std
    alpha = (low - mean) / std
    norm_const = 1.0 - 0.5 * (1.0 + erf(alpha / np.sqrt(2)))
    density = np.exp(-0.5 * z ** 2) / (std * np.sqrt(2 * np.pi) * norm_const)
    return np.where(x >= low, density, 0.0)


def get_prior_specs(free_param_names=None, specs=None):
    """
    Slice DEFAULT_PRIOR_SPECS (or a custom specs dict) down to free_param_names, default
    jansen_rit.FREE_PARAM_NAMES. Returns a dict param_name -> (mean, std) for build_model.
    """
    free_param_names = free_param_names or jansen_rit.FREE_PARAM_NAMES
    specs = specs or DEFAULT_PRIOR_SPECS
    missing = [p for p in free_param_names if p not in specs]
    if missing:
        raise KeyError(f'No prior spec for {missing}, add them to DEFAULT_PRIOR_SPECS or pass a custom specs dict')
    return {name: specs[name] for name in free_param_names}


##############################################################################
# NUTS KERNEL PRESETS
##############################################################################

def kernel_preset(name, free_param_names=None):
    """
    Two named NUTS configurations. Both init every chain from an independent prior draw
    (init_to_sample), not one shared start point, so R-hat is a meaningful convergence
    check. 'default' (target_accept_prob=0.75, max_tree_depth=14) is what the real fits and
    most sweeps use, chosen because a tighter/shallower setting failed the parameter
    recovery check (see the notebook). 'lower_accept' is the faster, looser option (0.6,
    12).
    """
    free_param_names = free_param_names or jansen_rit.FREE_PARAM_NAMES
    if name == 'default':
        return dict(target_accept_prob=0.75, max_tree_depth=14, adapt_step_size=True, init_strategy=init_to_sample())
    if name == 'lower_accept':
        return dict(target_accept_prob=0.6, max_tree_depth=12, adapt_step_size=True, init_strategy=init_to_sample())
    raise ValueError(f"Unknown kernel preset '{name}', expected 'default' or 'lower_accept'")


##############################################################################
# THE NUMPYRO MODEL
##############################################################################

def build_model(free_param_names=None, input_params=DEFAULT_INPUT_PARAMS, sigma=2):
    """
    Build a NumPyro model function fitting jansen_rit.simulate_erp_temporal to one observed
    waveform, sampling only free_param_names (default jansen_rit.FREE_PARAM_NAMES) and
    keeping the rest at jansen_rit.DEFAULT_PARAMS_TEMPORAL. Returns model(data, prior_specs,
    sigma_obs), where data comes from make_data_dict and prior_specs is a dict param_name ->
    (mean, std). input_params (the input pulse) and x_init (zero, a resting fixed point) are
    fixed, not inferred. Each parameter is sampled as TruncatedNormal(mean, std, low=0), not
    a plain Normal, since g can't physically be negative and a plain Normal's small mass
    below zero showed up often enough across a full run to matter. See the notebook's 'A
    numerical pitfall' section for why a two sided bound (the old log-uniform prior) caused
    real NUTS problems where this one-sided truncation doesn't.
    """
    free_param_names = free_param_names or jansen_rit.FREE_PARAM_NAMES

    def model(data, prior_specs, sigma_obs=sigma):
        ts_sim = data['ts_sim']
        n_burn = data['n_burn']
        sim_factor = data['sim_factor']
        ds = data['ds']
        nt_obs = data['nt_obs']
        xpy_obs = data['xpy_obs']

        free_params = {}
        for name in free_param_names:
            mean, std = prior_specs[name]
            free_params[name] = npr.sample(name, dist.TruncatedNormal(mean, std, low=0.0))

        xpy_full = jansen_rit.simulate_erp_temporal(ts_sim, input_params, free_params=free_params, integrator='rk4')
        # drop the burn-in, then downsample from the fine simulation grid down to the
        # data's own 1 ms grid (sim_factor), then to whatever the fit target itself
        # uses (ds), see make_data_dict.
        xpy_hat = xpy_full[n_burn:][::sim_factor][::ds]
        # an extreme draw (more likely with 10 free parameters than 4) can occasionally push the
        # integration to nan/inf. Swapping that for a large but finite value keeps the
        # likelihood (and anything that re-traces the model, like az.from_numpyro) well defined:
        # a nan loc crashes outright, a huge finite one just means "correctly, very unlikely".
        xpy_hat = jnp.nan_to_num(xpy_hat, nan=1e6, posinf=1e6, neginf=-1e6)

        with plate('data', size=nt_obs):
            xpy_model = npr.deterministic('xpy_model', xpy_hat)
            npr.sample('xpy_', dist.Normal(xpy_model, sigma_obs), obs=xpy_obs)

    return model


##############################################################################
# DATA / SIGNAL PREP FOR THE FIT
##############################################################################

def make_data_dict(time_ms, values, ds=1, burn_in_ms=10.0, dt_sim=1.0):
    """
    Build the data dict build_model's model expects, from a 1D observed waveform (time_ms in
    ms, stimulus at t=0). The model simulates on a finer grid (dt_sim) than the data itself,
    starting burn_in_ms before the data's first time point so the simulation never clips the
    input pulse's rising edge (x_init=0 is an exact fixed point before the pulse turns on,
    so this margin isn't for settling a transient). Every simulated trace then gets
    downsampled twice: sim_factor (the fine grid back to the data's own spacing) then ds (a
    further, optional coarsening). Both factors and n_burn are stored in the returned dict.
    """
    time_ms = np.asarray(time_ms, dtype=float)
    values = np.asarray(values, dtype=float)

    dt_data = float(time_ms[1] - time_ms[0])
    sim_factor = int(round(dt_data / dt_sim))
    if sim_factor < 1:
        raise ValueError(f'dt_sim ({dt_sim} ms) must be <= the data spacing ({dt_data:.3f} ms)')
    if not np.isclose(sim_factor * dt_sim, dt_data, rtol=0.02, atol=0):
        # a dt_sim that doesn't evenly divide dt_data makes sim_factor round to something that
        # reconstructs a slightly different spacing, which can cause erratic simulation error.
        # plain print, not warnings.warn, since the notebook's setup cell silences warnings.
        print(f'WARNING: dt_sim ({dt_sim} ms) does not evenly divide the data spacing ({dt_data:.3f} ms), '
              f'sim_factor={sim_factor} reconstructs {sim_factor * dt_sim:.4f} ms instead, this can '
              f'introduce real, erratic simulation error, pick a dt_sim that evenly divides dt_data.')
    n_burn = int(round(burn_in_ms / dt_sim))
    n_data = len(time_ms)
    # exactly n_burn burn-in points, then exactly enough fine-grid points to hit every
    # one of the n_data data points on the nose (avoids jnp.arange endpoint/rounding drift)
    n_sim = n_burn + (n_data - 1) * sim_factor + 1
    ts_sim = time_ms[0] - burn_in_ms + dt_sim * np.arange(n_sim)

    ts_obs = time_ms[::ds]
    xpy_obs = values[::ds]

    return dict(
        nt_obs=len(xpy_obs), ds=ds, sim_factor=sim_factor, n_burn=n_burn,
        ts_sim=jnp.asarray(ts_sim), ts_obs=ts_obs, xpy_obs=jnp.asarray(xpy_obs),
    )


def estimate_sigma_obs(time_ms, values, baseline_window=(-90.0, -10.0)):
    """
    Data driven estimate of the observation noise std, from the standard deviation of the
    pre-stimulus baseline (no evoked response there, just noise). Comes out smaller for
    trial-averaged data than a single trial automatically, from whatever trace is passed in.
    """
    time_ms = np.asarray(time_ms, dtype=float)
    values = np.asarray(values, dtype=float)
    lo, hi = baseline_window
    mask = (time_ms >= lo) & (time_ms <= hi)
    if mask.sum() < 2:
        raise ValueError(
            f'Not enough points in baseline window {baseline_window} to estimate sigma_obs '
            f'(trace covers {time_ms.min():.1f} to {time_ms.max():.1f} ms), pass a narrower/shifted window')
    return float(np.std(values[mask], ddof=1))


def simulate_on_data_grid(data, free_params, input_params=DEFAULT_INPUT_PARAMS, integrator='rk4'):
    """
    Run simulate_erp_temporal on data['ts_sim'] and slice/downsample it exactly like
    build_model's model does internally, so a plain forward simulation lines up with
    data['ts_obs']/data['xpy_obs']. Used for quick checks outside of a NUTS run.
    """
    xpy_full = jansen_rit.simulate_erp_temporal(data['ts_sim'], input_params, free_params=free_params, integrator=integrator)
    return xpy_full[data['n_burn']:][::data['sim_factor']][::data['ds']]


def select_window(time_ms, values, window):
    """Crop (time_ms, values) to a [lo, hi] ms window. values can be 1D or (time, channel)."""
    lo, hi = window
    mask = (time_ms >= lo) & (time_ms <= hi)
    return time_ms[mask], values[mask]


def crop_after_sigma_obs(time_ms, values, window=(0.0, 100.0), baseline_window=(-90.0, -10.0)):
    """
    estimate_sigma_obs needs the real pre-stimulus baseline, but the fit target doesn't (the
    model's response is an exact fixed point before the pulse turns on). Computes sigma_obs
    from the full trace, then crops to window (default (0, 100) ms) for everything
    downstream. Returns (time_ms_cropped, values_cropped, sigma_obs).
    """
    sigma_obs = estimate_sigma_obs(time_ms, values, baseline_window=baseline_window)
    t, v = select_window(time_ms, values, window)
    return t, v, sigma_obs


def apply_lowpass(signal, cutoff_hz=20, order=2, fs=500):
    """
    2nd order Butterworth low pass, zero phase (sosfiltfilt), used in the frequency filter
    sweep.
    """
    sos = butter(order, cutoff_hz, 'lp', fs=fs, output='sos')
    return sosfiltfilt(sos, signal)


##############################################################################
# RUNNING NUTS
##############################################################################

def run_nuts(model, data, prior_specs, rng_key, sigma_obs=2.0, n_warmup=500, n_samples=1000, n_chains=4,
             kernel_kwargs=None, chain_method='vectorized', verbose=True):
    """
    Build a NUTS kernel from kernel_kwargs (default kernel_preset('default')), run it, and
    return (mcmc, elapsed_seconds). chain_method='vectorized' runs every chain as one
    batched computation instead of sequentially, trading GPU memory for speed, first thing
    to switch back if a run runs out of memory.
    """
    kernel_kwargs = kernel_preset('default') if kernel_kwargs is None else kernel_kwargs
    kernel = NUTS(model, **kernel_kwargs)
    mcmc = MCMC(kernel, num_warmup=n_warmup, num_samples=n_samples, num_chains=n_chains,
                chain_method=chain_method, progress_bar=verbose)

    start = time.time()
    mcmc.run(rng_key, data, prior_specs, sigma_obs, extra_fields=('potential_energy', 'num_steps', 'diverging'))
    elapsed = time.time() - start

    if verbose:
        print(f'NUTS took {elapsed:.1f} sec')
        mcmc.print_summary(exclude_deterministic=True)

    return mcmc, elapsed


def sample_prior_predictive(model, data, prior_specs, n_prior=100, sigma_obs=2.0, rng_key=None):
    """
    Draw n_prior samples from the model's prior (Predictive with num_samples), for the prior
    predictive check.
    """
    rng_key = jax.random.PRNGKey(0) if rng_key is None else rng_key
    predictive = Predictive(model, num_samples=n_prior)
    return predictive(rng_key, data, prior_specs, sigma_obs)


def sample_posterior_predictive(model, data, prior_specs, posterior_samples, sigma_obs=2.0, rng_key=None):
    """
    Run the model forward at posterior_samples (filtered to array entries only), for the
    posterior predictive check.
    """
    rng_key = jax.random.PRNGKey(0) if rng_key is None else rng_key
    # Predictive doesn't like the non array entries numpyro.MCMC.get_samples() can carry, filter to be safe.
    filtered = {k: v for k, v in posterior_samples.items() if isinstance(v, jnp.ndarray)}
    predictive = Predictive(model, filtered)
    return predictive(rng_key, data, prior_specs, sigma_obs)


##############################################################################
# POSTERIOR DIAGNOSTICS
##############################################################################

def winsorize_pooled(chains_pooled, k=1.5):
    """
    Clip each parameter's pooled draws to its own Tukey whisker range (Q1 minus k*IQR to Q3
    plus k*IQR, floored at 0). Display only, for the plots further down, since a handful of
    draws can sit far from the rest even under the current prior. print_summary and the raw
    divergence count are never winsorized.
    """
    clipped = np.array(chains_pooled, dtype=float, copy=True)
    for i, row in enumerate(clipped):
        q1, q3 = np.percentile(row, [25, 75])
        iqr = q3 - q1
        lo = max(0.0, q1 - k * iqr)
        hi = q3 + k * iqr
        clipped[i] = np.clip(row, lo, hi)
    return clipped


def winsorize_idata(az_post, prm_names, k=1.5):
    """
    Same idea as winsorize_pooled, but returns a deep copy of an InferenceData with
    prm_names winsorized in place, for az.plot_trace / plotting.plot_trace_with_prior.
    """
    az_clean = az_post.copy()
    for prm in prm_names:
        vals = az_clean.posterior[prm].values
        q1, q3 = np.percentile(vals, [25, 75])
        iqr = q3 - q1
        lo = max(0.0, q1 - k * iqr)
        hi = q3 + k * iqr
        az_clean.posterior[prm].values[...] = np.clip(vals, lo, hi)
    return az_clean


def compute_map(chains_pooled, n_bins=50):
    """Histogram mode MAP estimate per parameter. chains_pooled: (n_params, n_draws) array."""
    if chains_pooled.shape[0] < 1:
        raise ValueError('chains_pooled must have shape (n_params, n_draws)')
    params_map = []
    for row in chains_pooled:
        hist, edges = np.histogram(row, bins=n_bins)
        idx = np.argmax(hist)
        params_map.append((edges[idx] + edges[idx + 1]) / 2)
    return np.array(params_map)


def low_prob_sample(prm_names, prior_predictions, rng=None):
    """
    Pick one initial value per parameter from the outer 2.5% tails of its prior predictive
    samples, for the 'sampling from the tail of the prior' experiment (does starting far
    from a typical draw help escape a mode the chains get stuck in). Returns (prior_q,
    tail_sample_vals, quantile_bounds), all needed by plotting.plot_prior_tail.
    """
    rng = np.random.default_rng() if rng is None else rng
    prior_q = np.array([prior_predictions[prm] for prm in prm_names]).T

    q_low_lim = np.quantile(prior_q, 0.0, axis=0)
    q_low = np.quantile(prior_q, 0.025, axis=0)
    q_high = np.quantile(prior_q, 0.975, axis=0)
    q_high_lim = np.quantile(prior_q, 1.0, axis=0)

    prob = np.array([q_low - q_low_lim, q_high_lim - q_high])
    prob = prob / prob.sum(axis=0)

    tail_sample_vals = []
    for iprm in range(len(prm_names)):
        tail_sample_vals.append(rng.choice(
            [rng.uniform(q_low_lim[iprm], q_low[iprm]), rng.uniform(q_high[iprm], q_high_lim[iprm])],
            p=prob[:, iprm]))

    return prior_q, tail_sample_vals, [q_low_lim, q_low, q_high, q_high_lim]


def stack_chains_by_elpd(az_post, prm_names, n_chains):
    """
    Split an InferenceData into one per chain, compare them by expected log predictive
    density (arviz stacking), and return the stacking weighted posterior draws plus each
    chain's weight. Used for the 'weighted multimodal posterior' experiment, weighting each
    chain by how well it predicts the data instead of pooling every chain equally.
    """
    chains_dict = {str(i): az_post.sel(chain=[i]) for i in range(n_chains)}
    comp_stack = az.compare(chains_dict, method='stacking')
    weights = np.array(comp_stack.sort_index()['weight'])

    list_chains = list(chains_dict.values())
    for i, idata in enumerate(list_chains):
        new_idata = az.InferenceData(
            posterior=idata.posterior, sample_stats=idata.sample_stats,
            log_likelihood=idata.log_likelihood if 'log_likelihood' in idata.groups() else None,
            observed_data=idata.observed_data if 'observed_data' in idata.groups() else None,
        )
        new_idata.add_groups(posterior_predictive=idata.posterior.copy())
        list_chains[i] = new_idata

    w_post = az.weight_predictions(list_chains, weights)
    chains_stacked = w_post.posterior_predictive[prm_names].to_array().values
    return chains_stacked, weights
