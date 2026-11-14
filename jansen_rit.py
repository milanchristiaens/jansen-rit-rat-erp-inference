"""
Jansen-Rit neural mass model (O. David's ERP formulation), JAX implementation.

The model has 9 state variables. State x8 is the observed output (the simulated
local field potential / ERP). It is driven by 10 physiological parameters:

    g_1, g_2, g_3, g_4   gain parameters (connectivity strengths between subpopulations)
    delta                sigmoid slope / firing threshold parameter
    tau_e, tau_i         excitatory / inhibitory synaptic time constants
    h_e, h_i             excitatory / inhibitory synaptic gains
    u                     constant external input

For this project we currently only infer g_1..g_4 and keep the rest fixed at
DEFAULT_PARAMS. Everything here is written so that adding more free parameters
later (e.g. delta, tau_e) is a matter of extending FREE_PARAM_NAMES, not
rewriting the model.

Sections in this file:
    parameter vector helpers   turning a dict of overrides into a full param array
    model dynamics, constant input   the original DCM ERP model, sigmoid + vector_field
    model dynamics, pulse input      same model with a Gaussian input pulse
    fixed step integrators     Euler and RK4
    simulation entry points    simulate_erp / simulate_erp_temporal
    canonical (textbook) Jansen-Rit   the classic 1995 formulation, kept separately

"""
from collections import OrderedDict

import jax
import jax.numpy as jnp

cpu_device = jax.devices('cpu')[0]
gpu_device = jax.devices('gpu')[0] if 'gpu' in [d.platform for d in jax.devices()] else None

# Canonical parameter order used everywhere in this module. Notebooks 2 and 3
# build parameter vectors in this order.
PARAM_NAMES = ['g_1', 'g_2', 'g_3', 'g_4', 'delta', 'tau_i', 'h_i', 'tau_e', 'h_e', 'u']

# The subset currently being inferred (MCMC + SBI). Extend this list (and pick
# matching priors in the inference notebooks) to infer more parameters later.
FREE_PARAM_NAMES = ['g_1', 'g_2', 'g_3', 'g_4']

# Default/fixed values for every parameter, taken from the original DCM fit.
DEFAULT_PARAMS = OrderedDict([
    ('g_1', 0.42),
    ('g_2', 0.76),
    ('g_3', 0.15),
    ('g_4', 0.16),
    ('delta', 12.13),
    ('tau_i', 7.77),
    ('h_i', 27.88),
    ('tau_e', 5.77),
    ('h_e', 1.63),
    ('u', 3.94),
])

# Defaults for the temporal-input variant of the model (used when the input is
# a Gaussian pulse instead of a constant). Slightly different delta/damping,
# matching the original notebook.
DEFAULT_PARAMS_TEMPORAL = OrderedDict([
    ('g_1', 0.42),
    ('g_2', 0.76),
    ('g_3', 0.15),
    ('g_4', 0.16),
    ('delta', 10.13),
    ('tau_i', 7.77),
    ('h_i', 27.88),
    ('tau_e', 5.77),
    ('h_e', 1.63),
    ('damping', 2.3),
])


# Same idea as PARAM_NAMES, but for the temporal (pulse input) model: the last
# entry is 'damping' instead of 'u', since the pulse replaces the constant
# drive. Used wherever a notebook wants to sample/vary every parameter of
# simulate_erp_temporal, not just FREE_PARAM_NAMES.
PARAM_NAMES_TEMPORAL = list(DEFAULT_PARAMS_TEMPORAL.keys())

N_STATES = 9
ALPHA_DEFAULT = -0.56  # sigmoid steepness constant, called `constants[0]`/`alpha` throughout


##############################################################################
# PARAMETER VECTOR HELPERS
##############################################################################

def build_param_vector(free_params=None, base_defaults=None):
    """
    Build a full, canonically-ordered parameter array from a dict of
    overrides on top of the defaults. E.g. build_param_vector({'g_1': 0.5})
    returns all 10 default params with g_1 replaced.

    This is the main entry point for keeping the "g-only for now" scope
    list/dict-driven: pass a dict with only the parameters you're currently
    sampling/inferring and everything else stays fixed.
    """
    base_defaults = DEFAULT_PARAMS if base_defaults is None else base_defaults
    free_params = free_params or {}
    values = dict(base_defaults)
    for name, val in free_params.items():
        if name not in values:
            raise KeyError(f"Unknown parameter '{name}', expected one of {list(base_defaults.keys())}")
        values[name] = val
    return jnp.array([values[name] for name in base_defaults.keys()])


##############################################################################
# MODEL DYNAMICS, CONSTANT INPUT
##############################################################################

def sigmoid(x1, x2, delta, alpha):
    """Sigmoidal firing-rate nonlinearity used by every population in the model."""
    return (1. / (1. + jnp.exp(alpha * (x1 - delta * x2)))) - 0.5
sigmoid = jax.jit(sigmoid, device=cpu_device)


def vector_field(state, t, alpha, params):
    """
    Right-hand side of the JR ODE system with a constant input `u`.

    params must be ordered as PARAM_NAMES: [g_1, g_2, g_3, g_4, delta, tau_i, h_i, tau_e, h_e, u]
    Use build_param_vector() to construct this from a dict of overrides.
    """
    x0, x1, x2, x3, x4, x5, x6, x7, x8 = state
    g_1, g_2, g_3, g_4, delta, tau_i, h_i, tau_e, h_e, u = params

    dx0 = x3
    dx1 = x4
    dx2 = x5
    dx3 = (1. / tau_e) * (h_e * (g_1 * sigmoid(x8, x4 - x5, delta, alpha) + u) - (x0 / tau_e) - 2 * x3)
    dx4 = (1. / tau_e) * (h_e * (g_2 * sigmoid(x0, x3, delta, alpha)) - (x1 / tau_e) - 2 * x4)
    dx5 = (1. / tau_i) * (h_i * (g_4 * sigmoid(x6, x7, delta, alpha)) - (x2 / tau_i) - 2 * x5)
    dx6 = x7
    dx7 = (1. / tau_e) * (h_e * (g_3 * sigmoid(x8, x4 - x5, delta, alpha)) - (x6 / tau_e) - 2 * x7)
    dx8 = x4 - x5

    return jnp.array([dx0, dx1, dx2, dx3, dx4, dx5, dx6, dx7, dx8])
vector_field = jax.jit(vector_field, device=cpu_device)


##############################################################################
# MODEL DYNAMICS, PULSE INPUT
##############################################################################

def temporal_input(t, amplitude, width, t0):
    """Gaussian input pulse, used instead of a constant `u` for evoked-response simulations."""
    return amplitude * jnp.exp(-((t - t0) ** 2) / (2 * width ** 2))
temporal_input = jax.jit(temporal_input, device=cpu_device)


def vector_field_temporal(state, t, alpha, params, input_params):
    """
    Right-hand side of the JR ODE system with a temporal (Gaussian pulse) input,
    used for evoked-response simulations. `params` here is ordered as
    [g_1, g_2, g_3, g_4, delta, tau_i, h_i, tau_e, h_e, damping] (see
    DEFAULT_PARAMS_TEMPORAL) -- the constant input `u` is replaced by the pulse.
    """
    x0, x1, x2, x3, x4, x5, x6, x7, x8 = state
    g_1, g_2, g_3, g_4, delta, tau_i, h_i, tau_e, h_e, damping = params
    amplitude, width, t0 = input_params

    u = temporal_input(t, amplitude, width, t0)

    dx0 = x3
    dx1 = x4
    dx2 = x5
    dx3 = (1. / tau_e) * (h_e * (g_1 * sigmoid(x8, x4 - x5, delta, alpha) + u) - (x0 / tau_e) - damping * x3)
    dx4 = (1. / tau_e) * (h_e * (g_2 * sigmoid(x0, x3, delta, alpha)) - (x1 / tau_e) - damping * x4)
    dx5 = (1. / tau_i) * (h_i * (g_4 * sigmoid(x6, x7, delta, alpha)) - (x2 / tau_i) - damping * x5)
    dx6 = x7
    dx7 = (1. / tau_e) * (h_e * (g_3 * sigmoid(x8, x4 - x5, delta, alpha)) - (x6 / tau_e) - damping * x7)
    dx8 = x4 - x5

    return jnp.array([dx0, dx1, dx2, dx3, dx4, dx5, dx6, dx7, dx8])
vector_field_temporal = jax.jit(vector_field_temporal, device=cpu_device)


##############################################################################
# FIXED STEP INTEGRATORS
##############################################################################

def odeint_euler(f, x0, t, *args):
    """Fixed-step forward Euler integration via jax.lax.scan."""
    def step(state, t_now):
        x_prev, t_prev = state
        h = t_now - t_prev
        x = x_prev + h * f(x_prev, t_prev, *args)
        return (x, t_now), x
    _, xs = jax.lax.scan(step, (x0, t[0]), t)
    return xs


def odeint_rk4(f, x0, t, *args):
    """Fixed-step classic RK4 integration via jax.lax.scan. More accurate than Euler, same cost pattern."""
    def step(state, t_now):
        x_prev, t_prev = state
        h = t_now - t_prev
        k1 = h * f(x_prev, t_prev, *args)
        k2 = h * f(x_prev + k1 / 2., t_prev + h / 2., *args)
        k3 = h * f(x_prev + k2 / 2., t_prev + h / 2., *args)
        k4 = h * f(x_prev + k3, t_now, *args)
        x = x_prev + (1. / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        return (x, t_now), x
    _, xs = jax.lax.scan(step, (x0, t[0]), t)
    return xs


##############################################################################
# SIMULATION ENTRY POINTS
##############################################################################

def simulate_erp(ts, free_params=None, x_init=None, alpha=ALPHA_DEFAULT, integrator='rk4',
                  base_defaults=None):
    """
    Convenience wrapper for the constant-input model: build the full parameter
    vector from a dict of overrides, integrate, and return the observed trace
    (state x8) at every time in `ts`.

    free_params: dict, e.g. {'g_1': 0.5, 'g_2': 0.8} -- anything not given
        falls back to DEFAULT_PARAMS. Pass {} or None for the pure default trace.
    """
    params = build_param_vector(free_params, base_defaults=base_defaults or DEFAULT_PARAMS)
    x0 = jnp.zeros(N_STATES) if x_init is None else jnp.asarray(x_init)
    integrate = odeint_rk4 if integrator == 'rk4' else odeint_euler
    xs = integrate(vector_field, x0, ts, alpha, params)
    return xs[:, 8]


def simulate_erp_temporal(ts, input_params, free_params=None, x_init=None, alpha=ALPHA_DEFAULT,
                           integrator='rk4', base_defaults=None, key=None, noise_params=(0., 0., 1)):
    """
    Convenience wrapper for the temporal-input (Gaussian pulse) model, optionally
    with additive noise (bias amplitude, white-noise amplitude, high frequency filter on white noise
    (larger value of noise_tau -> smaller update rate -> bigger filtering)) -- 
    matches the noise model used for the SBI simulations.
    """
    params = build_param_vector(free_params, base_defaults=base_defaults or DEFAULT_PARAMS_TEMPORAL)
    x0 = jnp.zeros(N_STATES) if x_init is None else jnp.asarray(x_init)
    integrate = odeint_rk4 if integrator == 'rk4' else odeint_euler
    xs = integrate(vector_field_temporal, x0, ts, alpha, params, input_params)
    x_py = xs[:, 8]

    wn_bias_amp, wn_amp, noise_tau = noise_params
    if key is None or (wn_bias_amp == 0 and wn_amp == 0):
        return x_py

    k1, k2 = jax.random.split(key, 2)
    white = wn_amp * jax.random.normal(k1, x_py.shape)
    bias = wn_bias_amp * jax.random.normal(k2, ())

    dt = ts[1] - ts[0]
    update_rate = 1.0 - jnp.exp(-dt / noise_tau)

    def lowpass_step(prev, x):
        y = update_rate * x + (1 - update_rate) * prev
        return y, y
    _, white_smooth = jax.lax.scan(lowpass_step, white[0], white)

    return x_py + white_smooth + bias


##############################################################################
# CANONICAL (TEXTBOOK) JANSEN-RIT
##############################################################################

def sigmoid_canonical(v, v0, r, e0):
    """Standard Jansen-Rit / David sigmoid, range 0 to 2*e0."""
    return (2 * e0) / (1. + jnp.exp(r * (v0 - v)))


def canonical_pulse(t, amplitude, width, t0):
    """Same Gaussian pulse shape as jansen_rit.temporal_input, own amplitude convention."""
    return amplitude * jnp.exp(-((t - t0) ** 2) / (2 * width ** 2))


def canonical_vector_field(state, t, canonical_params, pulse_params):
    """
    Right hand side of the textbook Jansen-Rit ODE system (8 states, single argument
    sigmoid), pulse driven. state is (y0a, dy0a, y0b, dy0b, y1, dy1, y2, dy2): y0a/y0b
    are the spiny stellate and inhibitory interneuron populations, y1/y2 the excitatory
    and inhibitory postsynaptic potentials on the pyramidal population. Output is y1-y2.
    """
    y0a, dy0a, y0b, dy0b, y1, dy1, y2, dy2 = state
    A, B, a, b, v0, r, e0, C1, C2, C3, C4 = canonical_params
    amplitude, width, t0 = pulse_params
    p_t = canonical_pulse(t, amplitude, width, t0)
    pyr_output = y1 - y2

    d_y0a = dy0a
    d_dy0a = a * A * C1 * sigmoid_canonical(pyr_output, v0, r, e0) - 2 * a * dy0a - a ** 2 * y0a
    d_y0b = dy0b
    d_dy0b = a * A * C3 * sigmoid_canonical(pyr_output, v0, r, e0) - 2 * a * dy0b - a ** 2 * y0b
    d_y1 = dy1
    d_dy1 = a * A * (p_t + C2 * sigmoid_canonical(y0a, v0, r, e0)) - 2 * a * dy1 - a ** 2 * y1
    d_y2 = dy2
    d_dy2 = b * B * C4 * sigmoid_canonical(y0b, v0, r, e0) - 2 * b * dy2 - b ** 2 * y2

    return jnp.array([d_y0a, d_dy0a, d_y0b, d_dy0b, d_y1, d_dy1, d_y2, d_dy2])


def add_pulse_noise(key, trace, dt, noise_params):
    """Same bias + lowpass white noise model as jansen_rit.simulate_erp_temporal, applied
    externally since the canonical simulator above is not part of jansen_rit.py."""
    wn_bias_amp, wn_amp, noise_tau = noise_params
    if key is None or (wn_bias_amp == 0 and wn_amp == 0):
        return trace
    k1, k2 = jax.random.split(key, 2)
    white = wn_amp * jax.random.normal(k1, trace.shape)
    bias = wn_bias_amp * jax.random.normal(k2, ())
    update_rate = 1.0 - jnp.exp(-dt / noise_tau)

    def lowpass_step(prev, x):
        y = update_rate * x + (1 - update_rate) * prev
        return y, y
    _, white_smooth = jax.lax.scan(lowpass_step, white[0], white)
    return trace + white_smooth + bias


def simulate_canonical_erp_temporal(ts_ms, pulse_params_ms, gains=None, key=None, noise_params=(0., 0., 1),
                                     A=3.25, B=22.0, a=100.0, b=50.0, v0=6.0, r=0.56, e0=2.5):
    """
    Textbook Jansen & Rit (1995) model, pulse driven. gains: (C1, C2, C3, C4) override,
    defaults to the canonical C1=135 ratio. ts_ms and pulse_params_ms (amplitude, width,
    center, width/center in ms) stay in millisecond convention for the caller, the seconds
    conversion needed for a/b/e0 happens internally.
    """
    C1, C2, C3, C4 = gains if gains is not None else (135.0, 0.8 * 135.0, 0.25 * 135.0, 0.25 * 135.0)
    canonical_params = (A, B, a, b, v0, r, e0, C1, C2, C3, C4)

    amplitude, width_ms, t0_ms = pulse_params_ms
    pulse_params_sec = (amplitude, width_ms / 1000.0, t0_ms / 1000.0)

    ts_sec = jnp.asarray(ts_ms) / 1000.0
    x0 = jnp.zeros(8)
    xs = odeint_rk4(canonical_vector_field, x0, ts_sec, canonical_params, pulse_params_sec)
    trace = xs[:, 4] - xs[:, 6]

    dt_sec = ts_sec[1] - ts_sec[0]
    return add_pulse_noise(key, trace, dt_sec, noise_params)
