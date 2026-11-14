"""
Shared plotting style used across all three notebooks (colors, axis style,
violin/legend helpers) so a figure looks the same no matter which notebook
it came from. Imports mcmc since plot_trace_with_prior draws the prior curve
with mcmc.truncated_normal_pdf.

Sections in this file:
    constants               house colors, shared g axis ticks, house DPI
    general helpers          axis spines/ticks, subject dropdown, legends
    violin plot building blocks   shared pieces used by the comparison plots
    posterior comparison plots   wake vs anesthesia violins, used by 02_mcmc.ipynb and 03_sbi.ipynb
    MCMC notebook plots (02_mcmc.ipynb)   trace, fitted vs observed, prior tail, convergence
    SBI / PCIst diagnostic plots (03_sbi.ipynb)   variance, similarity matrix, PCIst vs params

"""
# standard library
import re

# numeric / data
import numpy as np
import pandas as pd
import jax.numpy as jnp

# plotting stack
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import seaborn as sns
import arviz as az
from statannotations.Annotator import Annotator

# notebook widgets
import ipywidgets as widgets
from IPython.display import display

# local
import mcmc

# House palette: wake is always the neutral/grey tone, each anesthetic gets its
# own color, consistent across every plot in the project.
ANES_COLORS = {'wake': 'slategray', 'keta': 'cornflowerblue', 'prop': 'crimson', 'sevo': 'mediumseagreen'}
DATASET_PALETTES = {
    'keta': {'wake': 'slategray', 'keta': 'cornflowerblue'},
    'prop': {'wake': 'slategray', 'prop': 'crimson'},
    'sevo': {'wake': 'slategray', 'sevo': 'mediumseagreen'},
}

# Shared axis range for every plot of a g posterior (violins, KDEs, trace
# plots). Linear now, not log: the g prior switched from log-uniform (1e-2 to
# 1e2) to a normal centered on the DCM defaults, so posterior values sit in a
# much narrower, linear-scale range. Change this one line to widen/narrow the
# axis on every g plot in the project at once, or pass y_ticks explicitly to
# a single call if just that one plot needs something different.
G_AXIS_YTICKS = np.round(np.linspace(0, 1.4, 8), 2)  # 0 ... 1.4, covers the g_1/g_2 prior out to ~3 std

# Figures get pinned back to this DPI (matplotlib's own default) before layout,
# see the note above the violin comparison plots for why.
HOUSE_DPI = 100


##############################################################################
# GENERAL HELPERS
##############################################################################


##############################################################################
# GENERAL HELPERS
##############################################################################

def subject_explorer(plot_fn, subject_to_anes, default_subject='j12', description='Subject:'):
    """
    Dropdown widget that redraws plot_fn(subject, anes_list) whenever the selection changes.
    subject_to_anes comes from data.discover_available_conditions. Used for any per subject
    interactive plot.
    """
    subjects = sorted(subject_to_anes.keys(), key=lambda s: int(re.findall(r'\d+', s)[0]) if re.findall(r'\d+', s) else 0)
    default_subject = default_subject if default_subject in subjects else subjects[0]

    dropdown = widgets.Dropdown(options=subjects, value=default_subject, description=description)
    out = widgets.Output()

    def redraw(change=None):
        with out:
            out.clear_output(wait=True)
            plot_fn(dropdown.value, subject_to_anes[dropdown.value])

    dropdown.observe(redraw, names='value')
    display(dropdown, out)
    redraw()


def plot_condition_bars(ax, condition_labels, condition_values, condition_colors, wake_values,
                         condition_errs=None, wake_err=None, wake_label='Wake', ylabel='Value', width=0.6, capsize=4):
    """
    Bar chart with one pooled wake bar (mean and SEM across sessions) plus one bar per
    anesthesia condition. condition_errs/wake_err let a caller pass its own SEM (e.g. trial
    level) instead of recomputing it here.
    """
    wake_values = np.atleast_1d(np.asarray(wake_values, dtype=float))
    wake_mean = np.mean(wake_values)
    if wake_err is None:
        wake_err = np.std(wake_values, ddof=1) / np.sqrt(len(wake_values)) if len(wake_values) > 1 else None

    positions = np.arange(len(condition_labels) + 1)
    ax.bar(positions[0], wake_mean, yerr=wake_err, width=width, color=ANES_COLORS['wake'], capsize=capsize)
    ax.bar(positions[1:], condition_values, yerr=condition_errs, width=width, color=condition_colors, capsize=capsize)
    ax.set_xticks(positions)
    ax.set_xticklabels([wake_label] + [c.capitalize() for c in condition_labels])
    ax.set_ylabel(ylabel)
    modify_axis_spines(ax, which=['y'])
    return ax


def apply_house_style():
    """Call once at the top of a notebook to set the shared font and DPI."""
    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['font.size'] = 10
    plt.rcParams['svg.fonttype'] = 'none'
    plt.rcParams['figure.dpi'] = HOUSE_DPI  # see the comment on HOUSE_DPI above, avoids the annotator gap bug


def modify_axis_spines(ax, which=None, base=1.0, xticks=(), yticks=(), yaxis_left=True, xaxis_bot=True,
                        xlim=None, ylim=None):
    """
    House style: keep only the left/bottom spines, bounded to the tick range. Pass xlim/ylim
    to also fix the axis range; ticks are then recomputed with _nice_ticks_for_range unless
    an explicit xticks/yticks list is given.
    """
    which = which or []
    tick_locator = plticker.MultipleLocator(base=base)

    if yaxis_left:
        ax.spines.right.set(visible=False)
        yspine = ax.spines.left
    else:
        ax.spines.left.set(visible=False)
        yspine = ax.spines.right

    if xaxis_bot:
        ax.spines.top.set(visible=False)
        xspine = ax.spines.bottom
    else:
        ax.spines.bottom.set(visible=False)
        xspine = ax.spines.top

    if 'x' in which:
        xticks = list(xticks)
        if xlim is not None:
            ax.set_xlim(*xlim)
            if len(xticks) == 0:
                xticks = _nice_ticks_for_range(xlim[0], xlim[1])
        elif len(xticks) == 0:
            xticks = ax.get_xticks()
            ax.xaxis.set_major_locator(tick_locator)
        ax.set_xticks(xticks)
        xspine.set_bounds(ax.get_xticks()[0], ax.get_xticks()[-1])
    else:
        ax.spines.bottom.set(visible=False)

    if 'y' in which:
        yticks = list(yticks)
        if ylim is not None:
            ax.set_ylim(*ylim)
            if len(yticks) == 0:
                yticks = _nice_ticks_for_range(ylim[0], ylim[1])
        elif len(yticks) == 0:
            yticks = ax.get_yticks()
        ax.set_yticks(yticks)
        yspine.set_bounds(ax.get_yticks()[0], ax.get_yticks()[-1])
        if len(yticks) == 0:
            ax.yaxis.set_major_locator(tick_locator)
    else:
        ax.spines.left.set(visible=False)


def _nice_ticks_for_range(vmin, vmax, target=6):
    """
    Tick positions that start exactly at vmin and end exactly at vmax, spaced by a round
    step, so an axis spine can be bounded to real labeled ticks instead of trailing into
    blank space. Used by modify_axis_spines.
    """
    vmin, vmax = float(vmin), float(vmax)
    span = vmax - vmin
    if span <= 0:
        return np.array([vmin, vmax])

    raw_step = span / target
    magnitude = 10 ** np.floor(np.log10(raw_step))
    step = magnitude * 10  # fallback if nothing below fits (shouldn't happen)
    for base_step in (1, 2, 2.5, 5, 10):
        candidate = base_step * magnitude
        if candidate >= raw_step - 1e-9:
            step = candidate
            break

    ticks = np.arange(vmin, vmax, step)
    if len(ticks) == 0 or not np.isclose(ticks[-1], vmax):
        if len(ticks) > 1 and (vmax - ticks[-1]) < 0.4 * step:
            ticks = ticks[:-1]
        ticks = np.append(ticks, vmax)
    return ticks


def dedupe_legend(fig_or_ax, **legend_kwargs):
    """Legend that lists each label only once, built from every axis in a figure if needed."""
    if hasattr(fig_or_ax, 'get_legend_handles_labels'):
        handles, labels = fig_or_ax.get_legend_handles_labels()
    else:
        handles, labels = [], []
        for ax in fig_or_ax.axes:
            h, l = ax.get_legend_handles_labels()
            handles += h
            labels += l
    by_label = dict(zip(labels, handles))
    return fig_or_ax.legend(by_label.values(), by_label.keys(), **legend_kwargs)


def _remove_legend(plot_obj):
    """Drop a seaborn plot's own legend, used before building a custom one."""
    try:
        plot_obj.legend_.remove()
    except AttributeError:
        pass


##############################################################################
# VIOLIN PLOT BUILDING BLOCKS
##############################################################################

def violin_span_at_y(vertices, y0):
    """
    Left/right x edge of a violin's own outline at a given y, so a stat line can hug the
    violin's real shape instead of a fixed width bar.
    """
    v = np.asarray(vertices)
    mid = np.argmax(v[:, 1])
    left = v[1:mid + 1]        # ascending y, left (negative x) side
    right = v[mid:-1][::-1]    # ascending y, right (positive x) side
    y0c = min(max(y0, v[:, 1].min()), v[:, 1].max())
    x_left = np.interp(y0c, left[:, 1], left[:, 0])
    x_right = np.interp(y0c, right[:, 1], right[:, 0])
    return x_left, x_right


def _shrink_span(x_left, x_right, frac):
    """
    Pull a horizontal span in a bit on each side so its drawn line ends inside the violin
    outline.
    """
    pad = (x_right - x_left) * frac
    return x_left + pad, x_right - pad


def add_violin_mean_line(ax, body, mean_val, lw=1.8, color=None, line_shrink=0.03):
    """Draw one mean line across a violin body, matching that violin's width at the mean."""
    color = body.get_edgecolor()[0] if color is None else color
    x_left, x_right = violin_span_at_y(body.get_paths()[0].vertices, mean_val)
    x_left, x_right = _shrink_span(x_left, x_right, line_shrink)
    ax.hlines(mean_val, x_left, x_right, color=color, linewidth=lw, capstyle='butt', zorder=5)


def _collection_index_map(df, x_col, hue_col, order, hue_order, value_col='value', n_start=0):
    """
    Map each (x, hue) pair to its index in ax.collections, in the order seaborn draws dodged
    violins.
    """
    idx_map = {}
    idx = n_start
    for x_val in order:
        for hue_val in hue_order:
            sub = df.loc[(df[x_col] == x_val) & (df[hue_col] == hue_val), value_col]
            if len(sub) == 0:
                continue
            idx_map[(x_val, hue_val)] = idx
            idx += 1
    return idx_map


def _lighten_color(rgb, amount):
    """Blend an RGB(A) color towards white by `amount` (0 unchanged, 1 white)."""
    r, g, b = rgb[:3]
    return (r + (1 - r) * amount, g + (1 - g) * amount, b + (1 - b) * amount)


def _violin_stat_lines(ax, df, x_col, hue_col, order, hue_order, value_col='value', lw=1.2, n_start=0,
                        fill_lighten=0.55, line_shrink=0.03):
    """
    Draw the 25th percentile, mean and 75th percentile as short lines over each violin
    (replacing a boxplot's box/median), colored like the violin and lightening its fill so
    the lines stand out. Assumes the violins were drawn with inner=None.
    """
    n_hue = len(hue_order)
    idx_map = _collection_index_map(df, x_col, hue_col, order, hue_order, value_col, n_start=n_start)
    for i, x_val in enumerate(order):
        for j, hue_val in enumerate(hue_order):
            sub = df.loc[(df[x_col] == x_val) & (df[hue_col] == hue_val), value_col]
            if len(sub) == 0:
                continue
            coll = ax.collections[idx_map[(x_val, hue_val)]]
            vertices = coll.get_paths()[0].vertices
            face_rgba = coll.get_facecolor()[0]
            line_color = (face_rgba[0], face_rgba[1], face_rgba[2], 1.0)

            light_rgb = _lighten_color(face_rgba, fill_lighten)
            coll.set_facecolor((*light_rgb, face_rgba[3]))

            q1, q3, mean = sub.quantile(0.25), sub.quantile(0.75), sub.mean()
            for y0 in (q1, q3):
                x_left, x_right = violin_span_at_y(vertices, y0)
                x_left, x_right = _shrink_span(x_left, x_right, line_shrink)
                ax.hlines(y0, x_left, x_right, color=line_color, linewidth=lw, capstyle='butt', zorder=5)
            x_left, x_right = violin_span_at_y(vertices, mean)
            x_left, x_right = _shrink_span(x_left, x_right, line_shrink)
            ax.hlines(mean, x_left, x_right, color=line_color, linewidth=lw * 1.8, capstyle='butt', zorder=5)


def filter_positive(df, value_col='value'):
    """
    Keep only positive values in value_col. g can't be negative, so this is applied before
    every violin/effect size plot.
    """
    return df[df[value_col] > 0].copy()


def cohens_d(x, y):
    """
    Standardized mean difference (y minus x, in pooled standard deviations). Used instead of
    a raw Wasserstein distance so the effect size is comparable across parameters that live
    on very different scales.
    """
    x, y = np.asarray(x), np.asarray(y)
    nx, ny = len(x), len(y)
    pooled_std = np.sqrt(((nx - 1) * x.std(ddof=1) ** 2 + (ny - 1) * y.std(ddof=1) ** 2) / (nx + ny - 2))
    return (y.mean() - x.mean()) / pooled_std


##############################################################################
# POSTERIOR COMPARISON PLOTS (02_mcmc.ipynb and 03_sbi.ipynb)
##############################################################################
# The 4 functions below all call fig.set_dpi(HOUSE_DPI) and apply y_lim before the
# statannotations brackets are drawn. Both are needed: statannotations places its
# outside brackets wrong at any DPI other than matplotlib's default of 100 (happens
# for free under a Jupyter retina display setting), and applying y_lim after the
# brackets are drawn leaves a gap sized for the wrong axis range.

def plot_single_dataset_comparison(fig, ax, param_df, dataset_name, dataset_palette, lw=1.0, lgd_loc=(0.8, 0.75),
                                    legend=True, saturation=1.0, stat_loc='outside',
                                    ylabel='Pooled Posterior Value', custom_labels=None,
                                    width=0.5, labelpad=10, theta_column='theta_label', y_ticks=None, y_lim=None,
                                    fill_lighten=0.55, line_shrink=0.06):
    """
    Violin plot of every parameter for one anesthesia dataset, wake vs that anesthetic. See
    the section note above for the DPI/y_lim ordering.
    """
    fig.set_dpi(HOUSE_DPI)
    y_ticks = G_AXIS_YTICKS if y_ticks is None else y_ticks
    df_filtered = filter_positive(param_df[param_df['dataset'] == dataset_name])
    df_filtered['condition_cap'] = df_filtered['condition'].str.capitalize()
    theta_labels = df_filtered[theta_column].unique()
    custom_labels = custom_labels if custom_labels is not None else theta_labels

    unified_palette = {k.capitalize(): v for k, v in dataset_palette.items()}
    hue_order = ['Wake'] + [c.capitalize() for c in dataset_palette.keys() if c.lower() != 'wake']

    vio = sns.violinplot(x=theta_column, y='value', data=df_filtered, hue='condition_cap', palette=unified_palette,
                          ax=ax, linewidth=lw, hue_order=hue_order, dodge=True, cut=0, inner=None,
                          saturation=saturation, width=width, density_norm='width')
    _violin_stat_lines(ax, df_filtered, theta_column, 'condition_cap', theta_labels, hue_order, lw=lw,
                        fill_lighten=fill_lighten, line_shrink=line_shrink)

    if y_lim is not None:
        modify_axis_spines(ax, which=['y'], ylim=y_lim, yticks=y_ticks)

    sig_pairs, sig_annotations = [], []
    anes_name_cap = dataset_name.capitalize()
    for theta in theta_labels:
        sig_pairs.append(((theta, 'Wake'), (theta, anes_name_cap)))
        wake_samples = df_filtered.loc[(df_filtered[theta_column] == theta) & (df_filtered['condition'] == 'wake'), 'value'].to_numpy()
        anes_samples = df_filtered.loc[(df_filtered[theta_column] == theta) & (df_filtered['condition'] == dataset_name.lower()), 'value'].to_numpy()
        cohen_d = cohens_d(wake_samples, anes_samples)
        sig_annotations.append(f'd: {cohen_d:.2f}')

    annotator = Annotator(x=theta_column, y='value', hue='condition_cap', pairs=sig_pairs, data=df_filtered, ax=vio,
                           plot='violinplot', verbose=False)
    annotator.configure(text_format='simple', loc=stat_loc, line_width=lw)
    annotator.set_custom_annotations(sig_annotations)
    annotator.annotate()

    ax.set_xticks(range(len(theta_labels)))
    ax.set_xticklabels(custom_labels, rotation=0)
    ax.tick_params(axis='x', length=0)
    modify_axis_spines(ax, which=['y'], ylim=y_lim, yticks=y_ticks)

    if legend:
        ax.legend(loc=lgd_loc, frameon=False, ncol=len(hue_order))
    else:
        _remove_legend(vio)

    ax.set_ylabel(ylabel, labelpad=labelpad)
    ax.patch.set_alpha(0.0)
    ax.set_xlabel('')
    fig.tight_layout()


def plot_parameter_across_datasets(fig, ax, param_df, parameter_label, anes_keys, dataset_palettes, lw=1.0,
                                    lgd_loc=(0.8, 0.75), legend=True, saturation=1.0, stat_loc='outside',
                                    ylabel='Pooled Posterior Value', width=0.5, labelpad=10,
                                    theta_column='theta_label', y_ticks=None, y_lim=None, fill_lighten=0.55,
                                    line_shrink=0.03):
    """
    Violin plot of one parameter, wake vs anesthesia, side by side for every anesthesia
    dataset. See the section note above for the DPI/y_lim ordering.
    """
    fig.set_dpi(HOUSE_DPI)
    y_ticks = G_AXIS_YTICKS if y_ticks is None else y_ticks
    df_filtered = filter_positive(param_df[param_df[theta_column] == parameter_label])
    df_filtered['condition_cap'] = df_filtered['condition'].str.capitalize()
    df_filtered['role'] = np.where(df_filtered['condition'] == 'wake', 'Wake', 'Anesthesia')

    unified_palette = {}
    for ds_palette in dataset_palettes.values():
        for cond_name, color in ds_palette.items():
            unified_palette[cond_name.capitalize()] = color

    cap_anes_keys = [k.capitalize() for k in anes_keys]
    hue_order = ['Wake', 'Anesthesia']

    vio = sns.violinplot(x='dataset', y='value', data=df_filtered, hue='role', order=anes_keys,
                          hue_order=hue_order, palette={'Wake': unified_palette['Wake'], 'Anesthesia': 'lightgray'},
                          ax=ax, linewidth=lw, dodge=True, cut=0, inner=None, saturation=saturation, width=width,
                          density_norm='width')

    n_hue = len(hue_order)
    idx_map = _collection_index_map(df_filtered, 'dataset', 'role', anes_keys, hue_order)
    for i, dataset in enumerate(anes_keys):
        idx = idx_map.get((dataset, 'Anesthesia'))
        if idx is not None:
            ax.collections[idx].set_facecolor(sns.desaturate(unified_palette[dataset.capitalize()], saturation))

    _violin_stat_lines(ax, df_filtered, 'dataset', 'role', anes_keys, hue_order, lw=lw,
                        fill_lighten=fill_lighten, line_shrink=line_shrink)

    if y_lim is not None:
        modify_axis_spines(ax, which=['y'], ylim=y_lim, yticks=y_ticks)

    sig_pairs, sig_annotations = [], []
    for dataset in anes_keys:
        sig_pairs.append(((dataset, 'Wake'), (dataset, 'Anesthesia')))
        wake_samples = df_filtered.loc[(df_filtered['dataset'] == dataset) & (df_filtered['condition'] == 'wake'), 'value'].to_numpy()
        anes_samples = df_filtered.loc[(df_filtered['dataset'] == dataset) & (df_filtered['condition'] == dataset.lower()), 'value'].to_numpy()
        cohen_d = cohens_d(wake_samples, anes_samples)
        sig_annotations.append(f'd: {cohen_d:.2f}')

    annotator = Annotator(x='dataset', y='value', hue='role', order=anes_keys, hue_order=hue_order,
                           pairs=sig_pairs, data=df_filtered, ax=vio, plot='violinplot', verbose=False)
    annotator.configure(text_format='simple', loc=stat_loc, line_width=lw)
    annotator.set_custom_annotations(sig_annotations)
    annotator.annotate()

    ax.set_xticks(range(len(anes_keys)))
    ax.set_xticklabels(cap_anes_keys, rotation=0)
    ax.tick_params(axis='x', length=0)
    modify_axis_spines(ax, which=['y'], ylim=y_lim, yticks=y_ticks)

    if legend:
        legend_labels = ['Wake'] + cap_anes_keys
        handles = [mpatches.Patch(facecolor=sns.desaturate(unified_palette[lbl], saturation), edgecolor='black', linewidth=lw)
                   for lbl in legend_labels]
        ax.legend(handles, legend_labels, loc=lgd_loc, frameon=False, ncol=len(legend_labels))
    else:
        _remove_legend(vio)

    ax.set_ylabel(ylabel, labelpad=labelpad)
    ax.patch.set_alpha(0.0)
    ax.set_xlabel('')
    fig.tight_layout()


def plot_pooled_posteriors(fig, ax, param_df, anes_keys, dataset_palettes, lw=1.0, lgd_loc=(0.8, 0.75),
                            legend=False, saturation=1.0, stat_loc='outside',
                            ylabel='Pooled Posterior Value', width=0.5, labelpad=10, theta_column='parameter',
                            y_ticks=None, y_lim=None, fill_lighten=0.55, line_shrink=0.06):
    """
    Violin plot of every parameter for every anesthesia dataset at once, wake vs anesthesia,
    pooled across subjects. See the section note above.
    """
    fig.set_dpi(HOUSE_DPI)
    y_ticks = G_AXIS_YTICKS if y_ticks is None else y_ticks
    param_df = filter_positive(param_df)
    param_df['condition_cap'] = param_df['condition'].str.capitalize()
    param_df['x_plot'] = param_df['dataset'].astype(str) + ' ' + param_df[theta_column].astype(str)
    theta_labels = param_df[theta_column].unique()
    unique_conditions = param_df['condition'].unique()

    unified_palette = {}
    for ds_palette in dataset_palettes.values():
        for cond_name, color in ds_palette.items():
            unified_palette[cond_name.capitalize()] = color

    for cond in unique_conditions:
        if cond.capitalize() not in unified_palette:
            raise ValueError(f"Color for condition '{cond}' missing in the 'dataset_palettes' dictionary.")

    cap_anes_keys = [k.capitalize() for k in anes_keys]

    param_df['role'] = np.where(param_df['condition'] == 'wake', 'Wake', 'Anesthesia')
    x_order = [f'{dataset} {theta}' for dataset in anes_keys for theta in theta_labels]
    hue_order = ['Wake', 'Anesthesia']

    vio = sns.violinplot(x='x_plot', y='value', data=param_df, hue='role', order=x_order,
                          hue_order=hue_order, palette={'Wake': unified_palette['Wake'], 'Anesthesia': 'lightgray'},
                          ax=ax, linewidth=lw, dodge=True, cut=0, inner=None, saturation=saturation, width=width,
                          density_norm='width')

    n_hue = len(hue_order)
    idx_map = _collection_index_map(param_df, 'x_plot', 'role', x_order, hue_order)
    for i, x_val in enumerate(x_order):
        dataset = x_val.split(' ')[0]
        idx = idx_map.get((x_val, 'Anesthesia'))
        if idx is not None:
            ax.collections[idx].set_facecolor(sns.desaturate(unified_palette[dataset.capitalize()], saturation))

    _violin_stat_lines(ax, param_df, 'x_plot', 'role', x_order, hue_order, lw=lw,
                        fill_lighten=fill_lighten, line_shrink=line_shrink)

    if y_lim is not None:
        modify_axis_spines(ax, which=['y'], ylim=y_lim, yticks=y_ticks)

    sig_pairs, sig_annotations = [], []
    for dataset in anes_keys:
        for theta in theta_labels:
            x_val = f'{dataset} {theta}'
            sig_pairs.append(((x_val, 'Wake'), (x_val, 'Anesthesia')))
            wake_samples = param_df.loc[(param_df['dataset'] == dataset) & (param_df[theta_column] == theta) & (param_df['condition'] == 'wake'), 'value'].to_numpy()
            anes_samples = param_df.loc[(param_df['dataset'] == dataset) & (param_df[theta_column] == theta) & (param_df['condition'] == dataset.lower()), 'value'].to_numpy()
            cohen_d = cohens_d(wake_samples, anes_samples)
            sig_annotations.append(f'd: {cohen_d:.2f}')

    annotator = Annotator(x='x_plot', y='value', hue='role', order=x_order, hue_order=hue_order,
                           pairs=sig_pairs, data=param_df, ax=vio, plot='violinplot', verbose=False)
    annotator.configure(text_format='simple', loc=stat_loc, line_width=lw)
    annotator.set_custom_annotations(sig_annotations)
    annotator.annotate()

    ax.set_xticks(range(len(x_order)))
    ax.set_xticklabels(np.tile(theta_labels, len(anes_keys)), rotation=0)
    ax.tick_params(axis='x', length=0)
    modify_axis_spines(ax, which=['y'], ylim=y_lim, yticks=y_ticks)

    if legend:
        legend_labels = ['Wake'] + cap_anes_keys
        handles = [mpatches.Patch(facecolor=sns.desaturate(unified_palette[lbl], saturation), edgecolor='black', linewidth=lw)
                   for lbl in legend_labels]
        ax.legend(handles, legend_labels, title='Condition', loc=lgd_loc, frameon=False, ncol=len(legend_labels))
    else:
        _remove_legend(vio)

    ax.set_ylabel(ylabel, labelpad=labelpad)
    ax.set_xlabel('')
    ax.patch.set_alpha(0.0)
    fig.tight_layout()


def plot_weighted_vs_unweighted_comparison(fig, ax, df_weighted, df_unweighted, sel_param, anes_keys,
                                            dataset_palettes, ylabel, lw=1.2, saturation=0.9, width=0.7,
                                            stat_loc='outside', y_ticks=None, y_lim=None, suptitle=None):
    """
    Violin plot of one parameter, weighted vs unweighted gain matrix, wake vs anesthesia,
    one panel per anesthesia dataset (hatch marks the unweighted bars). See the section note
    above for the DPI/y_lim ordering.
    """
    fig.set_dpi(HOUSE_DPI)
    y_ticks = G_AXIS_YTICKS if y_ticks is None else y_ticks

    def prep(df, variant_label):
        d = df.loc[df['theta_label'] == sel_param, ['dataset', 'condition', 'value']].copy()
        d['variant'] = variant_label
        return d

    combined = pd.concat([prep(df_weighted, 'weighted'), prep(df_unweighted, 'unweighted')], ignore_index=True)
    combined = filter_positive(combined)
    combined['role'] = np.where(combined['condition'] == 'wake', 'Wake', 'Anesthesia')
    combined['group'] = combined['role'] + ' (' + combined['variant'] + ')'

    hue_order = ['Wake (weighted)', 'Wake (unweighted)', 'Anesthesia (weighted)', 'Anesthesia (unweighted)']
    unified_palette = {cond.capitalize(): color for palette in dataset_palettes.values() for cond, color in palette.items()}
    violin_palette = {'Wake (weighted)': unified_palette['Wake'], 'Wake (unweighted)': unified_palette['Wake'],
                       'Anesthesia (weighted)': 'lightgray', 'Anesthesia (unweighted)': 'lightgray'}

    vio = sns.violinplot(x='dataset', y='value', hue='group', data=combined, order=anes_keys, hue_order=hue_order,
                          palette=violin_palette, ax=ax, linewidth=lw, dodge=True, cut=0, inner=None,
                          saturation=saturation, width=width)

    n_hue = len(hue_order)
    for i, dataset in enumerate(anes_keys):
        ax.collections[i * n_hue + 2].set_facecolor(sns.desaturate(unified_palette[dataset.capitalize()], saturation))
        ax.collections[i * n_hue + 3].set_facecolor(sns.desaturate(unified_palette[dataset.capitalize()], saturation))
    for i in range(len(anes_keys)):
        ax.collections[i * n_hue + 1].set_hatch('//')
        ax.collections[i * n_hue + 3].set_hatch('//')

    _violin_stat_lines(ax, combined, 'dataset', 'group', anes_keys, hue_order, lw=lw)

    if y_lim is not None:
        modify_axis_spines(ax, which=['y'], ylim=y_lim, yticks=y_ticks)

    sig_pairs, sig_annotations = [], []
    for dataset in anes_keys:
        for variant in ['weighted', 'unweighted']:
            sig_pairs.append(((dataset, f'Wake ({variant})'), (dataset, f'Anesthesia ({variant})')))
            wake_samples = combined.loc[(combined['dataset'] == dataset) & (combined['variant'] == variant) & (combined['role'] == 'Wake'), 'value'].to_numpy()
            anes_samples = combined.loc[(combined['dataset'] == dataset) & (combined['variant'] == variant) & (combined['role'] == 'Anesthesia'), 'value'].to_numpy()
            cohen_d = cohens_d(wake_samples, anes_samples)
            sig_annotations.append(f'd: {cohen_d:.2f}')

    annotator = Annotator(x='dataset', y='value', hue='group', order=anes_keys, hue_order=hue_order,
                           pairs=sig_pairs, data=combined, ax=vio, plot='violinplot', verbose=False)
    annotator.configure(text_format='simple', loc=stat_loc, line_width=lw)
    annotator.set_custom_annotations(sig_annotations)
    annotator.annotate()

    ax.set_xticks(range(len(anes_keys)))
    ax.set_xticklabels([a.capitalize() for a in anes_keys])
    ax.tick_params(axis='x', length=0)
    ax.set_ylabel(ylabel)
    ax.set_xlabel('')
    modify_axis_spines(ax, which=['y'], ylim=y_lim, yticks=y_ticks)
    _remove_legend(vio)

    # figure level legends (figure fraction coordinates), not axes.legend(bbox_to_anchor=...): the
    # annotator above already resizes the axes to fit its brackets, so anchoring to axes-fraction
    # coordinates was fighting that resize. fig.legend also just stacks in fig.legends, no add_artist
    # trick needed for a second one, unlike ax.legend() which only keeps a single legend slot
    cap_anes_keys = [a.capitalize() for a in anes_keys]
    color_labels = ['Wake'] + cap_anes_keys
    color_handles = [mpatches.Patch(facecolor=sns.desaturate(unified_palette[lbl], saturation), edgecolor='black', linewidth=lw) for lbl in color_labels]
    variant_handles = [mpatches.Patch(facecolor='white', edgecolor='black', linewidth=lw, label='Weighted'),
                        mpatches.Patch(facecolor='white', edgecolor='black', linewidth=lw, hatch='//', label='Unweighted')]

    fig.legend(color_handles, color_labels, loc='upper center', bbox_to_anchor=(0.5, 0.97), frameon=False, ncol=len(color_labels))
    fig.legend(handles=variant_handles, loc='upper center', bbox_to_anchor=(0.5, 0.905), frameon=False, ncol=2)

    if suptitle:
        fig.suptitle(suptitle, y=0.99)
    fig.subplots_adjust(top=0.68)  # reserve room for the two stacked legends with subplots_adjust


##############################################################################
# MCMC NOTEBOOK PLOTS (02_mcmc.ipynb)
##############################################################################

def _param_grid(n_params, max_cols=4):
    """Row/col count for a per parameter panel grid, at most max_cols columns wide."""
    ncols = min(max_cols, n_params)
    nrows = int(np.ceil(n_params / ncols))
    return nrows, ncols


def _hide_unused_axes(ax, n_used, nrows, ncols):
    """Turn off any leftover empty panels in a per parameter grid."""
    for j in range(n_used, nrows * ncols):
        ax[j // ncols, j % ncols].axis('off')


def plot_trace_with_prior(az_post, prm_names, prior_specs, theta_ref=None, title=''):
    """
    Prior KDE plus one trace line per chain, one panel per parameter. See mcmc.py for
    chains_dict/prior_specs.
    """
    n = len(prm_names)
    axes = az.plot_trace(az_post, var_names=prm_names, compact=True, kind='trace',
                          backend_kwargs={'figsize': (10, 2 * n), 'layout': 'constrained'})

    for ax_row in axes:
        for ax in ax_row:
            ax.title.set_fontsize(12)

    if theta_ref is not None:
        for ax, ref_val in zip(axes[:, 0], theta_ref):
            ax.axvline(x=ref_val, color='red', linestyle='--')
        for ax, ref_val in zip(axes[:, 1], theta_ref):
            ax.axhline(y=ref_val, color='red', linestyle='--')

    for i, prm in enumerate(prm_names):
        mean, std = prior_specs[prm]
        post_samples = az_post.posterior[prm].values.flatten()
        x_lo = min(mean - 4 * std, post_samples.min())
        x_hi = max(mean + 4 * std, post_samples.max())
        x_vals = np.linspace(x_lo, x_hi, 300)
        axes[i, 0].plot(x_vals, mcmc.truncated_normal_pdf(x_vals, mean, std), color='lime')

    for ax in axes[:, 0]:
        ax.set_ylabel('Density', fontsize=11)
        ax.set_xlabel('Parameter value', fontsize=11)
    for ax in axes[:, 1]:
        ax.set_ylabel('Parameter value', fontsize=11)
        ax.set_xlabel('Draw', fontsize=11)

    handles = [mlines.Line2D([], [], color='lime', label='Prior'),
               mlines.Line2D([], [], color='steelblue', label='Posterior (per chain)')]
    if theta_ref is not None:
        handles.append(mlines.Line2D([], [], color='red', linestyle='--', label='Reference value'))

    fig = axes[0, 0].figure
    fig.suptitle(title, fontsize=14, y=1.03)
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 1.07), ncol=len(handles), frameon=False)
    return fig, axes


def plot_pooled_posterior(prm_names, prior_predictions, chains_pooled, params_map, theta_ref=None, title=''):
    """
    Prior vs pooled posterior KDE per parameter, with the MAP and an optional reference
    value marked.
    """
    nrows, ncols = _param_grid(len(prm_names))
    fig, ax = plt.subplots(ncols=ncols, nrows=nrows, figsize=(3.3 * ncols, 2.7 * nrows), squeeze=False)

    for iprm, prm in enumerate(prm_names):
        a = ax[iprm // ncols, iprm % ncols]
        a.set_xlabel(prm)
        if theta_ref is not None:
            a.axvline(theta_ref[iprm], color='r', label='reference', linestyle='--')
        a.axvline(params_map[iprm], color='deepskyblue', label='MAP', linestyle='--', lw=2.)
        sns.kdeplot(prior_predictions[prm], ax=a, color='lime', alpha=0.5, lw=2, label='prior', fill=True)
        sns.kdeplot(chains_pooled[iprm, :], ax=a, color='darkblue', alpha=0.2, label='posterior', lw=2., fill=True)

    _hide_unused_axes(ax, len(prm_names), nrows, ncols)
    handles, labels = ax[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, bbox_to_anchor=(1.02, 1), loc='upper left', frameon=False)
    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 1])
    return fig, ax


def plot_fitted_per_chain(data, az_posterior, title_prefix='', max_cols=4, xlim=None, ylim=None, xticks=(), yticks=()):
    """Observed data vs that chain's mean fitted trace, one panel per chain."""
    ts_obs = data['ts_obs']
    xpy_obs = data['xpy_obs']
    n_chains = az_posterior.sizes['chain']
    nrows, ncols = _param_grid(n_chains, max_cols=max_cols)

    fig, ax = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 4.5 * nrows), squeeze=False)
    plot_xlim = xlim if xlim is not None else (float(np.min(ts_obs)), float(np.max(ts_obs)))
    for ich in range(n_chains):
        a = ax[ich // ncols, ich % ncols]
        a.plot(ts_obs, xpy_obs, 'o', color='crimson', markersize=3, label='Observed')
        a.plot(ts_obs, az_posterior['xpy_model'][ich, :, :].mean(axis=0), lw=2.2, color='steelblue', label='Fit (chain mean)')
        a.set_title(f'{title_prefix}chain {ich + 1}', fontsize=12)
        a.set_xlabel('Time (ms)')
        if ich % ncols == 0:
            a.set_ylabel('Voltage (normalized)')
        a.legend(fontsize=10, frameon=False, loc='upper right')
        modify_axis_spines(a, which=['x', 'y'], xlim=plot_xlim, ylim=ylim, xticks=xticks, yticks=yticks)
    _hide_unused_axes(ax, n_chains, nrows, ncols)
    fig.tight_layout()
    return fig, ax


def plot_predictive_overlay(ax, time_points, xpy_obs, pred_traces, obs_label='Observed data',
                             pred_label='Posterior predictive', title='', obs_color='crimson',
                             pred_color='steelblue', lw=0.3, xlim=None, ylim=None, xticks=(), yticks=()):
    """
    Observed data as dots plus every row of pred_traces as a thin line. Shared by the
    prior/posterior predictive checks and the channel/window/filter sweeps in 02_mcmc.ipynb.
    """
    ax.plot(time_points, xpy_obs, 'o', color=obs_color, markersize=3, label=obs_label)
    for i, trace in enumerate(np.asarray(pred_traces)):
        ax.plot(time_points, trace, color=pred_color, lw=lw, label=pred_label if i == 0 else None)
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Voltage (normalized)')
    if title:
        ax.set_title(title)
    ax.legend(loc='best', frameon=False)
    xlim = xlim if xlim is not None else (float(np.min(time_points)), float(np.max(time_points)))
    modify_axis_spines(ax, which=['x', 'y'], xlim=xlim, ylim=ylim, xticks=xticks, yticks=yticks)
    return ax


def plot_prior_tail(prm_names, theta_ref, prior_q, tail_sample_vals, quantile_bounds,
                     title='Initial values from the tail of the prior'):
    """
    Prior density per parameter with the outer 2.5% tails shaded and the sampled tail value
    marked. See mcmc.low_prob_sample.
    """
    q_low_lim, q_low, q_high, q_high_lim = quantile_bounds
    nrows, ncols = _param_grid(len(prm_names))
    fig, ax = plt.subplots(ncols=ncols, nrows=nrows, figsize=(3.3 * ncols, 2.7 * nrows), squeeze=False)

    for iprm, prm in enumerate(prm_names):
        a = ax[iprm // ncols, iprm % ncols]
        sns.kdeplot(prior_q[:, iprm], color='lime', ax=a, cut=0, label='prior')
        kde_x, kde_y = a.lines[0].get_data()
        a.fill_between(kde_x, kde_y,
                        where=((kde_x <= q_low[iprm]) & (kde_x >= q_low_lim[iprm])) |
                              ((kde_x <= q_high_lim[iprm]) & (kde_x >= q_high[iprm])),
                        color='cyan', alpha=0.7)
        ylim = a.get_ylim()[1]
        a.annotate('', xy=(tail_sample_vals[iprm], 0), xytext=(tail_sample_vals[iprm], 0.5 * ylim),
                    horizontalalignment='center', arrowprops=dict(arrowstyle='->', lw=2, color='orange'))
        a.scatter(tail_sample_vals[iprm], 0.52 * ylim, color='orange', s=50, label='init sample', zorder=5)
        if theta_ref is not None:
            a.axvline(theta_ref[iprm], color='r', label='reference', linestyle='--')
        a.set_xlabel(prm)
        a.set_ylabel('Density' if iprm % ncols == 0 else '')

    _hide_unused_axes(ax, len(prm_names), nrows, ncols)
    handles, labels = ax[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, bbox_to_anchor=(1.02, 1), loc='upper left', frameon=False)
    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 1])
    return fig, ax


def plot_convergence(lp, n_samples, n_chains, title='Convergence of the chains', figsize=(10, 5)):
    """Log joint density per sample, one line per chain, with each chain's mean marked."""
    fig, ax = plt.subplots(figsize=figsize)
    for i in range(0, n_samples * n_chains, n_samples):
        chain_lp = lp[i:i + n_samples]
        ax.plot(chain_lp)
        ax.axhline(np.mean(chain_lp), color='cyan', linestyle='--', lw=1)
    ax.set_ylabel('Log joint density')
    ax.set_xlabel('Sample')
    ax.set_title(title)
    modify_axis_spines(ax, which=['x', 'y'], xlim=(0, n_samples - 1))
    return fig, ax


##############################################################################
# SBI / PCIST DIAGNOSTIC PLOTS (03_sbi.ipynb)
##############################################################################

def plot_cumulative_explained_variance(fig, ax, erp_data, p_color, p_label, n_comp_highlight=4, calpha=0.7, lw=1.5):
    """
    Cumulative explained variance vs number of SVD components, with a marker at
    n_comp_highlight.
    """
    _, s, _ = jnp.linalg.svd(erp_data, full_matrices=False)

    var_ratio = s ** 2 / jnp.sum(s ** 2)
    cumulative = np.array(jnp.cumsum(var_ratio))
    n_components = np.arange(1, len(cumulative) + 1)

    line = ax.plot(n_components, cumulative, c=p_color, alpha=calpha, lw=lw)
    line[0].set_label(p_label)

    if 1 <= n_comp_highlight <= len(cumulative):
        y_val = cumulative[n_comp_highlight - 1]
        ax.vlines(n_comp_highlight, ymin=0, ymax=y_val, color='crimson', linestyle='--', alpha=0.7)
        ax.hlines(y_val, xmin=1, xmax=n_comp_highlight, color='crimson', linestyle='--', alpha=0.7)
        ax.text(n_comp_highlight - 1.5, y_val, f"{y_val:.2f}", color='crimson', fontsize=10, va='bottom', ha='right')

    ax.set_xlim(0.5, len(cumulative) + 0.5)
    ax.set_xlabel('Number of components')
    ax.set_ylabel('Cumulative explained variance')
    ax.grid(False)
    fig.legend(frameon=False, bbox_to_anchor=(1, 0.85))

    return fig, ax


def plot_similarity_matrix(fig, ax, sim_mat, labels, title="Similarity matrix"):
    """
    Heatmap of a subspace similarity matrix, used to compare gain matrices across SBI
    pipelines.
    """
    im = ax.imshow(sim_mat, vmin=0, vmax=1)
    ax.set_title(title)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90)
    ax.set_yticklabels(labels)
    fig.colorbar(im)
    plt.tight_layout()
    return fig, ax


def _is_pair(x):
    """True if x is a single (lo, hi) pair rather than a list of pairs."""
    return len(x) == 2 and all(isinstance(v, (int, float, np.integer, np.floating)) for v in x)


def _broadcast_lims(value, n_panels, name):
    """
    Turn a single (lo, hi) pair or a list of them into one per panel entry, for
    plot_pci_vs_params.
    """
    if value is None:
        return [None] * n_panels
    if _is_pair(value):
        return [value] * n_panels
    value = list(value)
    if len(value) != n_panels:
        raise ValueError(f'{name} list has {len(value)} entries, expected {n_panels} (one per parameter)')
    return value


def _broadcast_ticks(value, n_panels, name):
    """
    Turn a single tick list or a list of them into one per panel entry, for
    plot_pci_vs_params.
    """
    if value is None:
        return [None] * n_panels
    value = list(value)
    if len(value) > 0 and isinstance(value[0], (list, tuple, np.ndarray)):
        if len(value) != n_panels:
            raise ValueError(f'{name} list has {len(value)} entries, expected {n_panels} (one per parameter)')
        return value
    return [value] * n_panels


def _broadcast_scalar(value, n_panels):
    """Turn a single int or a list of ints into one per panel entry, for plot_pci_vs_params."""
    if isinstance(value, (list, tuple, np.ndarray)):
        value = list(value)
        if len(value) != n_panels:
            raise ValueError(f'tick count list has {len(value)} entries, expected {n_panels} (one per parameter)')
        return value
    return [value] * n_panels


def plot_pci_vs_params(theta_arr, pci_values, prm_names, param_labels, title,
                        pipeline=0, encoding=None, mode='scatter', max_density_points=3000,
                        xlim=(0, 1.4), ylim=(0, 200), xticks=None, yticks=None,
                        n_xticks=5, n_yticks=5):
    """
    Each free parameter against its simulated PCIst value, one panel per parameter. pipeline
    controls the color grouping (see encode_condition in inference_utils.py): 0 is one
    group, 1 is wake vs anesthesia, 2 is one color per condition. mode='density' draws a KDE
    instead of a scatter, subsampled to max_density_points per group for speed.
    xlim/ylim/xticks/yticks accept one value for every panel or a list with one entry per
    parameter.
    """
    n_panels = len(prm_names)
    xlims = _broadcast_lims(xlim, n_panels, 'xlim')
    ylims = _broadcast_lims(ylim, n_panels, 'ylim')
    xticks_list = _broadcast_ticks(xticks, n_panels, 'xticks')
    yticks_list = _broadcast_ticks(yticks, n_panels, 'yticks')
    n_xticks_list = _broadcast_scalar(n_xticks, n_panels)
    n_yticks_list = _broadcast_scalar(n_yticks, n_panels)

    dot_size = 1.0
    dot_alpha = 0.6 if pipeline == 0 else 0.35

    # groups: list of (mask, color, legend_label or None)
    if pipeline == 0:
        groups = [(np.ones(len(pci_values), dtype=bool), 'steelblue', None)]
        legend_handles = None
    elif pipeline == 1:
        if encoding is None:
            raise ValueError('pipeline=1 needs the wake/anesthesia encoding column, pass it as `encoding`')
        state = np.asarray(encoding).reshape(-1)
        cond_colors = {'wake': ANES_COLORS['wake'], 'anesthesia': 'cornflowerblue'}
        groups = [(state == 0, cond_colors['wake'], 'Wake'),
                  (state == 1, cond_colors['anesthesia'], 'Anesthesia')]
        legend_handles = [mlines.Line2D([], [], marker='o', linestyle='', color=c, label=lbl)
                           for _, c, lbl in groups]
    elif pipeline == 2:
        if encoding is None:
            raise ValueError('pipeline=2 needs the 4-way condition one hot columns, pass it as `encoding`')
        one_hot = np.asarray(encoding)
        cond_labels = ['wake', 'keta', 'prop', 'sevo']
        cond_index = one_hot.argmax(axis=1)
        groups = [(cond_index == idx, ANES_COLORS[c], c.capitalize())
                  for idx, c in enumerate(cond_labels)]
        legend_handles = [mlines.Line2D([], [], marker='o', linestyle='', color=c, label=lbl)
                           for _, c, lbl in groups]
    else:
        raise ValueError(f'Invalid pipeline {pipeline}, expected 0, 1, or 2')

    fig, axes = plt.subplots(1, n_panels, figsize=(10, 3))
    rng = np.random.default_rng(0)
    single_group = len(groups) == 1
    for i, prm in enumerate(prm_names):
        x_vals = theta_arr[:, i]
        if mode == 'scatter':
            for mask, color, _ in groups:
                axes[i].scatter(x_vals[mask], pci_values[mask], alpha=dot_alpha, s=dot_size, c=color)
        elif mode == 'density':
            for mask, color, _ in groups:
                idx = np.flatnonzero(mask)
                if len(idx) > max_density_points:
                    idx = rng.choice(idx, size=max_density_points, replace=False)
                x_sub, y_sub = x_vals[idx], pci_values[idx]
                if single_group:
                    sns.kdeplot(x=x_sub, y=y_sub, ax=axes[i], fill=True, color=color,
                                alpha=0.55, levels=8, thresh=0.05, bw_adjust=1.2)
                else:
                    sns.kdeplot(x=x_sub, y=y_sub, ax=axes[i], fill=False, color=color,
                                linewidths=1.2, levels=4, thresh=0.15, bw_adjust=1.3)
        else:
            raise ValueError(f"mode must be 'scatter' or 'density', got {mode!r}")
        axes[i].set_xlabel(param_labels[prm])
        axes[i].set_ylabel('PCIst')
        panel_xlim = xlims[i]
        panel_ylim = ylims[i]
        panel_xticks = xticks_list[i] if xticks_list[i] is not None else np.linspace(panel_xlim[0], panel_xlim[1], n_xticks_list[i])
        panel_yticks = yticks_list[i] if yticks_list[i] is not None else np.linspace(panel_ylim[0], panel_ylim[1], n_yticks_list[i])
        modify_axis_spines(axes[i], which=['x', 'y'], xlim=panel_xlim, xticks=panel_xticks,
                            ylim=panel_ylim, yticks=panel_yticks)

    if legend_handles is not None:
        axes[-1].legend(handles=legend_handles, loc='upper right', frameon=False, fontsize=8)

    fig.suptitle(title)
    fig.tight_layout()
    return fig, axes


def plot_pci_kde_comparison(ax, series_dict, colors=None, xlim=(0, 200), bw_adjust=0.25,
                             xlabel='PCIst', ylabel='Density', title=''):
    """
    Filled KDE overlay of several PCIst distributions on one axis (simulated vs empirical,
    one pipeline vs another, etc). series_dict maps a label to its values.
    """
    default_colors = ['slategray', 'steelblue', 'mediumseagreen', 'coral', 'crimson']
    colors = colors or {}
    for i, (label, values) in enumerate(series_dict.items()):
        c = colors.get(label, default_colors[i % len(default_colors)])
        alpha = 0.5 if label.lower() == 'empirical' else 0.4
        sns.kdeplot(np.asarray(values), ax=ax, fill=True, color=c, alpha=alpha,
                    label=label, cut=0, bw_adjust=bw_adjust)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    modify_axis_spines(ax, which=['x', 'y'], xlim=xlim)
    dedupe_legend(ax, frameon=False)
    return ax
