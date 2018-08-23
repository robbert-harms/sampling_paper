import subprocess
from matplotlib import mlab, ticker
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pandas import Series

import mdt.component_templates.base
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import seaborn


def set_matplotlib_font_size(font_size):
    import matplotlib.pyplot as plt
    plt.rc('font', size=font_size)  # controls default text sizes
    plt.rc('axes', titlesize=font_size)  # fontsize of the axes title
    plt.rc('axes', labelsize=font_size)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=font_size)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=font_size)  # fontsize of the tick labels
    plt.rc('legend', fontsize=font_size)  # legend fontsize
    plt.rc('figure', titlesize=font_size)

set_matplotlib_font_size(18)


pjoin = mdt.make_path_joiner(r'/home/robbert/phd-data/papers/sampling_paper/single_slice/')

nmr_samples = 200
colors = ['#6e8cbe', '#c45054', '#55a868']
thinning_method_colors = {'Thinning': '#e66ce6', r'More samples ($\times 10^3$)': 'black'}


def autocorrelation_plot(series, ax=None, **kwds):
    """Autocorrelation plot for time series.

    Parameters:
    -----------
    series: Time series
    ax: Matplotlib axis object, optional
    kwds : keywords
        Options to pass to matplotlib plotting method

    Returns:
    -----------
    ax: Matplotlib axis object
    """
    from pandas.compat import lmap
    import matplotlib.pyplot as plt
    n = len(series)
    data = np.asarray(series)
    if ax is None:
        ax = plt.gca(xlim=(1, n), ylim=(-1.0, 1.0))
    mean = np.mean(data)
    c0 = np.sum((data - mean) ** 2) / float(n)

    def r(h):
        return ((data[:n - h] - mean) *
                (data[h:] - mean)).sum() / float(n) / c0
    x = np.arange(n) + 1
    y = lmap(r, x)
    z95 = 1.959963984540054
    z99 = 2.5758293035489004
    ax.axhline(y=z99 / np.sqrt(n), linestyle='--', color='grey', linewidth=2)
    # ax.axhline(y=z95 / np.sqrt(n), color='grey')
    ax.axhline(y=0.0, color='black', linewidth=2)
    # ax.axhline(y=-z95 / np.sqrt(n), color='grey')
    ax.axhline(y=-z99 / np.sqrt(n), linestyle='--', color='grey', linewidth=2)
    ax.set_xlabel("Lag")
    ax.set_ylabel("Autocorrelation")
    ax.plot(x, y, **kwds)
    if 'label' in kwds:
        ax.legend()
    ax.grid()
    return ax



def plot_chains(ax, samples, color, label):
    ax.plot(samples, c=color, label=label)
    ax.set_xlim(0, len(samples))

    ax.set_xlabel('Sample index')
    ax.set_ylabel('Volume Fraction \n(a.u.)')

    ax.xaxis.set_major_locator(ticker.LinearLocator(3))
    ax.yaxis.set_major_locator(ticker.LinearLocator(4))
    ax.get_xticklabels()[0].set_horizontalalignment('left')
    ax.get_xticklabels()[-1].set_horizontalalignment('right')


def plot_histogram(ax, samples, color, orientation='vertical'):
    sample_mean = np.mean(samples)
    sample_std = np.std(samples)

    n, _, _ = ax.hist(samples, 10, normed=True, orientation=orientation, color=color)
    [label.set_visible(False) for label in ax.get_xticklabels() + ax.get_yticklabels()]

    fit_x_coords = np.linspace(np.min(samples), np.max(samples) * 1.01, 100)
    if np.max(samples) < 0:
        fit_x_coords = np.linspace(np.max(samples), np.min(samples), 100)

    sample_predicted_dist = (fit_x_coords, norm.pdf(fit_x_coords, loc=sample_mean, scale=sample_std))
    sample_mean_point = (sample_mean, float(mlab.normpdf(sample_mean, sample_mean, sample_std)))

    if orientation == 'horizontal':
        sample_predicted_dist = sample_predicted_dist[::-1]
        sample_mean_point = sample_mean_point[::-1]

    ax.plot(*sample_predicted_dist, color=color, linewidth=1)
    ax.plot(*sample_mean_point, color=color, marker='o', label='Mean', markersize=12)

    if orientation == 'horizontal':
        ax.yaxis.offsetText.set_visible(False)
    else:
        ax.xaxis.offsetText.set_visible(False)


def create_plots(model_name, thinning_samples, thinning_methods_mean, thinning_methods_std, display_legend=True):
    fig = plt.figure(figsize=(5.5, 8))
    fig.suptitle(model_titles[model_name], y=1)

    gs = GridSpec(4, 1, height_ratios=[1, 1, 0.5, 0.5])

    ax_chain = plt.subplot(gs[0, 0])
    divider = make_axes_locatable(ax_chain)
    ax_histy = divider.append_axes("right", 0.6, pad=0.15, sharey=ax_chain)

    for ind, th in enumerate(thinning):
        plot_chains(ax_chain, thinning_samples[th], colors[ind], th)
        plot_histogram(ax_histy, thinning_samples[th], colors[ind], orientation='horizontal')


    ax_autocorrelation = plt.subplot(gs[1, 0])
    autocorrelation_plot(Series(thinning_samples[thinning[0]]), ax=ax_autocorrelation)
    autocorrelation_plot(Series(thinning_samples[thinning[1]]), ax=ax_autocorrelation)
    autocorrelation_plot(Series(thinning_samples[thinning[2]]), ax=ax_autocorrelation)
    ax_autocorrelation.grid(True)
    ax_autocorrelation.xaxis.set_major_locator(ticker.LinearLocator(3))
    ax_autocorrelation.yaxis.set_major_locator(ticker.LinearLocator(5))
    ax_autocorrelation.get_xticklabels()[0].set_horizontalalignment('left')
    ax_autocorrelation.get_xticklabels()[-1].set_horizontalalignment('right')
    ax_autocorrelation.set_ylim((-0.3, 0.9))

    ax_mean = plt.subplot(gs[2, 0])
    ax_std = plt.subplot(gs[3, 0], sharex=ax_mean)

    ax_std.set_xlabel('Thinning / More samples')
    items = [(ax_mean, thinning_methods_mean, 'Mean'),
             (ax_std, thinning_methods_std, 'Std.')]

    for ax, data, label in items:
        plot_thinning(ax, data, label)

    if display_legend:
        ax_chain.legend(loc="lower right", bbox_to_anchor=(1.02, 1.02), borderaxespad=0., ncol=3, handletextpad=0.6)
        ax_mean.legend(loc="lower right", bbox_to_anchor=(1.02, 1.02), borderaxespad=0., ncol=3, handletextpad=0.6)


    fig.subplots_adjust(hspace=0.5, top=0.85, bottom=0.18, left=0.3, right=0.98)

    ax_chain_pos = ax_chain.get_position()
    pos2 = [ax_chain_pos.x0, ax_chain_pos.y0 + 0.05, ax_chain_pos.width, ax_chain_pos.height]
    ax_chain.set_position(pos2)

    ax_mean_pos = ax_mean.get_position()
    pos2 = [ax_mean_pos.x0, ax_mean_pos.y0 - 0.08, ax_mean_pos.width, ax_mean_pos.height]
    ax_mean.set_position(pos2)

    ax_std_pos = ax_std.get_position()
    pos2 = [ax_std_pos.x0, ax_std_pos.y0 - 0.08, ax_std_pos.width, ax_std_pos.height]
    ax_std.set_position(pos2)


    return fig


def plot_thinning(axes, thinnings, label):
    for thinning_method, values in thinnings.items():
        axes.plot(range(1, 21), values, c=thinning_method_colors[thinning_method], label=thinning_method)
    axes.set_ylabel(label)
    # axes.xaxis.set_major_locator(ticker.LinearLocator(3))
    axes.yaxis.set_major_locator(ticker.LinearLocator(3))
    axes.set_xlim(0, 21)
    axes.get_xticklabels()[0].set_horizontalalignment('left')
    axes.get_xticklabels()[-1].set_horizontalalignment('right')


img_pjoin = mdt.make_path_joiner('/tmp/sampling_paper/thinning/voxel_demo/', make_dirs=True)

param_names = {'NODDI': 'w_ic.w', 'BallStick_r1': 'w_stick0.w'}
thinning = [1, 10, 20]
model_titles = {
    'BallStick_r1': 'BallStick_in1 (FS)',
    'NODDI': 'NODDI (FR)',
}

for model_name in ['BallStick_r1', 'NODDI']:
    thinning_samples = {}
    thinning_method_means = {}
    thinning_method_stds = {}

    samples = mdt.load_sample(pjoin('figure_8', model_name, 'samples', param_names[model_name]))

    for th in thinning:
        thinning_samples[th] = samples[0, (nmr_samples * th):th]

    thinning_method_means['Thinning'] = [np.mean(samples[0, :(nmr_samples * th):th]) for th in range(1, 21)]
    thinning_method_stds['Thinning'] = [np.std(samples[0, :(nmr_samples * th):th]) for th in range(1, 21)]

    thinning_method_means[r'More samples ($\times 10^3$)'] = [np.mean(samples[0, :(nmr_samples * th)]) for th in range(1, 21)]
    thinning_method_stds[r'More samples ($\times 10^3$)'] = [np.std(samples[0, :(nmr_samples * th)]) for th in range(1, 21)]

    fig = create_plots(model_name, thinning_samples, thinning_method_means, thinning_method_stds,
                       display_legend=model_name=='NODDI')

    fig.savefig(img_pjoin('{}.png'.format(model_name)))

# plt.show()

subprocess.Popen('''
    convert BallStick_r1.png NODDI.png +append thinning_chain_demo.png
''', shell=True, cwd=img_pjoin()).wait()