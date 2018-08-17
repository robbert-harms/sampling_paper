import subprocess

from matplotlib import mlab, ticker
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import mdt
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from pandas import Series
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
    ax.axhline(y=z99 / np.sqrt(n), linestyle='--', color='grey', linewidth=2, zorder=0)
    # ax.axhline(y=z95 / np.sqrt(n), color='grey')
    ax.axhline(y=0.0, color='black', linewidth=1, zorder=0)
    # ax.axhline(y=-z95 / np.sqrt(n), color='grey')
    ax.axhline(y=-z99 / np.sqrt(n), linestyle='--', color='grey', linewidth=2, zorder=0)
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

    n, _, _ = ax.hist(samples, 25, normed=True, orientation=orientation, color=color)
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


def create_plots(model_name, method_samples, method_means, method_stds, display_legend=False, plot_mean_std=True):
    fig = plt.figure(figsize=(5, 8))
    fig.suptitle(model_titles[model_name], y=1)

    gs = GridSpec(4, 1, height_ratios=[1, 0.5, 0.5, 1])

    ax_chain = plt.subplot(gs[0, 0])

    for ind, method in enumerate(methods):
        plot_chains(ax_chain, method_samples[method], colors[ind], method_labels[method])

    ax_mean = plt.subplot(gs[1, 0])
    ax_std = plt.subplot(gs[2, 0])

    ax_std.set_xlabel('# Burn-in')
    items = [(ax_mean, method_means, 'Mean'),
             (ax_std, method_stds, 'Std.')]

    for ax, data, label in items:
        plot_burnin(ax, data, label)
    ax_std.set_ylim((0, 0.3))

    if display_legend:
        ax_chain.legend(loc="lower right", bbox_to_anchor=(1.05, 1.02), borderaxespad=0., ncol=3, handletextpad=0.5)

    ax_autocorrelation = plt.subplot(gs[3, 0])
    autocorrelation_plot(Series(method_samples['powell']), ax=ax_autocorrelation)
    autocorrelation_plot(Series(method_samples['0.4']), ax=ax_autocorrelation)
    autocorrelation_plot(Series(method_samples['0.8']), ax=ax_autocorrelation)
    ax_autocorrelation.grid(True)
    ax_autocorrelation.xaxis.set_major_locator(ticker.LinearLocator(3))
    ax_autocorrelation.yaxis.set_major_locator(ticker.LinearLocator(5))
    ax_autocorrelation.get_xticklabels()[0].set_horizontalalignment('left')
    ax_autocorrelation.get_xticklabels()[-1].set_horizontalalignment('right')
    ax_autocorrelation.set_ylim((-0.6, 1))

    fig.subplots_adjust(hspace=0.5, top=0.85, bottom=0.15, left=0.25, right=0.95)

    ax_chain_pos = ax_chain.get_position()
    pos2 = [ax_chain_pos.x0, ax_chain_pos.y0 + 0.05, ax_chain_pos.width, ax_chain_pos.height]
    ax_chain.set_position(pos2)

    ax_ar_pos = ax_autocorrelation.get_position()
    pos2 = [ax_ar_pos.x0, ax_ar_pos.y0 - 0.05, ax_ar_pos.width, ax_ar_pos.height]
    ax_autocorrelation.set_position(pos2)

    return fig


def plot_burnin(axes, method_values, label):
    for ind, key in enumerate(methods):
        axes.plot(range(0, 1100, 100), method_values[key], c=colors[ind], label=method_labels[key])
    axes.set_ylabel(label)
    axes.xaxis.set_major_locator(ticker.LinearLocator(3))
    axes.yaxis.set_major_locator(ticker.LinearLocator(3))
    axes.get_xticklabels()[0].set_horizontalalignment('left')
    axes.get_xticklabels()[-1].set_horizontalalignment('right')


set_matplotlib_font_size(18)

pjoin = mdt.make_path_joiner(r'/home/robbert/phd-data/papers/sampling_paper/single_slice/')

nmr_samples = 1000
colors = ['#6e8cbe', '#c45054', '#55a868']

param_names = {'NODDI': 'w_ic.w',
               'BallStick_r1': 'w_stick0.w'}
methods = ['powell', '0.4', '0.8']
model_titles = {
    'BallStick_r1': 'BallStick_in1 (FS)',
    'NODDI': 'NODDI (FR)',
}

method_labels = {'powell': 'MLE',
                 '0.4': '0.4',
                 '0.8': '0.8'}

for model_name in ['BallStick_r1', 'NODDI']:
    method_samples = {}
    method_means = {}
    method_stds = {}

    for method in methods:
        samples = mdt.load_sample(pjoin('figure_7', 'init_points', method, model_name, 'samples', param_names[model_name]))
        method_samples[method] = samples[0, :nmr_samples]

        method_means[method] = [np.mean(samples[0, burnin:(burnin + nmr_samples)]) for burnin in range(0, 1100, 100)]
        method_stds[method] = [np.std(samples[0, burnin:(burnin + nmr_samples)]) for burnin in range(0, 1100, 100)]

    fig = create_plots(model_name, method_samples, method_means, method_stds,
                       display_legend=model_name == 'NODDI')

    # plt.show()

    mdt.make_path_joiner('/tmp/sampling_paper/burnin/chain_demo/', make_dirs=True)
    fig.savefig('/tmp/sampling_paper/burnin/chain_demo/{}.png'.format(model_name))


subprocess.Popen('''
    convert BallStick_r1.png NODDI.png +append burnin_chain_demo.png
''', shell=True, cwd='/tmp/sampling_paper/burnin/chain_demo/').wait()
