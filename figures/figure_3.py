from matplotlib import mlab, ticker
from mpl_toolkits.axes_grid1 import make_axes_locatable
import mdt.component_templates.base
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import seaborn

"""
Creates all elements of figure 3 of the paper.
"""

pjoin = mdt.make_path_joiner('/home/robbert/phd-data/papers/sampling_paper/single_slice/')


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

nmr_samples = 1000
burnin = 1000
model_name = 'BallStick_r1'
ap_methods = ['AMWG', 'MWG', 'FSL', 'SCAM']

ap_method_names = {
    'MWG': 'None',
    'FSL': 'FSL',
    'SCAM': 'SCAM',
    'AMWG': 'AMWG'
}

chains = {}
for method_name in ap_methods:
    samples = mdt.load_samples(pjoin('figure_3', 'sampling_methods', method_name, model_name, 'samples'))
    chains[ap_method_names[method_name]] = samples['w_stick0.w'][0, burnin:burnin + nmr_samples]


def plot_chain(ax, samples):
    ax.plot(range(burnin, burnin+nmr_samples), samples)
    ax.set_xlabel('Sample index')
    ax.set_ylabel('Volume Fraction \n(a.u.)')
    ax.yaxis.set_major_locator(ticker.MaxNLocator(5))
    ax.xaxis.set_major_locator(ticker.LinearLocator(3))


def plot_histogram(ax, samples, sample_mean, sample_std, orientation='vertical'):
    ax.hist(samples, 25, normed=True, orientation=orientation, color='lightgray')
    [label.set_visible(False) for label in ax.get_xticklabels() + ax.get_yticklabels()]

    fit_x_coords = np.linspace(np.min(samples), np.max(samples) * 1.01, 100)
    if np.max(samples) < 0:
        fit_x_coords = np.linspace(np.max(samples), np.min(samples), 100)

    sample_predicted_dist = (fit_x_coords, norm.pdf(fit_x_coords, loc=sample_mean, scale=sample_std))
    sample_mean_point = (sample_mean, float(mlab.normpdf(sample_mean, sample_mean, sample_std)))

    if orientation == 'horizontal':
        sample_predicted_dist = sample_predicted_dist[::-1]
        sample_mean_point = sample_mean_point[::-1]

    ax.plot(*sample_predicted_dist, color='blue', linewidth=1)
    ax.plot(*sample_mean_point, color='blue', marker='o', label='Mean', markersize=12)

    if orientation == 'horizontal':
        ax.yaxis.offsetText.set_visible(False)
    else:
        ax.xaxis.offsetText.set_visible(False)


def create_chain_hist_fig():
    fig, ax_scatter = plt.subplots(figsize=(7, 3))

    divider = make_axes_locatable(ax_scatter)
    ax_histy = divider.append_axes("right", 1, pad=0.15, sharey=ax_scatter)

    return fig, [ax_scatter, ax_histy]


for method_name, samples in chains.items():
    fig, ax = create_chain_hist_fig()
    fig.suptitle('{}'.format(method_name), y=1)
    fig.subplots_adjust(top=0.9)

    sample_mean = np.mean(samples)
    sample_std = np.std(samples)

    plot_chain(ax[0], samples)

    plot_histogram(ax[1], samples,
                   sample_mean, sample_std,
                   orientation='horizontal')
    ax[1].set_ylim(0.58, 0.74)
    fig.tight_layout()
plt.show()