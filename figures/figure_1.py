import mdt
import numpy as np
from matplotlib.ticker import MaxNLocator
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib import ticker
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.stats import norm
import seaborn

__author__ = 'Robbert Harms'
__date__ = '2018-08-15'
__maintainer__ = 'Robbert Harms'
__email__ = 'robbert.harms@maastrichtuniversity.nl'
__licence__ = 'LGPL v3'


"""
Creates all elements of figure 1 of the paper.
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


def create_chain_histogram():
    set_matplotlib_font_size(22)
    y_limits = (0.56, 0.76)
    model_title = 'FS'

    mask = mdt.load_brain_mask(pjoin('mgh_1003_slice_44_mask'))

    # selecting the voxel by volume index
    volume_ind = (67, 44, 0)
    roi_ind = mdt.volume_index_to_roi_index(volume_ind, mask)

    # the samples of the give voxel
    samples = mdt.load_sample(pjoin('figure_1', 'BallStick_r1', 'samples', 'w_stick0.w'))[roi_ind]

    def plot_chain(ax, samples):
        ax.plot(samples)
        ax.set_xlim(0, len(samples))

        ax.xaxis.set_major_locator(ticker.LinearLocator(3))
        ax.yaxis.set_major_locator(ticker.LinearLocator(3))

        ax.yaxis.get_major_ticks()[0].label1.set_verticalalignment('bottom')
        ax.yaxis.get_major_ticks()[-1].label1.set_verticalalignment('top')

        ax.xaxis.set_visible(False)
        ax.set_ylim(y_limits)

        ax.set_ylabel(model_title, x=-0.5)

    def plot_histogram(ax, samples, orientation='vertical'):
        sample_mean = np.mean(samples)
        sample_std = np.std(samples)

        ax.hist(samples, 200, normed=True, orientation=orientation, color='lightgray')
        [label.set_visible(False) for label in ax.get_xticklabels() + ax.get_yticklabels()]

        fit_x_coords = np.linspace(np.min(samples), np.max(samples) * 1.01, 100)
        if np.max(samples) < 0:
            fit_x_coords = np.linspace(np.max(samples), np.min(samples), 100)

        sample_predicted_dist = (fit_x_coords, norm.pdf(fit_x_coords, loc=sample_mean, scale=sample_std))
        sample_mean_point = (sample_mean, float(mlab.normpdf(sample_mean, sample_mean, sample_std)))

        if orientation == 'horizontal':
            sample_predicted_dist = sample_predicted_dist[::-1]
            sample_mean_point = sample_mean_point[::-1]

        ax.plot(*sample_predicted_dist, color='black', linewidth=1)
        ax.plot(*sample_mean_point, color='black', marker='o', label='Mean', markersize=12)

        if orientation == 'horizontal':
            ax.yaxis.offsetText.set_visible(False)
        else:
            ax.xaxis.offsetText.set_visible(False)

    def create_chain_hist_fig(samples):
        fig, ax_chain = plt.subplots(figsize=(7, 3.235))

        divider = make_axes_locatable(ax_chain)
        ax_histy = divider.append_axes("right", 1, pad=0.15, sharey=ax_chain)

        plot_chain(ax_chain, samples)
        plot_histogram(ax_histy, samples, orientation='horizontal')

        plt.tight_layout()

        return fig, [ax_chain, ax_histy]

    fig, axes = create_chain_hist_fig(samples)
    plt.show()


def create_mean_std_maps():
    mdt.view_maps(pjoin('figure_1', 'BallStick_r1', 'samples', 'model_defined_maps'), config='''
    annotations:
    - arrow_width: 0.7
      font_size: 20
      marker_size: 3.5
      text_distance: 0.08
      text_location: upper left
      text_template: '{value:.3f}'
      voxel_index: [67, 44, 0]
    colorbar_settings: {location: null, nmr_ticks: 3, power_limits: null, round_precision: 3,
      visible: true}
    font: {family: sans-serif, size: 28}
    grid_layout:
    - Rectangular
    - cols: null
      rows: null
      spacings: {bottom: 0.03, hspace: 0.15, left: 0.1, right: 0.8, top: 0.97, wspace: 0.4}
    map_plot_options:
      FS:
        scale: {use_max: true, use_min: true, vmax: 0.5, vmin: 0.2}
        title: Mean
      FS.std:
        scale: {use_max: true, use_min: true, vmax: 0.02, vmin: 0.0}
        title: Std.
    maps_to_show: [FS, FS.std]
    rotate: 270
    zoom:
      p0: {x: 19, y: 17}
      p1: {x: 117, y: 128}
    ''')

def create_covariance_map():
    mask = mdt.load_brain_mask(pjoin('mgh_1003_slice_44_mask'))
    samples = mdt.load_samples(pjoin('figure_1', 'BallStick_r1', 'samples'))

    covariances = np.zeros(samples['S0.s0'].shape[0])
    correlations = np.zeros(samples['S0.s0'].shape[0])

    for voxel_ind in range(samples['S0.s0'].shape[0]):
        cov = np.cov(samples['S0.s0'][voxel_ind, :],
                     samples['w_stick0.w'][voxel_ind, :])
        covariances[voxel_ind] = cov[0, 1]
        correlations[voxel_ind] = covariances[voxel_ind] / (np.sqrt(cov[0, 0] * cov[1, 1]))

    mdt.view_maps(
        mdt.restore_volumes({'Covariance': covariances, 'Correlations': correlations}, mask),
        config='''
    annotations:
    - arrow_width: 0.7
      font_size: 20
      marker_size: 3.5
      text_distance: 0.08
      text_location: upper left
      text_template: '{value:.3g}'
      voxel_index: [67, 44, 0]
    colorbar_settings:
      location: right
      nmr_ticks: 4
      power_limits: [-3, 4]
      round_precision: 3
      visible: true
    font: {family: sans-serif, size: 28}
    grid_layout:
    - Rectangular
    - cols: null
      rows: null
      spacings: {bottom: 0.0, hspace: 0.15, left: 0.1, right: 0.86, top: 0.8, wspace: 0.4}
    map_plot_options:
      Correlations:
        scale: {use_max: true, use_min: true, vmax: 0.0, vmin: -0.6}
        title: Correlation S0 - FS
        title_spacing: 0.05
    maps_to_show: [Correlations]
    rotate: 270
    slice_index: 0
    zoom:
      p0: {x: 19, y: 17}
      p1: {x: 117, y: 128}
    ''')


def create_scatter_plot():
    def plot_cov_ellipse(cov, pos, nstd=2, ax=None, **kwargs):
        """
        Plots an `nstd` sigma error ellipse based on the specified covariance
        matrix (`cov`). Additional keyword arguments are passed on to the
        ellipse patch artist.
        Parameters
        ----------
            cov : The 2x2 covariance matrix to base the ellipse on
            pos : The location of the center of the ellipse. Expects a 2-element
                sequence of [x0, y0].
            nstd : The radius of the ellipse in numbers of standard deviations.
                Defaults to 2 standard deviations.
            ax : The axis that the ellipse will be plotted on. Defaults to the
                current axis.
            Additional keyword arguments are pass on to the ellipse patch.
        Returns
        -------
            A matplotlib ellipse artist
        """

        def eigsorted(cov):
            vals, vecs = np.linalg.eigh(cov)
            order = vals.argsort()[::-1]
            return vals[order], vecs[:, order]

        if ax is None:
            ax = plt.gca()

        vals, vecs = eigsorted(cov)
        theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))

        # Width and height are "full" widths, not radius
        width, height = 2 * nstd * np.sqrt(vals)
        ellip = Ellipse(xy=pos, width=width, height=height, angle=theta, **kwargs)

        ax.add_artist(ellip)
        return ellip

    def scatter_plot(ax, samples_x, samples_y, colors, sample_means, sample_covar):
        ax.scatter(samples_x, samples_y, c=colors, cmap='hot')

        ax.set_xlim(np.min(samples_x) * 0.95, np.max(samples_x) * 1.06)
        ax.set_ylim(np.min(samples_y) * 0.95, np.max(samples_y) * 1.05)

        plot_cov_ellipse(sample_covar, sample_means, nstd=1, ax=ax, linewidth=3, facecolor='none', edgecolor='black')
        ax.plot(*sample_means, color='black', marker='o', markersize=15, label='Mean')

        ax.xaxis.offsetText.set_visible(False)
        ax.yaxis.offsetText.set_visible(False)

    def plot_histogram(ax, title, samples, sample_mean, sample_std, orientation='vertical'):
        if orientation == 'horizontal':
            ax.set_title(title, rotation=270, x=1.15, y=0.5)
        else:
            ax.set_title(title, y=1.0)

        ax.hist(samples, 200, normed=True, orientation=orientation, color='lightgray')
        [label.set_visible(False) for label in ax.get_xticklabels() + ax.get_yticklabels()]

        fit_x_coords = np.linspace(np.min(samples), np.max(samples) * 1.01, 100)
        if np.max(samples) < 0:
            fit_x_coords = np.linspace(np.max(samples), np.min(samples), 100)

        sample_predicted_dist = (fit_x_coords, norm.pdf(fit_x_coords, loc=sample_mean, scale=sample_std))
        sample_mean_point = (sample_mean, float(mlab.normpdf(sample_mean, sample_mean, sample_std)))

        if orientation == 'horizontal':
            sample_predicted_dist = sample_predicted_dist[::-1]
            sample_mean_point = sample_mean_point[::-1]

        ax.plot(*sample_predicted_dist, color='black', linewidth=1)
        ax.plot(*sample_mean_point, color='black', marker='o', label='Mean', markersize=12)

        if orientation == 'horizontal':
            ax.yaxis.offsetText.set_visible(False)
        else:
            ax.xaxis.offsetText.set_visible(False)

    def create_scatter_hist_fig():
        fig, ax_scatter = plt.subplots(figsize=(10, 5))

        divider = make_axes_locatable(ax_scatter)
        ax_histx = divider.append_axes("top", 1, pad=0.15, sharex=ax_scatter)
        ax_histy = divider.append_axes("right", 1, pad=0.15, sharey=ax_scatter)

        ax_scatter.xaxis.set_major_locator(MaxNLocator(5, integer=True))

        fig.subplots_adjust(top=0.9, right=0.85)

        return fig, [ax_scatter, ax_histx, ax_histy]

    set_matplotlib_font_size(22)

    mask = mdt.load_brain_mask(pjoin('mgh_1003_slice_44_mask'))
    samples_dict = mdt.load_samples(pjoin('figure_1', 'BallStick_r1', 'samples'))

    # the parameters to show on the x and y axis
    name_x = 'S0.s0'
    name_y = 'w_stick0.w'

    # better parameter names for in the paper
    param_nicknames = {'S0.s0': 'S0', 'w_stick0.w': 'FS'}

    # selecting the voxel by volume index
    volume_ind = (67, 44, 0)
    roi_ind = mdt.volume_index_to_roi_index(volume_ind, mask)

    # using the likelihoods for coloring the points
    likelihoods = samples_dict['LogLikelihood'][roi_ind]
    weights = likelihoods + np.abs(np.min(likelihoods))
    weights /= np.max(weights)
    weights = weights ** 2

    # computing the means, stds and covariances
    samples_array = np.array([samples_dict[p][roi_ind] for p in [name_x, name_y]])
    cov = np.cov(samples_array)
    means = np.average(samples_array, axis=1)

    # create the figure an axii
    fig, ax = create_scatter_hist_fig()

    scatter_plot(ax[0], samples_array[0], samples_array[1], weights, means, cov)
    plot_histogram(ax[1], param_nicknames[name_x], samples_array[0], means[0], np.sqrt(cov[0, 0]))
    plot_histogram(ax[2], param_nicknames[name_y], samples_array[1], means[1], np.sqrt(cov[1, 1]),
                   orientation='horizontal')

    plt.show()


create_chain_histogram()
create_mean_std_maps()
create_covariance_map()
create_scatter_plot()

