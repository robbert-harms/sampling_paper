from matplotlib.compat import subprocess

import mdt
import numpy as np
import matplotlib.pyplot as plt
import seaborn


__author__ = 'Robbert Harms'
__date__ = "2017-03-10"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"



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

pjoin = mdt.make_path_joiner('/home/robbert/phd-data/papers/sampling_paper/simulations/')
nmr_trials = 10
simulations_unweighted_signal_height = 1e4
nmr_samples = 100000
protocols = [
    'hcp_mgh_1003',
    'rheinland_v3a_1_2mm'
]
noise_snrs = [30]
model_names = [
    'BallStick_r1',
    'BallStick_r2',
    'BallStick_r3',
    'Tensor',
    'NODDI',
    'CHARMED_r1',
    'CHARMED_r2',
    'CHARMED_r3'
]
ap_methods = ['MWG', 'SCAM', 'FSL', 'AMWG']


protocol_names = {'hcp_mgh_1003': 'HCP MGH',
                  'rheinland_v3a_1_2mm': 'RLS'}

ap_method_names = {
    'MWG': 'None',
    'FSL': 'FSL',
    'SCAM': 'SCAM',
    'AMWG': 'AMWG'
}

model_titles = {
    'BallStick_r1': 'BallStick_in1',
    'BallStick_r2': 'BallStick_in2',
    'BallStick_r3': 'BallStick_in3',
    'Tensor': 'Tensor',
    'NODDI': 'NODDI',
    'CHARMED_r1': 'CHARMED_in1',
    'CHARMED_r2': 'CHARMED_in2',
    'CHARMED_r3': 'CHARMED_in3'
}


def get_ess_results():
    protocol_results = {}
    for protocol_name in protocols:
        method_results = {}
        for method_name in ap_methods:
            model_results = {}
            for model_name in model_names:
                for snr in noise_snrs:

                    list_of_ideal = []

                    for trial_ind in range(nmr_trials):
                        current_pjoin = pjoin.create_extended(protocol_name, model_name, 'suppl_figure_1_2', str(snr),
                                                              method_name, str(trial_ind), model_name, 'samples')

                        ess = mdt.load_nifti(current_pjoin('multivariate_ess', 'MultivariateESS')).get_data()
                        ess[ess > nmr_samples] = 0
                        list_of_ideal.append(np.squeeze(ess))

                    model_results[model_name] = (np.mean(list_of_ideal),
                                                 np.mean(np.std(list_of_ideal, axis=0) / np.sqrt(len(list_of_ideal))))

            method_results[method_name] = model_results
        protocol_results[protocol_name] = method_results
    return protocol_results


protocol_results = get_ess_results()

width = 0.35       # the width of the bars
colors = ['#e6bae6', '#8cb8db', '#fdc830', '#65e065']


def plot_protocol_results(data, model_name):
    x_locations = np.array([0, 1, 2, 3])  # the x locations for the groups

    f, axarr = plt.subplots(1, 2, figsize=(8, 5))
    f.subplots_adjust(wspace=0.4, right=0.98, left=0.15, top=0.88, bottom=0.18)

    for ind, protocol_name in enumerate(protocols):
        ax = axarr[ind]
        bars = []
        max_min = []
        for method_ind, method_name in enumerate(ap_methods):
            bars.append(ax.bar(x_locations[method_ind] * width, data[protocol_name][method_name][model_name][0],
                               width, color=colors[method_ind], yerr=data[protocol_name][method_name][model_name][1],
                               label=method_name,
                               error_kw=dict(ecolor='gray', lw=2, capsize=5, capthick=2)))
            max_min.append(data[protocol_name][method_name][model_name][0] +
                           data[protocol_name][method_name][model_name][1])
            max_min.append(data[protocol_name][method_name][model_name][0] -
                           data[protocol_name][method_name][model_name][1])

        if ind == 0:
            ax.set_ylabel('ESS')
        ax.set_xticks((x_locations) * width)
        ax.set_xticklabels([ap_method_names[ap_method] for ap_method in ap_methods])
        for tick in ax.get_xticklabels():
            tick.set_rotation(45)
        ax.set_xlim(-0.45, 1.5)
        ax.set_ylim((np.min(max_min) - 0.15 * (np.max(max_min) - np.min(max_min)),
                     np.max(max_min) + 0.15 * (np.max(max_min) - np.min(max_min))))
        ax.ticklabel_format(useOffset=False, axis='y')
        ax.set_title(protocol_names[protocol_name])

    f.suptitle(model_titles[model_name], y=1)
    f.savefig(mdt.make_path_joiner('/tmp/sampling_paper/suppl_figures/ess/all/', make_dirs=True)('{}.png'.format(model_name)))


for model_name in model_names:
    plot_protocol_results(protocol_results, model_name)
# plt.show()

for model_name in model_names:
    subprocess.Popen('convert -bordercolor white -border 25x35 {0}.png {0}.png'.format(model_name), shell=True,
                     cwd='/tmp/sampling_paper/suppl_figures/ess/all/').wait()

commands = """
convert BallStick_r1.png Tensor.png +append \( NODDI.png CHARMED_r1.png +append \) -append {0}.png
convert -trim {0}.png {0}.png
convert {0}.png -splice 0x100 {0}.png
convert {0}.png -font /usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf -gravity center -pointsize 28 -fill '#282828' -annotate +0-530 'ESS of Adaptive Proposals' {0}.png
convert -trim {0}.png {0}.png
""".format('suppl_figures_ess')
subprocess.Popen(commands, shell=True, cwd='/tmp/sampling_paper/suppl_figures/ess/all/').wait()

commands = """
convert BallStick_r2.png BallStick_r3.png +append \( CHARMED_r2.png CHARMED_r3.png +append \) -append {0}.png
convert -trim {0}.png {0}.png
convert {0}.png -splice 0x100 {0}.png
convert {0}.png -font /usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf -gravity center -pointsize 28 -fill '#282828' -annotate +0-530 'ESS of Adaptive Proposals' {0}.png
convert -trim {0}.png {0}.png
""".format('suppl_figures_ess_multidir')
subprocess.Popen(commands, shell=True, cwd='/tmp/sampling_paper/suppl_figures/ess/all/').wait()