import subprocess

from matplotlib import ticker
import mdt
from mdt.lib.post_processing import DTIMeasures
import numpy as np
import matplotlib.pyplot as plt
import seaborn

__author__ = 'Robbert Harms'
__date__ = '2018-02-03'
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


pjoin = mdt.make_path_joiner('/mnt/storage2/robbert/papers/sampling_papers/simulations/')
nmr_trials = 10
simulations_unweighted_signal_height = 1e4
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
ap_methods = [
    'MWG',
    'SCAM',
    'FSL',
    'AMWG'
]


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


def get_ground_truth_measures(model_name, original_parameters):
    if model_name == 'Tensor':
        return DTIMeasures.fractional_anisotropy(
            original_parameters[..., 1],
            original_parameters[..., 2],
            original_parameters[..., 3])
    elif model_name == 'BallStick_r1':
        return original_parameters[..., 1]
    elif model_name == 'BallStick_r2':
        return original_parameters[..., 1]
    elif model_name == 'BallStick_r3':
        return original_parameters[..., 1]
    elif model_name == 'NODDI':
        return original_parameters[..., 1]
    elif model_name == 'CHARMED_r1':
        return original_parameters[..., 7]
    elif model_name == 'CHARMED_r2':
        return original_parameters[..., 7]
    elif model_name == 'CHARMED_r3':
        return original_parameters[..., 7]


def get_results(model_name, samples_dir):
    model_defined_maps = mdt.load_volume_maps(samples_dir + '/model_defined_maps/')
    univariate_normal = mdt.load_volume_maps(samples_dir + '/univariate_normal/')

    if model_name == 'BallStick_r1':
        return univariate_normal['w_stick0.w']
    if model_name == 'BallStick_r2':
        return univariate_normal['w_stick0.w']
    if model_name == 'BallStick_r3':
        return univariate_normal['w_stick0.w']
    elif model_name == 'Tensor':
        return model_defined_maps['Tensor.FA']
    elif model_name == 'NODDI':
        return univariate_normal['w_ic.w']
    elif model_name == 'CHARMED_r1':
        return univariate_normal['w_res0.w']
    elif model_name == 'CHARMED_r2':
        return univariate_normal['w_res0.w']
    elif model_name == 'CHARMED_r3':
        return univariate_normal['w_res0.w']


def get_protocol_results():
    protocol_results = {}
    for protocol_name in protocols:
        method_results = {}
        for method_name in ap_methods:
            model_results = {}
            for model_name in model_names:

                if protocol_name == 'rheinland_v3a_1_2mm' and model_name.startswith('CHARMED'):
                    continue

                for snr in noise_snrs:
                    current_pjoin = pjoin.create_extended(protocol_name, model_name)

                    ground_truth_map = np.squeeze(get_ground_truth_measures(
                        model_name, mdt.load_nifti(current_pjoin('original_parameters')).get_data()))

                    trial_means = []
                    trial_stds = []
                    for trial_ind in range(nmr_trials):
                        trial_pjoin = current_pjoin.create_extended('figure_4_5', str(snr), method_name,
                                                                    str(trial_ind), model_name, 'samples')

                        trial_results = np.squeeze(get_results(model_name, trial_pjoin()))
                        trial_diffs = np.abs(trial_results - ground_truth_map)

                        trial_diffs = trial_diffs[np.abs(trial_diffs - np.mean(trial_diffs)) > 2 * np.std(trial_diffs)]

                        trial_means.append(np.mean(trial_diffs))
                        trial_stds.append(np.std(trial_diffs))

                    model_results[model_name] = {}
                    model_results[model_name]['accuracy'] = (
                        float(np.mean(1 / np.array(trial_means))),
                        float(np.std(1 / np.array(trial_means)) / np.sqrt(nmr_trials))
                    )

                    prec = 1 / np.array(trial_stds)
                    prec[np.isinf(prec)] = 0

                    model_results[model_name]['precision'] = (
                        float(np.mean(prec)),
                        float(np.std(prec) / np.sqrt(nmr_trials))
                    )

            method_results[method_name] = model_results
        protocol_results[protocol_name] = method_results
    return protocol_results


protocol_results = get_protocol_results()


x_locations = np.array([0, 1, 2, 3])  # the x locations for the groups
width = 0.35       # the width of the bars
colors = ['#e6bae6', '#8cb8db', '#fdc830', '#65e065']

offsets = {
    'BallStick_r1': {
        'hcp_mgh_1003': {
            'accuracy': 2.95e2,
            'precision': True
        },
        'rheinland_v3a_1_2mm': {
            'accuracy': 1.57e2,
            'precision': 2.08e2,
        }},
    'Tensor': {
        'hcp_mgh_1003': {
            'accuracy': 22,
            'precision': 18,
        },
        'rheinland_v3a_1_2mm': {
            'accuracy': 17,
            'precision': 15,
        }},
    'NODDI': {
        'hcp_mgh_1003': {
            'accuracy': True,
            'precision': True,
        },
        'rheinland_v3a_1_2mm': {
            'accuracy': 1.38e2,
            'precision': 1.75e2,
        }},
    'CHARMED_r1': {
        'hcp_mgh_1003': {
            'accuracy': 1,
            'precision': 38,
        },
        'rheinland_v3a_1_2mm': {
            'accuracy': 79,
            'precision': 34,
        }},
}


def plot_protocol_results(data, protocol_name, model_name):
    f, axarr = plt.subplots(1, 2, figsize=(6, 5))
    f.subplots_adjust(wspace=0.1, right=0.85, left=0.35, top=0.88, bottom=0.18)

    for ind, plot_type in enumerate(['accuracy', 'precision']):
        ax = axarr[ind]
        bars = []
        max_min = []
        for method_ind, method_name in enumerate(ap_methods):
            kwargs = {}
            if plot_type == 'precision':
                kwargs.update(hatch='///')

            bars.append(ax.bar(x_locations[method_ind] * width,
                               data[protocol_name][method_name][model_name][plot_type][0],
                               width, color=colors[method_ind],
                               yerr=data[protocol_name][method_name][model_name][plot_type][1],
                               label=method_name, error_kw=dict(ecolor='gray', lw=2, capsize=5, capthick=2),
                               **kwargs))

            max_min.append(data[protocol_name][method_name][model_name][plot_type][0] +
                           data[protocol_name][method_name][model_name][plot_type][1])
            max_min.append(data[protocol_name][method_name][model_name][plot_type][0] -
                           data[protocol_name][method_name][model_name][plot_type][1])

        tick_locations = (x_locations + 0.35) * width - 0.1
        tick_locations[2] += 0.1
        ax.set_xticks(tick_locations)
        ax.set_xticklabels([ap_method_names[ap_method] for ap_method in ap_methods])
        for tick in ax.get_xticklabels():
            tick.set_rotation(45)
        ax.set_xlim(-0.1, 1.5)
        ax.set_ylim((np.min(max_min) - 0.15 * (np.max(max_min) - np.min(max_min)),
                     np.max(max_min) + 0.15 * (np.max(max_min) - np.min(max_min))))

        # ax.ticklabel_format(useOffset=offsets[model_name][protocol_name][plot_type], style='sci', scilimits=(-2, 2), axis='y')
        # ax.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
        ax.yaxis.set_major_locator(ticker.MaxNLocator(5))

        if plot_type == 'precision':
            ax.yaxis.tick_right()
        # ax.set_title(plot_type.capitalize())

    f.suptitle(model_titles[model_name] + ' - ' + protocol_names[protocol_name], y=0.98, x=0.6)
    f.savefig(mdt.make_path_joiner('/tmp/sampling_paper/adaptive_proposals/acc_prec/', make_dirs=True)('{}_{}.png'.format(protocol_name, model_name)))

    subprocess.Popen("""
    convert -trim {protocol_name}_{model_name}.png {protocol_name}_{model_name}.png
    convert -bordercolor white -border 25x30 {protocol_name}_{model_name}.png {protocol_name}_{model_name}.png
    """.format(protocol_name=protocol_name, model_name=model_name), shell=True,
                     cwd='/tmp/sampling_paper/adaptive_proposals/acc_prec/').wait()


for protocol_name in protocols:
    for model_name in model_names:
        if protocol_name == 'rheinland_v3a_1_2mm' and model_name.startswith('CHARMED'):
            continue
        plot_protocol_results(protocol_results, protocol_name, model_name)

plt.show()

for model_name in model_names:
    subprocess.Popen("""
        convert +append hcp_mgh_1003_{model_name}.png rheinland_v3a_1_2mm_{model_name}.png {model_name}.png

        convert -trim {model_name}.png {model_name}.png
        convert -bordercolor white -border 40x30 {model_name}.png {model_name}.png
        convert {model_name}.png -splice 0x40 {model_name}.png
        convert {model_name}.png -font /usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf \\
            -gravity center -pointsize 28 -fill '#282828' -annotate +110-250 '{model_title}' {model_name}.png

        """.format(model_name=model_name, model_title=model_titles[model_name]), shell=True,
                     cwd='/tmp/sampling_paper/adaptive_proposals/acc_prec/').wait()

subprocess.Popen("""
convert BallStick_r1.png Tensor.png +append \( NODDI.png CHARMED_r1.png +append \) -append adaptive_proposals_acc_prec.png
""", shell=True, cwd='/tmp/sampling_paper/adaptive_proposals/acc_prec/').wait()

subprocess.Popen("""
convert BallStick_r2.png BallStick_r3.png +append \( CHARMED_r2.png CHARMED_r3.png +append \) -append adaptive_proposals_acc_prec_multidir.png
""", shell=True, cwd='/tmp/sampling_paper/adaptive_proposals/acc_prec/').wait()
