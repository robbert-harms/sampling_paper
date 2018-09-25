import glob
import os
import pickle
import matplotlib.pyplot as plt
import mdt
from mdt.lib.batch_utils import SelectedSubjects, SimpleBatchProfile, BatchFitProtocolLoader, SimpleSubjectInfo
import numpy as np
from mot.mcmc_diagnostics import minimum_multivariate_ess
import seaborn

__author__ = 'Robbert Harms'
__date__ = '2018-01-18'
__maintainer__ = 'Robbert Harms'
__email__ = 'robbert.harms@maastrichtuniversity.nl'
__licence__ = 'LGPL v3'


class RheinLandBatchProfile(SimpleBatchProfile):

    def __init__(self, *args, resolutions_to_use=None, **kwargs):
        """Construct the Rheinland study batch profile.

        Args:
            resolutions_to_use (list of str): the list of resolutions to use, should contain
                'data_ms15' and/or 'data_ms20'. If not set, we will use both resolutions.
        """
        super(RheinLandBatchProfile, self).__init__(*args, **kwargs)
        self._auto_append_mask_name_to_output_sub_dir = False
        self._resolutions_to_use = resolutions_to_use or ['data_ms15', 'data_ms20']

    def _get_subjects(self, data_folder):
        dirs = sorted([os.path.basename(f) for f in glob.glob(os.path.join(data_folder, '*'))])
        subjects = []

        for directory in dirs:
            for resolution in self._resolutions_to_use:
                subject_pjoin = mdt.make_path_joiner(data_folder, directory, resolution)

                if os.path.exists(subject_pjoin()):
                    niftis = glob.glob(subject_pjoin('*.nii*'))

                    dwi_fname = list(filter(lambda v: '_mask' not in v and 'grad_dev' not in v, niftis))[0]
                    mask_fname = list(sorted(filter(lambda v: '_mask' in v, niftis)))[0]

                    protocol_fname = glob.glob(subject_pjoin('*prtcl'))[0]
                    protocol_loader = BatchFitProtocolLoader(subject_pjoin(), protocol_fname=protocol_fname)

                    subjects.append(SimpleSubjectInfo(subject_pjoin(),
                                                      directory + '_' + resolution,
                                                      dwi_fname, protocol_loader, mask_fname))

        return subjects

    def __str__(self):
        return 'Rheinland'


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

nmr_samples = {
    'BallStick_r1': 15000,
    'NODDI': 20000,
    'Tensor': 20000,
    'CHARMED_r1': 30000
}

model_names = [
    'CHARMED_r1',
    'NODDI',
    'BallStick_r1',
    'Tensor'
]


def func(subject_info, model_name, samples_output_dir):
    subject_id = subject_info.subject_id
    base_folder = subject_info.subject_base_folder

    wm_mask = mdt.load_brain_mask(base_folder + '/output/optimization_paper/wm_mask.nii.gz')
    ess_data = mdt.load_nifti(samples_output_dir + subject_id +
                              '/' + model_name + '/samples/multivariate_ess/MultivariateESS').get_data()

    ess = np.mean(mdt.create_roi(ess_data, wm_mask))

    nmr_params = len(mdt.get_model(model_name)().get_free_param_names())

    more_samples_required = (minimum_multivariate_ess(nmr_params, alpha=0.05, epsilon=0.1) - ess) / (ess / nmr_samples[model_name])
    ideal_nmr_samples = nmr_samples[model_name] + more_samples_required

    print('subject_id, model_name, ess, more_samples_required, ideal_nmr_samples')
    print(subject_id, model_name, ess, more_samples_required, ideal_nmr_samples)
    return ideal_nmr_samples


mgh_results = {}
rls_results = {}

for model_name in model_names:
    ideal_samples_per_subject = mdt.batch_apply(
        func, '/home/robbert/phd-data/rheinland/',
        batch_profile=RheinLandBatchProfile(resolutions_to_use=['data_ms20']),
        subjects_selection=SelectedSubjects(indices=range(10)),
        extra_args=[model_name,
                    '/home/robbert/phd-data/papers/sampling_paper/ess/rheinland/'
                    ])
    rls_results[model_name] = np.array([float(v) for v in ideal_samples_per_subject.values()])

    ideal_samples_per_subject = mdt.batch_apply(
        func, '/home/robbert/phd-data/hcp_mgh/',
        batch_profile=mdt.get_batch_profile('HCP_MGH')(),
        subjects_selection=SelectedSubjects(indices=range(10)),
        extra_args=[model_name,
                    '/home/robbert/phd-data/papers/sampling_paper/ess/hcp_mgh/'])
    mgh_results[model_name] = np.array([float(v) for v in ideal_samples_per_subject.values()])

# plt.show()
# exit(0)
# with open('/tmp/tmp_results_estimate_ess.pkl', 'wb') as f:
#     pickle.dump({'mgh': mgh_results, 'rls': rls_results}, f, pickle.HIGHEST_PROTOCOL)

with open('/tmp/tmp_results_estimate_ess.pkl', 'rb') as f:
    d = pickle.load(f)
    mgh_results = d['mgh']
    rls_results = d['rls']

print('rls:', {k: np.mean(rls_results[k]) for k in model_names})
print('mgh:', {k: np.mean(mgh_results[k]) for k in model_names})

print('rls:', {k: np.std(rls_results[k]) for k in model_names})
print('mgh:', {k: np.std(mgh_results[k]) for k in model_names})

f, ax = plt.subplots()
f.subplots_adjust(left=0.2)
f.suptitle(r'Estimated minimum number of MCMC samples', y=1)

x_locations = np.array(range(1, 3 * len(model_names), 3))  # the x locations for the groups
width = 0.35       # the width of the bars

mgh_rects = ax.bar((x_locations + 0) * width, [np.mean(mgh_results[k]) for k in model_names],
                   width, color='#fdc830', yerr=[np.std(mgh_results[k]) for k in model_names],
                   error_kw=dict(ecolor='gray', lw=2, capsize=5, capthick=2))

rls_rects = ax.bar((x_locations + 1) * width, [np.mean(rls_results[k]) for k in model_names],
                   width, color='#8db8da', yerr=[np.std(rls_results[k]) for k in model_names],
                   error_kw=dict(ecolor='gray', lw=2, capsize=5, capthick=2))

ax.set_ylabel('Number of samples')
ax.set_xticks((x_locations + 1) * width)
ax.set_xticklabels(model_names)
ax.legend((mgh_rects[0], rls_rects[0]), ('HCP MGH', 'RLS'), loc='upper left')
ax.set_ylim([9000, 35000])
# ax.title(r'Estimated minimum number of MCMC samples')
plt.show()

