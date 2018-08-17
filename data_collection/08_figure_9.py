import glob
import os
import mdt
from mdt.lib.batch_utils import SimpleBatchProfile, BatchFitProtocolLoader, SimpleSubjectInfo

__author__ = 'Robbert Harms'
__date__ = '2018-01-18'
__maintainer__ = 'Robbert Harms'
__email__ = 'robbert.harms@maastrichtuniversity.nl'
__licence__ = 'LGPL v3'


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


def func(subject_info, model_name, opt_output_dir, samples_output_dir):
    subject_id = subject_info.subject_id
    input_data = subject_info.get_input_data()
    base_folder = subject_info.subject_base_folder

    starting_point = mdt.fit_model(model_name + ' (Cascade)',
                                   input_data,
                                   opt_output_dir + '/' + subject_id)

    wm_mask = mdt.load_brain_mask(base_folder + '/output/optimization_paper/wm_mask.nii.gz')

    wm_input_data = input_data.copy_with_updates(input_data.protocol, input_data.signal4d,
                                                 wm_mask, input_data.nifti_header)

    print('Subject {}'.format(subject_id))
    mdt.sample_model(model_name,
                     wm_input_data,
                     samples_output_dir + '/' + subject_id,
                     nmr_samples=nmr_samples[model_name],
                     initialization_data={'inits': starting_point},
                     store_samples=False,
                     post_processing={'multivariate_ess': True,
                                      'model_defined_maps': False})


for model_name in model_names:
    mdt.batch_apply(func, '/home/robbert/phd-data/rheinland/',
                    batch_profile=RheinLandBatchProfile(resolutions_to_use=['data_ms20']),
                    subjects_selection=range(10),
                    extra_args=[model_name,
                                '/home/robbert/phd-data/rheinland_output/',
                                '/home/robbert/phd-data/papers/sampling_paper/ess/rheinland/'
                                ])

    mdt.batch_apply(func, '/home/robbert/phd-data/hcp_mgh/',
                    batch_profile='HCP_MGH',
                    subjects_selection=range(10),
                    extra_args=[model_name,
                                '/home/robbert/phd-data/hcp_mgh_output/',
                                '/home/robbert/phd-data/papers/sampling_paper/ess/hcp_mgh/'])