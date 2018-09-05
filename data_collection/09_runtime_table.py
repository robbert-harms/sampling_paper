import glob
import os
import time
import numpy as np
import mdt
from mdt.lib.batch_utils import SimpleBatchProfile, BatchFitProtocolLoader, SimpleSubjectInfo


__author__ = 'Robbert Harms'
__date__ = '2018-01-18'
__maintainer__ = 'Robbert Harms'
__email__ = 'robbert.harms@maastrichtuniversity.nl'
__licence__ = 'LGPL v3'


"""
Computes the optimization and sampling times for table 7 of the paper.

This computes CPU and GPU statistics for a single slice over 10k datapoints and extrapolates the rest up to 
the total number of white matter voxels and minimum number of samples.  
"""

pjoin = mdt.make_path_joiner(r'/home/robbert/phd-data/papers/sampling_paper/runtime_table_cpu_gpu/')

if not os.path.exists(pjoin()):
    os.makedirs(pjoin())


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


def func(subject_info, model_name, dataset_name, opt_output_dir, samples_output_dir):
    subject_id = subject_info.subject_id
    input_data = subject_info.get_input_data()

    mask = np.copy(input_data.mask)
    mask[:, :, :mask.shape[2]//2] = 0
    mask[:, :, mask.shape[2] // 2 + 1:] = 0

    input_data = input_data.copy_with_updates(input_data.protocol, input_data.signal4d, mask)

    print('{}, {}, {}'.format(dataset_name, subject_id, model_name))

    device_names = ['CPU', 'GPU']

    for device_ind in [gpu_device_ind, cpu_device_ind]:
        start = time.time()
        starting_point = mdt.fit_model(model_name + ' (Cascade)',
                                       input_data,
                                       '{}/{}/{}'.format(opt_output_dir, device_names[device_ind], subject_id),
                                       post_processing={'uncertainties': False},
                                       cl_device_ind=device_ind)

        with open(pjoin('times.txt'), 'a') as f:
            f.write('{}, {}, {}, {}, {}, {}\n'.format(device_names[device_ind], 'optimization',
                                                      dataset_name, model_name, subject_id, time.time() - start))

        start = time.time()
        mdt.sample_model(model_name,
                         input_data,
                         '{}/{}/{}'.format(samples_output_dir, device_names[device_ind], subject_id),
                         method='AMWG',
                         initialization_data={'inits': starting_point},
                         store_samples=False,
                         nmr_samples=10000,
                         burnin=0,
                         thinning=0,
                         cl_device_ind=device_ind,
                         post_processing={'multivariate_ess': False,
                                          'model_defined_maps': False})

        with open(pjoin('times.txt'), 'a') as f:
            f.write('{}, {}, {}, {}, {}, {}\n'.format(device_names[device_ind], 'sampling',
                                                      dataset_name, model_name, subject_id, time.time() - start))


cpu_device_ind = 0
gpu_device_ind = 1

model_names = [
    'BallStick_r1',
    'BallStick_r2',
    'BallStick_r3',
    'NODDI',
    'Tensor',
    'CHARMED_r1',
    'CHARMED_r2',
    'CHARMED_r3'
]


for model_name in model_names:
    mdt.batch_apply(func, '/home/robbert/phd-data/rheinland/',
                    batch_profile=RheinLandBatchProfile(resolutions_to_use=['data_ms20']),
                    subjects_selection=[0],
                    extra_args=[model_name,
                                'rheinland',
                                pjoin('rheinland'),
                                pjoin('rheinland')
                                ])

    mdt.batch_apply(func, '/home/robbert/phd-data/hcp_mgh/',
                    batch_profile='HCP_MGH',
                    subjects_selection=['mgh_1003'],
                    extra_args=[model_name,
                                'hcp_mgh',
                                pjoin('hcp_mgh'),
                                pjoin('hcp_mgh')])