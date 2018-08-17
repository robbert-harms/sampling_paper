import numpy as np
import mdt

__author__ = 'Robbert Harms'
__date__ = '2018-02-14'
__maintainer__ = 'Robbert Harms'
__email__ = 'robbert.harms@maastrichtuniversity.nl'
__licence__ = 'LGPL v3'


pjoin = mdt.make_path_joiner(r'/home/robbert/phd-data/papers/sampling_paper/single_slice/')

# only need to sample a single voxel
mask = np.zeros_like(mdt.load_brain_mask(pjoin('mgh_1003_slice_44_mask')))
mask[67, 44, 0] = 1

input_data = mdt.load_input_data(
    pjoin('mgh_1003_slice_44'),
    pjoin('protocol.prtcl'),
    mask,
    noise_std=44.19256591796875)  # noise std was pre-calculated using the entire dataset

# set the sampling proposal of the weights to a large value for demonstration purposes
class w(mdt.get_template('parameters', 'w')):
    sampling_proposal_std = 0.25


# Optimization starting point
fit_results = mdt.fit_model('BallStick_r1 (Cascade)', input_data, pjoin('figure_3'))

# Sample the model
for method in ['AMWG', 'MWG', 'FSL', 'SCAM']:
    mdt.sample_model(
        'BallStick_r1',
        input_data,
        pjoin('figure_3', 'sampling_methods', method),
        method=method,
        nmr_samples=2000,
        burnin=0,
        initialization_data={'inits': fit_results},
        store_samples=True
    )
