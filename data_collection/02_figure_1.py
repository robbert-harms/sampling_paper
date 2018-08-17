import mdt

__author__ = 'Robbert Harms'
__date__ = '2018-02-14'
__maintainer__ = 'Robbert Harms'
__email__ = 'robbert.harms@maastrichtuniversity.nl'
__licence__ = 'LGPL v3'

"""
This script is for optimizing the Ball&Stick model for the intro figure. This script does not create the figure, it 
merely creates the right output data.

This script requires the single slice data from the HCP MGH consortium, which is bundled with this repository.
"""

pjoin = mdt.make_path_joiner(r'/home/robbert/phd-data/papers/sampling_paper/single_slice/')

# the single slice input data
input_data = mdt.load_input_data(
    pjoin('mgh_1003_slice_44'),
    pjoin('protocol.prtcl'),
    pjoin('mgh_1003_slice_44_mask'),
    noise_std=44.19256591796875)  # noise std was pre-calculated using the entire dataset

# Optimization starting point
fit_results = mdt.fit_model('BallStick_r1 (Cascade)', input_data, pjoin('figure_1'))

# Sample the model
mdt.sample_model(
    'BallStick_r1',
    input_data,
    pjoin('figure_1'),
    method='AMWG',
    nmr_samples=10000,
    burnin=200,
    initialization_data={'inits': fit_results},
    store_samples=True
)