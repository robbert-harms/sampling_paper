import numpy as np
import mdt

__author__ = 'Robbert Harms'
__date__ = '2018-02-14'
__maintainer__ = 'Robbert Harms'
__email__ = 'robbert.harms@maastrichtuniversity.nl'
__licence__ = 'LGPL v3'


"""
Generate the data for figure 7 of the article. Due to randomness in the MCMC algorithm, the results may not look
exactly the same as in the article. Run this script several times for different outputs.
"""

pjoin = mdt.make_path_joiner(r'/home/robbert/phd-data/papers/sampling_paper/single_slice/')

mask = np.zeros_like(mdt.load_brain_mask(pjoin('mgh_1003_slice_44_mask')))
mask[67, 44, 0] = 1

input_data = mdt.load_input_data(
    pjoin('mgh_1003_slice_44'),
    pjoin('protocol.prtcl'),
    mask,
    noise_std=44.19256591796875)  # noise std was pre-calculated using the entire dataset

init_methods = ['powell', '0.4', '0.8']
inits = {
    'BallStick_r1': {
        'powell': mdt.fit_model('BallStick_r1 (Cascade)', input_data, pjoin('figure_7')),
        '0.4': {'w_stick0.w': 0.4},
        '0.8': {'w_stick0.w': 0.8}
    },
    'NODDI': {
        'powell': mdt.fit_model('NODDI (Cascade)', input_data, pjoin('figure_7')),
        '0.4': {'w_ic.w': 0.4, 'w_ec.w': 0.2},
        '0.8': {'w_ic.w': 0.8, 'w_ec.w': 0.1}
    }
}

for model in ['BallStick_r1', 'NODDI']:
    for init_method in init_methods:
        mdt.sample_model(
            model,
            input_data,
            pjoin('figure_7', 'init_points', init_method),
            nmr_samples=2000,
            initialization_data={'inits': inits[model][init_method]},
            store_samples=True,
            recalculate=False
        )
