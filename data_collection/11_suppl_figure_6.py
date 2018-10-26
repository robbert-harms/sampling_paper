import numpy as np
import mdt

__author__ = 'Robbert Harms'
__date__ = '2018-02-14'
__maintainer__ = 'Robbert Harms'
__email__ = 'robbert.harms@maastrichtuniversity.nl'
__licence__ = 'LGPL v3'


"""
Generate the data for supplementary figure 6 of the article. 
"""

pjoin = mdt.make_path_joiner(r'/home/robbert/phd-data/papers/sampling_paper/single_slice/')

mask = np.zeros_like(mdt.load_brain_mask(pjoin('mgh_1003_slice_44_mask')))
mask[90, 40, 0] = 1

input_data = mdt.load_input_data(
    pjoin('mgh_1003_slice_44'),
    pjoin('protocol.prtcl'),
    mask,
    noise_std=44.19256591796875)  # noise std was pre-calculated using the entire dataset

inits = {
    'CHARMED_r3': mdt.fit_model('CHARMED_r3 (Cascade)', input_data, pjoin('suppl_figure_6')),
    'NODDI': mdt.fit_model('NODDI (Cascade)', input_data, pjoin('suppl_figure_6')),
}

nmr_samples = 1000
max_thinning = 20

for model in ['CHARMED_r3', 'NODDI']:
    mdt.sample_model(
        model,
        input_data,
        pjoin('suppl_figure_6'),
        nmr_samples=nmr_samples * (max_thinning + 1),
        initialization_data={'inits': inits[model]},
        post_processing={'model_defined_maps': False},
        store_samples=True,
        recalculate=False
    )