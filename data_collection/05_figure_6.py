import mdt

__author__ = 'Robbert Harms'
__date__ = '2018-02-14'
__maintainer__ = 'Robbert Harms'
__email__ = 'robbert.harms@maastrichtuniversity.nl'
__licence__ = 'LGPL v3'

"""
Sample the results needed for figure 6 of the article. 

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
fit_results = mdt.fit_model('NODDI (Cascade)', input_data, pjoin('figure_6'))

# Sample the model
for burnin in [0, 1000, 3000]:

    # with the fit results as initialization
    mdt.sample_model(
        'NODDI',
        input_data,
        pjoin('figure_6', 'powell_initialized', str(burnin)),
        method='AMWG',
        nmr_samples=10000,
        burnin=burnin,
        store_samples=False,
        initialization_data={'inits': fit_results}
    )

    # with the MDT defaults as initialization
    mdt.sample_model(
        'NODDI',
        input_data,
        pjoin('figure_6', 'default_initialized', str(burnin)),
        method='AMWG',
        nmr_samples=10000,
        burnin=burnin,
        store_samples=False,
        initialization_data={}
    )