import numpy as np
import mdt

__author__ = 'Robbert Harms'
__date__ = '2018-02-14'
__maintainer__ = 'Robbert Harms'
__email__ = 'robbert.harms@maastrichtuniversity.nl'
__licence__ = 'LGPL v3'


"""
Generate the data for supplementary figure 5.
"""

pjoin = mdt.make_path_joiner(r'/home/robbert/phd-data/papers/sampling_paper/single_slice/')

mask = np.zeros_like(mdt.load_brain_mask(pjoin('mgh_1003_slice_44_mask')))
mask[90, 40, 0] = 1

input_data = mdt.load_input_data(
    pjoin('mgh_1003_slice_44'),
    pjoin('protocol.prtcl'),
    mask,
    noise_std=44.19256591796875)  # noise std was pre-calculated using the entire dataset

full_input_data = mdt.load_input_data(
    pjoin('mgh_1003_slice_44'),
    pjoin('protocol.prtcl'),
    pjoin('mgh_1003_slice_44_mask'),
    noise_std=44.19256591796875)  # noise std was pre-calculated using the entire dataset

bs3_results = mdt.fit_model('BallStick_r3 (Cascade)', full_input_data, pjoin('suppl_figure_5', 'slice'),
                            recalculate=False)

def view_voxel_select_maps():
    angle_map = np.arccos(np.sum(bs3_results['Stick0.vec0'] * bs3_results['Stick1.vec0'], axis=-1))
    mdt.apply_mask(angle_map, full_input_data.mask)
    bs3_results['angle_map'] = angle_map
    mdt.view_maps(bs3_results, config='''
        annotations:
        - arrow_width: 1.0
          font_size: null
          marker_size: 1.0
          text_distance: 0.05
          text_location: upper left
          text_template: '{voxel_index}
        
            {value:.3g}'
          voxel_index: [90, 40, 0]
        colorbar_settings:
          location: right
          nmr_ticks: 3
          power_limits: [-2, 2]
          round_precision: 3
          visible: true
        font: {family: sans-serif, size: 28}
        grid_layout:
        - Rectangular
        - cols: 2
          rows: null
          spacings: {bottom: 0.03, hspace: 0.15, left: 0.1, right: 0.86, top: 0.97, wspace: 0.4}
        maps_to_show: [Stick0.vec0, Stick1.vec0, angle_map, w_stick0.w, w_stick1.w]
        rotate: 270
        zoom:
          p0: {x: 19, y: 17}
          p1: {x: 117, y: 128}
    ''')

view_voxel_select_maps()

init_methods = ['powell', '0.4', '0.8']
inits = {
    'BallStick_r1': {
        'powell': mdt.fit_model('BallStick_r1 (Cascade)', input_data, pjoin('suppl_figure_5')),
        '0.4': {'w_stick0.w': 0.4},
        '0.8': {'w_stick0.w': 0.8}
    },
    'NODDI': {
        'powell': mdt.fit_model('NODDI (Cascade)', input_data, pjoin('suppl_figure_5')),
        '0.4': {'w_ic.w': 0.4, 'w_ec.w': 0.2},
        '0.8': {'w_ic.w': 0.8, 'w_ec.w': 0.1}
    }
}

for model in ['BallStick_r1', 'NODDI']:
    for init_method in init_methods:
        mdt.sample_model(
            model,
            input_data,
            pjoin('suppl_figure_5', 'init_points', init_method),
            nmr_samples=2000,
            initialization_data={'inits': inits[model][init_method]},
            store_samples=True,
            recalculate=False
        )
