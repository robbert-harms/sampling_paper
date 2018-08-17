import mdt

__author__ = 'Robbert Harms'
__date__ = '2018-02-15'
__maintainer__ = 'Robbert Harms'
__email__ = 'robbert.harms@maastrichtuniversity.nl'
__licence__ = 'LGPL v3'


pjoin = mdt.make_path_joiner(r'/home/robbert/phd-data/papers/sampling_paper/single_slice/')

data = {}
for method in ['powell_initialized', 'default_initialized']:
    for burnin in [0, 1000, 3000]:
        maps = mdt.load_volume_maps(pjoin('figure_6', method, str(burnin), 'NODDI', 'samples', 'model_defined_maps'))
        data['{}_{}_mean'.format(method, burnin)] = maps['w_ic.w']
        data['{}_{}_std'.format(method, burnin)] = maps['w_ic.w.std']

mdt.view_maps(data, config='''
annotations:
- arrow_width: 0.5
  font_size: 20
  marker_size: 3.5
  text_distance: 0.08
  text_location: upper left
  text_template: '{value:.3f}'
  voxel_index: [67, 44, 0]
colorbar_settings:
  location: right
  nmr_ticks: 3
  power_limits: [-2, 2]
  round_precision: 3
  visible: true
font: {family: sans-serif, size: 28}
grid_layout:
- Rectangular
- cols: 4
  rows: null
  spacings: {bottom: 0.03, hspace: 0.15, left: 0.1, right: 0.86, top: 0.97, wspace: 0.4}
maps_to_show: [
    default_initialized_0_mean, default_initialized_0_std, 
    powell_initialized_0_mean, powell_initialized_0_std,
    default_initialized_1000_mean, default_initialized_1000_std, 
    powell_initialized_1000_mean, powell_initialized_1000_std,
    default_initialized_3000_mean, default_initialized_3000_std, 
    powell_initialized_3000_mean, powell_initialized_3000_std]
map_plot_options:
  default_initialized_0_mean:
    scale: {use_max: true, use_min: true, vmax: 0.6, vmin: 0.3}
    title: ' '
    colorbar_settings: {visible: false}
  default_initialized_1000_mean:
    scale: {use_max: true, use_min: true, vmax: 0.6, vmin: 0.3}
    title: ' '
    colorbar_settings: {visible: false}
  default_initialized_3000_mean:
    scale: {use_max: true, use_min: true, vmax: 0.6, vmin: 0.3}
    title: ' '
    colorbar_settings: {visible: true, location: bottom, nmr_ticks: 3}
  powell_initialized_0_mean:
    scale: {use_max: true, use_min: true, vmax: 0.6, vmin: 0.3}
    title: ' '
    colorbar_settings: {visible: false}
  powell_initialized_1000_mean:
    scale: {use_max: true, use_min: true, vmax: 0.6, vmin: 0.3}
    title: ' '
    colorbar_settings: {visible: false}
  powell_initialized_3000_mean:
    scale: {use_max: true, use_min: true, vmax: 0.6, vmin: 0.3}
    title: ' '
    colorbar_settings: {visible: true, location: bottom, nmr_ticks: 3}
  default_initialized_0_std:
    scale: {use_max: true, use_min: true, vmax: 0.05, vmin: 0.0}
    title: ' '
    colorbar_settings: {visible: false}
  default_initialized_1000_std:
    scale: {use_max: true, use_min: true, vmax: 0.05, vmin: 0.0}
    title: ' '
    colorbar_settings: {visible: false}
  default_initialized_3000_std:
    scale: {use_max: true, use_min: true, vmax: 0.05, vmin: 0.0}
    title: ' '
    colorbar_settings: {visible: true, location: bottom, nmr_ticks: 3}
  powell_initialized_0_std:
    scale: {use_max: true, use_min: true, vmax: 0.05, vmin: 0.0}
    title: ' '
    colorbar_settings: {visible: false}
  powell_initialized_1000_std:
    scale: {use_max: true, use_min: true, vmax: 0.05, vmin: 0.0}
    title: ' '
    colorbar_settings: {visible: false}
  powell_initialized_3000_std:
    scale: {use_max: true, use_min: true, vmax: 0.05, vmin: 0.0}
    title: ' '
    colorbar_settings: {visible: true, location: bottom, nmr_ticks: 3}
rotate: 270
zoom:
  p0: {x: 19, y: 17}
  p1: {x: 117, y: 128}
''')
