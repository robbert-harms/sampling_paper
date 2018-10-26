import mdt

__author__ = 'Robbert Harms'
__date__ = '2018-02-15'
__maintainer__ = 'Robbert Harms'
__email__ = 'robbert.harms@maastrichtuniversity.nl'
__licence__ = 'LGPL v3'


pjoin = mdt.make_path_joiner(r'/home/robbert/phd-data/papers/sampling_paper/single_slice/')
maps = mdt.load_volume_maps(pjoin('suppl_figure_5', 'slice', 'BallStick_r3'))

mdt.view_maps(maps, config='''
annotations:
- arrow_width: 0.5
  font_size: 20
  marker_size: 3.5
  text_distance: 0.08
  text_location: upper left
  text_template: '{value:.3f}'
  voxel_index: [90, 40, 0]
colorbar_settings:
  location: right
  nmr_ticks: 3
  power_limits: [-2, 2]
  round_precision: 3
  visible: true
font: {family: sans-serif, size: 28}
map_plot_options:
  FS: {title: FS}
maps_to_show: [FS]
rotate: 270
zoom:
  p0: {x: 19, y: 17}
  p1: {x: 117, y: 128}

''')
