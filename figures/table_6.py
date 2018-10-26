from collections import defaultdict
import mdt
import csv

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

computed_nmr_samples = 10000
computed_nmr_voxels = {'hcp_mgh': 7892, 'rheinland': 4646}

min_nmr_samples = {
    'BallStick_r1': 11000,
    'BallStick_r2': 15000,
    'BallStick_r3': 25000,
    'NODDI': 15000,
    'Tensor': 13000,
    'CHARMED_r1': 17000,
    'CHARMED_r2': 25000,
    'CHARMED_r3': 30000
}

model_names = [
    'CHARMED_r1',
    'CHARMED_r2',
    'CHARMED_r3',
    'BallStick_r3',
    'BallStick_r2',
    'BallStick_r1',
    'NODDI',
    'Tensor'
]

nmr_voxels = {'hcp_mgh': 410000, 'rheinland': 204993}

autodict = lambda: defaultdict(autodict)
results = autodict()

with open(pjoin('times.txt'), 'r') as f:
    for row in csv.reader(f):
        device_type, method, dataset_name, model_name, subject_id, time = [r.strip() for r in row]
        results[model_name][dataset_name][device_type] = 0


with open(pjoin('times.txt'), 'r') as f:
    for row in csv.reader(f):
        device_type, method, dataset_name, model_name, subject_id, time = [r.strip() for r in row]

        time_all_voxels = float(time) * nmr_voxels[dataset_name] / computed_nmr_voxels[dataset_name]

        if method == 'optimization':
            # total_time = time_all_voxels
            # not counting optimization time in the final results
            total_time = 0
        else:
            total_time = time_all_voxels * min_nmr_samples[model_name] / computed_nmr_samples

        results[model_name][dataset_name][device_type] += total_time


for model_name, a in results.items():
    for dataset_name, b in a.items():
        for device_type, c in b.items():
            print(model_name, dataset_name, device_type, round(c / (60 * 60), ndigits=1))