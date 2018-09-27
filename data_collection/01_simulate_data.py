import os
import numpy as np
import mdt

__author__ = 'Robbert Harms'
__date__ = '2018-01-01'
__maintainer__ = 'Robbert Harms'
__email__ = 'robbert.harms@maastrichtuniversity.nl'
__licence__ = 'LGPL v3'

"""
This script is meant to generate the simulation data needed for figure 4 and 5 of the article.

Before running this script, please copy the protocol files "hcp_mgh_1003.prtcl" and "rheinland_v3a_1_2mm.prtcl"
to the directory where you wish to store the simulation data and results. Set the same path here in this script. 
"""

# the path to where you stored the protocol files.
data_storage_path = r'/home/robbert/phd-data/papers/sampling_paper/simulations/'

# an utility for easily appending to paths
pjoin = mdt.make_path_joiner(data_storage_path)

# The names of the protocol files we wish to simulate data for
protocols = [
    'hcp_mgh_1003',
    'rheinland_v3a_1_2mm'
]

# The models for which we want to simulate the data
models = [
    'BallStick_r1',
    'BallStick_r2',
    'BallStick_r3',
    'NODDI',
    'Tensor',
    'CHARMED_r1',
    'CHARMED_r2',
    'CHARMED_r3'
]

# The different SNR's we wish to simulate. Each additional SNR will generate a copy of the simulated data with
# noise corresponding to the desired SNR.
noise_snrs = [30]

# The b0 signal intensity, fixed such that we can easily generate the noise (noise_std = unweighted_signal_height / SNR)
unweighted_signal_height = 1e4


def create_parameter_combinations(nmr_voxels, randomize_parameter_indices, default_values, lower_bounds, upper_bounds,
                                  seed=None):
    """Create a list of parameters combinations.

    This first generates a grid of (nmr_voxels, nmr_parameters) holding for each parameter the default value. Next,
    it will randomize the indicated columns with random variables, uniformly distributed between the lower and upper
    bounds.

    Args:
        nmr_voxels (int): the number of unique parameter combinations.
        randomize_parameter_indices (list of int): the indices of the parameter we sample randomly using a uniform
            distribution on the half open interval between the [lower, upper) bounds. See np.random.uniform.
        default_values (list of float): the default values for each of the parameters
        lower_bounds (list of float): the lower bounds used for the generation of the grid
        upper_bounds (list of float): the upper bounds used for the generation of the grid
        seed (int): if given the seed for the random number generator, this makes the random parameters predictable.

    Returns:
        ndarray: a two dimensional list of parameter combinations
    """
    grid = np.array(default_values)[None, :]
    grid = np.repeat(grid, nmr_voxels, axis=0)

    random_state = np.random.RandomState(seed)

    for param_ind in randomize_parameter_indices:
        grid[:, param_ind] = random_state.uniform(lower_bounds[param_ind], upper_bounds[param_ind], size=nmr_voxels)

    return grid


def prepare_noddi_params(params_cube):
    """Normalizes the w_ic.w and w_ec.w if > 1, else leaves it as is."""
    param_names = mdt.get_model('NODDI')().get_free_param_names()

    w_ec_w = params_cube[:, param_names.index('w_ec.w')]
    w_ic_w = params_cube[:, param_names.index('w_ic.w')]

    summed = w_ec_w + w_ic_w
    w_ec_w[summed > 1] = (w_ec_w / summed)[summed > 1]
    w_ic_w[summed > 1] = (w_ic_w / summed)[summed > 1]

    params_cube[:, param_names.index('w_ec.w')] = w_ec_w
    params_cube[:, param_names.index('w_ic.w')] = w_ic_w


def prepare_charmed_r1_params(params_cube):
    """Set the primary direction of the Tensor to the first CHARMED Restricted compartment. \
        Sorts the diffusivities of the Tensor
    """
    param_names = mdt.get_model('CHARMED_r1')().get_free_param_names()
    params_cube[..., param_names.index('Tensor.theta')] = params_cube[
        ..., param_names.index('CHARMEDRestricted0.theta')]
    params_cube[..., param_names.index('Tensor.phi')] = params_cube[..., param_names.index('CHARMEDRestricted0.phi')]
    params_cube[:, 1:4] = np.sort(params_cube[:, 1:4], axis=1)[:, ::-1]


def prepare_ballstick2_params(params_cube):
    """Make sure the weights sum to 1."""
    weights_sum = np.sum(params_cube[:, [1, 4]], axis=1)
    indices = weights_sum > 1
    params_cube[indices, 1] /= weights_sum[indices]
    params_cube[indices, 4] /= weights_sum[indices]


def prepare_charmed2_params(params_cube):
    """Make sure the weights sum to 1 and sort the Tensor diffusivities"""
    weights_sum = np.sum(params_cube[:, [7, 11]], axis=1)
    indices = weights_sum > 1
    params_cube[indices, 7] /= weights_sum[indices]
    params_cube[indices, 11] /= weights_sum[indices]

    params_cube[:, 1:4] = np.sort(params_cube[:, 1:4], axis=1)[:, ::-1]


def prepare_charmed3_params(params_cube):
    """Make sure the weights sum to 1 and sort the Tensor diffusivities"""
    weights_sum = np.sum(params_cube[:, [7, 11, 15]], axis=1)
    indices = weights_sum > 1
    params_cube[indices, 7] /= weights_sum[indices]
    params_cube[indices, 11] /= weights_sum[indices]
    params_cube[indices, 15] /= weights_sum[indices]

    params_cube[:, 1:4] = np.sort(params_cube[:, 1:4], axis=1)[:, ::-1]


def prepare_ballstick3_params(params_cube):
    """Make sure the weights sum to 1."""
    weights_sum = np.sum(params_cube[:, [1, 4, 7]], axis=1)
    indices = weights_sum > 1
    params_cube[indices, 1] /= weights_sum[indices]
    params_cube[indices, 4] /= weights_sum[indices]
    params_cube[indices, 7] /= weights_sum[indices]


def prepare_tensor_cb(params_cube):
    """Sort the eigenvalues (diffusivities) such that the highest is d, then dperp0, then dperp1."""
    params_cube[:, 1:4] = np.sort(params_cube[:, 1:4], axis=1)[:, ::-1]


# Model specific simulation settings
simulations = {
    'BallStick_r1': dict(
        # Available parameters:
        # ['S0.s0', 'w_stick0.w', 'Stick0.theta', 'Stick0.phi']
        randomize_parameters=['w_stick0.w', 'Stick0.theta', 'Stick0.phi'],
        prepare_params_cube_cb=None,
        lower_bounds=[1e2, 0.2, 0, 0],
        upper_bounds=[1e5, 0.8, np.pi, np.pi]
    ),
    'BallStick_r2': dict(
        # Available parameters:
        # ['S0.s0', 'w_stick0.w', 'Stick0.theta', 'Stick0.phi', 'w_stick1.w', 'Stick1.theta', 'Stick1.phi']
        randomize_parameters=['w_stick0.w', 'Stick0.theta', 'Stick0.phi', 'w_stick1.w', 'Stick1.theta', 'Stick1.phi'],
        prepare_params_cube_cb=prepare_ballstick2_params,
        lower_bounds=[1e2, 0.2, 0, 0, 0.2, 0, 0],
        upper_bounds=[1e5, 0.8, np.pi, np.pi, 0.8, np.pi, np.pi]
    ),
    'BallStick_r3': dict(
        # Available parameters:
        # ['S0.s0', 'w_stick0.w', 'Stick0.theta', 'Stick0.phi', 'w_stick1.w', 'Stick1.theta', 'Stick1.phi',
        # 'w_stick2.w', 'Stick2.theta', 'Stick2.phi']
        randomize_parameters=['w_stick0.w', 'Stick0.theta', 'Stick0.phi', 'w_stick1.w', 'Stick1.theta', 'Stick1.phi',
                              'w_stick2.w', 'Stick2.theta', 'Stick2.phi'],
        prepare_params_cube_cb=prepare_ballstick3_params,
        lower_bounds=[1e2, 0.2, 0, 0, 0.2, 0, 0, 0.2, 0, 0],
        upper_bounds=[1e5, 0.8, np.pi, np.pi, 0.8, np.pi, np.pi, 0.8, np.pi, np.pi]
    ),
    'NODDI': dict(
        # Available parameters:
        # ['S0.s0', 'w_ic.w', 'NODDI_IC.theta', 'NODDI_IC.phi', 'NODDI_IC.kappa', 'w_ec.w']
        randomize_parameters=['w_ic.w', 'NODDI_IC.theta', 'NODDI_IC.phi', 'NODDI_IC.kappa', 'w_ec.w'],
        prepare_params_cube_cb=prepare_noddi_params,
        lower_bounds=[1e2, 0.2, 0, 0, 0.1, 0.2],
        upper_bounds=[1e5, 0.8, np.pi, np.pi, 2 * np.pi, 0.8]
    ),
    'CHARMED_r1': dict(
        # Available parameters:
        # ['S0.s0', 'Tensor.d', 'Tensor.dperp0', 'Tensor.dperp1', 'Tensor.theta', 'Tensor.phi',
        #  'Tensor.psi', 'w_res0.w', 'CHARMEDRestricted0.d', 'CHARMEDRestricted0.theta', 'CHARMEDRestricted0.phi']
        randomize_parameters=['Tensor.d', 'Tensor.dperp0', 'Tensor.dperp1',
                              'Tensor.theta', 'Tensor.phi', 'Tensor.psi', 'w_res0.w',
                              'CHARMEDRestricted0.d', 'CHARMEDRestricted0.theta', 'CHARMEDRestricted0.phi'],
        prepare_params_cube_cb=prepare_charmed_r1_params,
        lower_bounds=[1e3, 5e-11, 5e-11, 5e-11, 0, 0, 0, 0.2, 5e-11, 0, 0],
        upper_bounds=[1e9, 5e-9, 5e-9, 5e-9, np.pi, np.pi, np.pi, 0.8, 5e-9, np.pi, np.pi],
    ),
    'CHARMED_r2': dict(
        # Available parameters:
        # ['S0.s0', 'Tensor.d', 'Tensor.dperp0', 'Tensor.dperp1', 'Tensor.theta', 'Tensor.phi', 'Tensor.psi', 'w_res0.w',
        #  'CHARMEDRestricted0.d', 'CHARMEDRestricted0.theta', 'CHARMEDRestricted0.phi', 'w_res1.w',
        #  'CHARMEDRestricted1.d', 'CHARMEDRestricted1.theta', 'CHARMEDRestricted1.phi']
        randomize_parameters=['Tensor.d', 'Tensor.dperp0', 'Tensor.dperp1', 'Tensor.theta', 'Tensor.phi', 'Tensor.psi',
                              'w_res0.w',
                              'CHARMEDRestricted0.d', 'CHARMEDRestricted0.theta', 'CHARMEDRestricted0.phi', 'w_res1.w',
                              'CHARMEDRestricted1.d', 'CHARMEDRestricted1.theta', 'CHARMEDRestricted1.phi'],
        prepare_params_cube_cb=prepare_charmed2_params,
        lower_bounds=[1e3, 5e-11, 5e-11, 5e-11, 0, 0, 0, 0.2, 5e-11, 0, 0, 0.2, 5e-11, 0, 0],
        upper_bounds=[1e9, 5e-9, 5e-9, 5e-9, np.pi, np.pi, np.pi, 0.8, 3e-9, np.pi, np.pi, 0.8, 3e-9, np.pi, np.pi],
    ),
    'CHARMED_r3': dict(
        # Available parameters:
        # ['S0.s0', 'Tensor.d', 'Tensor.dperp0', 'Tensor.dperp1', 'Tensor.theta', 'Tensor.phi', 'Tensor.psi', 'w_res0.w',
        #  'CHARMEDRestricted0.d', 'CHARMEDRestricted0.theta', 'CHARMEDRestricted0.phi', 'w_res1.w',
        #  'CHARMEDRestricted1.d', 'CHARMEDRestricted1.theta', 'CHARMEDRestricted1.phi', 'w_res2.w',
        #  'CHARMEDRestricted2.d', 'CHARMEDRestricted2.theta', 'CHARMEDRestricted2.phi']
        randomize_parameters=['Tensor.d', 'Tensor.dperp0', 'Tensor.dperp1', 'Tensor.theta', 'Tensor.phi', 'Tensor.psi',
                              'w_res0.w',
                              'CHARMEDRestricted0.d', 'CHARMEDRestricted0.theta', 'CHARMEDRestricted0.phi', 'w_res1.w',
                              'CHARMEDRestricted1.d', 'CHARMEDRestricted1.theta', 'CHARMEDRestricted1.phi', 'w_res2.w',
                              'CHARMEDRestricted2.d', 'CHARMEDRestricted2.theta', 'CHARMEDRestricted2.phi'],
        prepare_params_cube_cb=prepare_charmed3_params,
        lower_bounds=[1e3, 5e-11, 5e-11, 5e-11, 0, 0, 0, 0.2, 5e-11, 0, 0, 0.2, 5e-11, 0, 0, 0.2, 5e-11, 0, 0],
        upper_bounds=[1e9, 5e-9, 5e-9, 5e-9, np.pi, np.pi, np.pi, 0.8, 5e-9, np.pi, np.pi, 0.8, 5e-9, np.pi, np.pi,
                      0.8, 3e-9, np.pi, np.pi],
    ),
    'Tensor': dict(
        # Available parameters:
        # ['S0.s0', 'Tensor.d', 'Tensor.dperp0', 'Tensor.dperp1', 'Tensor.theta', 'Tensor.phi', 'Tensor.psi']
        randomize_parameters=['Tensor.d', 'Tensor.dperp0', 'Tensor.dperp1',
                              'Tensor.theta', 'Tensor.phi', 'Tensor.psi'],
        prepare_params_cube_cb=prepare_tensor_cb,
        lower_bounds=[1e2, 5e-11, 5e-11, 5e-11, 0, 0, 0],
        upper_bounds=[1e5, 5e-9, 5e-9, 5e-9, np.pi, np.pi, np.pi]
    )
}


def create_simulations(protocol_name, model_name):
    output_pjoin = pjoin.create_extended(protocol_name, model_name)
    if os.path.exists(output_pjoin()):
        return
    else:
        output_pjoin.make_dirs()

    model = mdt.get_model(model_name)(volume_selection=False)
    param_names = model.get_free_param_names()

    model_config = simulations[model_name]

    parameters = create_parameter_combinations(
        10000,
        [param_names.index(name) for name in model_config['randomize_parameters']],
        model.get_initial_parameters(),
        model_config['lower_bounds'],
        model_config['upper_bounds'],
        seed=0)

    if model_config['prepare_params_cube_cb'] is not None:
        model_config['prepare_params_cube_cb'](parameters)

    simulated_signals = mdt.simulate_signals(model, mdt.load_protocol(pjoin(protocol_name)), parameters)

    mdt.write_nifti(parameters[None, :, None, :], output_pjoin('original_parameters.nii'))
    mdt.write_nifti(simulated_signals[None, :, None, :], output_pjoin('simulated_signals.nii'))

    for snr in noise_snrs:
        noisy_signals = mdt.add_rician_noise(simulated_signals, unweighted_signal_height / snr, seed=0)
        mdt.write_nifti(noisy_signals[None, :, None, :], output_pjoin('noisy_signals_{}.nii'.format(snr)))

    mdt.create_blank_mask(output_pjoin('noisy_signals_{}.nii'.format(noise_snrs[0])),
                          output_pjoin('mask.nii'))


for protocol_name in protocols:
    for model_name in models:
        create_simulations(protocol_name, model_name)
