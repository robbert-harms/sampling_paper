import mdt

__author__ = 'Robbert Harms'
__date__ = "2018-08-14"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"

pjoin = mdt.make_path_joiner('/home/robbert/phd-data/papers/sampling_paper/simulations/')
nmr_trials = 10
simulations_unweighted_signal_height = 1e4
nmr_samples = 100000
protocols = ['hcp_mgh_1003', 'rheinland_v3a_1_2mm']
noise_snrs = [30]
model_names = ['BallStick_r1', 'BallStick_r2', 'BallStick_r3', 'Tensor',
               'NODDI', 'CHARMED_r1', 'CHARMED_r2', 'CHARMED_r3']
ap_methods = ['AMWG', 'MWG', 'FSL', 'SCAM']

for model_name in model_names:
    for protocol_name in protocols:
        for snr in noise_snrs:
            for method_name in ap_methods:
                    noise_std = simulations_unweighted_signal_height / snr
                    current_pjoin = pjoin.create_extended(protocol_name, model_name)

                    input_data = mdt.load_input_data(
                        current_pjoin('noisy_signals_{}.nii'.format(snr)),
                        pjoin(protocol_name + '.prtcl'),
                        current_pjoin('mask.nii'),
                        noise_std=noise_std)

                    fit_results = mdt.fit_model(
                        model_name + ' (Cascade)',
                        input_data,
                        current_pjoin('output', str(snr)))

                    for trial_ind in range(nmr_trials):
                        print('Going to process', method_name, protocol_name, model_name, snr, trial_ind)

                        mdt.sample_model(
                            model_name,
                            input_data,
                            current_pjoin('figure_5', str(snr), method_name, str(trial_ind)),
                            method=method_name,
                            nmr_samples=nmr_samples,
                            burnin=0,
                            thinning=0,
                            initialization_data={'inits': fit_results},
                            store_samples=False,
                            post_processing={
                                'model_defined_maps': True,
                                'multivariate_ess': False,
                                'proposal_state': True}
                        )