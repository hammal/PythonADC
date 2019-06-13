from pathlib import Path
import time

prefix_file = Path('condor_prefix')
with prefix_file.open() as file:
    condor_prefix = file.read()


submit_condor = Path('./submit.condor')


timestamp = time.strftime("%d%m%Y_%H%M%S")
sampling_period_array = [8e-5]
OSR_array = [16]
M_array = [1,2,4,8]
N_array = [1,2]
L_array = [1,2,4,8]
input_phase_array = [0]
input_amplitude_array = [0.5,1]
input_frequency_array = [1./(sampling_period_array[0]*OSR_array[0])]
beta_array = [6250]
primary_signal_dimension_array = [0]
systemtype_array = ['ParallelIntegratorChain']
eta2_magnitude_array = [1]
kappa_array = [1]
sigma2_thermal_array = [1e-6]
sigma2_reconst_array = [1e-6]
num_periods_in_simulation_array = [20]


# timestamp, M, N, L, beta, f_sig, Ts, Ax, phi, sigma2_thermal, sigma2_reconst, eta2, kappa, OSR, systemtype, sig_dim, n_sim
with submit_condor.open(mode='w') as f:
    f.write(condor_prefix)
    for M in M_array:
        for N in N_array:
            for L in L_array:
                if L == M or L == 1:
                    for beta in beta_array:
                        for f_sig in input_frequency_array:
                            for Ts in sampling_period_array:
                                for Ax in input_amplitude_array:
                                    for phi in input_phase_array:
                                        for sigma2_thermal in sigma2_thermal_array:
                                            for sigma2_reconst in sigma2_reconst_array:
                                                for eta2 in eta2_magnitude_array:
                                                    for kappa in kappa_array:
                                                        for OSR in OSR_array:
                                                            for systemtype in systemtype_array:
                                                                for sig_dim in primary_signal_dimension_array:
                                                                    for n_sim in num_periods_in_simulation_array:
                                                                        f.write("arguments = -id {0}_$(Process) -d /home/olafurt/adc_experiments/{1}_$(Process) -M {2} -N {3} -L {4} -beta {5} -f_sig {6} -Ts {7} -Ax {8} -phi {9} -sigma2_thermal {10} -sigma2_reconst {11} -eta2 {12} -kappa {13} -OSR {14} -systemtype {15} -sig_dim {16} -n_sim {17}\nqueue\n\n".format(timestamp, timestamp, M, N, L, beta, f_sig, Ts, Ax, phi, sigma2_thermal, sigma2_reconst, eta2, kappa, OSR, systemtype, sig_dim, n_sim))
