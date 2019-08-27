from run_experiment import ExperimentRunner
import numpy as np
import scipy as sp
import pickle as pkl
from pathlib import Path
from gram_schmidt import gram_schmidt

def hadamardMatrix(n):
    return sp.linalg.hadamard(n)/np.sqrt(n)

M = 4

def run(experiment_id, newInputVector, systemtype):
    DATA_STORAGE_PATH = Path('./test_random_noise_directions')#'/home/olafurt/cyclic_vs_parallel_random_thermal_noise_directions')
    M = 4
    N = 4
    L = 1
    OSR = 32
    sampling_period = 8e-5
    beta = 6250
    input_phase = 0
    input_amplitude = 1
    input_frequency = 1./(2*sampling_period*OSR*16)
    kappa = 1
    sigma_thermal = 1e-8
    sigma_reconst = 1e-6
    num_periods_in_simulation = 15
    controller = "diagonalController"
    bitsPerControl = 1
    leaky = True
    dither = True
    mismatch = ''
    beta_hat = beta
    beta_tilde = beta
    nonuniformNoise = None
    
    gaussian_random_matrix = np.random.randn(M*N,M*N)
    noise_basis = gram_schmidt(gaussian_random_matrix)
    # covariance_matrix_thermal = sum(np.outer(noise_basis[:,i],noise_basis[:,i]) for i in range(M*N))*(sigma_thermal**2)/(M*N)
    options = {'noise_basis':noise_basis}


    runner = ExperimentRunner(experiment_id=experiment_id,
                              data_dir=DATA_STORAGE_PATH,
                              M=M,
                              N=N,
                              L=L,
                              input_phase=input_phase,
                              input_amplitude=input_amplitude,
                              input_frequency=input_frequency,
                              beta=beta,
                              sampling_period=sampling_period,
                              primary_signal_dimension=0,
                              systemtype=systemtype,
                              OSR=OSR,
                              kappa=kappa,
                              sigma_thermal=sigma_thermal,
                              sigma_reconst=sigma_reconst,
                              num_periods_in_simulation=num_periods_in_simulation,
                              controller=controller,
                              bitsPerControl=bitsPerControl,
                              leaky=leaky,
                              dither=dither,
                              mismatch=mismatch,
                              beta_hat=beta_hat,
                              beta_tilde=beta_tilde,
                              newInputVector=newInputVector,
                              nonuniformNoise=nonuniformNoise,
                              options=options)

    runner.run_simulation()
    runner.run_reconstruction()
    runner.log(f'Saving results to "{DATA_STORAGE_PATH}"')
    runner.saveAll()
    with open(DATA_STORAGE_PATH / f'{experiment_id}_results.pkl', 'wb') as f:
      pkl.dump(runner, f, protocol=pkl.HIGHEST_PROTOCOL)

H = hadamardMatrix(M)

input_ch = np.zeros_like(H[:,0])
input_ch[0] = 1
input_parallel = H[:,0]

for i in range(1):
  run(experiment_id=f'CyclicHadamard_{i}', newInputVector=input_ch, systemtype='CyclicHadamard')
for i in range(1):
  run(experiment_id=f'Parallel_{i}', newInputVector=input_parallel, systemtype='ParallelIntegratorChains')
