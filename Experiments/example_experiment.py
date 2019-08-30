import numpy as np
from pathlib import Path
import pickle as pkl
from run_experiment import ExperimentRunner

global M
global name
name = 'example_experiment'
M = 1

def run(experiment_id, newInputVector, systemtype):
    DATA_STORAGE_PATH = Path(f'./experiments/{name}')
    if not DATA_STORAGE_PATH.exists():
        DATA_STORAGE_PATH.mkdir(parents=True)
    N = 3
    OSR = 32
    sampling_period = 8e-5
    beta = 6250
    input_amplitude = 1
    input_frequency = 1./(2*sampling_period*OSR*16)
    sigma_thermal = 1e-6
    sigma_reconst = 1e-6
    num_periods_in_simulation = 1
    controller = "diagonalController"


    runner = ExperimentRunner(experiment_id=experiment_id,
                              data_dir=DATA_STORAGE_PATH,
                              M=M,
                              N=N,
                              OSR=OSR,
                              beta=beta,
                              input_amplitude=input_amplitude,
                              input_frequency=input_frequency,
                              sampling_period=sampling_period,
                              systemtype=systemtype,
                              sigma_thermal=sigma_thermal,
                              sigma_reconst=sigma_reconst,
                              num_periods_in_simulation=num_periods_in_simulation,
                              controller=controller,
                              newInputVector=newInputVector)

    runner.run_simulation()
    runner.run_reconstruction()
    runner.log(f'Saving results to "{DATA_STORAGE_PATH}"')
    runner.saveAll()
    with open(DATA_STORAGE_PATH / f'{experiment_id}_results.pkl', 'wb') as f:
      pkl.dump(runner, f, protocol=pkl.HIGHEST_PROTOCOL)

input_ch = np.zeros(M)
input_ch[0] = 1
input_parallel = 0.5*np.ones(M)

# for i in range(1):
#   run(experiment_id=f'{name}_CyclicHadamard_{i}', newInputVector=input_ch, systemtype='CyclicHadamard')
for i in range(1):
  run(experiment_id=f'{name}_Parallel_{i}', newInputVector=input_parallel, systemtype='ParallelIntegratorChains')