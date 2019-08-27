from run_experiment import ExperimentRunner
import AnalogToDigital.evaluation as evaluation
import numpy as np
import scipy as sp
import pickle as pkl
from pathlib import Path
import matplotlib.pyplot as plt


data_dir = Path("./test_random_noise_directions")
save_dir = data_dir / 'results'
if not save_dir.exists():
    save_dir.mkdir(parents=True)

parallel_estimates_list = []
cyclic_estiamtes_list = []

for i, results_file in enumerate(data_dir.glob("*Parallel*results*")):
    with open(results_file, 'rb') as f:
        runner = pkl.load(f)
        parallel_estimates_list.append(runner.input_estimates[:,0])
    parallelsystem = runner.sys_nominal

for i,results_file in enumerate(data_dir.glob("*Cyclic*results*")):
    with open(results_file, 'rb') as f:
        runner = pkl.load(f)
        cyclic_estiamtes_list.append(runner.input_estimates[:,0])
    cyclicsystem = runner.sys_nominal

parallel_psd_list = []
for est in parallel_estimates_list:
    sigmadelta = evaluation.SigmaDeltaPerformance(system=parallelsystem, estimate=est, fs=1)
    freq = sigmadelta.freq
    parallel_psd_list.append(sigmadelta.spec)


cyclic_psd_list = []
for est in cyclic_estiamtes_list:
    sigmadelta = evaluation.SigmaDeltaPerformance(system=cyclicsystem, estimate= est, fs=1)
    freq = sigmadelta.freq
    cyclic_psd_list.append(sigmadelta.spec)

cyclic_arr = np.array(cyclic_psd_list)
parallel_arr = np.array(parallel_psd_list)
print(cyclic_arr.shape)
print(parallel_arr.shape)
print(np.mean(cyclic_arr,axis=0).shape)
print(np.mean(parallel_arr,axis=0).shape)

plt.semilogx(freq,10*np.log10(np.mean(cyclic_arr, axis=0)), 'b-')
plt.semilogx(freq,10*np.log10(np.mean(parallel_arr, axis=0)), 'r-')
plt.ioff()
plt.show()
