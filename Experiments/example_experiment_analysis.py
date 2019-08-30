from run_experiment import ExperimentRunner
import AnalogToDigital.evaluation as evaluation
import numpy as np
import scipy as sp
import pickle as pkl
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt


name = 'example_experiment'
data_dir = Path(f"./experiments/{name}")
save_dir = data_dir / 'results'
if not save_dir.exists():
    save_dir.mkdir(parents=True)

parallel_estimates_list = []
cyclic_estiamtes_list = []

parallel_psd_list = []
cyclic_psd_list = []

par_dict = {}
cyc_dict = {}

for i, results_file in enumerate(data_dir.glob("*Parallel*results*")):
    with open(results_file, 'rb') as f:
        runner = pkl.load(f)
    est = runner.input_estimates[:,0]
    parallel_estimates_list.append(est)
    parallelsystem = runner.sys_nominal
    sigmadelta = evaluation.SigmaDeltaPerformance(system=parallelsystem, estimate=est, fs=1)
    freq = sigmadelta.freq
    spec = sigmadelta.spec
    parallel_psd_list.append(spec)
    par_dict[i] = 10*np.log10(spec)

for i,results_file in enumerate(data_dir.glob("*Cyclic*results*")):
    with open(results_file, 'rb') as f:
        runner = pkl.load(f)
    est = runner.input_estimates[:,0]
    cyclic_estiamtes_list.append(est)
    cyclicsystem = runner.sys_nominal
    sigmadelta = evaluation.SigmaDeltaPerformance(system=cyclicsystem, estimate=est, fs=1)
    freq = sigmadelta.freq
    spec = sigmadelta.spec
    cyclic_psd_list.append(spec)
    cyc_dict[i] = 10*np.log10(spec)

if len(parallel_psd_list) > 0:
    par_dict['freq'] = freq
    par_dict['freq'][0] = 1e-5
    par_df = pd.DataFrame(par_dict).set_index('freq')
    fIndex = sigmadelta.fIndex

    # Reducing the number of samples in the PSD,
    # but keeping all samples around the peaks
    tmp = np.concatenate((np.arange(1, fIndex-5, 4),
                              np.arange(fIndex-5, fIndex+5),
                              np.arange(fIndex+5, par_df.index.size, 4)))
    decimationMask = np.zeros(par_df.index.size, dtype=bool)
    decimationMask[tmp] = True
    par_df[decimationMask].to_csv(save_dir / f'{name}_parallel_psd.csv', sep=' ')

if len(cyclic_psd_list) > 0:
    cyc_dict['freq'] = freq
    cyc_dict['freq'][0] = 1e-5
    cyc_df = pd.DataFrame(cyc_dict).set_index('freq')
    fIndex = sigmadelta.fIndex
    tmp = np.concatenate((np.arange(1, fIndex-5, 4),
                              np.arange(fIndex-5, fIndex+5),
                              np.arange(fIndex+5, cyc_df.index.size, 4)))
    decimationMask = np.zeros(cyc_df.index.size, dtype=bool)
    decimationMask[tmp] = True
    cyc_df[decimationMask].to_csv(save_dir / f'{name}_cyclic_psd.csv', sep=' ')

cyclic_arr = np.array(cyclic_psd_list)
parallel_arr = np.array(parallel_psd_list)

if len(cyclic_psd_list) > 0:
    plt.semilogx(freq,10*np.log10(np.mean(cyclic_arr, axis=0)), label='Cyclic Hadamard System', color='xkcd:cadet blue')
if len(parallel_psd_list) > 0:
    plt.semilogx(freq,10*np.log10(np.mean(parallel_arr, axis=0)), label='Parallel System', color='xkcd:scarlet')

plt.legend()
ax = plt.gca()

# Set x and y limits on the plot
ax.set_ylim(bottom=-170, top=0)
ax.set_xlim(left=1e-4, right=5e-1)

fig = plt.gcf()
ax.set_xlabel('$fT$')
ax.set_ylabel('PSD [dB]')
ax.set_title(f'{name}')
manager = plt.get_current_fig_manager()
manager.window.showMaximized()
plt.tight_layout()

fig.savefig(f'{save_dir}/{name}.pdf', format="pdf")
plt.ioff()
plt.show()


# This can be used for each simulation to plot the state traces and their PSDs
def states_and_their_PSDs(runner):
    t = np.linspace(0,(runner.size-1)*runner.sampling_period, runner.size)
    fig,ax = plt.subplots(runner.N,2)
    ax = ax.reshape(runner.N,2)
    label = [f'x_{i}' for i in range(len(runner.result['output']))]

    output = runner.result['output']
    freq, spec = sp.signal.welch(output.T, fs=1./runner.sampling_period, nperseg=(1<<9))
    
    for i in range(runner.N):
      ax[i,0].plot([0,len(t)],[0,0])
      for j in range(0,runner.M):
        ax[i,0].plot(np.arange(len(t)), output[:,i*runner.M+j], label=label[i*runner.M+j])
        ax[i,1].semilogx(freq, 10*np.log10(spec[i*runner.M+j,:]), label=label[i*runner.M+j])

        # Plotting guide lines for easier reading
        ax[i,0].plot([0,len(t)],[-1,-1], 'b--')
        ax[i,0].plot([0,len(t)],[1,1], 'b--')

        ax[i,0].plot([0,len(t)],[1./np.sqrt(2),1./np.sqrt(2)], 'g--')
        ax[i,0].plot([0,len(t)],[-1./np.sqrt(2),-1./np.sqrt(2)], 'g--')

        ax[i,0].legend()
        ax[i,1].legend()
        ax[i,1].set_ylim(bottom=-80,top=-10)

    ax[0,0].set_title(f"{runner.systemtype}")
    ax[0,1].set_title("PSDs of the state traces")
    plt.draw()
    plt.ioff()
    plt.show()

# states_and_their_PSDs(runner)
