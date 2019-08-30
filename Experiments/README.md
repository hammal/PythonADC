# ADC Experiments

## Contents
The main workhorse is the ExperimentRunner class in run_experiment.py
Any experiment that is to be run must be specified by passing parameters into this class.


## An example experiment
The "default" experiment is run if only `experiment_id` and `data_dir` are passed to the ExperimentRunner:
```python
def run():
    runner = ExperimentRunner(experiment_id='simple_experiment',
                              data_dir='./')
    runner.run_experiment()
    runner.run_reconstruction()
    runner.saveAll()
    with open(f'./{experiment_id}_results.pkl', 'wb') as f:
      pkl.dump(runner, f, protocol=pkl.HIGHEST_PROTOCOL)
```
You can see all the default parameters in the `__init__()` function of ExperimentRunner.
Essentially it's a single integrator, sinusoidal input with amplitude 1 and frequency at the maximum frequency ($`1/(2\times T_s\times OSR`$).

A bit more involved example:

```python
name = 'example_experiment'
def run(experiment_id, newInputVector, systemtype):
    data_dir = Path(f'./experiments/{name}')
    if not data_dir.exists():
        data_dir.mkdir(parents=True)
    N = 3
    OSR = 32
    sampling_period = 8e-5
    beta = 6250
    input_amplitude = 1
    input_frequency = 1./(2*sampling_period*OSR*16)
    sigma_thermal = 1e-6
    sigma_reconst = 1e-6
    num_periods_in_simulation = 15
    controller = "diagonalController"

    runner = ExperimentRunner(experiment_id=experiment_id,
                              data_dir=data_dir,
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
    runner.saveAll()
    with open(data_dir / f'{experiment_id}_results.pkl', 'wb') as f:
      pkl.dump(runner, f, protocol=pkl.HIGHEST_PROTOCOL)

input_ch = np.zeros(M)
input_ch[0] = 1
input_parallel = 0.5*np.ones(M)

for i in range(5):
  run(experiment_id=f'{name}_CyclicHadamard_{i}', newInputVector=input_ch, systemtype='CyclicHadamard')
for i in range(5):
  run(experiment_id=f'{name}_Parallel_{i}', newInputVector=input_parallel, systemtype='ParallelIntegratorChains')
```

### Specifying experiment length
The experiment length is specified with the `num_periods_in_simulation` parameter.
The idea was that making the simulation length be an integer number of periods of the sampling period, the results of the FFT is cleaner.
So e.g. if `sampling_period = 8e-5` and `num_periods_in_simulation = 1` we have `$1/8\times10^{-5} = 12.500$` samples.

This way of specifying the length is maybe not strictly required for nice FFTs when the experiments are "long enough", but I find it a nice way.


### Loading and analysing experiments
Simply load the pickle file `{experiment_id}_results.pkl`. The simulation output is stored in `ExperimentRunner.result` and the input estimates are `ExperimentRunner.input_estimates`.

The simplest way of analysing a single simulation is using the `AnalogToDigital.evaluation.SigmaDeltaPerformance` class.


