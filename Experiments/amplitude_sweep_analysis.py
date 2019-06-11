import argparse
import numpy as np
import scipy as sp
import pandas as pd
import pickle as pkl
from pathlib import Path

import AnalogToDigital.evaluation as evaluation
import AnalogToDigital.system as system


# Edit this path
DATA_DIR = Path('./amplitude_sweep')
if not DATA_DIR.exists():
    DATA_DIR.mkdir(parents=True)


def main(experiment_id):
    experiment_directory = DATA_DIR / experiment_id
    save_dir = experiment_directory / experiment_id
    if not save_dir.exists():
        save_dir.mkdir()

    with open(list(experiment_directory.glob("*.params.pkl"))[0], 'rb') as f:
        tmp_params = pkl.load(f)
        if 'M' in tmp_params:
            M = tmp_params['M']
        if 'N' in tmp_params:
            N = tmp_params['N']
        if 'L' in tmp_params:
            L = tmp_params['L']
        if 'bpc' in tmp_params:
            bpc = tmp_params['bpc']
        # M,N,L,bpc = tmp_params['M'], tmp_params['N'], tmp_params['L'], tmp_params['bpc']
    
    estimates = {l:{} for l in range(L)}
    EvaluationObjects = {l:[] for l in range(L)}
    PSDs = {l:{} for l in range(L)}

    for results_file in experiment_directory.glob("*results.pkl"):
            with open(results_file, 'rb') as f:
                current_runner = pkl.load(f)
            for l in range(L):
                estimates[l][current_runner.input_amplitude] = current_runner.input_estimates[:,l]
            # continue

    system, fs, OSR, fmax = (current_runner.sys,
                       1./current_runner.sampling_period,
                       current_runner.OSR,
                       current_runner.input_frequency)
    estimates_list = []
    for l in range(L):
        amplitudes = list(estimates[l].keys())
        amplitudes.sort()
        estimates_list.append([estimates[l][amp] for amp in amplitudes])
        EvaluationObjects[l] = evaluation.SNRvsAmplitude(system,
                                                         estimates_list[l],
                                                         fs=1,
                                                         OSR=OSR,
                                                         fmax=1./(2*OSR))

        EvaluationObjects[l].ToTextFile(save_dir / f'InputPower_vs_SNR_{l}.csv', delimiter=' ')
        for k, est in enumerate(EvaluationObjects[l].estimates):
            if k == 0:
                PSDs[l]['freq'] = est['performance'].freq
                PSDs[l]['freq'][0] = 1e-5
            est['performance'].ToTextFile(save_dir / f'DeltaSigmaPerformance_{amplitudes[k]}.csv', delimiter=' ', OSR=16)
            PSDs[l][amplitudes[k]] = 10*np.log10(est['performance'].spec)
        df = pd.DataFrame(PSDs[l])
        df = df.set_index('freq')
        fIndex = est['performance'].fIndex
        tmp = np.concatenate((np.arange(0,fIndex-5,4),
                              np.arange(fIndex-5, fIndex+5),
                              np.arange(fIndex+5,df.index.size,4)))
        decimationMask = np.zeros(df.index.size,dtype=bool)
        decimationMask[tmp] = True
        #np.mod(np.arange(df.index.size),2)!=0
        # decimationMask[est['performance'].fIndex -5 : est['performance'].fIndex +5] = True
        # df[decimationMask].to_csv(save_dir / f'PSDs_{l}_subsampled.csv', sep=' ')
        for i, column in enumerate(df):
            df[column][decimationMask].to_csv(save_dir / f'PSDs_0_dB_Ax_{df.columns[i]}.csv', sep=' ')


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("-id", "--experiment_id", required=True, type=str)
    args = vars(arg_parser.parse_args())
    main(**args)