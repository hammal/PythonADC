###############################
#      Standard Packages      #
###############################
import argparse
import numpy as np
import scipy as sp
import pickle as pkl
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from pathlib import Path
import time
import os
import re
import io

import boto3
import uuid

###############################
#         ADC Packages        #
###############################
import AnalogToDigital.system as system
import AnalogToDigital.simulator as simulator
import AnalogToDigital.reconstruction as reconstruction
import AnalogToDigital.filters as filters
import AnalogToDigital.evaluation as evaluation

from run_experiment import ExperimentRunner

unit_test_dir = "/Users/olafurjonthoroddsen/polybox/mastersverkefni/pythonADC/Experiments/unit_tests"
DATA_STORAGE_PATH = Path(r'/Volumes/WD Passport/adc_data/final/jitter')
# DATA_STORAGE_PATH = ""

M = 1
L = 1
bitsPerControl = 1
def unitTest():
    runner = ExperimentRunner(experiment_id="unitTest",
                              data_dir=DATA_STORAGE_PATH,
                              M=M,
                              N=3,
                              L=L,
                              input_phase=0,
                              input_amplitude=1,
                              input_frequency=None,
                              num_periods_in_simulation=1,
                              sigma2_thermal=1e-6,
                              sigma2_reconst=1e-6,
                              systemtype='ParallelIntegratorChain',
                              controller='diagonalController',
                              bitsPerControl=bitsPerControl)
    # print(runner.eta2_magnitude)
    # print(runner.ctrlMixingMatrix)
    runner.unitTest()
    # params = runner.getParams()
    # for key in params.keys():
    #   print(f'{key}: {params[key]}')

    
    # runner.saveAll()
    # with open(DATA_STORAGE_PATH / f'jitterTest_results.pkl', 'wb') as f:
    #   pkl.dump(runner, f, protocol=pkl.HIGHEST_PROTOCOL)

    # sigmadeltaperformance = evaluation.SigmaDeltaPerformance(
    #   system=runner.sys,
    #   estimate=runner.input_estimates[:,:]
    #   fs=1./(runner.sampling_period),
    #   osr=runner.OSR,
    #   fmax=None)

    print(runner.input_estimates.shape)
    estimates = [runner.input_estimates[:,i] for i in range(L)]
    
    snrvsamplitude = evaluation.SNRvsAmplitude(system=runner.sys, estimates=estimates,OSR=runner.OSR,fs=1.)#/runner.sampling_period)
    print("IP", "SNR", "TMSNR", "TSNR", "THDN")
    print(snrvsamplitude.snrVsAmp)
    
    sigmadeltaperformance.ToTextFile(filename=DATA_STORAGE_PATH/f'jitterTest_PSD.csv', OSR=16)

    # sigmadeltaperformance2 = evaluation.SigmaDeltaPerformance(
    #   system=runner.sys,
    #   estimate=runner.input_estimates[:,0].reshape(1,-1),
    #   fs=1./runner.sampling_period,
    #   osr=runner.OSR,
    #   fmax=runner.input_frequency)    

    t = np.linspace(0,(runner.size-1)*runner.sampling_period, runner.size)
    fig,ax = plt.subplots(runner.N,2)
    label= [f'x_{i}' for i in range(len(runner.result['output']))]
    output = runner.result['output']

    #[:,0:-1:M]
    freq, spec = sp.signal.welch(output.T, fs=1./runner.sampling_period, nperseg=(1<<9))
    
    for i in range(runner.N):
      for j in range(runner.M):
        ax[i,0].plot(t, output[:,i*runner.M+j], label=label[i*runner.M+j])
        # ax[i,0].legend()

        ax[i,1].semilogx(freq, 10*np.log10(spec[i*runner.M+j,:]), label=[i*runner.M+j])
        # ax[i,1].legend()

    plt.draw()
    fig.savefig("stateTrajectory.png", dpi=300)
    # ax.legend(loc='lower left', ncol=M) #loc='center left', bbox_to_anchor=(1, 0.5))
    fig_psd, ax_psd = plt.subplots()
    for i in range(L):
      snrvsamplitude.estimates[i]['performance'].PlotPowerSpectralDensity(ax=ax_psd, label=f'signal_{i}')
    # sigmadeltaperformance2.PlotPowerSpectralDensity(ax=ax[1],label='signal2')
    plt.draw()
    fig_psd.savefig("Power Spectral Densities.png", dpi=300)
    
if __name__ == "__main__":
    unitTest()