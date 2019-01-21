#!/home/olafurt/miniconda3/bin/python


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

BUCKET_NAME = 'paralleladcexperiments5b70cd4e-74d3-4496-96fa-f4025220d48c'


def hadamardMatrix(n):
    return sp.linalg.hadamard(n)/np.sqrt(n)


def create_s3_filename(file_name):
    return ''.join([str(uuid.uuid4().hex[:6]), file_name])


def writeCSVDataFrameToS3(s3_connection, bucket_name, file_name, df):
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    (s3_connection
      .Object(bucket_name, file_name)
      .put(Body=csv_buffer.getvalue()))
    string_buffer.close()


def writeStringToS3(s3_connection, bucket_name, file_name, string):
    string_buffer = io.StringIO()
    string_buffer.write(string)
    (s3_connection
      .Object(bucket_name, file_name)
      .put(Body=string_buffer.getvalue()))
    string_buffer.close()


def uploadTos3(s3_connection, bucket_name, file_name, obj):
    pickle_buffer = io.BytesIO()
    pkl.dump(obj, pickle_buffer)
    (s3_connection
      .Object(bucket_name, file_name)
      .put(Body=pickle_buffer.getvalue()))
    pickle_buffer.close()


class ExperimentRunner():
    """ Class to handle running experiments"""

    def __init__(self,
                 experiment_id,
                 data_dir,
                 M,
                 N,
                 L,
                 input_phase,
                 input_amplitude,
                 input_frequency=None,
                 beta=6250,
                 sampling_period=8e-5,
                 primary_signal_dimension=0,
                 systemtype='ParallelIntegratorChain',
                 OSR=16,
                 eta2_magnitude=1,
                 kappa=1,
                 sigma2_thermal=1e-6,
                 sigma2_reconst=1e-6,
                 num_periods_in_simulation=100):

        #print("Initializing Experiment")
        self.experiment_id = experiment_id
        self.data_dir = Path(data_dir)
        self.M = M
        self.N = N
        self.L = L
        self.input_phase = input_phase
        self.input_amplitude = input_amplitude
        self.input_frequency = input_frequency
        self.beta = beta
        self.sampling_period = sampling_period
        self.primary_signal_dimension = primary_signal_dimension
        self.systemtype = systemtype
        self.OSR = OSR
        self.kappa = kappa
        self.sigma2_thermal = sigma2_thermal
        self.sigma2_reconst = sigma2_reconst
        self.num_periods_in_simulation = num_periods_in_simulation
        self.size = round(num_periods_in_simulation/sampling_period)
        
        self.eta2_magnitude = ((beta * sampling_period * OSR)/ (np.pi))**(2*N) * (M**(N-2))

        self.border = np.int(self.size //100)

        self.logstr = ("{0}: EXPERIMENT LOG\n{0}: Experiment ID: {1}\n".format(time.strftime("%d/%m/%Y %H:%M:%S"), experiment_id))
        self.log("eta2_magnitude set to max(|G(s)b|^2) = {:.5e}".format(self.eta2_magnitude))
        self.finished_simulation = False
        self.finished_reconstruction = False
        if not self.data_dir.exists():
            self.data_dir.mkdir(parents=True)

        if self.primary_signal_dimension > self.M:
            self.log("Primary Signal Dimension cannot be larger than M, setting to 0 (first dim)")
            self.primary_signal_dimension = 0


        if self.input_frequency == None:
            self.log("Setting f_sig = f_s/(2*OSR)")
            self.input_frequency = 1./(self.sampling_period * 2 * self.OSR)


        if self.systemtype == "ParallelIntegratorChain":
            self.t = np.linspace(0,(self.size-1)*self.sampling_period, self.size)

            self.A = np.zeros((self.N*self.M, self.N*self.M))
            mixingPi = np.empty((self.N-1, self.M, self.M))
            H = hadamardMatrix(self.M)
            
            if N > 1:
                if L == M:
                    for k in range(N-1):
                        mixingPi[k] = beta*(np.outer(H[:,0],H[:,0])*L)/np.sqrt(L)
                        self.A[(k+1)*self.M:(k+2)*self.M, (k)*self.M:(k+1)*self.M] = mixingPi[k]
                elif L == 1:
                    for k in range(N-1):
                        mixingPi[k] = self.beta*(sum(np.outer(H[:,i],H[:,i]) for i in range(self.M)))
                        self.A[(k+1)*self.M:(k+2)*self.M, (k)*self.M:(k+1)*self.M] = mixingPi[k]
                else:
                    raise NotImplemented
        else:
            raise NotImplemented


        # Define input signals:
        self.input_signals = []
        self.input_frequencies = np.zeros(M)
        self.input_frequencies[self.primary_signal_dimension] = self.input_frequency
        allowed_disturbance_signal_frequencies = self.input_frequency * (0.5**np.arange(2,M+1))
        for i in range(self.M):
            if i == self.primary_signal_dimension: continue
            k = np.random.randint(0,M-1)
            self.input_frequencies[i] = allowed_disturbance_signal_frequencies[k]

        for i in range(self.M):
            self.vector = np.zeros(self.M*self.N)
            self.vector[0:self.M] = self.beta*(H[:,i])
            self.input_signals.append(system.Sin(self.sampling_period,
                                                 amplitude=self.input_amplitude,
                                                 frequency=self.input_frequencies[i],
                                                 phase=self.input_phase,
                                                 steeringVector=self.vector))
        self.input_signals = tuple(self.input_signals)

<<<<<<< HEAD
        #print("A = \n%s\nb = \n%s" % (self.A, self.input_signals[self.primary_signal_dimension].steeringVector))
=======
        print("A = \n%s\nb = \n%s" % (self.A, self.input_signals[self.primary_signal_dimension].steeringVector))
>>>>>>> Experiment runner and helper files

        self.c = np.eye(self.N * self.M)
        self.sys = system.System(self.A, self.c)

        self.ctrlMixingMatrix = - self.kappa * self.beta * np.eye(self.N * self.M)
        self.ctrl = system.Control(self.ctrlMixingMatrix, self.size)

    
    def log(self,message=""):
        timestamp = r'\d{2}/\d{2}/\d{4} [0-2][0-9]:[0-5][0-9]:[0-5][0-9]'
        regex = re.compile(timestamp)
        if regex.match(message):
            tmp = message if message[-1:] == "\n" else message + "\n"
        else:
            tmp = "{}: {}\n".format(time.strftime("%d/%m/%Y %H:%M:%S"), message)
        self.logstr += tmp


    def saveLog(self):
        with (self.data_dir / f'{self.experiment_id}.log').open(mode='w') as outfile:
            outfile.write(self.logstr)


    def saveSimulation(self):
        # Save simulation results, dictionary with:
        #     't': time sequence
        #     'control': control object (control[:] holds the values of the s_k(t) signals)
        #     'output': time domain state values
        #     'system': A,c matrices
        #     'state': value of last state (unnecessary)
        #     'options': options dictionary passed to the simulator
        sim_file_path = self.data_dir / 'result_obj.pkl'
        with sim_file_path.open(mode='wb') as outfile:
            pkl.dump(self.result, outfile)
        self.log("Simulation file saved at {}".format(sim_file_path))


    def saveInputSignals(self):
        # Save input signals used in the simulation.
        #   Load with: input_signals = pkl.load('input_signals.pkl')
        #   Primary Signal is then: input_signals[primary_signal_dimension]
        inputs_file_path = self.data_dir / 'input_signals.pkl'
        with inputs_file_path.open(mode='wb') as outfile:
            pkl.dump(self.input_signals, outfile)
        self.log("Input signals saved at \"{}\"".format(inputs_file_path))


    def saveInputEstimates(self):
        input_estimates_file_path = self.data_dir / 'input_estimates.pkl'
        with input_estimates_file_path.open(mode='wb') as outfile:
            pkl.dump(self.input_estimates, outfile)
        self.log("Input Estimates saved at \"{}\"".format(input_estimates_file_path))


    def saveAll(self):
        file_path = self.data_dir / 'ExperimentRunner.pkl'
        with file_path.open(mode='wb') as outfile:
            pkl.dump(self.__dict__, outfile)
        self.log("ExperimentRunner saved at \"{}\"".format(file_path))
        self.saveLog()


    def run_simulation(self):
        self.sim_start_time = time.time()

        self.simulation_options = {'stateBound':(self.sampling_period * self.beta * self.kappa) / (1. - (self.sampling_period * self.beta / np.sqrt(self.L))),
                                   'stateBoundInputs': (self.sampling_period * self.beta * self.kappa) / (1. - (self.sampling_period * self.beta / np.sqrt(self.L))),
                                   'num_parallel_converters': self.M,
                                   'noise':[{'std':self.sigma2_thermal, 'steeringVector': self.beta*np.eye(self.N * self.M)[:,i]}  for i in range(self.M * self.N)]}
        
        self.sim = simulator.Simulator(self.sys, self.ctrl, options=self.simulation_options)
        self.result = self.sim.simulate(self.t, self.input_signals)

        self.sim_run_time = time.time() - self.sim_start_time
        self.log(self.result['log'])
        self.log("Simulation run time: {:.2f} seconds".format(self.sim_run_time))
        self.finished_simulation = True


    def run_reconstruction(self):
        self.recon_time_start = time.time()

        self.eta2 = np.ones(self.M * self.N) * self.eta2_magnitude
        self.reconstruction_options = {'eta2':self.eta2,
                                       'sigmaU2':[1.]*self.M,
                                       'noise':[{'std':self.sigma2_reconst,
                                                 'steeringVector': self.beta*np.eye(self.N * self.M)[:,i], 'name':'noise_{}'.format(i)} for i in range(self.N * self.M)]}
        self.reconstruction = reconstruction.WienerFilter(self.t, self.sys, self.input_signals, self.reconstruction_options)
        tmp_estimates, recon_log = self.reconstruction.filter(self.ctrl)

        self.input_estimates = tmp_estimates[self.border:-self.border]
        self.recon_run_time = time.time() - self.recon_time_start
        self.log(recon_log)
        self.log("Reconstruction run time: {:.2f} seconds".format(self.recon_run_time))
        self.finished_reconstruction = True

    
    def unitTest(self):
        #for key in self.__dict__.keys():
        #   print("{} = {}".format(key, self.__dict__[key]))

        #print("\n\n")
        self.log("This message should have a timestamp")
        self.log("00/00/0000 00:00:00: This message should not have a timestamp")
        #print(self.logstr)

        self.size = 5000#round(1./self.sampling_period)
        self.t = np.linspace(0,(self.size-1)*self.sampling_period, self.size)
        self.data_dir = self.data_dir.parent / ('unitTest_' + self.data_dir.name)
        if not self.data_dir.exists():
            self.data_dir.mkdir(parents=True)

        self.run_simulation()
        # self.run_reconstruction()

    def getParams(self):
        return {'M':self.M,
                'N':self.N,
                'L':self.L,
                'beta':self.beta,
                'sampling_period':self.sampling_period,
                'input_frequency':self.input_frequency,
                'eta2':self.eta2_magnitude,
                'disturbance_frequencies':self.input_frequencies[1:],
                'size': "{:e}".format(self.size),
                'num_oob': self.result['num_oob'],
                'oob_rate': self.results['num_oob'] / self.size}


def main(experiment_id,
         data_dir,
         M, 
         N,
         L,
         input_phase,
         input_amplitude,
         input_frequency=None,
         beta=6250,
         sampling_period=8e-5,
         primary_signal_dimension=1,
         systemtype='ParallelIntegratorChain',
         OSR=16,
         eta2_magnitude=1,
         kappa=1,
         sigma2_thermal=1e-6,
         sigma2_reconst=1e-6,
         num_periods_in_simulation=20):
    
    runner = ExperimentRunner(experiment_id,
                              data_dir,
                              M, 
                              N,
                              L,
                              input_phase,
                              input_amplitude,
                              input_frequency,
                              beta,
                              sampling_period,
                              primary_signal_dimension,
                              systemtype,
                              OSR,
                              eta2_magnitude,
                              kappa,
                              sigma2_thermal,
                              sigma2_reconst,
                              num_periods_in_simulation)

    # runner.unitTest()
    runner.run_simulation()
    runner.run_reconstruction()

    s3_resource = boto3.resource('s3')
    s3_file_name_prefix = uuid.uuid4().hex[:6]

    runner.log("Saving results to S3")
    runner.log("S3 file name: \"{}\"".format(''.join([s3_file_name_prefix, experiment_id])))
    params = runner.getParams()
    params_string = ''
    for key in params.keys():
      params_string = ''.join([params_string, f'{key}: {params[key]}\n'])

    uploadTos3(
      s3_connection=s3_resource,
      bucket_name=BUCKET_NAME,
      file_name=''.join([s3_file_name_prefix,experiment_id,'_results.pkl']),
      obj=runner)

    writeStringToS3(
      s3_connection=s3_resource,
      bucket_name=BUCKET_NAME,
      file_name=f'{s3_file_name_prefix}{experiment_id}.log',
      string=runner.logstr)

    writeStringToS3(
      s3_connection=s3_resource,
      bucket_name=BUCKET_NAME,
      file_name=f'{s3_file_name_prefix}{experiment_id}.params',
      string=params_string)


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(description="Parallel ADC\
                                                      Experiment Runner")

        # Required arguments
    arg_parser.add_argument("-id", "--experiment_id", required=True, type=str)
    arg_parser.add_argument("-d", "--data_dir", required=True, type=str)
    arg_parser.add_argument("-M", required=True, type=int)
    arg_parser.add_argument("-N", required=True, type=int)
    arg_parser.add_argument("-L", required=True, type=int)
    arg_parser.add_argument("-beta", required=True, type=int)
    arg_parser.add_argument("-Ts", "--sampling_period", required=True, type=float)
    arg_parser.add_argument("-Ax", "--input_amplitude", required=True,  type=float)
    arg_parser.add_argument("-phi", "--input_phase", required=True, type=float)
    arg_parser.add_argument("-sigma2_thermal", required=True, type=float)
    arg_parser.add_argument("-sigma2_reconst", required=True, type=float)


    # Optional arguments, things that could change later
    # or are currently set to some fixed value
    arg_parser.add_argument("-f_sig", "--input_frequency", default=None)
    arg_parser.add_argument("-eta2", "--eta2_magnitude", type=float, default=1)
    arg_parser.add_argument("-kappa", type=float, default=1)
    arg_parser.add_argument("-OSR", type=int, default=16)
    arg_parser.add_argument("-systemtype", type=str, default="ParallelIntegratorChain")
    arg_parser.add_argument("-sig_dim", "--primary_signal_dimension", type=int, default=0)
    arg_parser.add_argument("-n_sim", "--num_periods_in_simulation", type=int, default=20)

    args = vars(arg_parser.parse_args())

    main(**args)
