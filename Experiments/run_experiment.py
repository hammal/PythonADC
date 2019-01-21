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
# DATA_STORAGE_PATH = Path('/itet-stor/olafurt/net_scratch/adc_data')
DATA_STORAGE_PATH = Path(r'/Volumes/WD Passport/adc_data')


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

        self.border = np.int(self.size //100)
        self.all_input_signal_amplitudes = np.zeros(L)
        self.all_input_signal_amplitudes[primary_signal_dimension] = input_amplitude

        self.logstr = ("{0}: EXPERIMENT LOG\n{0}: Experiment ID: {1}\n".format(time.strftime("%d/%m/%Y %H:%M:%S"), experiment_id))

        self.finished_simulation = False
        self.finished_reconstruction = False
        if not self.data_dir.exists():
            self.data_dir.mkdir(parents=True)

        if self.primary_signal_dimension > self.M:
            self.log("Primary Signal Dimension cannot be larger than M, setting to 0 (first dim)")
            self.primary_signal_dimension = 0

        if self.input_frequency == None:
            self.input_frequency = 1./(self.sampling_period * 2 * self.OSR)# + np.random.randn()
            self.log(f'Setting f_sig = f_s/(2*OSR) = {self.input_frequency}')


        if self.systemtype == "ParallelIntegratorChain":

            self.A = np.zeros((self.N*self.M, self.N*self.M))
            mixingPi = np.zeros((self.N-1, self.M, self.M))
            H = hadamardMatrix(self.M)
            
            if N > 1:
                # L=1 means just one of M dimensions is used and there is
                # only one input signal => We scale up the input vector by sqrt(M)
                if L == 1:
                    for k in range(N-1):
                        mixingPi[k] = beta * np.sqrt(M) * (np.outer(H[:,0],H[:,0])) + beta * np.sqrt(M) * sum(np.outer(H[:,i],H[:,i]) for i in range(1,self.M)) * 1e-3
                        self.A[(k+1)*self.M:(k+2)*self.M, (k)*self.M:(k+1)*self.M] = mixingPi[k]
                # L=M means M input signals
                elif L == M:
                    for k in range(N-1):
                        mixingPi[k] = self.beta * (sum(np.outer(H[:,i],H[:,i]) for i in range(self.M)))
                        self.A[(k+1)*self.M:(k+2)*self.M, (k)*self.M:(k+1)*self.M] = mixingPi[k]
                else:
                    for k in range(N-1):
                        mixingPi[k] = self.beta * np.sqrt(M/L) * (sum(np.outer(H[:,i],H[:,i]) for i in range(self.L)))
                        self.A[(k+1)*self.M:(k+2)*self.M, (k)*self.M:(k+1)*self.M] = mixingPi[k]
                    # raise NotImplemented
                print("A = {}".format(self.A))
            else:
              mixingPi = [np.zeros((M,M))]


            # Define input signals:
            self.input_signals = []
            self.all_input_signal_frequencies = np.zeros(L)
            self.all_input_signal_frequencies[self.primary_signal_dimension] = self.input_frequency
            allowed_signal_frequencies = self.input_frequency * (0.5**np.arange(2,3*M))
            for i in range(self.L):
                if i == self.primary_signal_dimension: continue
                k = np.random.randint(0,L-1)
                self.all_input_signal_frequencies[i] = allowed_signal_frequencies[k]
                self.all_input_signal_amplitudes[i] = input_amplitude

            # M x L selector matrix for the b-vectors
            # if L == 2:
            #   selectorMatrix = (np.array([
            #                         [1, 0],
            #                         [0, 1],
            #                         [1, 0],
            #                         [0, 1]
            #                               ]))
            if L == M:
              inputVectorMatrix = np.eye(M)/np.sqrt(M)
            elif L == 1:
              inputVectorMatrix = H

            # uPi,sPi,vhPi = np.linalg.svd(mixingPi[0])
            # vhPi[np.abs(vhPi) < 1e-16] = 0
            for i in range(self.L):
                vector = np.zeros(self.M*self.N)
                # Scale the input vector up by sqrt(number of channels per signal)
                vector[0:self.M] =  beta * np.sqrt(M) * inputVectorMatrix[:,i] # self.beta * np.sqrt(M/L) * vhPi[i]
                self.input_signals.append(system.Sin(self.sampling_period,
                                                     amplitude=self.all_input_signal_amplitudes[i],
                                                     frequency=self.all_input_signal_frequencies[i],
                                                     phase=self.input_phase,
                                                     steeringVector=vector))
                print(f'b_{i} = {self.input_signals[i].steeringVector}')
            self.input_signals = tuple(self.input_signals)



        elif self.systemtype == "CyclicIntegratorChain":

            self.A = np.zeros((self.N*self.M, self.N*self.M))
            mixingPi = np.zeros((self.N, self.M, self.M))
            H = hadamardMatrix(self.M)
            

            self.all_input_signal_frequencies = np.zeros(L)
            self.all_input_signal_frequencies[self.primary_signal_dimension] = self.input_frequency
            self.all_input_signal_amplitudes[self.primary_signal_dimension] = self.input_amplitude

            if N > 1:
                if L == 1:
                    # Start with the Pi_N, in the top right corner
                    mixingPi[-1] = beta * np.sqrt(M) * sum(np.outer(H[:,i],H[:,i]) for i in range(self.N-1))
                    self.A[ 0 : self.M, -self.M : self.M*self.N] = mixingPi[-1]

                    # Iterate just like before, down the first sub-diagonal, always summing over all but the k'th pi vector
                    for k in range(N-1):
                        mixingPi[k] = beta * np.sqrt(M)* sum(np.outer(H[:,i],H[:,i]) for i in range(self.N) if i != k)
                        self.A[(k+1)*self.M:(k+2)*self.M, (k)*self.M:(k+1)*self.M] = mixingPi[k]
                else:
                  raise NotImplemented
            else:
              mixingPi = [np.zeros((M,M))]
            # Define input signals:
            self.input_signals = []
            if self.L==1:
              selector = np.zeros(self.M)
              selector[0] = 1
              vector = np.zeros(self.M*self.N)
              vector[0:self.M*self.N] = beta * np.sqrt(M) * np.dot(np.vstack((np.outer(H[:,i],H[:,i]) for i in range(self.N))),
                                                                   selector.reshape(-1,1)).flatten()

              self.input_signals.append(system.Sin(self.sampling_period,
                                                   amplitude=self.input_amplitude,
                                                   frequency=self.input_frequency,
                                                   phase=self.input_phase,
                                                   steeringVector=vector))
            else:
              raise NotImplemented

        else:
            raise NotImplemented


        # Define input signals:
        self.input_signals = []
        self.all_input_signal_frequencies = np.zeros(L)
        self.all_input_signal_frequencies[self.primary_signal_dimension] = self.input_frequency
        allowed_signal_frequencies = self.input_frequency * (0.5**np.arange(2,3*M))
        for i in range(self.L):
            if i == self.primary_signal_dimension: continue
            k = np.random.randint(0,L-1)
            self.all_input_signal_frequencies[i] = allowed_signal_frequencies[k]
            self.all_input_signal_amplitudes[i] = input_amplitude

        # M x L selector matrix for the b-vectors
        if L == 2:
          selectorMatrix = ((1./np.sqrt(2)) * np.array([
                                [1, 0],
                                [0, 1],
                                [1, 0],
                                [0, 1]
                                      ]))
        elif L == M or L == 1:
          selectorMatrix = np.eye(M)

        # uPi,sPi,vhPi = np.linalg.svd(mixingPi[0])
        # vhPi[np.abs(vhPi) < 1e-16] = 0
        for i in range(self.L):
            vector = np.zeros(self.M*self.N)
            # Scale the input vector up by sqrt(number of channels per signal)
            vector[0:self.M] =  np.dot(mixingPi[0], selectorMatrix[:,i]) * np.sqrt(M/L) # self.beta * np.sqrt(M/L) * vhPi[i]
            self.input_signals.append(system.Sin(self.sampling_period,
                                                 amplitude=self.all_input_signal_amplitudes[i],
                                                 frequency=self.all_input_signal_frequencies[i],
                                                 phase=self.input_phase,
                                                 steeringVector=vector))
            print(f'b_{i} = {self.input_signals[i].steeringVector}')
        self.input_signals = tuple(self.input_signals)
        #print("A = \n%s\nb = \n%s" % (self.A, self.input_signals[self.primary_signal_dimension].steeringVector))

        self.c = np.eye(self.N * self.M)
        self.sys = system.System(A=self.A, c=self.c, b=self.input_signals[primary_signal_dimension].steeringVector)

        systemResponse = lambda f: np.dot(self.sys.frequencyResponse(f), self.sys.b)
        self.eta2_magnitude = np.max(np.abs(systemResponse(1./(2. * sampling_period * OSR)))**2)
        self.log("eta2_magnitude set to max(|G(s)b|^2) = {:.5e}".format(self.eta2_magnitude))
        print("eta2_magnitude set to max(|G(s)b|^2) = {:.5e}".format(self.eta2_magnitude))

        diagonalControl = True
        dither = True
        blockDiagonalControl = False
        blockDiagonalControl_with_DiagonalPart_and_Dither = False

        self.ctrlMixingMatrix = np.zeros((N*M , N*M))

        if diagonalControl:
            self.ctrlMixingMatrix = - self.kappa * self.beta * np.eye(self.N * self.M)

        elif blockDiagonalControl_with_DiagonalPart_and_Dither:
          lambd = 0.5
          self.ctrlMixingMatrix = ((np.random.randint(2,size=(self.N * self.M, self.N * self.M))*2 - 1) 
                                    * beta  * 0*1e-3 / (self.M**2 * (self.N**2 - self.N)))

          for k in range(N):
            self.ctrlMixingMatrix[k * self.M: (k+1) * self.M,
                                  k * self.M:(k+1) * self.M] = - beta*(np.sqrt(M) ** (k))*np.outer(H[:,0],H[:,0])
                                    # ((1.-lambd) * self.beta / ((M-1))
                                    #                              * (np.ones((M,M)) - np.eye(M))
                                    #                              + lambd * self.beta
                                    #                              * np.eye(self.M))

        elif blockDiagonalControl:
          # scaling = 
          for k in range(N):
            self.ctrlMixingMatrix[k * self.M: (k+1) * self.M,
                                  k * self.M:(k+1) * self.M] = (-self.beta
                                                                * np.outer(H[:,0],H[:,0]))
        else:
          raise NotImplemented

        if dither:
          self.ctrlMixingMatrix += (np.random.randint(2,size=(self.N * self.M, self.N * self.M))*2 - 1) * beta  * 1e1 / (self.M*self.N)**2

        # print("######################")
        # print("######################\n")
        # print(f'beta*sqrt(M) = {self.beta*np.sqrt(M)}')
        # for i in range(N*M):
        #   print(f'sum(B_{i}) = {sum(self.ctrlMixingMatrix[:,i])}')
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
      params = self.getParams()
      params_string = ''
      for key in params.keys():
        params_string = ''.join([params_string, f'{key}: {params[key]}\n'])

      with open(self.data_dir / f'{self.experiment_id}.params', 'w') as f:
        f.write(params_string)

      with open(self.data_dir / f'{self.experiment_id}.params.pkl', 'wb') as f:
        pkl.dump(params, f)

      with open(self.data_dir / f'{self.experiment_id}.log', 'w') as f:
        f.write(self.logstr)
        # file_path = self.data_dir / 'ExperimentRunner.pkl'
        # with file_path.open(mode='wb') as outfile:
        #     pkl.dump(self.__dict__, outfile)
        # self.log("ExperimentRunner saved at \"{}\"".format(file_path))
        # self.saveLog()


    def run_simulation(self):
        t = np.linspace(0,(self.size-1)*self.sampling_period, self.size)      
        self.sim_start_time = time.time()

        self.simulation_options = {'stateBound':(self.sampling_period * self.beta * self.kappa) / (1. - (self.sampling_period * self.beta / np.sqrt(self.M))),
                                   'stateBoundInputs': (self.sampling_period * self.beta * self.kappa) / (1. - (self.sampling_period * self.beta / np.sqrt(self.M))),
                                   'num_parallel_converters': self.M,
                                   'noise':[{'std':self.sigma2_thermal, 'steeringVector': self.beta*np.eye(self.N * self.M)[:,i]}  for i in range(self.M * self.N)]}
        
        initalState = np.random.rand(self.N*self.M)*1e-3
        # for k in range(self.N):
        #   initalState[k*self.M:self.M*(k+1)] = np.ones(self.M) * np.random.randint(2) * 2. - 1.
        sim = simulator.Simulator(self.sys, self.ctrl, options=self.simulation_options, initalState=initalState)
        self.result = sim.simulate(t, self.input_signals)

        self.sim_run_time = time.time() - self.sim_start_time
        self.log(self.result['log'])
        self.log("Simulation run time: {:.2f} seconds".format(self.sim_run_time))
        self.finished_simulation = True


    def run_reconstruction(self):
        t = np.linspace(0,(self.size-1)*self.sampling_period, self.size)      
        recon_time_start = time.time()
        self.eta2 = np.ones(self.M * self.N) * self.eta2_magnitude
        self.reconstruction_options = {'eta2':self.eta2,
                                       'sigmaU2':[1.]*self.M,
                                       'noise':[{'std':self.sigma2_reconst,
                                                 'steeringVector': self.beta*np.eye(self.N * self.M)[:,i], 'name':'noise_{}'.format(i)} for i in range(self.N * self.M)]}
        self.reconstruction = reconstruction.WienerFilter(t, self.sys, self.input_signals, self.reconstruction_options)
        tmp_estimates, recon_log = self.reconstruction.filter(self.ctrl)

        self.input_estimates = tmp_estimates[self.border:-self.border]
        self.recon_run_time = time.time() - recon_time_start
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

        # self.size = round(1./self.sampling_period)
        # self.data_dir = self.data_dir.parent / ('unitTest_' + self.data_dir.name)
        # if not self.data_dir.exists():
        #     self.data_dir.mkdir(parents=True)

        self.run_simulation()
        self.run_reconstruction()
        self.log(f'Size: {self.size}')
        self.log(f'Number of OOBs: {self.result["num_oob"]}')
        self.log(f'OOB Rate: {self.result["num_oob"]/self.size}')

        print()


        print(self.logstr)

    def getParams(self):
        input_steering_vectors = {f'b_{i}': self.input_signals[i].steeringVector for i in range(self.L)}
        params = {'M':self.M,
                  'N':self.N,
                  'L':self.L,
                  'beta':self.beta,
                  'sampling_period':self.sampling_period,
                  'primary_input_frequency':self.input_frequency,
                  'primary_input_amplitude':self.input_amplitude,
                  'eta2':self.eta2_magnitude,
                  'other_input_frequencies':self.all_input_signal_frequencies[1:],
                  'other_input_amplitudes':self.all_input_signal_amplitudes[1:],
                  'size': "{:e}".format(self.size),
                  'num_oob': self.result['num_oob'],
                  'oob_rate': self.result['num_oob'] / self.size,
                  'sigma2_thermal': self.sigma2_thermal,
                  'sigma2_reconst': self.sigma2_reconst}
        return {**params, **input_steering_vectors}


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
                              DATA_STORAGE_PATH,
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

    # s3_resource = boto3.resource('s3')
    # s3_file_name_prefix = uuid.uuid4().hex[:6]

    # runner.log("Saving results to S3")
    # runner.log("S3 file name: \"{}\"".format(''.join([s3_file_name_prefix, experiment_id])))
    runner.log(f'Saving results to "{DATA_STORAGE_PATH}"')
    runner.saveAll()
    with open(DATA_STORAGE_PATH / f'{experiment_id}_results.pkl', 'wb') as f:
      pkl.dump(runner, f, protocol=pkl.HIGHEST_PROTOCOL)

    # uploadTos3(
    #   s3_connection=s3_resource,
    #   bucket_name=BUCKET_NAME,
    #   file_name=''.join([s3_file_name_prefix,experiment_id,'_results.pkl']),
    #   obj=runner)

    # writeStringToS3(
    #   s3_connection=s3_resource,
    #   bucket_name=BUCKET_NAME,
    #   file_name=f'{s3_file_name_prefix}{experiment_id}.log',
    #   string=runner.logstr)

    # writeStringToS3(
    #   s3_connection=s3_resource,
    #   bucket_name=BUCKET_NAME,
    #   file_name=f'{s3_file_name_prefix}{experiment_id}.params',
    #   string=params_string)

    # uploadTos3(
    #   s3_connection=s3_resource,
    #   bucket_name=BUCKET_NAME,
    #   file_name=''.join([s3_file_name_prefix,experiment_id,'.params.pkl']),
    #   obj=params)


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
    arg_parser.add_argument("-f_sig", "--input_frequency", type=float, default=None)
    arg_parser.add_argument("-eta2", "--eta2_magnitude", type=float, default=1)
    arg_parser.add_argument("-kappa", type=float, default=1)
    arg_parser.add_argument("-OSR", type=int, default=16)
    arg_parser.add_argument("-systemtype", type=str, default="ParallelIntegratorChain")
    arg_parser.add_argument("-sig_dim", "--primary_signal_dimension", type=int, default=0)
    arg_parser.add_argument("-n_sim", "--num_periods_in_simulation", type=int, default=20)

    args = vars(arg_parser.parse_args())

    main(**args)