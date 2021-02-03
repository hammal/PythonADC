#!/home/olafurt/miniconda3/bin/python

###############################
#      Standard Packages      #
###############################
import argparse
import numpy as np
import scipy as sp
import pickle as pkl
from pathlib import Path
import time
import re

###############################
#         ADC Packages        #
###############################
import AnalogToDigital.system as system
import AnalogToDigital.simulator as simulator
import AnalogToDigital.reconstruction as reconstruction


from gram_schmidt import gram_schmidt

randomSeed1 = int(np.random.randint(2**32 - 1))
randomSeed2 = int(np.random.randint(2**32 - 1))
randomSeed3 = int(np.random.randint(2**32 - 1))
randomSeed4 = int(np.random.randint(2**32 - 1))


# Edit this path
# DATA_STORAGE_PATH = Path('/itet-stor/olafurt/net_scratch/adc_data')
DATA_STORAGE_PATH = Path('./unit_tests')

def hadamardMatrix(n):
    return sp.linalg.hadamard(n)/np.sqrt(n)


class ExperimentRunner():
    """ Class to handle running experiments"""

    def __init__(self,
                 experiment_id,
                 data_dir,
                 N=1,
                 M=1,
                 L=1,
                 input_amplitude=1,
                 input_frequency=None,
                 input_phase=0,
                 beta=6250,
                 sampling_period=8e-5,
                 primary_signal_dimension=0,
                 systemtype='ParallelIntegratorChains',
                 OSR=16,
                 eta2_magnitude=1,
                 kappa=1,
                 sigma_thermal=1e-6,
                 sigma_reconst=1e-6,
                 size = 1 << 14,
                 controller='diagonalController',
                 bitsPerControl=1,
                 delayChain = None,
                 leaky=False,
                 dither=False,
                 ditherScale=1e-3,
                 mismatch='',
                 beta_hat=6250,
                 beta_tilde=6250,
                 newInputVector=np.ones(1),
                 nonuniformNoise=False,
                 options={},
                 fullFeedback = False,
                 roll = False,
                 controlReferences=None,
                 subSample = 1,
                 reconstruct_with_all_inputs = False,
                 ctrlOffsets = None,
                 multiSystem = None,
                 ):

        print("Initializing Experiment | ID: %s" % experiment_id)
        self.experiment_id = experiment_id
        self.data_dir = Path(data_dir)
        self.M = M
        self.N = N
        self.L = L
        self.input_phase = input_phase
        self.input_amplitude = input_amplitude
        self.input_frequency = input_frequency
        if type(beta) is list: 
          self.beta = beta[0]
          self.betas = beta
        else:
          self.beta = beta
          self.betas = np.arange(N) * beta
        self.sampling_period = sampling_period
        self.primary_signal_dimension = primary_signal_dimension
        self.systemtype = systemtype
        self.OSR = OSR
        self.kappa = kappa
        self.sigma_thermal = sigma_thermal
        self.sigma_reconst = sigma_reconst
        self.leakyIntegrators = leaky
        self.size = size
        self.nonuniformNoise = nonuniformNoise

        self.beta_hat = beta_hat
        self.beta_tilde = beta_tilde

        self.controller = controller
        self.bitsPerControl = bitsPerControl

        if controlReferences is None:
          self.ControlReferences = np.zeros(M * N)
        else:
          self.ControlReferences = controlReferences

        self.border = np.int(self.size // 50)
        self.all_input_signal_amplitudes = np.zeros(L)
        self.all_input_signal_amplitudes[primary_signal_dimension] = input_amplitude

        self.logstr = ("{0}: EXPERIMENT LOG\n{0}: Experiment ID: {1}\n".format(time.strftime("%d/%m/%Y %H:%M:%S"), experiment_id))

        self.finished_simulation = False
        self.finished_reconstruction = False
        if not self.data_dir.exists():
            self.data_dir.mkdir(parents=True)

        self.reconstruct_with_all_inputs = reconstruct_with_all_inputs
        self.subSample = subSample
        self.mismatch = {}
        if isinstance(mismatch, str):
          if mismatch:
            affected, mismatchtype, mismatch_error = mismatch.split(',')
            self.mismatch[affected] = {mismatchtype: float(mismatch_error)}
        # assume dictionary
        else:
          for key in mismatch:
            self.mismatch[key] = mismatch[key]
  
        self.newInputVector = newInputVector
        self.options = options
        #################################################
        #     System and input signal specifications    #
        #################################################

        if self.primary_signal_dimension > self.M:
            self.log("Primary Signal Dimension cannot be larger than M, setting to 0 (first dim)")
            self.primary_signal_dimension = 0

        if self.input_frequency == None:
            self.input_frequency = 1./(self.sampling_period * 2 * self.OSR)
            self.log(f'Setting f_sig = f_s/(2*OSR) = {self.input_frequency}')
        if np.log2(self.M).is_integer():
          self.H = hadamardMatrix(self.M)
          print(self.H, self.M)
        else:
          print(f"No Hadamard matrix computed since M={self.M}")
          self.H = None

        # Define input signals:
        self.input_signals = []
        self.input_signals_mismatch = []
        self.all_input_signal_frequencies = np.zeros(L)
        self.all_input_signal_frequencies[self.primary_signal_dimension] = self.input_frequency
        allowed_signal_frequencies = self.input_frequency * (0.5**np.arange(1,3*self.M))
        for i in range(self.L):
            if i == self.primary_signal_dimension: continue
            k = np.random.randint(0,L-1)
            self.all_input_signal_frequencies[i] = allowed_signal_frequencies[i]
            self.all_input_signal_amplitudes[i] = self.input_amplitude
            print("frequency_{} = {}".format(i, allowed_signal_frequencies[i]))

        self.all_input_signal_amplitudes[self.primary_signal_dimension] = self.input_amplitude
        self.A_simulation = np.zeros((self.N*self.M, self.N*self.M))
        self.A_nominal = np.zeros((self.N*self.M, self.N*self.M))

        if self.systemtype == "BraidedChains" or self.systemtype == "BraidedChainsFewControls":
            print(f"System Type: {self.systemtype}")
            mixingPi = np.zeros((self.N-1, self.M, self.M))
            
            if N > 1:
                # L=1 means just one of M dimensions is used and there is
                # only one input signal => We scale up the input vector by sqrt(M)
                if L == 1:
                    for k in range(N-1):
                        print(k, self.H, N, M, self.betas)
                        if self.bitsPerControl > 0:
                          if self.betas:
                            mixingPi[k] = self.betas[k+1] * (np.outer(self.H[:,0],self.H[:,0]))
                          else:
                            mixingPi[k] = self.beta * (np.outer(self.H[:,0],self.H[:,0]))
                        else:
                          mixingPi[k] = self.beta * (np.outer(self.H[:,0],self.H[:,0]))
                        self.A_nominal[(k+1)*self.M:(k+2)*self.M, (k)*self.M:(k+1)*self.M] = mixingPi[k]
                        self.A_simulation[(k+1)*self.M:(k+2)*self.M, (k)*self.M:(k+1)*self.M] = mixingPi[k]
                # L=M means M input signals
                elif L == M:
                    for k in range(N-1):
                        mixingPi[k] = self.beta * np.eye(self.M)
                        self.A_nominal[(k+1)*self.M:(k+2)*self.M, (k)*self.M:(k+1)*self.M] = mixingPi[k]
                        self.A_simulation[(k+1)*self.M:(k+2)*self.M, (k)*self.M:(k+1)*self.M] = mixingPi[k]
                else:
                    # for k in range(N-1):
                    #     mixingPi[k] = self.beta * np.sqrt(M/L) * (sum(np.outer(H[:,i],H[:,i]) for i in range(self.L)))
                    #     self.A[(k+1)*self.M:(k+2)*self.M, (k)*self.M:(k+1)*self.M] = mixingPi[k]
                    for k in range(N-1):
                        mixingPi[k] = self.beta * np.sqrt(self.M) / np.sqrt(self.L) * (np.outer(self.H[:,0],self.H[:,0]))
                        self.A_nominal[(k+1)*self.M:(k+2)*self.M, (k)*self.M:(k+1)*self.M] = mixingPi[k]
                        self.A_simulation[(k+1)*self.M:(k+2)*self.M, (k)*self.M:(k+1)*self.M] = mixingPi[k]
            else:
              mixingPi = [np.zeros((M,M))]

            # Limit the low frequency gain of the filter
            # at approximately the thermal noise level
            self.rho = self.compute_rho()

            if 'A' in self.mismatch:
              # set seed
              np.random.seed(randomSeed1)
              print("Mismatch in A")
              A_mismatch = np.zeros_like(mixingPi)
              for key in self.mismatch['A']:
                if key == 'element':
                  faulty_component = np.random.randint(self.M)
                  print("Faulty component: %s" % (faulty_component,))
                  A_mismatch[1][faulty_component,faulty_component] = (np.random.randint(2)*2-1)*self.mismatch['A']['element']
                  print(A_mismatch[1])
                elif key == 'module':
                  A_mismatch[1][:self.M, :self.M] = (np.random.randint(2,size=(self.M,self.M))*2-1)*self.mismatch['A']['module']
                  print(A_mismatch[1])
                elif key =='system':
                  print("System mismatch")
                  for k in range(self.N - 1):
                    # A_mismatch[k] = (np.random.rand(self.M,self.M) * 2 - 1.) * self.mismatch['A']['system']
                    v1 = (self.H[:,0] * (np.random.rand(self.M) * 2 - 1) * self.mismatch['A']['system'])
                    v2 = (self.H[:,0] * (np.random.rand(self.M) * 2 - 1) * self.mismatch['A']['system'])
                    A_mismatch[k] =  self.beta * np.outer(v1,v2)

                    #(np.random.randint(2,size=(self.M,self.M))* 2 - 1)*self.mismatch['A']['system']

            if fullFeedback:
              for n in range(N):
                tmp = np.zeros((M,M))
                for i in range(M-1):
                  if self.betas:
                    tmp += - self.betas[n] * np.outer(self.H[:,i+1], self.H[:,i+1])
                  else:
                    tmp += - self.beta * np.outer(self.H[:,i+1], self.H[:,i+1])
                # for i in range(M):
                  # print(f"Full feedback Factor from {i}={self.rho * (2**i)}")
                  # tmp += - self.rho * (2**i) * np.outer(self.H[:,i], self.H[:,i])
                print(f"FullFeedback: {tmp}")
                self.A_nominal[n*M:(n+1)*M, n*M:(n+1)*M] = tmp
                self.A_simulation[n*M:(n+1)*M, n*M:(n+1)*M] = tmp
              # Normal local feedback
            else:
              self.A_nominal -= np.eye(N*M)*self.rho
              self.A_simulation -= np.eye(N*M)*self.rho

            if "A" in self.mismatch:
              for k in range(N-1):
                  self.A_simulation[(k+1)*self.M:(k+2)*self.M, (k)*self.M:(k+1)*self.M] = self.A_nominal[(k+1)*self.M:(k+2)*self.M, (k)*self.M:(k+1)*self.M] + A_mismatch[k]

            # Define input steeringVectors
            if L == 1:
              inputVectorMatrix = self.beta * self.H
            # elif L == 2:
              # The outer product sum results in a checkerboard matrix with 1/0 entries.
              # We pick input vectors from there and scale up by M (undoing the attenuation from the outer products)
              # inputVectorMatrix = self.beta * (M/2) * ( np.outer(H[:,0],H[:,0]) + np.outer(H[:,1],H[:,1]) ) * 2
            # elif L == M:
            #   inputVectorMatrix = self.beta * np.sqrt(M) * (np.hstack( (np.outer(H[:,i],H[:,i])[:,0].reshape(-1,1) for i in range(L)) ))
            else:
              inputVectorMatrix = self.beta / np.sqrt(L) * self.H

            print("Nominal InputVectorMatrix: {}".format(inputVectorMatrix[:,:self.L]))
            if 'inputMatrix' in self.mismatch and 'system' in self.mismatch['inputMatrix']:
              print("Mismatch in inputVector")
              np.random.seed(randomSeed2)
              inputVectorMatrixMismatch = inputVectorMatrix *  (1. + (np.random.rand(inputVectorMatrix.shape[0], inputVectorMatrix.shape[1])*2-1)*self.mismatch['inputMatrix']['system'])
              print("Simulated InputVectorMatrix: {}".format(inputVectorMatrixMismatch[:,:self.L]))
              

            for i in range(self.L):
                vector = np.zeros(self.M*self.N)
                vector[0:self.M] = inputVectorMatrix[:,i]
                self.input_signals.append(system.Sin(self.sampling_period,
                                                     amplitude=self.all_input_signal_amplitudes[i],
                                                     frequency=self.all_input_signal_frequencies[i],
                                                     phase=self.input_phase,#+ (np.pi/2)*i,
                                                     steeringVector=vector))
                if 'inputMatrix' in self.mismatch and 'system' in self.mismatch['inputMatrix']:
                  vector = np.zeros(self.M*self.N)
                  vector[0:self.M] = inputVectorMatrixMismatch[:,i]
                  self.input_signals_mismatch.append(system.Sin(self.sampling_period,
                                                     amplitude=self.all_input_signal_amplitudes[i],
                                                     frequency=self.all_input_signal_frequencies[i],
                                                     phase=self.input_phase,#+ (np.pi/2)*i,
                                                     steeringVector=vector))
                else:
                  self.input_signals_mismatch = self.input_signals
                print(f'b_{i} = {self.input_signals[i].steeringVector}')
            
            if reconstruct_with_all_inputs:
              for i in range(1, self.M):
                vector = np.zeros(self.M*self.N)
                vector[0:self.M] = inputVectorMatrix[:,i]
                self.input_signals.append(system.Sin(self.sampling_period,
                                                     amplitude=self.all_input_signal_amplitudes[0],
                                                     frequency=self.all_input_signal_frequencies[0],
                                                     phase=self.input_phase,#+ (np.pi/2)*i,
                                                     steeringVector=vector))
        
        elif self.systemtype == "CyclicHadamard":
            print("System Type: Cyclic Hadamard")
            # [   0                                mixingPi_1]
            # [mixingPi_2                              0     ]
            # [   0       mixingPi_3                   0     ]
            # [   .     .            .                 .     ]
            # [   .         .             .            .     ]
            # [   .                                    .     ]
            # [                                        0     ]
            # [   0                        mixingPi_N  0     ]


            # mixingPi_i = beta*(sqrt(M) / M)(  sum(h_j, h_j.T) for j != i  )

            mixingPi = np.zeros((self.N, self.M, self.M))

            if N > 1:
                # Iterate just like before, down the first sub-diagonal, always summing over all but the k'th pi vector
                for k in range(self.N):
                    mixingPi[k] = self.beta_tilde * sum(np.outer(self.H[:,i],self.H[:,i]) for i in range(self.M) if i != np.mod(k,M))

                self.rho = self.compute_rho()
                print("rho = {}".format(self.rho))
                self.A_nominal -= np.eye(N*M)*self.rho
                self.A_simulation -= np.eye(N*M)*self.rho

                A_mismatch = np.zeros_like(mixingPi)
                if 'A' in self.mismatch:
                  np.random.seed(randomSeed1)
                  for key in self.mismatch['A']:
                    if key == 'element':
                      faulty_component = self.random_coordinate(self.M)
                      print("Faulty component: %s" % (faulty_component,))
                      A_mismatch[1][faulty_component] = (np.random.randint(2)*2-1)*self.mismatch['A']['element']
                      print(A_mismatch[1])
                    elif key == 'module':
                      A_mismatch[1][:self.M, :self.M] = (np.random.rand(self.M,self.M)*2-1)*self.mismatch['A']['module']
                      print(A_mismatch[1])
                    elif key =='system':
                      for k in range(self.M):
                        A_mismatch[k] = (np.random.rand(self.M,self.M) * 2 - 1.) * self.mismatch['A']['system']
                        # (np.random.randint(2,size=(mixingPi[0].shape))* 2 - 1)*self.mismatch['A']['system']

                self.A_nominal[ 0 : self.M, -self.M : self.M*self.N] = mixingPi[0]
                self.A_simulation[ 0 : self.M, -self.M : self.M*self.N] = mixingPi[0] + np.multiply(mixingPi[0],A_mismatch[0])                
                for k in range(self.N-1):
                  self.A_nominal[(k+1)*self.M:(k+2)*self.M, (k)*self.M:(k+1)*self.M] = mixingPi[k+1]
                  self.A_simulation[(k+1)*self.M:(k+2)*self.M, (k)*self.M:(k+1)*self.M] = mixingPi[k+1] + np.multiply(mixingPi[k+1],A_mismatch[k+1])

                # vec = np.hstack((self.H[:,i] for i in np.append(np.arange(1,self.M),0)))
                # if np.any(np.abs(np.dot(self.A_nominal,vec))):
                #   print("Error in building cyclic system. A_nominal.dot(vec) != 0")
                #   print(np.dot(self.A_nominal,vec))
            else:
              mixingPi = [np.zeros((M,M))]
            # Define input signals:
            self.input_signals = []
            self.input_phases = []
            self.input_frequencies = []
            if self.L==1:
              # self.newInputVector = vec#self.H[:,0] # CASE II
              vector = np.zeros((self.M*self.N, self.L))
              for k in range(self.N):
                vector[k*self.M:(k+1)*self.M, 0] = M * self.beta_hat * np.dot(np.outer(self.H[:,np.mod(k,M)],self.H[:,np.mod(k,M)]), self.newInputVector)
              phase = 2*np.pi*np.random.random()
              self.input_phases.append(phase)
              self.input_frequencies.append(self.input_frequency)

              # self.input_signals.append(system.Constant(amplitude=self.input_amplitude,
              #                                           steeringVector=vector))
              # self.input_signals.append(system.Sin(self.sampling_period,
              #                                      amplitude=self.input_amplitude,
              #                                      frequency=self.input_frequency,
              #                                      phase=self.input_phase,
              #                                      steeringVector=vector))
              # print(f'b_{i} = {self.input_signals[i].steeringVector}')
              self.input_amplitude = [self.input_amplitude]

            elif self.L == self.M:
              vector = np.zeros((self.M*self.N, self.L))
            # else:
              # input_basis = self.options['randomInputBasis']
              # print("M signals, using random input basis")

              # pibasis = np.vstack((np.outer(self.H[:,np.mod(k,M)],self.H[:,np.mod(k,M)]) for k in range(M)))
              for k in range(self.L):
                vector[k*self.M:(k+1)*self.M, k] = np.sqrt(self.M) * self.beta_hat * self.H[:,k]# / np.sqrt(self.M)
                #np.dot(np.outer(self.H[:,np.mod(k,M)],self.H[:,np.mod(k,M)]),self.H[:,np.mod(k,M)])
                #np.dot(pibasis, input_basis[:,0])
                
                phase = 2*np.pi*np.random.random()
                self.input_phases.append(phase)

                bw = (1./(2*self.sampling_period*self.OSR))
                # frequency = np.random.random()*(49/100)*bw + bw/1000
                frequency = bw / (2** M) * (2 ** k)
                self.input_frequencies.append(frequency)
              self.input_amplitude = np.ones(self.L) * input_amplitude
              # self.input_amplitude = np.random.randn(self.L) * 1e0
              # self.input_amplitude /= np.linalg.norm(self.input_amplitude) / input_amplitude #/ np.sqrt(self.L)
              
              

            elif self.L == self.M // 2:
              vector = np.zeros(self.M*self.N, self.L)
                
            # else:
              # input_basis = self.options['randomInputBasis']
              # print("M signals, using random input basis")

              # pibasis = np.vstack((np.outer(self.H[:,np.mod(k,M)],self.H[:,np.mod(k,M)]) for k in range(M)))
              for k in range(self.L):
                vector[k*self.M:(k+1)*self.M, k] = self.beta_hat * self.H[:,k]
                vector[(k + 2)*self.M:(k+3)*self.M, k] = self.beta_hat * self.H[:,k+2]

                #np.dot(np.outer(self.H[:,np.mod(k,M)],self.H[:,np.mod(k,M)]),self.H[:,np.mod(k,M)])
                #np.dot(pibasis, input_basis[:,0])
                
                phase = 2*np.pi*np.random.random()
                self.input_phases.append(phase)

                bw = (1./(2*self.sampling_period*self.OSR))
                frequency = np.random.random()*(49/100)*bw + bw/100
                self.input_frequencies.append(frequency)

            # noiseVector = np.zeros(self.M*self.N)
            # noiseVector[0] = mixingPi[0,0,0]
            # self.input_signals.append(system.Noise(standardDeviation=1e-4, steeringVector=noiseVector))
            print("Nominal Inputvector: {}".format(vector))
            print("Input amplitudes: {}".format(self.input_amplitude))
            np.random.seed(randomSeed2)
            for l in range(self.L):

              self.input_signals.append(system.Sin(self.sampling_period,
                                                  amplitude=self.input_amplitude[l],
                                                  frequency=self.input_frequencies[l],
                                                  phase=self.input_phases[l],
                                                  steeringVector=vector[:,l]))
              if 'inputMatrix' in self.mismatch and 'system' in self.mismatch['inputMatrix']:
                vectorMismatch = vector[:,l] * (1. + (np.random.rand(vector.shape[0])*2-1) * self.mismatch['inputMatrix']['system'])
                print("Simulated inputVector: {}".format(vectorMismatch))
                self.input_signals_mismatch.append(system.Sin(self.sampling_period,
                                                  amplitude=self.input_amplitude[l],
                                                  frequency=self.input_frequencies[l],
                                                  phase=self.input_phases[l],
                                                  steeringVector=vectorMismatch))
              else:
                self.input_signals_mismatch = self.input_signals

        elif self.systemtype == "ParallelIntegratorChains":
            print("System Type: Parallel Integrator Chains")
            A_mismatch = np.zeros((self.N, self.M, self.M))
            if 'A' in self.mismatch:
              np.random.seed(randomSeed1)
              print("Mismatch in A")
              for key in self.mismatch['A']:
                if key == 'element':
                  faulty_component = np.random.randint(self.M)
                  print("Faulty component: %s" % (faulty_component,))
                  A_mismatch[1][faulty_component,faulty_component] = (np.random.randint(2)*2-1)*self.mismatch['A']['element']
                  print(A_mismatch[1])
                elif key == 'module':
                  A_mismatch[1][:self.M, :self.M] = (np.random.randint(2,size=(self.M,self.M))*2-1)*self.mismatch['A']['module']
                  print(A_mismatch[1])
                elif key =='system':
                  print("System mismatch")
                  for k in range(self.N):
                    A_mismatch[k] = (np.random.rand(self.M,self.M) * 2 - 1.) * self.mismatch['A']['system']
                    #(np.random.randint(2,size=(self.M,self.M))* 2 - 1)*self.mismatch['A']['system']

            self.rho = self.compute_rho()
            self.A_nominal -= np.eye(N*M)*self.rho
            self.A_simulation -= np.eye(N*M)*self.rho


            for k in range(N-1):
                self.A_simulation[(k+1)*self.M:(k+2)*self.M, (k)*self.M:(k+1)*self.M] = self.beta * np.eye(self.M) + np.multiply(self.beta*np.eye(self.M),A_mismatch[k])
                self.A_nominal[(k+1)*self.M:(k+2)*self.M, (k)*self.M:(k+1)*self.M] = self.beta * np.eye(self.M)


            
            vector = np.zeros((self.M * self.N, self.L))
            for l in range(self.L):
              rem = self.M - self.L + 1
              vector[l * rem: (l+1)*rem,l] = np.ones(rem) * self.beta
            
            print("Nominal Inputvector: {}".format(vector))

            if 'inputMatrix' in self.mismatch and 'system' in self.mismatch['inputMatrix']:
                np.random.seed(randomSeed2)
                print("Mismatch in inputVector")
                vectorMismatch = vector * ( 1. + (np.random.rand(vector.shape[0], vector.shape[1])*2-1)*self.mismatch['inputMatrix']['system'])
                print("Simulated inputVector: {}".format(vectorMismatch))

            for l in range(self.L):
              self.input_signals.append(system.Sin(self.sampling_period,
                                                     amplitude=self.all_input_signal_amplitudes[l],
                                                     frequency=self.all_input_signal_frequencies[l],
                                                     phase=self.input_phase,#+ (np.pi/2)*i,
                                                     steeringVector=vector[:,l]))
              if 'inputMatrix' in self.mismatch and 'system' in self.mismatch['inputMatrix']:
                self.input_signals_mismatch.append(system.Sin(self.sampling_period,
                                                     amplitude=self.all_input_signal_amplitudes[l],
                                                     frequency=self.all_input_signal_frequencies[l],
                                                     phase=self.input_phase,#+ (np.pi/2)*i,
                                                     steeringVector=vectorMismatch[:,l]))
              else:
                self.input_signals_mismatch = self.input_signals
              print(f'b_{i} = {self.input_signals[i].steeringVector}')

            # if self.L == 1:
            #   vector = np.zeros(self.M*self.N)
            #   vector[0:self.M] = self.beta_hat * self.newInputVector
            #   self.input_signals.append(system.Sin(self.sampling_period,
            #                                        amplitude=self.all_input_signal_amplitudes[i],
            #                                        frequency=self.all_input_signal_frequencies[i],
            #                                        phase=self.input_phase,#+ (np.pi/2)*i,
            #                                        steeringVector=vector))
            # elif self.L == self.M:
            #   for i in range(self.L):
            #     vector = np.zeros(self.M*self.N)
            #     vector[i] = self.beta
            #     self.input_signals.append(system.Sin(self.sampling_period,
            #                                          amplitude=self.all_input_signal_amplitudes[i],
            #                                          frequency=self.all_input_signal_frequencies[i],
            #                                          phase=self.input_phase,#+ (np.pi/2)*i,
            #                                          steeringVector=vector))

            
            # noiseVector = np.zeros(self.M*self.N)
            # noiseVector[0] = self.beta
            # self.input_signals.append(system.Noise(standardDeviation=1e-4, steeringVector=noiseVector))
        elif self.systemtype == "CyclicGramSchmidt":
            print("System Type: Cyclic Gram-Schmidt")
            # [   0                                mixingPi_1]
            # [mixingPi_2                              0     ]
            # [   0       mixingPi_3                   0     ]
            # [   .     .            .                 .     ]
            # [   .         .             .            .     ]
            # [   .                                    .     ]
            # [                                        0     ]
            # [   0                        mixingPi_N  0     ]

            mixingPi = np.zeros((self.N, self.M, self.M))

            self.GramSchmidtBasis = gram_schmidt(np.random.randn(self.M,self.M))

            if N > 1:
                # Iterate just like before, down the first sub-diagonal, always summing over all but the k'th pi vector
                for k in range(self.M):
                    mixingPi[k] = self.beta * sum(np.outer(self.GramSchmidtBasis[:,i], self.GramSchmidtBasis[:,i]) for i in range(self.N) if i != k)

                self.A[ 0 : self.M, -self.M : self.M*self.N] = mixingPi[0]
                for k in range(self.M-1):
                  self.A[(k+1)*self.M:(k+2)*self.M, (k)*self.M:(k+1)*self.M] = mixingPi[k+1]
                # print(self.A)

                vec = np.hstack((self.GramSchmidtBasis[:,i] for i in np.append(np.arange(1,self.M),0)))
                if np.any(np.abs(np.dot(self.A/self.beta,vec))):
                  print("Error in building cyclic system. A.dot(vec) != 0")
                  print(np.dot(self.A,vec))
            else:
              mixingPi = [np.zeros((M,M))]

            self.rho = self.compute_rho()
            self.A -= np.eye(N*M)*self.rho

            # Define input signals:
            self.input_signals = []
            if self.L==1:
              vector = np.zeros(self.M*self.N)
              for k in range(self.M):
                vector[k*self.M:(k+1)*self.M] = self.beta * self.GramSchmidtBasis[:,k]

              self.input_signals.append(system.Sin(self.sampling_period,
                                                   amplitude=self.input_amplitude,
                                                   frequency=self.input_frequency,
                                                   phase=self.input_phase,
                                                   steeringVector=vector))
              # system.Constant(amplitude=0,
              #                                           steeringVector=vector))

            else:
              pass
        elif self.systemtype == 'StandardBasis':
            print("System Type: Cyclic Standard Basis")
            # [   0                                mixingPi_1]
            # [mixingPi_2                              0     ]
            # [   0       mixingPi_3                   0     ]
            # [   .     .            .                 .     ]
            # [   .         .             .            .     ]
            # [   .                                    .     ]
            # [                                        0     ]
            # [   0                        mixingPi_N  0     ]

            mixingPi = np.zeros((self.N, self.M, self.M))

            basis = np.eye(self.M)

            if N > 1:
                # Iterate just like before, down the first sub-diagonal, always summing over all but the k'th pi vector
                for k in range(self.M):
                    mixingPi[k] = self.beta * sum(np.outer(basis[:,i], basis[:,i]) for i in range(self.N) if i != k)


                A_mismatch = np.zeros_like(mixingPi)
                if 'A' in self.mismatch:
                  for key in self.mismatch['A']:
                    if key == 'element':
                      tmpidx = np.random.randint(self.M-1)
                      tmparr = np.arange(1,self.M)
                      faulty_component = tmparr[tmpidx]
                      print("Faulty component: %s" % (faulty_component,))
                      A_mismatch[1][faulty_component, faulty_component] = (np.random.randint(2)*2-1)*self.mismatch['A']['element']
                      print(A_mismatch[1])
                    elif key == 'module':
                      A_mismatch[1][:self.M, :self.M] = (np.random.randint(2,size=(self.M,self.M))*2-1)*self.mismatch['A']['module']
                      print(A_mismatch[1])
                    elif key =='system':
                      for k in range(self.M):
                        A_mismatch[k] = (np.random.randint(2,size=(mixingPi[0].shape))* 2 - 1)*self.mismatch['A']['system']

                self.A_simulation[ 0 : self.M, -self.M : self.M*self.N] = mixingPi[0] + np.multiply(mixingPi[0],A_mismatch[0])
                self.A_nominal[ 0 : self.M, -self.M : self.M*self.N] = mixingPi[0]
                for k in range(self.M-1):
                  self.A_simulation[(k+1)*self.M:(k+2)*self.M, (k)*self.M:(k+1)*self.M] = mixingPi[k+1] + np.multiply(mixingPi[k+1],A_mismatch[k+1])
                  self.A_nominal[(k+1)*self.M:(k+2)*self.M, (k)*self.M:(k+1)*self.M] = mixingPi[k+1]
                # print(self.A)

                vec = np.hstack((basis[:,i] for i in np.append(np.arange(1,self.M),0)))
                if np.any(np.abs(np.dot(self.A_nominal/self.beta,vec))):
                  print("Error in building cyclic system. A.dot(vec) != 0")
                  print(np.dot(self.A,vec))
            else:
              mixingPi = [np.zeros((M,M))]

            self.rho = self.compute_rho()
            self.A -= np.eye(N*M)*self.rho

            # Define input signals:
            self.input_signals = []
            if self.L==1:
              vector = np.zeros(self.M*self.N)
              for k in range(self.M):
                vector[k*self.M:(k+1)*self.M] = self.beta * basis[:,k]

              self.input_signals.append(system.Sin(self.sampling_period,
                                                   amplitude=self.input_amplitude,
                                                   frequency=self.input_frequency,
                                                   phase=self.input_phase,
                                                   steeringVector=vector))

            # noiseVector = np.zeros(self.M*self.N)
            # noiseVector[0] = 1
            # self.input_signals.append(system.Noise(standardDeviation=1e-3, steeringVector=noiseVector))
        else:
            raise Exception("Invalid system type: '{}'".format(self.systemtype))

        self.input_signals = tuple(self.input_signals)
        self.c = np.eye(self.N * self.M)
        self.sys_nominal = system.System(A=self.A_nominal, c=self.c, b=self.input_signals[primary_signal_dimension].steeringVector)
        self.sys_simulation = system.System(A=self.A_simulation, c=self.c, b=self.input_signals[primary_signal_dimension].steeringVector)

        self.eta2_magnitude = self.compute_eta2()
        print("eta2_magnitude = {:.5e}".format(self.eta2_magnitude))
        print("A_nominal = \n{}\n".format(self.A_nominal))
        if mismatch:
          print("A_simulation = \n{}\n".format(self.A_simulation))
          print("Difference: \n{}\n".format(self.A_nominal - self.A_simulation))

        #################################################
        #           Controller specification            #
        #################################################
        self.dither = dither
        self.ditherScale = ditherScale
        """
          The subspace controller is only implemented for L=1 signal right now.
        """
        self.ctrlInputMatrix = np.zeros((self.N*self.M,self.N*self.M))
        ctrlMismatch = np.zeros_like(self.ctrlInputMatrix)
        self.ctrlObservationMatrix = np.zeros((self.N*self.M, self.N * self.M))
        ditherMatrix = np.zeros_like(self.ctrlInputMatrix)  
            
        
        if controller == 'subspaceController':
          print("Using subspaceController")
          print("Constructing controller for system: {}".format(self.systemtype))
          if self.systemtype == 'BraidedChains':
            self.ctrlObservationMatrix = np.zeros((self.N * self.M, self.N * self.M))
            self.ctrlInputMatrix = np.zeros((self.N*self.M, self.M * self.N))
            ctrlMismatch = np.zeros_like(self.ctrlInputMatrix)
            ditherMatrix = np.zeros_like(self.ctrlInputMatrix)  
            # if L>1:
            #   raise "Multi-Bit controller not implemented for L>1 input signals"
            if dither:
              np.random.seed(randomSeed3)
              for l in range(self.L):
                for m in range(self.N):
                  for n in range(self.N):
                    # self.ctrlInputMatrix[(m) * self.M:(m+1)*self.M, n + self.N * l] = (np.random.randint(2) * 2 - 1) * self.beta * self.kappa / (self.L * self.M) * self.ditherScale * self.H[:,l]
                    ditherMatrix =  (np.random.randint(2,size=(self.N*self.M, self.M * self.N)) * 2 - 1)* (self.beta * self.kappa / ((self.N * self.M)**2) *  self.ditherScale)
              # self.ctrlInputMatrix =  (np.random.rand(self.N*self.M, self.L * self.N) * 2 - 1)* (self.beta * self.kappa / (self.N * self.M) *  self.ditherScale)
            
            for i in range(N):
              for m in range(M):
                self.ctrlInputMatrix[i*M:(i+1)*M,i*M + m] = - self.kappa * self.H[:,0] #/ M
            self.nominalCtrlInputMatrix = np.copy(self.ctrlInputMatrix)
              
            self.ctrlObservationMatrix = -np.dot(self.ctrlInputMatrix, np.diag(1/(np.linalg.norm(self.ctrlInputMatrix, ord=2, axis=0)))).transpose()
            self.ctrlObservationMatrix /= np.sqrt(M)
            # self.ctrlObservationMatrix = -self.ctrlInputMatrix.transpose() / self.beta / self.kappa  / np.sqrt(M)
            # print(f"CtrlObservation Matrix: {self.ctrlObservationMatrix}")
            # print(self.ctrlInputMatrix)

          if self.systemtype == 'BraidedChainsFewControls':
            self.ctrlObservationMatrix = np.zeros((self.N * self.L, self.N * self.M))
            self.ctrlInputMatrix = np.zeros((self.N*self.M, self.L * self.N))
            ctrlMismatch = np.zeros_like(self.ctrlInputMatrix)
            ditherMatrix = np.zeros_like(self.ctrlInputMatrix)  
            # if L>1:
            #   raise "Multi-Bit controller not implemented for L>1 input signals"
            if dither:
              np.random.seed(randomSeed3)
              for l in range(self.L):
                for m in range(self.N):
                  for n in range(self.N):
                    # self.ctrlInputMatrix[(m) * self.M:(m+1)*self.M, n + self.N * l] = (np.random.randint(2) * 2 - 1) * self.beta * self.kappa / (self.L * self.M) * self.ditherScale * self.H[:,l]
                    ditherMatrix =  (np.random.randint(2,size=(self.N*self.M, self.L * self.N)) * 2 - 1)* ( self.kappa / ((self.N * self.M)**2) *  self.ditherScale)
              # self.ctrlInputMatrix =  (np.random.rand(self.N*self.M, self.L * self.N) * 2 - 1)* (self.beta * self.kappa / (self.N * self.M) *  self.ditherScale)
            
            if L==1:
              for i in range(N):
                self.ctrlInputMatrix[i*M:(i+1)*M,i] = - self.kappa * self.H[:,0]
              self.nominalCtrlInputMatrix = np.copy(self.ctrlInputMatrix)
            else:
              for l in range(self.L):
                for n in range(self.N):
                  self.ctrlInputMatrix[n*M:(n+1)*M,n + l * N] = - self.beta * self.kappa * self.H[:,l]
              self.nominalCtrlInputMatrix = np.copy(self.ctrlInputMatrix)
            self.ctrlObservationMatrix = -np.dot(self.ctrlInputMatrix, np.diag(1/(np.linalg.norm(self.ctrlInputMatrix, ord=2, axis=0)))).transpose()
            # self.ctrlObservationMatrix = -self.ctrlInputMatrix.transpose() / self.beta / self.kappa  / np.sqrt(M)
            print(f"CtrlObservation Matrix: {self.ctrlObservationMatrix}")
          elif self.systemtype == 'CyclicHadamard':
            """ Control mixing matrix for the cyclic parallel system:
                        [  H^T   0     ...   0  ]
                        [  0    H^T    ...   0  ]
                        [  .        .        .  ]
               beta * ( [  .           .     .  ] )
                        [  .              .  0  ]
                        [  0                H^T ]

            """           
            
            if dither:
              np.random.seed(randomSeed3)
              ditherMatrix = (np.random.randint(2,size=(self.N*self.M,self.N*self.M ))*2 - 1) * ( self.kappa / ((self.N * self.M)**2) *  self.ditherScale)
            
            for k in range(self.M):
              self.ctrlInputMatrix[k*self.M:(k+1)*self.M, k*self.M:(k+1)*self.M] = - self.beta * self.H.transpose() * self.kappa# / np.sqrt(2)
              self.ctrlObservationMatrix[k*self.M:(k+1)*self.M, k*self.M:(k+1)*self.M] = self.H / np.sqrt(M)


            # self.ctrlObservationMatrix = np.dot(self.ctrlInputMatrix, np.diag(1/(np.linalg.norm(self.ctrlInputMatrix, ord=2, axis=0)))).transpose()
            print("ctrlObservationMatrix: %s" % (self.ctrlObservationMatrix,))
            self.nominalCtrlInputMatrix = np.copy(self.ctrlInputMatrix)

            # for k in range(self.N):
            #   self.ctrlMixingMatrix[k*self.M:(k+1)*self.M, k*(self.M-1):(k+1)*(self.M-1)] = - beta * np.delete(H,k, axis=1)
          elif self.systemtype == 'CyclicGramSchmidt':
            if dither:
               ditherMatrix = (np.random.randint(2,size=(self.N*self.M,self.N*self.M ))*2 - 1) * self.kappa / ((self.N * self.M **2)) * self.ditherScale
            for k in range(self.M):
              self.ctrlInputMatrix[k*self.M:(k+1)*self.M, k*self.M:(k+1)*self.M] = - self.beta * self.GramSchmidtBasis.transpose()

            self.ctrlObservationMatrix = np.dot(self.ctrlInputMatrix, np.diag(1/(np.linalg.norm(self.ctrlInputMatrix, ord=2, axis=0)))).transpose()
            self.nominalCtrlInputMatrix = np.copy(self.ctrlInputMatrix)
          elif self.systemtype == 'StandardBasis':
            """ Control mixing matrix for the cyclic parallel system:
                        [  H    0    ...      0 ]
                        [  0    H    ...      0 ]
                        [  .        .         . ]
               beta * ( [  .            .     . ] )
                        [  .               .  0 ]
                        [  0                  H ]

            """

            if dither:
              ditherMatrix = (np.random.randint(2,size=(self.N*self.M,self.N*self.M ))*2 - 1) * self.kappa / ((self.M * self.N) ** 2) *  self.ditherScale

            for k in range(self.M):
              self.ctrlInputMatrix[k*self.M:(k+1)*self.M, k*self.M:(k+1)*self.M] = - self.beta * np.eye(self.M)

            self.ctrlObservationMatrix = np.dot(self.ctrlInputMatrix, np.diag(1/(np.linalg.norm(self.ctrlInputMatrix, ord=2, axis=0)))).transpose()
            self.nominalCtrlInputMatrix = np.copy(self.ctrlInputMatrix)

          if 'controlInputMatrix' in self.mismatch:
            np.random.seed(randomSeed4)
            for key in self.mismatch['controlInputMatrix']:
                if key == 'element':
                  faulty_component = self.random_coordinate(self.M)
                  print(f"Single faulty component: {faulty_component}")
                  ctrlMismatch[faulty_component] = (np.random.randint(2)*2-1)*self.mismatch['controlInputMatrix']['element']

                elif key == 'module':
                  print(f"First module faulty")
                  # ctrlMismatch[:self.M, :self.M] = (np.random.randint(2,size=(self.M,self.M))*2-1)*self.mismatch['controlInputMatrix']['module']
                  ctrlMismatch[:self.M, :self.N] = (np.random.rand(self.M,self.N) * 2 - 1.)*self.mismatch['controlInputMatrix']['module']

                elif key =='system':
                  print("Entire system faulty")
                  # ctrlMismatch = (np.random.randint(2,size=(self.ctrlInputMatrix.shape))* 2 - 1)*self.mismatch['controlInputMatrix']['system']
                  ctrlMismatch = (np.random.rand(self.ctrlInputMatrix.shape[0], self.ctrlInputMatrix.shape[1])* 2 - 1.) * self.mismatch['controlInputMatrix']['system']
                  # ctrlMismatch = 

        elif controller == 'ditherController':
            print("Using ditherController")
            
            self.ctrlInputMatrix = np.zeros((self.M * self.N, self.M * self.N + dither))
            np.random.seed(randomSeed3)
            # ditherMatrix = (np.random.randint(2,size=(self.N * self.M, self.N * self.M))*2 - 1) * self.beta * self.kappa / (self.N * self.M) *  self.ditherScale
            self.ctrlInputMatrix[:,self.M * self.N:] = (np.random.randn(self.N * self.M, dither)) * self.kappa / (self.N * self.M * dither)  *  self.ditherScale
            
            ditherMatrix = np.zeros_like(self.ctrlInputMatrix)

            # nondithered = np.zeros_like(self.ctrlInputMatrix)
            for i in range(self.M*self.N):
              self.ctrlInputMatrix[i,i] = - self.kappa #* np.sqrt(2)
              # nondithered[i,i] = -self.kappa * self.beta #* np.sqrt(2)
            self.nominalCtrlInputMatrix = np.copy(self.ctrlInputMatrix)
            
            self.ctrlObservationMatrix = np.zeros_like(self.ctrlInputMatrix) #/ np.sqrt(M)#-np.dot(nondithered, np.diag(1/(np.linalg.norm(nondithered, ord=2, axis=0)))).transpose()
            for i in range(self.M*self.N):
              self.ctrlObservationMatrix[i,i] = 1.

            self.ctrlObservationMatrix = self.ctrlObservationMatrix.transpose()


            
            print("control observation matrix: \n{}\n".format(self.ctrlObservationMatrix))

            ctrlMismatch = np.zeros_like(self.ctrlInputMatrix)
            if 'controlInputMatrix' in self.mismatch:
              np.random.seed(randomSeed4)
              for key in self.mismatch['controlInputMatrix']:
                if key == 'element':
                  coordinate = np.random.randint(self.M)
                  faulty_component = (coordinate, coordinate)
                  print(f"Single faulty component: ({faulty_component})")
                  ctrlMismatch[faulty_component] = (np.random.randint(2)*2-1)*self.mismatch['controlInputMatrix']['element']

                elif key == 'module':
                  print(f"First module faulty")
                  for i in range(self.M):
                    # ctrlMismatch[i,i] = (np.random.randint(2)*2-1)*self.mismatch['controlInputMatrix']['module']
                    ctrlMismatch[:self.M, :self.M] = (np.random.rand(self.M,self.M) * 2. - 1.)*self.mismatch['controlInputMatrix']['module']

                elif key =='system':
                  print("Entire system faulty")
                  # ctrlMismatch = (np.random.randint(2,size=(self.ctrlInputMatrix.shape))* 2 - 1)*self.mismatch['controlInputMatrix']['system']
                  ctrlMismatch = (np.random.rand(self.ctrlInputMatrix.shape[0], self.ctrlInputMatrix.shape[1])* 2. - 1.)*self.mismatch['controlInputMatrix']['system']

              print("Control Mismatch: \n{}\n".format(ctrlMismatch))
            # if self.mismatch and 'controlInputMatrix' in self.robustness:
            #   coordinate = np.random.randint(self.M)
            #   faulty_component = (coordinate,coordinate)
            #   print(f"Single faulty component: {faulty_component}")
            #   ctrlMismatch = np.zeros_like(self.ctrlInputMatrix)
            #   ctrlMismatch[faulty_component] = (np.random.randint(2)*2-1)*self.robustness['controlInputMatrix']
            #   self.ctrlInputMatrix = self.ctrlInputMatrix + np.multiply(ctrlMismatch, self.ctrlInputMatrix)

        elif controller == 'diagonalController':
            print("Using diagonalController")
            
            if dither:
              np.random.seed(randomSeed3)
              # ditherMatrix = (np.random.randint(2,size=(self.N * self.M, self.N * self.M))*2 - 1) * self.beta * self.kappa / (self.N * self.M) *  self.ditherScale

              # ditherMatrix = sp.linalg.block_diag(*[np.random.randn(self.M, self.M) * self.kappa / (self.M * self.M) *  self.ditherScale for n in range(self.N)]  )
              ditherMatrix = (np.random.randn(self.N * self.M, self.N * self.M)) * self.kappa / ((self.N * self.M)**2) *  self.ditherScale
              ditherMatrix = ditherMatrix - np.diag(np.diag(ditherMatrix))
            
            # nondithered = np.zeros_like(self.ctrlInputMatrix)
            for i in range(self.M*self.N):
              self.ctrlInputMatrix[i,i] = - self.kappa #* np.sqrt(2)
              # nondithered[i,i] = -self.kappa * self.beta #* np.sqrt(2)
            self.nominalCtrlInputMatrix = np.copy(self.ctrlInputMatrix)
            
            self.ctrlObservationMatrix = np.eye(self.ctrlInputMatrix.shape[0]) #/ np.sqrt(M)#-np.dot(nondithered, np.diag(1/(np.linalg.norm(nondithered, ord=2, axis=0)))).transpose()
            
            print("control observation matrix: \n{}\n".format(self.ctrlObservationMatrix))

            if 'controlInputMatrix' in self.mismatch:
              np.random.seed(randomSeed4)
              for key in self.mismatch['controlInputMatrix']:
                if key == 'element':
                  coordinate = np.random.randint(self.M)
                  faulty_component = (coordinate, coordinate)
                  print(f"Single faulty component: ({faulty_component})")
                  ctrlMismatch[faulty_component] = (np.random.randint(2)*2-1)*self.mismatch['controlInputMatrix']['element']

                elif key == 'module':
                  print(f"First module faulty")
                  for i in range(self.M):
                    # ctrlMismatch[i,i] = (np.random.randint(2)*2-1)*self.mismatch['controlInputMatrix']['module']
                    ctrlMismatch[:self.M, :self.M] = (np.random.rand(self.M,self.M) * 2. - 1.)*self.mismatch['controlInputMatrix']['module']

                elif key =='system':
                  print("Entire system faulty")
                  # ctrlMismatch = (np.random.randint(2,size=(self.ctrlInputMatrix.shape))* 2 - 1)*self.mismatch['controlInputMatrix']['system']
                  ctrlMismatch = (np.random.rand(self.ctrlInputMatrix.shape[0], self.ctrlInputMatrix.shape[0])* 2. - 1.)*self.mismatch['controlInputMatrix']['system']

              print("Control Mismatch: \n{}\n".format(ctrlMismatch))
            # if self.mismatch and 'controlInputMatrix' in self.robustness:
            #   coordinate = np.random.randint(self.M)
            #   faulty_component = (coordinate,coordinate)
            #   print(f"Single faulty component: {faulty_component}")
            #   ctrlMismatch = np.zeros_like(self.ctrlInputMatrix)
            #   ctrlMismatch[faulty_component] = (np.random.randint(2)*2-1)*self.robustness['controlInputMatrix']
            #   self.ctrlInputMatrix = self.ctrlInputMatrix + np.multiply(ctrlMismatch, self.ctrlInputMatrix)
        elif controller == 'diagonalScaled':
            self.ctrlInputMatrix = np.zeros((N*M,N*M))
            if dither:
              ditherMatrix = (np.random.randint(2,size=(self.N * self.M, self.N * self.M))*2 - 1) * self.beta * self.kappa *  self.ditherScale / (self.M * self.N)
            scale1 = [1., 2./(self.N), 3./(self.N), 1./self.N]
            scale2 = [1., 2./(self.N), 1./self.N, 3./(self.N)]
            for i in range(self.M*self.N):
              if i < self.M or (i > 2*self.M-1 and i < 3*self.M):
                self.ctrlInputMatrix[i,i] = np.sqrt(2)*self.kappa * self.beta * scale1[np.mod(i,4)]
              else:
                self.ctrlInputMatrix[i,i] = np.sqrt(2)*self.kappa * self.beta * scale2[np.mod(i,4)]

            self.nominalCtrlInputMatrix = np.copy(self.ctrlInputMatrix)

            ctrlMismatch = np.zeros_like(self.ctrlInputMatrix)
            if 'controlInputMatrix' in self.mismatch:
              for key in self.mismatch:
                if key == 'element':
                  coordinate = np.random.randint(self.M)
                  faulty_component = (coordinate, coordinate)
                  print(f"Single faulty component: ({faulty_component})")
                  ctrlMismatch[faulty_component] = (np.random.randint(2)*2-1)*self.mismatch['controlInputMatrix']['element']

                elif key == 'module':
                  print(f"First module faulty")
                  for i in range(self.M):
                    ctrlMismatch[i,i] = (np.random.randint(2)*2-1)*self.mismatch['controlInputMatrix']['module']

                elif key =='system':
                  print("Entire system faulty")
                  ctrlMismatch = (np.random.randint(2,size=(self.ctrlInputMatrix.shape))* 2 - 1)*self.mismatch['controlInputMatrix']['system']
            # if self.mismatch and 'controlInputMatrix' in self.robustness:
            #   coordinate = np.random.randint(self.M)
            #   faulty_component = (coordinate,coordinate)
            #   print(f"Single faulty component: {faulty_component}")
            #   ctrlMismatch = np.zeros_like(self.ctrlInputMatrix)
            #   ctrlMismatch[faulty_component] = (np.random.randint(2)*2-1)*self.robustness['controlInputMatrix']
            #   self.ctrlInputMatrix = self.ctrlInputMatrix + np.multiply(ctrlMismatch, self.ctrlInputMatrix)

            self.ctrlObservationMatrix = np.dot(self.nominalCtrlInputMatrix, np.diag(1/(np.linalg.norm(self.nominalCtrlInputMatrix, ord=2, axis=0)))).transpose()
            # print(f"CTRL {self.ctrlObservationMatrix}")
            # self.ctrlObservationMatrix = self.ctrlObservationMatrix * np.sqrt(M)
            # print(f"CTRL {self.ctrlObservationMatrix}")
        # Analog Dithering
        # self.ctrlInputMatrix += ditherMatrix
        # Digital Dithering
        self.nominalCtrlInputMatrix = self.ctrlInputMatrix + ditherMatrix
        self.ctrlInputMatrix = self.ctrlInputMatrix + np.multiply(ctrlMismatch, self.ctrlInputMatrix) + ditherMatrix
        if ctrlOffsets is None:
          offsets = np.zeros(self.nominalCtrlInputMatrix.shape[1])
        else:
          offsets = ctrlOffsets

        self.ctrlOptions = {
            'bitsPerControl':bitsPerControl,
            'projectionMatrix': self.ctrlObservationMatrix,
            'nominalCtrlInputMatrix':self.nominalCtrlInputMatrix,
            'bound':1,
            'offsets': offsets,
            'references': self.ControlReferences,
        }

        if roll:
          self.ctrlOptions["roll"] = self.rollFunction
          # print(" Test Rolling ")
          # for i in range(3):
          #   print(self.rollFunction(i))
          #   print(np.dot(self.rollFunction(i), np.arange(M * N * M * N).reshape((M*N, M*N))))
          # exit(1)


        if delayChain:
          self.ctrlOptions["delayChain"] = delayChain
          tmp = np.zeros_like(self.ctrlObservationMatrix)
          for i in range(N):
              for m in range(M):
                tmp[i*M:(i+1)*M,i*M + m] = self.H[:,0]
          self.ctrlOptions["projectionMatrix"] = tmp.transpose()
          self.delayChain = True
        else:
          self.delayChain = False

        if controller == 'ditherController':
          self.ctrlOptions["dither"] = dither


        self.ctrl = system.Control(self.ctrlInputMatrix, self.size, options=self.ctrlOptions)
        self.defineSimulationNoise()
        # self.ctrl_old = system.Control(self.ctrlInputMatrix, self.size, options=self.ctrlOptions)

    def rollFunction(self, index):
      offset = np.random.randint(self.M)
      # offset = 0
      can = np.zeros((self.M, self.M))
      howMany = self.M
      for i in range(howMany):
        can[i,i] = self.M / howMany
      tmp = np.roll(can, (index + offset) % self.M, axis=1)
      return sp.linalg.block_diag(*[tmp for x in range(self.N)])
          

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


    # def saveSimulation(self):
    #     # Save simulation results, dictionary with:
    #     #     't': time sequence
    #     #     'control': control object (control[:] holds the values of the s_k(t) signals)
    #     #     'output': time domain state values
    #     #     'system': A,c matrices
    #     #     'state': value of last state (unnecessary)
    #     #     'options': options dictionary passed to the simulator
    #     sim_file_path = self.data_dir / 'result_obj.pkl'
    #     with sim_file_path.open(mode='wb') as outfile:
    #         pkl.dump(self.result, outfile)
    #     self.log("Simulation file saved at {}".format(sim_file_path))


    # def saveInputSignals(self):
    #     # Save input signals used in the simulation.
    #     #   Load with: input_signals = pkl.load('input_signals.pkl')
    #     #   Primary Signal is then: input_signals[primary_signal_dimension]
    #     inputs_file_path = self.data_dir / 'input_signals.pkl'
    #     with inputs_file_path.open(mode='wb') as outfile:
    #         pkl.dump(self.input_signals, outfile)
    #     self.log("Input signals saved at \"{}\"".format(inputs_file_path))


    # def saveInputEstimates(self):
    #     input_estimates_file_path = self.data_dir / 'input_estimates.pkl'
    #     with input_estimates_file_path.open(mode='wb') as outfile:
    #         pkl.dump(self.input_estimates, outfile)
    #     self.log("Input Estimates saved at \"{}\"".format(input_estimates_file_path))

    def random_coordinate(self,m):
      return (np.random.randint(m), np.random.randint(m))


    # def plotFrequencyResponse(self,f,ax):
    #   fcrit = 1./(2.*self.sampling_period*self.OSR)
    #   freqs = np.linspace(1e-1, fcrit+1e3 ,5e3)
    #   vals = [10*np.log10(np.linalg.norm(f(val))) for val in freqs]
    #   ax.semilogx(freqs, vals, label=f.__name__)
    #   ax.plot([fcrit, fcrit], [-20,0], label='f_crit')
    #   ax.set_title(self.experiment_id)


    def compute_eta2(self):
      """
        Compute eta2_magnitude depending on the system type.
      """
      # import matplotlib.pyplot as plt
      print(f"Computing eta2 for {self.systemtype}")
      systemResponse = lambda f: np.dot(self.sys_nominal.frequencyResponse(f), self.sys_nominal.b)
      systemResponse_sim = lambda f: np.dot(self.sys_simulation.frequencyResponse(f), self.sys_nominal.b)
      g0 = np.linalg.norm(systemResponse(0),2)
      print("1/sigma_thermal = {}".format(1./self.sigma_thermal))
      print("norm(G(0)b) = {}".format(g0))
      print("relative = {}".format(1./(g0*self.sigma_thermal)))

      gsim_fs = np.linalg.norm(systemResponse_sim(1./(2*self.OSR*self.sampling_period)),2)
      gnom_fs = np.linalg.norm(systemResponse(1./(2*self.OSR*self.sampling_period)),2)
      print("norm(Gsim(fs)b) = {}".format(gsim_fs))
      print("norm(Gnom(fs)b) = {}".format(gnom_fs))


      # stf = lambda x: np.dot(np.dot(np.dot(self.sys.b.T,self.sys.frequencyResponse(x).conjugate().T),
      #                         (np.linalg.pinv(eta2*np.eye(self.M*self.N) + np.outer(np.dot(self.sys.frequencyResponse(x),self.sys.b),np.dot(self.sys.b.T,self.sys.frequencyResponse(x).conjugate().T))))),
      #                          np.dot(self.sys.frequencyResponse(x), self.sys.b))
      # stf.__name__ = 'stf'
      # ntf = lambda x: np.dot(np.dot(self.sys.b.T,self.sys.frequencyResponse(x).conjugate().T),
      #                         (np.linalg.pinv(eta2*np.eye(self.M*self.N) + np.outer(np.dot(self.sys.frequencyResponse(x),self.sys.b),np.dot(self.sys.b.T,self.sys.frequencyResponse(x).conjugate().T)))))

      # ntf.__name__ = 'ntf'

      if self.systemtype in ['CyclicHadamard', 'CyclicGramSchmidt', 'StandardBasis']:
        eta2 = np.sum(np.abs(systemResponse(1./(2. * self.sampling_period * self.OSR)))**2)
        # print(f"eta2 = {10*np.log10(eta2)}")
        # fig,ax = plt.subplots()
        # self.plotFrequencyResponse(stf,ax)
        # self.plotFrequencyResponse(ntf,ax)
        # plt.legend()
        # plt.show()
        # exit()
        self.log("eta2_magnitude set to sum(|G(s)b|^2) = {:.5e}".format(eta2))
        return eta2

      elif self.systemtype in ['BraidedChains', 'BraidedChainsFewControls', 'ParallelIntegratorChains']:
        eta2 = np.sum(np.abs(systemResponse(1./(2. * self.sampling_period * self.OSR)))**2)
        # print(f"eta2 = {10*np.log10(eta2)}")
        # fig,ax = plt.subplots()
        # self.plotFrequencyResponse(stf,ax)
        # self.plotFrequencyResponse(ntf,ax)
        # plt.legend()
        # plt.show()
        # exit()
        self.log("eta2_magnitude set to max(|G(s)b|^2) = {:.5e}".format(eta2))
        return eta2

      else:
        raise Exception("Invalid system type: '{}'".format(self.systemtype))

    def compute_rho(self):
      """
        Limit the low frequency gain of the system at
        approximately the thermal noise level
      """
      if self.leakyIntegrators and self.sigma_thermal > 0:
        return self.beta/(self.sigma_thermal**(-1/self.N)) * 1e-2
      else:
        return 0


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


    def run_simulation(self):
        t = np.linspace(0,(self.size-1)*self.sampling_period, self.size)      
        self.sim_start_time = time.time()
        print("\n\nThermal noise:\n")
        print(self.simulation_noise)
        self.simulation_options = {'noise':self.simulation_noise,
                                  #  'numberOfAdditionalPoints': 4,
                                   #'jitter':{'range':self.sampling_period*1e-3}
                                   }
        initalState = (2*np.random.rand(self.N*self.M) - np.ones(self.N*self.M))*1e-3
        # initalState *= 0
        print("Initial state: ", initalState)
        # for k in range(self.N):
        #   initalState[k*self.M:self.M*(k+1)] = np.ones(self.M) * np.random.randint(2) * 2. - 1.
        sim = simulator.Simulator(self.sys_simulation, self.ctrl, options=self.simulation_options, initalState=initalState)
        # sim_old = simulator.Simulator(self.sys, self.ctrl_old, options=self.simulation_options, initalState=initalState)
        # print("\n#################\nNew simulator")
        # self.result = sim.simulate(t, self.input_signals_mismatch)
        self.result = sim.simulate(t, self.input_signals_mismatch)
        
        # print("\n#################\nOld simulator")
        # self.result = sim.simulate_old(t, self.input_signals)
        self.powerConsumption = self.result["power"]


        self.sim_run_time = time.time() - self.sim_start_time
        self.log(self.result['log'])
        self.log("Simulation run time: {:.2f} seconds".format(self.sim_run_time))
        self.finished_simulation = True


    def run_reconstruction(self):
        t = np.linspace(0,(self.size-1)*self.sampling_period, self.size)      
        recon_time_start = time.time()
        self.eta2 = np.ones(self.M * self.N) * self.eta2_magnitude
        if 'knownNoiseStructure' in self.options:
          print('Reconstruction with known thermal noise covariance')
          self.reconstruction_noise = self.simulation_noise
        else:
          self.reconstruction_noise = [{'std':self.sigma_reconst,
                                        # 'steeringVector': np.eye(self.N * self.M)[:,i] * self.beta,
                                        'steeringVector': np.eye(self.N * self.M)[:,i],
                                        'name':'noise_{}'.format(i)} for i in range(self.N * self.M)]

        self.reconstruction_options = {'eta2':self.eta2,
                                      #  'sigmaU2':[1.]*self.L,
                                       'noise':self.reconstruction_noise,
                                       'mismatch':self.mismatch}

        print("Input Signals:")
        print(self.input_signals)
        self.reconstruction = reconstruction.WienerFilter(t, self.sys_nominal, self.input_signals, self.reconstruction_options)
        tmp_estimates, recon_log = self.reconstruction.filter(self.ctrl)
        # tmp_estimates_old, recon_log_old = self.reconstruction.filter(self.ctrl_old)

        if self.reconstruct_with_all_inputs:
          tmp_estimates = np.sum(tmp_estimates, axis=1).reshape((tmp_estimates.shape[0],1))
          # print(tmp_estimates.shape)
        if tmp_estimates.shape[1] > 0:
          self.input_estimates = tmp_estimates[self.border:-self.border,:]
        else:
          self.input_estimates = tmp_estimates[self.border:-self.border]

        if self.subSample:
          self.input_estimates = self.input_estimates[::self.subSample]
        # self.input_estimates_old = tmp_estimates_old[self.border:-self.border]
        self.recon_run_time = time.time() - recon_time_start
        self.log(recon_log)
        self.log("Reconstruction run time: {:.2f} seconds".format(self.recon_run_time))
        self.finished_reconstruction = True
  

    def defineSimulationNoise(self):
        if 'noise_basis' in self.options:
          print(f"noise basis in self.options")
          self.simulation_noise = [{'std':self.sigma_thermal, #/np.sqrt(self.M),
                                    'steeringVector':self.options['noise_basis'][:,i],
                                    'name':'noise_{}'.format(i)} for i in range(self.M)]
        elif 'zero_one_noise' in self.options:
          sigmas = self.sigma_thermal * np.array([1, 1e-6]*(self.M*self.N//2))/np.sqrt(self.M*self.N/2)
          self.simulation_noise = [{'std':sigmas[i],
                                    'steeringVector':np.eye(self.N*self.M)[:,i] * self.beta,
                                    'name':'noise_{}'.format(i)} for i in range(self.M*self.M)]
        elif 'large_first_then_small' in self.options:
          p = self.M
          q = self.M*self.N - self.M
          b = 1e-3
          a = np.sqrt(1. / (p+(q*b**2)))
          tmp = np.array([a]*p)
          sigmas = np.concatenate((tmp,np.array([a*b]*q)))*self.sigma_thermal
          self.simulation_noise = [{'std':sigmas[i],
                                    'steeringVector':np.eye(self.N*self.M)[:,i] * self.beta,
                                    'name':'noise_{}'.format(i)} for i in range(self.M*self.N)]
        elif 'random_diagonal_thermal_noise' in self.options:
          tmp1 = np.random.random(size=self.M*self.N)*100
          tmp2 = tmp1/np.linalg.norm(tmp1)
          sigmas = self.sigma_thermal * tmp2
          print("Sigmas: ")
          print(sigmas)
          self.simulation_noise = [{'std':sigmas[i],
                                    'steeringVector':np.eye(self.N*self.M)[:,i] * self.beta,
                                    'name':'noise_{}'.format(i)} for i in range(self.M*self.N)]
        else:
          if self.nonuniformNoise == 'exponential':
            grid = 2**((np.arange(self.M*self.N)+1))
          elif self.nonuniformNoise == 'linear':
            grid = np.arange(self.M*self.N)+1
          else:
            grid = np.ones(self.M*self.N)

          grid = grid[::-1]
          x = (self.M*self.N*(self.sigma_thermal**2))/sum(grid)
          sigmas = grid*x
          assert np.allclose(sum(sigmas), self.M*self.N*self.sigma_thermal**2)
          # print("Sum(sigmas) = %s " % (sum(sigmas),))
          # print("M*N*sigma_thermal**2 = %s" % (self.N*self.M*self.sigma_thermal**2))
          # print("Noise vector = %s" % (sigmas,))

          self.simulation_noise = [{'std':np.sqrt(sigmas[i]), 'steeringVector': np.eye(self.N * self.M)[:,i] }  for i in range(self.M * self.N)]
          # self.simulation_noise = [{'std':np.sqrt(sigmas[i]), 'steeringVector': np.eye(self.N * self.M)[:,i] * self.beta}  for i in range(self.M * self.N)]

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
                  'sigma_thermal': self.sigma_thermal,
                  'sigma_reconst': self.sigma_reconst,
                  'bpc': self.bitsPerControl,
                  'controller': self.controller,
                  'systemtype': self.systemtype,
                  'mismatch':self.mismatch}
        return {**params, **input_steering_vectors}


def main(experiment_id, 
         N,
         M=1,
         L=1,
         input_amplitude=1,
         input_frequency=None,
         beta=6250,
         sampling_period=8e-5,
         primary_signal_dimension=0,
         systemtype='ParallelIntegratorChains',
         OSR=16,
         eta2_magnitude=1,
         kappa=1,
         sigma_thermal=1e-6,
         sigma_reconst=1e-6,
         input_phase=0,
         num_periods_in_simulation=20,
         controller='diagonalController',
         bitsPerControl=1,
         leaky=False,
         dither=False,
         mismatch='',
         beta_hat=6250,
         beta_tilde=6250,
         newInputVector=np.ones(1),
         nonuniformNoise=False,

         options={}):
    

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
                              primary_signal_dimension=primary_signal_dimension,
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


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(description="Parallel ADC\
                                                      Experiment Runner")

        # Required arguments
    arg_parser.add_argument("-id", "--experiment_id", required=True, type=str)
    # arg_parser.add_argument("-d", "--data_dir", type=str)
    arg_parser.add_argument("-M", required=True, type=int)
    arg_parser.add_argument("-N", required=True, type=int)
    arg_parser.add_argument("-L", required=True, type=int)
    arg_parser.add_argument("-beta", required=True, type=int)
    arg_parser.add_argument("-Ts", "--sampling_period", required=True, type=float)
    arg_parser.add_argument("-Ax", "--input_amplitude", required=True,  type=float)
    # arg_parser.add_argument("-phi", "--input_phase", required=True, type=float)
    arg_parser.add_argument("-sigma_thermal", required=True, type=float)
    arg_parser.add_argument("-sigma_reconst", required=True, type=float)
    arg_parser.add_argument("-c","--controller", type=str, default='diagonalController')
    arg_parser.add_argument("-bpc","--bitsPerControl", type=int, default=1)

    # Optional arguments, things that could change later
    # or are currently set to some fixed value
    # arg_parser.add_argument("-f_sig", "--input_frequency", type=float, default=None)
    # arg_parser.add_argument("-eta2", "--eta2_magnitude", type=float, default=1)
    # arg_parser.add_argument("-kappa", type=float, default=1)
    # arg_parser.add_argument("-OSR", type=int, default=16)
    arg_parser.add_argument("-systemtype", type=str, default="ParallelIntegratorChains")
    # arg_parser.add_argument("-sig_dim", "--primary_signal_dimension", type=int, default=0)
    arg_parser.add_argument("-n_sim", "--num_periods_in_simulation", type=int, default=1)
    arg_parser.add_argument("-leaky", type=bool, default=False)
    arg_parser.add_argument("-dither", type=bool, default=False)
    arg_parser.add_argument("-mismatch", type=str, default='')

    args = vars(arg_parser.parse_args())


    main(**args)
