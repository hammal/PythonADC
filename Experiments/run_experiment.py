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


# Edit this path
DATA_STORAGE_PATH = Path('/itet-stor/olafurt/net_scratch/adc_data')


def hadamardMatrix(n):
    return sp.linalg.hadamard(n)/np.sqrt(n)


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
                 num_periods_in_simulation=100,
                 controller='subspaceController',
                 bitsPerControl=1,
                 leaky=False,
                 dither=False,
                 mismatch=False):

        print("Initializing Experiment | ID: %s" % experiment_id)
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
        self.leakyIntegrators = leaky
        self.num_periods_in_simulation = num_periods_in_simulation
        self.size = round(num_periods_in_simulation/sampling_period)

        self.controller = controller
        self.bitsPerControl = bitsPerControl

        self.border = np.int(self.size // 100)
        self.all_input_signal_amplitudes = np.zeros(L)
        self.all_input_signal_amplitudes[primary_signal_dimension] = input_amplitude

        self.logstr = ("{0}: EXPERIMENT LOG\n{0}: Experiment ID: {1}\n".format(time.strftime("%d/%m/%Y %H:%M:%S"), experiment_id))

        self.finished_simulation = False
        self.finished_reconstruction = False
        if not self.data_dir.exists():
            self.data_dir.mkdir(parents=True)

        if mismatch:
          self.mismatch = True
          self.robustness = {'controlInputMatrix':0.01}
        else:
          self.mismatch = False
          self.robustness = []

        #################################################
        #     System and input signal specifications    #
        #################################################

        if self.primary_signal_dimension > self.M:
            self.log("Primary Signal Dimension cannot be larger than M, setting to 0 (first dim)")
            self.primary_signal_dimension = 0

        if self.input_frequency == None:
            self.input_frequency = 1./(self.sampling_period * 2 * self.OSR)
            self.log(f'Setting f_sig = f_s/(2*OSR) = {self.input_frequency}')
        self.H = hadamardMatrix(self.M)

        # Define input signals:
        self.input_signals = []
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
        self.A = np.zeros((self.N*self.M, self.N*self.M))

        if self.systemtype == "ParallelIntegratorChain":
            mixingPi = np.zeros((self.N-1, self.M, self.M))
            
            if N > 1:
                # L=1 means just one of M dimensions is used and there is
                # only one input signal => We scale up the input vector by sqrt(M)
                if L == 1:
                    for k in range(N-1):
                        mixingPi[k] = self.beta * np.sqrt(self.M) * (np.outer(self.H[:,0],self.H[:,0]))
                        self.A[(k+1)*self.M:(k+2)*self.M, (k)*self.M:(k+1)*self.M] = mixingPi[k]
                # L=M means M input signals
                elif L == M:
                    for k in range(N-1):
                        mixingPi[k] = self.beta * np.sqrt(self.M) * np.eye(self.M)
                        self.A[(k+1)*self.M:(k+2)*self.M, (k)*self.M:(k+1)*self.M] = mixingPi[k]
                else:
                    # for k in range(N-1):
                    #     mixingPi[k] = self.beta * np.sqrt(M/L) * (sum(np.outer(H[:,i],H[:,i]) for i in range(self.L)))
                    #     self.A[(k+1)*self.M:(k+2)*self.M, (k)*self.M:(k+1)*self.M] = mixingPi[k]
                    raise NotImplemented
            else:
              mixingPi = [np.zeros((M,M))]

            # Limit the low frequency gain of the filter
            # at approximately the thermal noise level
            self.rho = self.compute_rho()
            self.A -= np.eye(N*M)*self.rho

            # Define input steeringVectors
            if L == 1:
              inputVectorMatrix = self.beta * np.sqrt(M) * self.H
            # elif L == 2:
            #   # The outer product sum results in a checkerboard matrix with 1/0 entries.
            #   # We pick input vectors from there and scale up by M (undoing the attenuation from the outer products)
            #   inputVectorMatrix = self.beta * (M/2) * ( np.outer(H[:,0],H[:,0]) + np.outer(H[:,1],H[:,1]) ) * 2
            # elif L == M:
            #   inputVectorMatrix = self.beta * np.sqrt(M) * (np.hstack( (np.outer(H[:,i],H[:,i])[:,0].reshape(-1,1) for i in range(L)) ))
            else:
              raise NotImplemented

            for i in range(self.L):
                vector = np.zeros(self.M*self.N)
                vector[0:self.M] = inputVectorMatrix[:,i]
                self.input_signals.append(system.Sin(self.sampling_period,
                                                     amplitude=self.all_input_signal_amplitudes[i],
                                                     frequency=self.all_input_signal_frequencies[i],
                                                     phase=self.input_phase,#+ (np.pi/2)*i,
                                                     steeringVector=vector))
                print(f'b_{i} = {self.input_signals[i].steeringVector}')
            self.input_signals = tuple(self.input_signals)
        elif self.systemtype == "CyclicHadamard":
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
                for k in range(self.M):
                    mixingPi[k] = self.beta * sum(np.outer(self.H[:,i],self.H[:,i]) for i in range(self.N) if i != k)
                self.A[ 0 : self.M, -self.M : self.M*self.N] = mixingPi[0]
                for k in range(self.M-1):
                  self.A[(k+1)*self.M:(k+2)*self.M, (k)*self.M:(k+1)*self.M] = mixingPi[k+1]
                # print(self.A)

                vec = np.hstack((self.H[:,i] for i in np.append(np.arange(1,self.M),0)))
                if np.any(np.abs(np.dot(self.A,vec))):
                  print("Error in building cyclic system. A.dot(vec) != 0")
                  print(np.dot(self.A,vec))
            else:
              mixingPi = [np.zeros((M,M))]

            self.rho = self.compute_rho()
            print("rho = {}".format(self.rho))
            self.A -= np.eye(N*M)*self.rho

            # Define input signals:
            self.input_signals = []
            if self.L==1:
              vector = np.zeros(self.M*self.N)
              for k in range(self.M):
                vector[k*self.M:(k+1)*self.M] = self.beta * self.H[:,k]

              # self.input_signals.append(system.Constant(amplitude=self.input_amplitude,
              #                                           steeringVector=vector))
              self.input_signals.append(system.Sin(self.sampling_period,
                                                   amplitude=self.input_amplitude,
                                                   frequency=self.input_frequency,
                                                   phase=self.input_phase,
                                                   steeringVector=vector))

            elif self.L == self.M:
              for k in range(self.L):
                vector = np.zeros(self.M*self.N)
                vector[k*self.M:(k+1)*self.M] = self.beta * self.H[:,k]

                self.input_signals.append(system.Sin(self.sampling_period,
                                                   amplitude=self.input_amplitude,
                                                   frequency=self.all_input_signal_frequencies[k],
                                                   phase=self.input_phase,# + 2*np.pi*np.random.rand(),
                                                   steeringVector=vector))
        elif self.systemtype == "FullyParallelSystem":
            for k in range(N-1):
              self.A[(k+1)*self.M:(k+2)*self.M, (k)*self.M:(k+1)*self.M] = self.beta * np.eye(self.M)

            self.rho = self.compute_rho()
            self.A -= np.eye(N*M)*self.rho

            vector = np.zeros(self.M*self.N)
            vector[0:self.M] = self.beta * np.ones(M)
            self.input_signals.append(system.Sin(self.sampling_period,
                                                 amplitude=self.all_input_signal_amplitudes[i],
                                                 frequency=self.all_input_signal_frequencies[i],
                                                 phase=self.input_phase,#+ (np.pi/2)*i,
                                                 steeringVector=vector))
            print(f'b_{i} = {self.input_signals[i].steeringVector}')
            self.input_signals = tuple(self.input_signals)
        elif self.systemtype == "CyclicGramSchmidt":
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
        else:
            raise Exception("Invalid system type: '{}'".format(self.systemtype))

        self.c = np.eye(self.N * self.M)
        self.sys = system.System(A=self.A, c=self.c, b=self.input_signals[primary_signal_dimension].steeringVector)

        self.eta2_magnitude = self.compute_eta2()
        print("eta2_magnitude = {:.5e}".format(self.eta2_magnitude))
        print("A = {}".format(self.A))

        #################################################
        #           Controller specification            #
        #################################################
        self.dither = dither
        """
          The subspace controller is only implemented for L=1 signal right now.
        """
        if controller == 'subspaceController':
          if self.systemtype == 'ParallelIntegratorChain':
            self.ctrlInputMatrix = np.zeros((self.N*self.M, self.N))
            # if L>1:
            #   raise "Multi-Bit controller not implemented for L>1 input signals"
            if dither:
              self.ctrlInputMatrix =  (np.random.randint(2,size=(N*M , N))*2 - 1)*self.beta*1e-3

            if L==1:
              for i in range(N):
                self.ctrlInputMatrix[i*M:(i+1)*M,i] = - np.sqrt(self.M) * self.beta * self.H[:,0]
              self.nominalCtrlInputMatrix = np.copy(self.ctrlInputMatrix)
            else:
              raise NotImplemented
            self.ctrlObservationMatrix = np.dot(self.ctrlInputMatrix, np.diag(1/(np.linalg.norm(self.ctrlInputMatrix, ord=2, axis=0)))).transpose()

          elif self.systemtype == 'CyclicHadamard':
            """ Control mixing matrix for the cyclic parallel system:
                        [  H    0    ...      0 ]
                        [  0    H    ...      0 ]
                        [  .        .         . ]
               beta * ( [  .            .     . ] )
                        [  .               .  0 ]
                        [  0                  H ]

                # Xi_k eqdef  [pi_1^T ... pi_{k-1}^T pi_{k+1}^T ... pi_M^T]^T
            """

            self.ctrlInputMatrix = np.zeros((self.N*self.M, self.N * self.M))
            if dither:
              self.ctrlInputMatrix = (np.random.randint(2,size=(self.N*self.M,self.N*self.M ))*2 - 1)*self.beta*1e-3

            for k in range(self.M):
              self.ctrlInputMatrix[k*self.M:(k+1)*self.M, k*self.M:(k+1)*self.M] = - self.beta * self.H.transpose()

            self.ctrlObservationMatrix = np.dot(self.ctrlInputMatrix, np.diag(1/(np.linalg.norm(self.ctrlInputMatrix, ord=2, axis=0)))).transpose()
            self.nominalCtrlInputMatrix = np.copy(self.ctrlInputMatrix)

            if self.mismatch and 'controlInputMatrix' in self.robustness:
              # ctrlMismatch = (np.random.randint(2,size=(self.ctrlInputMatrix.shape))* 2 - 1)*self.robustness['controlInputMatrix']
              ctrlMismatch = np.zeros_like(self.ctrlInputMatrix)
              faulty_component = self.random_coordinate(self.M)
              print(f"Single faulty component: {faulty_component}")
              ctrlMismatch[faulty_component] = (np.random.randint(2)*2-1)*self.robustness['controlInputMatrix']
              # This is an entire faulty module: ctrlMismatch[:self.ctrlInputMatrix.shape[0], :self.ctrlInputMatrix.shape[1]]

              self.ctrlInputMatrix = self.ctrlInputMatrix + np.multiply(ctrlMismatch, self.ctrlInputMatrix)
              print(self.ctrlInputMatrix)

            # for k in range(self.N):
            #   self.ctrlMixingMatrix[k*self.M:(k+1)*self.M, k*(self.M-1):(k+1)*(self.M-1)] = - beta * np.delete(H,k, axis=1)
          elif self.systemtype == 'CyclicGramSchmidt':
            self.ctrlInputMatrix = np.zeros((self.N*self.M, self.N * self.M))
            if dither:
              self.ctrlInputMatrix = (np.random.randint(2,size=(self.N*self.M,self.N*self.M ))*2 - 1)*self.beta*1e-3
            for k in range(self.M):
              self.ctrlInputMatrix[k*self.M:(k+1)*self.M, k*self.M:(k+1)*self.M] = - self.beta * self.GramSchmidtBasis.transpose()

            self.ctrlObservationMatrix = np.dot(self.ctrlInputMatrix, np.diag(1/(np.linalg.norm(self.ctrlInputMatrix, ord=2, axis=0)))).transpose()
        elif controller == 'diagonalController':
            self.ctrlInputMatrix = np.zeros((N*M,N*M))
            if dither:
              self.ctrlInputMatrix = (np.random.randint(2,size=(self.N * self.M, self.N * self.M))*2 - 1) * self.beta*1e-3
            for i in range(self.M*self.N):
              self.ctrlInputMatrix[i,i] = - self.kappa * self.beta
            self.nominalCtrlInputMatrix = np.copy(self.ctrlInputMatrix)

            if self.mismatch and 'controlInputMatrix' in self.robustness:
              coordinate = np.random.randint(self.M)
              faulty_component = (coordinate,coordinate)
              print(f"Single faulty component: {faulty_component}")
              ctrlMismatch = np.zeros_like(self.ctrlInputMatrix)
              ctrlMismatch[faulty_component] = (np.random.randint(2)*2-1)*self.robustness['controlInputMatrix']
              self.ctrlInputMatrix = self.ctrlInputMatrix + np.multiply(ctrlMismatch, self.ctrlInputMatrix)

            self.ctrlObservationMatrix = np.dot(self.ctrlInputMatrix, np.diag(1/(np.linalg.norm(self.ctrlInputMatrix, ord=2, axis=0)))).transpose()

        self.ctrlOptions = {
            'bitsPerControl':bitsPerControl,
            'projectionMatrix':self.ctrlObservationMatrix,
            'nominalCtrlInputMatrix':self.nominalCtrlInputMatrix,
            'bound':1
        }

        self.ctrl = system.Control(self.ctrlInputMatrix, self.size, options=self.ctrlOptions)
        # self.ctrl_old = system.Control(self.ctrlInputMatrix, self.size, options=self.ctrlOptions)

        print("ctrlInputMatrix: %s\n\n" % (self.ctrlInputMatrix,))
        # print("ctrlInputMatrix: %s\n" % (ctrlInputMatrix,))


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

    def compute_eta2(self):
      """
        Compute eta2_magnitude depending on the system type.
      """
      print(f"Computing eta2 for {self.systemtype}")
      systemResponse = lambda f: np.dot(self.sys.frequencyResponse(f), self.sys.b)

      if self.systemtype in ['CyclicHadamard', 'CyclicGramSchmidt']:
        eta2 = np.sum(np.abs(systemResponse(1./(2. * self.sampling_period * self.OSR))**2))
        self.log("eta2_magnitude set to sum(|G(s)b|^2) = {:.5e}".format(eta2))
        return eta2

      elif self.systemtype in ['ParallelIntegratorChain', 'FullyParallelSystem']:
        eta2 = np.max(np.abs(systemResponse(1./(2. * self.sampling_period * self.OSR))**2))
        self.log("eta2_magnitude set to max(|G(s)b|^2) = {:.5e}".format(eta2))
        return eta2

      else:
        raise Exception("Invalid system type: '{}'".format(self.systemtype))

    def compute_rho(self):
      """
        Limit the low frequency gain of the system at
        approximately the thermal noise level
      """
      if self.leakyIntegrators and self.sigma2_thermal > 0:
        return (self.beta/((self.sigma2_thermal)**(-1/(self.N))))
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

        self.simulation_options = {'noise':[{'std':self.sigma2_thermal, 'steeringVector': np.eye(self.N * self.M)[:,i] * self.beta}  for i in range(self.M * self.N)],
                                   'numberOfAdditionalPoints': 4
                                   #'jitter':{'range':self.sampling_period*1e-3}
                                   }
        initalState = (2*np.random.rand(self.N*self.M) - np.ones(self.N*self.M))*1e-3
        print("Initial state: ", initalState)
        # for k in range(self.N):
        #   initalState[k*self.M:self.M*(k+1)] = np.ones(self.M) * np.random.randint(2) * 2. - 1.
        sim = simulator.Simulator(self.sys, self.ctrl, options=self.simulation_options, initalState=initalState)
        # sim_old = simulator.Simulator(self.sys, self.ctrl_old, options=self.simulation_options, initalState=initalState)
        # print("\n#################\nNew simulator")
        self.result = sim.simulate(t, self.input_signals)
        # print("\n#################\nOld simulator")
        # self.result_old = sim_old.simulate_old(t, self.input_signals)

        self.sim_run_time = time.time() - self.sim_start_time
        self.log(self.result['log'])
        self.log("Simulation run time: {:.2f} seconds".format(self.sim_run_time))
        self.finished_simulation = True


    def run_reconstruction(self):
        t = np.linspace(0,(self.size-1)*self.sampling_period, self.size)      
        recon_time_start = time.time()
        self.eta2 = np.ones(self.M * self.N) * self.eta2_magnitude
        self.reconstruction_options = {'eta2':self.eta2,
                                       'sigmaU2':[1.]*self.L,
                                       'noise':[{'std':self.sigma2_reconst,
                                                 'steeringVector': np.eye(self.N * self.M)[:,i] * self.beta,
                                                 'name':'noise_{}'.format(i)} for i in range(self.N * self.M)],
                                       'mismatch':self.mismatch}

        self.reconstruction = reconstruction.WienerFilter(t, self.sys, self.input_signals, self.reconstruction_options)
        tmp_estimates, recon_log = self.reconstruction.filter(self.ctrl)
        # tmp_estimates_old, recon_log_old = self.reconstruction.filter(self.ctrl_old)

        self.input_estimates = tmp_estimates[self.border:-self.border]
        # self.input_estimates_old = tmp_estimates_old[self.border:-self.border]
        self.recon_run_time = time.time() - recon_time_start
        self.log(recon_log)
        self.log("Reconstruction run time: {:.2f} seconds".format(self.recon_run_time))
        self.finished_reconstruction = True
    

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
                  'sigma2_reconst': self.sigma2_reconst,
                  'bpc': self.bitsPerControl,
                  'controller': self.controller,
                  'systemtype': self.systemtype}
        return {**params, **input_steering_vectors}


def main(experiment_id,
         M, 
         N,
         L,
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
         input_phase=0,
         num_periods_in_simulation=20,
         controller='subspaceController',
         bitsPerControl=1,
         leaky=False,
         dither=False):
    
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
                              sigma2_thermal=sigma2_thermal,
                              sigma2_reconst=sigma2_reconst,
                              num_periods_in_simulation=num_periods_in_simulation,
                              controller=controller,
                              bitsPerControl=bitsPerControl,
                              leaky=leaky,
                              dither=dither)

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
    arg_parser.add_argument("-sigma2_thermal", required=True, type=float)
    arg_parser.add_argument("-sigma2_reconst", required=True, type=float)
    arg_parser.add_argument("-c","--controller", type=str)
    arg_parser.add_argument("-bpc","--bitsPerControl", type=int)

    # Optional arguments, things that could change later
    # or are currently set to some fixed value
    # arg_parser.add_argument("-f_sig", "--input_frequency", type=float, default=None)
    # arg_parser.add_argument("-eta2", "--eta2_magnitude", type=float, default=1)
    # arg_parser.add_argument("-kappa", type=float, default=1)
    # arg_parser.add_argument("-OSR", type=int, default=16)
    arg_parser.add_argument("-systemtype", type=str, default="ParallelIntegratorChain")
    # arg_parser.add_argument("-sig_dim", "--primary_signal_dimension", type=int, default=0)
    arg_parser.add_argument("-n_sim", "--num_periods_in_simulation", type=int)#, default=20)
    arg_parser.add_argument("-leaky", type=bool, default=False)
    arg_parser.add_argument("-dither", type=bool, default=False)

    args = vars(arg_parser.parse_args())


    main(**args)
