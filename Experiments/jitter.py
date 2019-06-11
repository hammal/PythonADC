###############################
#      Standard Packages      #
###############################
import numpy as np
import scipy as sp
import pickle as pkl
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from pathlib import Path

###############################
#         ADC Packages        #
###############################
import AnalogToDigital.system as system
import AnalogToDigital.simulator as simulator
import AnalogToDigital.reconstruction as reconstruction
import AnalogToDigital.filters as filters
import AnalogToDigital.evaluation as evaluation

# Edit this path
data_dir = Path(r'./jitter_results')
if not data_dir.exists():
    data_dir.mkdir(parents=True)


Ts = 8e-5
osr = 16
N = 3
phase = 0
amplitude = 1
frequency = 1./(2.*osr*Ts)
size = (1<<12)
t = np.linspace(0,(size-1)*Ts, size)
kappa = 1
sigma2_thermal = 1e-6
sigma2_reconst = 1e-6

border = int(size//100)

beta = 6250

jitter_size = 1e-3


number_of_jitter_estimation_states = N


for sigma2_jitter in np.logspace(-8,-3,10): 
    for RECONSTRUCTION_METHOD in ['JitterAsGaussianNoise','AugmentedSSM', 'NoJitterEstimation', 'CleanSimulation']:


        if RECONSTRUCTION_METHOD == 'JitterAsGaussianNoise':
            A = np.eye(N,k=-1)*beta
            input_vector = np.zeros(N)
            input_vector[0] = beta
            c = np.eye(N)

            sigmaU2 = [1.]
            

            [sigmaU2.append(sigma2_jitter) for i in range(1)]
            simulationOptions = {'noise': [{'std':sigma2_thermal, 'steeringVector': beta * np.eye(N)[:,i]} for i in range(N)],
                                 'jitter': {'range':Ts*jitter_size}}

            mixingMatrix = - kappa * beta * np.eye(N)
            ctrl = system.Control(mixingMatrix, size)
            input_signal = system.Sin(Ts, amplitude=amplitude, frequency=frequency, phase=phase, steeringVector=input_vector)
            all_inputs = [input_signal]

            order = N
            jitterInputs = [system.Input(Ts=Ts, coefficients=np.zeros(len(t)), steeringVector=np.ones(N)*beta)]
            for i in range(1):
                all_inputs.append(jitterInputs[i])


        elif RECONSTRUCTION_METHOD == 'AugmentedSSM':
            A_tmp1 = np.hstack( (np.eye(N,k=-1)*beta , -np.eye(N)))
            A_tmp2 = np.hstack( (np.zeros((number_of_jitter_estimation_states, N)), np.eye(N,k=-1)*beta) )
            A = np.vstack( (A_tmp1, A_tmp2) )
            print(A)    

            input_vector = np.zeros(N+number_of_jitter_estimation_states)
            input_vector[0] = beta
            c = np.eye(N+number_of_jitter_estimation_states)
            sigmaU2 = [1.]
            eta2 = np.ones(N + number_of_jitter_estimation_states)
            simulationOptions = {'noise': [{'std':sigma2_thermal, 'steeringVector': beta * np.eye(N+number_of_jitter_estimation_states)[:,i]} for i in range(N+number_of_jitter_estimation_states)],
                                 'jitter': {'range':Ts*jitter_size}}
            mixingMatrix = - kappa * beta * np.eye(N+number_of_jitter_estimation_states)
            
            input_signal = system.Sin(Ts, amplitude=amplitude, frequency=frequency, phase=phase, steeringVector=input_vector)
            all_inputs = [input_signal]

            order = N+number_of_jitter_estimation_states

        elif RECONSTRUCTION_METHOD == 'NoJitterEstimation':
            A = np.eye(N,k=-1)*beta
            input_vector = np.zeros(N)
            input_vector[0] = beta
            c = np.eye(N)

            sigmaU2 = [1.]
            simulationOptions = {'noise': [{'std':sigma2_thermal, 'steeringVector': np.eye(N)[:,i]} for i in range(N)],
                                 'jitter': {'range':Ts*jitter_size}}

            mixingMatrix = - kappa * beta * np.eye(N)
            ctrl = system.Control(mixingMatrix, size)
            input_signal = system.Sin(Ts, amplitude=amplitude, frequency=frequency, phase=phase, steeringVector=input_vector)
            all_inputs = [input_signal]
            eta2 = np.ones(N)
            order = N
            
        else:
            print("Simulating without jitter for comparison")
            A = np.eye(N,k=-1)*beta
            input_vector = np.zeros(N)
            input_vector[0] = beta
            c = np.eye(N)

            sigmaU2 = [1.]

            simulationOptions = {'noise': [{'std':sigma2_thermal, 'steeringVector': beta * np.eye(N)[:,i]} for i in range(N)]}

            mixingMatrix = - kappa * beta * np.eye(N)
            ctrl = system.Control(mixingMatrix, size)
            input_signal = system.Sin(Ts, amplitude=amplitude, frequency=frequency, phase=phase, steeringVector=input_vector)
            all_inputs = [input_signal]
            order = N

        
        sys = system.System(A=A,c=c,b=input_vector)
        systemResponse = lambda f: np.dot(sys.frequencyResponse(f), sys.b)
        eta2_magnitude = np.max(np.abs(systemResponse(1./(2. * Ts * osr)))**2)
        print("Eta2 = %d" % eta2_magnitude)
        eta2 = np.ones(order)*eta2_magnitude

        
        ctrl = system.Control(mixingMatrix, size)
        sim = simulator.Simulator(sys, ctrl,  options=simulationOptions)
        res = sim.simulate(t, (input_signal,))
        

        reconstructionOptions = {'eta2':eta2,
                                 'sigmaU2':sigmaU2,
                                 'noise':[{'std': sigma2_reconst, 'steeringVector': beta * np.eye(order)[:,i], 'name':f'Bastard_{i}'} for i in range(order)]}

        

        print(f'Reconstructing using: {RECONSTRUCTION_METHOD}')
        recon = reconstruction.WienerFilter(t, sys, tuple(all_inputs), reconstructionOptions)
        input_estimates ,_ = recon.filter(ctrl)

        
        snrVsAmplitude = evaluation.SNRvsAmplitude(system=sys, estimates=[input_estimates[:,0]], OSR=osr)
        freq, spec = snrVsAmplitude.estimates[0]['performance'].freq, snrVsAmplitude.estimates[0]['performance'].spec

        snrVsAmplitude.ToTextFile(data_dir/f'{RECONSTRUCTION_METHOD}_{sigma2_jitter}_SNR.csv', delimiter=' ')
        f = lambda x: 10*np.log10(x)
        df = pd.DataFrame({'freq':freq, 'spec':spec})
        fIndex = df.idxmax().spec
        df = df.apply(f)

        tmp = np.concatenate((np.arange(0,fIndex-5,4),
                                  np.arange(fIndex-5, fIndex+5),
                                  np.arange(fIndex+5,df.index.size,4)))
        decimationMask = np.zeros(df.index.size,dtype=bool)
        decimationMask[tmp] = True

        pd.DataFrame(df[decimationMask]).to_csv(data_dir/f'{RECONSTRUCTION_METHOD}_sigma2jitter_{sigma2_jitter}_PSD.csv', sep=' ')

        # nperseg = min([512 * 2 * osr, input_estimates.shape[0]])
        # window = 'hanning'

        # freq, spec = signal.welch(input_estimates, 1./Ts, nperseg=nperseg, noverlap=None, nfft=None, return_onesided=True, scaling='spectrum', axis=0)


        plt.figure()
        plt.semilogx(freq, 10*np.log10(np.abs(spec)))
        plt.grid()
        plt.draw()
        plt.savefig(data_dir /f'{RECONSTRUCTION_METHOD}_sigma2jitter_{sigma2_jitter}_PSD.png', dpi=300)


        experiment_information = {'jitter_size': jitter_size,
                   'Reconstruction_method':RECONSTRUCTION_METHOD,
                   'sigma2_thermal': sigma2_thermal,
                   'sigma2_reconst': sigma2_reconst,
                   'sigma2_jitter': sigma2_jitter,
                   'number_of_jitter_estimation_states':number_of_jitter_estimation_states,
                   }
        params_string = ''
        for key in experiment_information.keys():
            params_string = ''.join([params_string, f'{key}: {experiment_information[key]}\n'])

        with open(data_dir / f'{RECONSTRUCTION_METHOD}_{sigma2_jitter}.params', 'w') as f:
            f.write(params_string)