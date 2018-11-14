###############################
###        References       ###
###############################
"""
    [1] Control-based Analog-to-digital Conversion Using a Wiener Filter
"""


###############################
###     Standard Packages   ###
###############################
import argparse
import numpy as np
import scipy as sp
import pickle as pkl
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
import time
import os
import shutil
import warnings
from ruamel.yaml import YAML


###############################
###       ADC Packages      ###
###############################
import AnalogToDigital.system as system
import AnalogToDigital.simulator as simulator
import AnalogToDigital.reconstruction as reconstruction
import AnalogToDigital.filters as filters
import AnalogToDigital.evaluation as evaluation


###############################
###        Constants        ###
###############################
pi = np.pi


def plotPowerSpectralDensity(spectralDensity, sinusoidFrequency, N_FFT, Ts, f, num_signal_bins = 5, signalBand=None, SNR=None):
    if signalBand is None:
        signalBand = [0., 2. * sinusoidFrequency * 7] # Set the signal band = [0, 7*f_nyquist]

    center_bin = int(np.rint(sinusoidFrequency * N_FFT*Ts))
    signal_bins = np.arange(center_bin-num_signal_bins//2, center_bin + num_signal_bins//2 + 1, 1)
    cutoff_frequency_idx = int(signalBand[1] * N_FFT*Ts)    

    noise_freqs = np.delete(f, signal_bins)
    noise_spectrum = np.delete(spectralDensity, signal_bins)
    plt.semilogx(noise_freqs[10:], 10 * np.log10(noise_spectrum[10:]), 'bx-', label='Noise')
    plt.semilogx(f[signal_bins], 10 * np.log10(spectralDensity[signal_bins]), 'rx-', label='Signal')

    # plt.semilogx([10./(Ts*N_FFT), 10./(Ts*N_FFT)], [-200,0], color='red')
    # plt.semilogx([signalBand[1],signalBand[1]], [-200,0], color='red')

    textSNR =r'SNR $=%.2f$' %(SNR,)
    textCutFreq = r'$f_{cut} = %.2f$' % (signalBand[1],)
    props = dict(boxstyle='square', facecolor='wheat', alpha=0.5)
    plt.text(f[center_bin]+10, 10*np.log10(spectralDensity[center_bin])-10, textSNR, fontsize=10, bbox=props)
    plt.text(f[cutoff_frequency_idx]+100, -20, textCutFreq, fontsize=10, bbox=props)
    plt.legend()


def estimateSNR(spectralDensity, sinusoidFrequency, N_FFT, Ts, f=None, num_signal_bins = 5, signalBand=None):
    if signalBand is None:
        signalBand = [0., 2. * sinusoidFrequency * 7] # Set the signal band = [0, 7*f_nyquist]

    center_bin = int(np.rint(sinusoidFrequency * N_FFT*Ts))
    signal_bins = np.arange(center_bin-num_signal_bins//2, center_bin + num_signal_bins//2 + 1, 1)
    
    cutoff_frequency_idx = int(signalBand[1] * N_FFT*Ts)
    
    noise_spectrum = np.delete(spectralDensity, signal_bins)

    signal_power = np.linalg.norm(spectralDensity[signal_bins], 2)
    noise_power  = np.linalg.norm(noise_spectrum[10:cutoff_frequency_idx], 2)
    SNR = 20*np.log10(signal_power/noise_power)

    return SNR

<<<<<<< HEAD
=======

>>>>>>> Working files (i.e. not nice clean code) for simulations and data processing
def estimateNoiseFloor(spectralDensity, sinusoidFrequency, N_FFT, Ts):
    signal_bin = int(np.rint((Ts*sinusoidFrequency) * N_FFT))

    return 10 * np.log10(np.mean(spectralDensity[10: signal_bin//2]))


def performSimulation(Ts, beta, size, order, frequency, amplitude, phase, kappa, jitterAmount, sigma_noise, data_dir, NumberOfJitterEstimationStates=0):
    output_dir_name = "noise{}_jitter{}".format(sigma_noise, jitterAmount)

    if NumberOfJitterEstimationStates > order:
        warnings.warn("More jitter estimation states than actual system.\
                       Setting NumberOfJitterEstimationStates = order")
        NumberOfJitterEstimationStates = order

    A_system = np.eye(order, k=-1)*beta
    if NumberOfJitterEstimationStates:
        # tmp_top = np.hstack((A_system, -np.eye(order)[:,:NumberOfJitterEstimationStates]))
        tmp_top = np.hstack((A_system, np.zeros((order,NumberOfJitterEstimationStates))))
        tmp_btm = np.hstack((np.zeros((NumberOfJitterEstimationStates, order)),
                             A_system[0:NumberOfJitterEstimationStates,0:NumberOfJitterEstimationStates]))
        A_augmented = np.vstack((tmp_top, tmp_btm))
        num_states = order + NumberOfJitterEstimationStates
    else:
        A_augmented = A_system
        num_states = order

    print("Augmented A matrix: \n{}".format(A_augmented))
    c = np.eye(num_states)
    sys = system.System(A_augmented, c)

    
    vector = np.zeros(num_states)
    vector[0] = beta    # This is \beta_1 in Figure 2 of [1]

    inp = system.Sin(Ts, amplitude=amplitude, frequency=frequency, phase=phase, steeringVector=vector)
    
    if NumberOfJitterEstimationStates:
        inp_jitter = system.Input(Ts, coefficients=np.zeros(size), steeringVector=np.ones(len(vector)))
        inputs = (inp, inp_jitter)
        output_dir_name += "_JitterEstimation_{}_States".format(NumberOfJitterEstimationStates)
    else:
        inputs = (inp,)
    
    # tmp1 = np.vstack((np.eye(order),np.eye(order)[:NumberOfJitterEstimationStates,:]))
    # print(tmp1)
    # tmp2 = np.zeros((order+NumberOfJitterEstimationStates,order))
    # print(tmp2)
    
    mixingMatrix = - kappa * beta * np.eye(num_states)#np.hstack((tmp1,tmp2))
    # print(mixingMatrix)
    # exit()
    #                                            
    # mixingMatrix = - kappa * beta * np.hstack(((np.vstack((np.eye(order),
    #                                            np.eye(order)[0:NumberOfJitterEstimationStates,:]))),
    #                                           np.zeros((order+NumberOfJitterEstimationStates,
    #                                                     NumberOfJitterEstimationStates))))

    ctrl = system.Control(mixingMatrix, size)


    print("Simulation sampling period: {} s".format(Ts))
    print("Input sinusoid frequency: {} Hz".format(frequency))
    print("OSR = {}".format(1./(2.*frequency*Ts)))
    print("State Bound: |x(t)| < b = {:.5f}".format((Ts*beta*kappa)/(1. - Ts*beta) + 0.1))


    ### Simulator setup

    if sigma_noise:
        sim = simulator.Simulator(sys, ctrl, options={'jitter':{'range':(Ts*jitterAmount)},
                                                      'noise':[{'std':sigma_noise, 'steeringVector':np.ones(num_states)}],
                                                      'stateBound': (Ts*beta*kappa)/(1. - Ts*beta)+0.1})
    else:
        sim = simulator.Simulator(sys, ctrl, options={'jitter':{'range':(Ts*jitterAmount)}})

    t = np.linspace(0., (size-1)*Ts, size)  # Simulation time steps

    ### File input/output directories
    
    # if jitterAmount and sigma_noise:
    #     output_dir = os.path.join(data_dir, "noise{}_jitter{}".format(sigma_noise, jitterAmount))
    # elif jitterAmount:
    #     output_dir = os.path.join(data_dir, "noise0_jitter{}".format(jitterAmount))
    # elif jitterAmount:
    #     output_dir = os.path.join(data_dir, "noise{}_jitter0".format(sigma_noise))
    # else:
    output_dir = os.path.join(data_dir, output_dir_name)


    if os.path.exists(output_dir):
        for file in os.listdir(output_dir):
            try:
                shutil.rmtree(os.path.join(output_dir,file))
            except:
                os.remove(os.path.join(output_dir,file))
    else:
        os.makedirs(output_dir)

    experiment_specifications = {"sinusoid frequency": [frequency],
                                 "amplitude": [amplitude],
                                 "phase": [phase],
                                 "beta": [beta],
                                 "kappa": [kappa],
                                 "Ts": [Ts],
                                 "order": [order],
                                 "simulation #samples": [size],
                                 "OSR": [1./(2.*frequency*Ts)],
                                 "state bound": [(Ts*beta*kappa)/(1. - Ts*beta)],
                                 "sigma_noise": [sigma_noise]}

    df = pd.DataFrame(experiment_specifications)
    df.to_csv(os.path.join(output_dir, "specifications.csv"), index=False)

    ### Simulate the system, the return variable contains the values of all states at all simulation times,
    ### the control sequence and more (see simulator.py)
    print("Starting Simulation")
    res = sim.simulate(t, inputs)
    plt.figure()
    plt.plot(res['t'], res['output'][:,(5,6,7,8,9)], label="JitterEstimationStates")
    plt.legend()
    plt.show()


    ### Dump results to files:
    ### Simulation result object
    ### Control sequence (Must be stored for reproducing u_hat)
    simulation_filename = "simulation_results.pkl"
    ctrl_filename = "ctrl_sequence.pkl"    
    sim_file = open(os.path.join(output_dir, simulation_filename), 'wb')
    ctrl_file = open(os.path.join(output_dir, ctrl_filename), 'wb')
    pkl.dump(res, sim_file)
    pkl.dump(ctrl, ctrl_file)
    sim_file.close()
    ctrl_file.close()

    with open(os.path.join(output_dir, "recon_required.pkl") ,'wb') as file:
        pkl.dump([t,sys,inputs], file)

    return output_dir


def digtalFilter(t, sys, inp, Ts, order, eta2, ctrl_sequence, jitter_correction_method, sigma_noise, recon_dir):

    options = {'eta2':eta2,
               'sigmaU2':[1.],
               'noise':[{'std':sigma_noise, 'steeringVector':np.ones(order),'name':'Bastard'}]}
    inputs = inp
    
    sigma_jitter = 0
    if "jitterAsGaussianNoise" in jitter_correction_method:
        sigma_jitter = jitter_correction_method['jitterAsGaussianNoise']['sigma_jitter']
        WhiteNoiseJitter = system.Input(Ts, coefficients=np.zeros(len(t)), steeringVector=np.ones(order))
        
        inputs += (WhiteNoiseJitter,)
        options['sigmaU2'].append(sigma_jitter)
    if "jitterEstimation" in jitter_correction_method:
        num_states = jitter_correction_method['jitterEstimation']['num_states']
        # options['eta2'] = np.concatenate((options['eta2'],np.zeros((num_states-order))))
        options['eta2'] = np.ones(num_states)
        options['sigmaU2'] = [1., 1.]
        # tmp = options['eta2']
        # tmp1 = np.vstack((tmp, np.zeros((num_states - order, tmp.shape[0]))))
        # options['eta2'] = np.hstack((tmp1, np.zeros((tmp1.shape[0], num_states-order))))
    
    recon = reconstruction.WienerFilter(t, sys, inputs, options)

    ### Reconstruction is performed using the control sequence contained in the 'ctrl' object,
    ### which got updated when sim.simulate(...) was run
    
    u_hat_filename = "u_hat.csv"
    
    u_hat = recon.filter(ctrl_sequence)
    # plt.figure()
    # plt.plot(t,u_hat[:,1])
    # plt.show()
    
    with open(os.path.join(recon_dir, u_hat_filename), 'w') as u_hat_file:
        pd.DataFrame(u_hat).to_csv(u_hat_file, index=False)
        # u_hat.tofile(u_hat_file, sep="\n")


def main(boolSim=True, boolRec=True, jitterAmount=0, jitterAsGaussianNoise=False, sigma_jitter=0, sigma_noise=0):
    if not boolSim and not boolRec:
        warnings.warn("Neither simulation nor reconstruction is performed")

    ### Simulation sampling rate
    Ts = 80e-6

    ### Analog system parameter
    beta = 6250
    order = 5

    ### Length of simulation sequence
    size = 20000

    ### Input signal: Sinusoid
    frequency = 80
    amplitude = 1.
    phase = pi * 7. / 8. # Rad


    ### Control specifications
    kappa = 1.

    NumberOfJitterEstimationStates = 5
    data_dir = os.path.join(os.getcwd(), "simulation_data")

    simulation_time_start = time.time()
    if boolSim:
        output_dir = performSimulation(Ts,
                                       beta,
                                       size,
                                       order,
                                       frequency,
                                       amplitude,
                                       phase,
                                       kappa,
                                       jitterAmount,
                                       sigma_noise,
                                       data_dir,
                                       NumberOfJitterEstimationStates)

    total_simulation_time = time.time() - simulation_time_start
    if boolRec:
        ### Retrieve control sequence for reconstruction
        experiment_directory = os.path.join(data_dir, "noise{}_jitter{}".format(sigma_noise,jitterAmount))
        if not os.path.exists(experiment_directory):
            warnings.warn("Experiment directory '{}' not found".format(experiment_directory))
            # os.makedirs(experiment_directory)    
        
        try:
            with open(os.path.join(experiment_directory, "ctrl_sequence.pkl"), 'rb') as ctrl_file:
                ctrl = pkl.load(ctrl_file)
            ### Retrieve simulation times, analog system and input signal
            with open(os.path.join(experiment_directory, "recon_required.pkl"), 'rb') as recon_required_file:
                t, sys, inp = pkl.load(recon_required_file)
            with open(os.path.join(experiment_directory, "simulation_results.pkl"), 'rb') as sim_res_file:
                sim_res = pkl.load(sim_res_file)

        except IOError:
            print("Simulation data not found in {}\n".format(experiment_directory))

        # plt.figure()
        # plt.plot(sim_res['t'], sim_res['output'])
        # plt.legend(["%s" % x for x in range(order+NumberOfJitterEstimationStates)])


        ### Reconstruction methods:
        ###  To be implemented, somehow parse which methods should be used to reconstruct from the ctrl sequences

        recon_dir = os.path.join(experiment_directory, "reconstructions")
        if not os.path.exists(recon_dir):
            os.makedirs(recon_dir)
        # else:
        #     [shutil.rmtree(file) for file in os.listdir(recon_dir)]

        ### Digital Filter
        eta2 = np.ones(order)#*(beta/(2.*pi*500))**(order*2)

        if jitterAsGaussianNoise:
            jitter_correction_method = {'jitterAsGaussianNoise':{'sigma_jitter':sigma_jitter}}
        elif NumberOfJitterEstimationStates:
            jitter_correction_method = {'jitterEstimation':{'num_states': NumberOfJitterEstimationStates+order}}
        else:
            jitter_correction_method = {}

        reconstruction_time_start = time.time()
        
        digtalFilter(t, sys, inp, Ts, order+NumberOfJitterEstimationStates, eta2, ctrl, jitter_correction_method, sigma_noise, recon_dir)

        total_reconstruction_time = time.time() - reconstruction_time_start

def configParser():
    pass


def randomWalkPlusSinusoid():

    ### Simulation sampling rate
    Ts = 80e-6

    ### Analog system parameter
    beta = 6250
    order = 5

    NumberOfJitterEstimationStates = 5

    ### Length of simulation sequence
    size = 10000

    ### Input signal: Sinusoid
    frequency = 80
    amplitude = 1.
    phase = pi * 7. / 8. # Rad

    kappa = 1.

    sigma_noise = 5e-2

    t = np.linspace(0, (size-1)*Ts, size)
    inp_sin_coeffs = amplitude * np.sin(2.*np.pi*frequency*t + phase)
    inp_rand_coeffs = np.random.randn(size)*sigma_noise


    # N_fft = (1<<12)
    # freq, randsinspec = signal.welch(inp_rand_coeffs+inp_sin_coeffs,
    #                                1./Ts,
    #                                axis=0,
    #                                nperseg = N_fft,
    #                                nfft = (1<<16),
    #                                scaling='density')

    # freq, rand_spec = signal.welch(inp_rand_coeffs,
    #                                1./Ts,
    #                                axis=0,
    #                                nperseg = N_fft,
    #                                nfft = (1<<16),
    #                                scaling='density')

    # freq, sin_spec = signal.welch(inp_sin_coeffs,
    #                                1./Ts,
    #                                axis=0,
    #                                nperseg = N_fft,
    #                                nfft = (1<<16),
    #                                scaling='density')

    # plt.semilogx(freq, 10*np.log10(randsinspec), label="$Sin(2\pi f + \phi)$ + random walk")
    # plt.semilogx(freq, 10*np.log10(rand_spec), label="random walk")
    # plt.semilogx(freq, 10*np.log10(sin_spec), label="$Sin(2\pi f + \phi)$")
    # plt.legend()
    # plt.show()
    # exit()

    A_system = np.eye(order, k=-1)*beta
    c = np.eye(order)

    tmp1 = np.hstack( (A_system, np.zeros((order, NumberOfJitterEstimationStates))) )
    tmp2 = np.hstack( (np.zeros((NumberOfJitterEstimationStates, order)), A_system) )
    A_augmented = np.vstack( (tmp1, tmp2) )

    print(A_augmented)


    c_augmented = np.eye(order+NumberOfJitterEstimationStates)

    sys = system.System(A_system, c)
    sys_augmented = system.System(A_augmented, c_augmented)

    vector = np.zeros(order)
    vector[0] = beta    # This is \beta_1 in Figure 2 of [1]

    vector_augmented = np.zeros(order+NumberOfJitterEstimationStates)
    vector_augmented[0] = beta

    ### Define the corrupted sinusoidal signal
    inp_sin = system.Sin(Ts, amplitude=amplitude, frequency=frequency, phase=phase, steeringVector=vector)
    input_random_walk = system.Input(Ts, coefficients=inp_rand_coeffs, steeringVector=np.ones(order))

    corrupted_sinusoid = system.Input(Ts, coefficients=inp_rand_coeffs + inp_sin_coeffs, steeringVector=vector)

    inp_sin_augmented = system.Sin(Ts, amplitude=amplitude, frequency=frequency, phase=phase, steeringVector=vector_augmented)
    input_random_walk_augmented = system.Input(Ts, coefficients=inp_rand_coeffs, steeringVector=np.ones(order+NumberOfJitterEstimationStates))

    mixingMatrix = - kappa * beta * np.eye(order)
    ctrl = system.Control(mixingMatrix, size)

    mixingMatrix_augmented = -kappa * beta * np.eye(order+NumberOfJitterEstimationStates)
    ctrl_augmented = system.Control(mixingMatrix_augmented, size)


    sim = simulator.Simulator(sys, ctrl, options={'stateBound': (Ts*beta*kappa)/(1. - Ts*beta)+1.})
    sim_augmented = simulator.Simulator(sys_augmented, ctrl_augmented, options={'stateBound': (Ts*beta*kappa)/(1. - Ts*beta)+10})

    #'noise':[{'std':sigma_noise, 'steeringVector':np.ones(order)}],
    res = sim.simulate(t, (corrupted_sinusoid,))
    res_augmented = sim_augmented.simulate(t, (inp_sin_augmented, input_random_walk_augmented))

    plt.figure()
    plt.plot(res['t'], res['output'])

    plt.figure()
    plt.plot(res_augmented['t'], res_augmented['output'])
    plt.show()

    eta2 = np.ones(order)

    options = {'eta2':eta2,
               'sigmaU2':[1., 1.]}
               #'noise':[{'std':1e-5, 'steeringVector':np.ones(order),'name':'Bastard'}]}
    options2 = {'eta2':eta2,
               'sigmaU2':[1.]}

    options_augmented = {'eta2': np.ones(order+NumberOfJitterEstimationStates),
                         'sigmaU2':[1., 1.]}

    recon = reconstruction.WienerFilter(t, sys, (inp_sin, input_random_walk), options)
    recon_augmented = reconstruction.WienerFilter(t, sys_augmented, (inp_sin_augmented, input_random_walk_augmented), options_augmented)
    
    u_hat_no_noise = recon.filter(ctrl)
    u_hat_with_noise = recon_augmented.filter(ctrl_augmented)
    


    fig, ax = plt.subplots(2,1)
    ax[0].plot(t,u_hat_with_noise[:,0])
    ax[1].plot(t,u_hat_with_noise[:,1])

    

    N_fft = (1<<11)
    freq, uhat_spec = signal.welch(u_hat_with_noise[:,0],
                                   1./Ts,
                                   axis=0,
                                   nperseg = N_fft,
                                   nfft = (1<<16),
                                   scaling='density')
    freq, randomwalk_spec = signal.welch(u_hat_with_noise[:,1],
                                   1./Ts,
                                   axis=0,
                                   nperseg = N_fft,
                                   nfft = (1<<16),
                                   scaling='density')


    freq, uhat_spec_no_noise = signal.welch(u_hat_no_noise[:,0],
                                   1./Ts,
                                   axis=0,
                                   nperseg = N_fft,
                                   nfft = (1<<16),
                                   scaling='density')

    freq, random_walk_spec_no_noise = signal.welch(u_hat_no_noise[:,1],
                                   1./Ts,
                                   axis=0,
                                   nperseg = N_fft,
                                   nfft = (1<<16),
                                   scaling='density')


    fig2, ax2 = plt.subplots(2,1)
    ax2[0].semilogx(freq,10*np.log10(uhat_spec),'b', label="$\hat{u}$ PSD - Noise estimated by auxiliary states")
    ax2[0].semilogx(freq,10*np.log10(randomwalk_spec),'r', label="Estimated Noise PSD")
    

    ax2[1].semilogx(freq, 10*np.log10(uhat_spec_no_noise), 'b', label="$\hat{u}$ PSD - Noise assumed in reconstruction, no auxiliary states")
    ax2[1].semilogx(freq, 10*np.log10(random_walk_spec_no_noise), 'r', label="Estimated Noise PSD")
    fig2.legend()
    ax2[0].grid()
    ax2[1].grid()
    plt.show()


    


    # plt.figure()
    # plt.plot(res['t'], res['output'][:,(5,6,7,8,9)], label="JitterEstimationStates")
    # plt.legend()

    # plt.figure()
    # N_fft = (1<<12)
    # spec = np.empty((5,(1<<15)+1))
    # for i,sig in enumerate(res['output'][:,[5,6,7,8,9]].transpose()):
    #     print("i = {}".format(i))
    #     freq, tmp_spec = signal.welch(sig,
    #                                1./Ts,
    #                                axis=0,
    #                                nperseg = N_fft,
    #                                nfft = (1<<16),
    #                                scaling='density')
    #     print("Spec shape: {}".format(tmp_spec.shape))
    #     spec[i,:] = tmp_spec

    # [plt.semilogx(freq,10*np.log10(spec[i,:])) for i in range(5)]
    # plt.legend(["%s" % x for x in range(5)])
    # plt.show()



<<<<<<< HEAD
def multipleInputExperiment():
    run_time_start = time.time()

    # Input signals, let's try two sinusoids
    Ts = 80e-6
    size = 10000
    nfft = (1<<16)
    order = 5
    beta = 6250

    hz_per_bin = 1./(nfft * Ts)

    freq_1 = 420*hz_per_bin
    phase_1 = np.pi/3.
    amplitude_1 = 1.

    freq_2 = 700*hz_per_bin
    phase_2 = np.pi/5.
    amplitude_2 = 0.5

    print("Frequency 1: {}".format(freq_1))
    print("Frequency 2: {}".format(freq_2))

    t = np.linspace(0,(size-1)*Ts, size)

    sin_1 = amplitude_1 * np.sin(2.*np.pi*freq_1*t + phase_1)
    sin_2 = amplitude_2 * np.sin(2.*np.pi*freq_2*t + phase_2)

    sigma_noise = 1e-5
    random_walk = np.random.randn(size)*sigma_noise

    input_vector_1 = np.zeros(order)
    input_vector_1[0] = beta
    input_vector_2 = np.zeros(order)
    input_vector_2[1] = beta
    
    inp_sin_1 = system.Sin(Ts, amplitude=amplitude_1, frequency=freq_1, phase=phase_1, steeringVector=input_vector_1)
    inp_sin_2 = system.Sin(Ts, amplitude=amplitude_2, frequency=freq_2, phase=phase_2, steeringVector=input_vector_2)
    inp_random_walk = system.Input(Ts, coefficients=random_walk, steeringVector=np.ones(order))

    A_system = np.eye(order, k=-1)*beta
    c = np.eye(order)
    sys = system.System(A_system, c)

    kappa = 1.

    mixingMatrix = - kappa * beta * np.eye(order)
    ctrl_only_sin_1 = system.Control(mixingMatrix, size)
    ctrl_only_sin_2 = system.Control(mixingMatrix, size)
    ctrl_sin_1_plus_sin_2 = system.Control(mixingMatrix, size)

    ctrl_sin_1_plus_random_walk = system.Control(mixingMatrix, size)

    sim_only_sin_1 = simulator.Simulator(sys, ctrl_only_sin_1, options={'stateBound': (Ts*beta*kappa)/(1. - Ts*beta)+1.})
    sim_only_sin_2 = simulator.Simulator(sys, ctrl_only_sin_2, options={'stateBound': (Ts*beta*kappa)/(1. - Ts*beta)+1.})
    sim_sin_1_plus_sin_2 = simulator.Simulator(sys, ctrl_sin_1_plus_sin_2, options={'stateBound': (Ts*beta*kappa)/(1. - Ts*beta)+1.})
    sim_sin_1_plus_random_walk = simulator.Simulator(sys, ctrl_sin_1_plus_random_walk, options={'stateBound': (Ts*beta*kappa)/(1. - Ts*beta)+1.})
    
    res_only_sin_1 = sim_only_sin_1.simulate(t, (inp_sin_1,))
    res_only_sin_2 = sim_only_sin_2.simulate(t, (inp_sin_2,))
    res_sim_sin_1_plus_sin_2 = sim_sin_1_plus_sin_2.simulate(t, (inp_sin_1, inp_sin_2))
    res_sin_1_plus_random_walk = sim_sin_1_plus_random_walk.simulate(t, (inp_sin_1, inp_random_walk))


    eta2 = np.ones(order)

    options = {'eta2':eta2,
               'sigmaU2':[1., 25e-3]}
               #'noise':[{'std':1e-5, 'steeringVector':np.ones(order),'name':'Bastard'}]}
    options2 = {'eta2':eta2,
               'sigmaU2':[1.]}

    recon_only_sin_1 = reconstruction.WienerFilter(t, sys, (inp_sin_1,), options2)
    recon_only_sin_2 = reconstruction.WienerFilter(t, sys, (inp_sin_2,), options2)
    recon_sin_1_plus_sin_2 = reconstruction.WienerFilter(t, sys, (inp_sin_1, inp_sin_2), options)
    recon_sin_1_plus_random_walk = reconstruction.WienerFilter(t, sys, (inp_sin_1, inp_random_walk), options)
    
    
    u_hat_sin_1 = recon_only_sin_1.filter(ctrl_only_sin_1)
    u_hat_sin_2 = recon_only_sin_2.filter(ctrl_only_sin_2)
    u_hat_sin_1_plus_sin_2 = recon_sin_1_plus_sin_2.filter(ctrl_sin_1_plus_sin_2)
    u_hat_sin_1_plus_random_walk = recon_sin_1_plus_random_walk.filter(ctrl_sin_1_plus_random_walk)

    plt.plot(t, u_hat_sin_1_plus_sin_2[:,0], alpha=0.7, label="f = {}".format(freq_1))
    plt.plot(t, u_hat_sin_1_plus_sin_2[:,1], alpha=0.7, label="f = {}".format(freq_2))
    # plt.plot(t, u_hat_sin_1_plus_sin_2[:,2], alpha=0.7, label="Random Walk")

    print("Squared norm: {}".format(np.linalg.norm(u_hat_sin_1_plus_sin_2[:,0] - u_hat_sin_1_plus_sin_2[:,1])))

    plt.figure()
    # inp_sin = system.Sin(Ts, amplitude=amplitude, frequency=frequency, phase=phase, steeringVector=vector)
    # input_random_walk = system.Input(Ts, coefficients=inp_rand_coeffs, steeringVector=np.ones(order))
    # corrupted_sinusoid = system.Input(Ts, coefficients=inp_rand_coeffs + inp_sin_coeffs, steeringVector=vector)

    # inp_sin_coeffs = amplitude * np.sin(2.*np.pi*frequency*t + phase)
    # inp_rand_coeffs = np.random.randn(size)*sigma_noise

    # plt.plot(t,sin_1)
    # plt.figure()
    
    freq, spec_sin_1 = signal.welch(u_hat_sin_1, 1./Ts, axis=0, nperseg = (1<<12), nfft = nfft, scaling='density')
    freq, spec_sin_2 = signal.welch(u_hat_sin_2, 1./Ts, axis=0, nperseg = (1<<12), nfft = nfft, scaling='density')
    freq, spec_sin_1_2 = signal.welch(u_hat_sin_1_plus_sin_2[:,0], 1./Ts, axis=0, nperseg = (1<<12), nfft = nfft, scaling='density')
    freq, spec_sin_2_2 = signal.welch(u_hat_sin_1_plus_sin_2[:,1], 1./Ts, axis=0, nperseg = (1<<12), nfft = nfft, scaling='density')
    freq, spec_sin_1_no_random_walk = signal.welch(u_hat_sin_1_plus_random_walk[:,0], 1./Ts, axis=0, nperseg = (1<<12), nfft = nfft, scaling='density')
    freq, spec_random_walk_no_sin_1 = signal.welch(u_hat_sin_1_plus_random_walk[:,1], 1./Ts, axis=0, nperseg = (1<<12), nfft = nfft, scaling='density')
    # freq, spec_random_walk = signal.welch(u_hat_sin_1_plus_sin_2[:,2], 1./Ts, axis=0, nperseg = (1<<12), nfft = nfft, scaling='density')

    plt.semilogx(freq, 10*np.log10(np.abs(spec_sin_1)), label="$f = {:.2f}$Hz".format(freq_1), alpha=0.7)
    plt.semilogx(freq, 10*np.log10(np.abs(spec_sin_2)), label="$f = {:.2f}$Hz".format(freq_2), alpha=0.7)
    plt.semilogx(freq, 10*np.log10(np.abs(spec_sin_1_2)), label="$f_1 = {:.2f}$Hz".format(freq_1), alpha=0.7)
    plt.semilogx(freq, 10*np.log10(np.abs(spec_sin_2_2)), label="$f_2 = {:.2f}$Hz".format(freq_2), alpha=0.7)
    plt.legend()

    plt.figure()
    plt.semilogx(freq, 10*np.log10(np.abs(spec_sin_1_no_random_walk)), label="$f_2 = {:.2f}$Hz".format(freq_1), alpha=0.7)
    plt.semilogx(freq, 10*np.log10(np.abs(spec_random_walk_no_sin_1)), label="Random walk", alpha=0.7)    
    # plt.semilogx(freq, 10*np.log10(np.abs(spec_random_walk)), label="Random Walk", alpha=0.7)

    print("Run time: {} seconds".format(time.time() - run_time_start))

    plt.legend()
    plt.show()


def hadamardMatrix(n):
    return sp.linalg.hadamard(n)/np.sqrt(n)


np.seterr(all='raise')

def checkCovarianceMatrixConvergence(A,b,eta2=1):
    # Initialize Covariance matrix as identity:
    V = np.eye(A.shape[0])*1e5
    V_tmp = np.zeros_like(V)
    tau = 1e-8

    print("Checking Covariance Matrices")
    k = 0
    while not np.allclose(V,V_tmp):
        V_tmp = V
        try:
            V = V + tau * (np.dot(A,V) + np.transpose(np.dot(A,V)) + np.outer(b,b) - (1./eta2) * np.dot(V, V))
            # print(np.linalg.norm(V_frw - V_tmp))
            # if k > 500000:
            #     exit()
            # k+=1
        except FloatingPointError:
            print("V_frw:\n{}\n".format(V))
            print("V_frw.dot(V_frw):\n{}".format(np.dot(V, V)))
            return
    print("V condition number: %s" % (np.linalg.cond(V)))
    print("Covariance Matrices Converge To: \n%s"   % (V,))



def piBlockSystem(M=1, N=1, L=1, eta2_magnitude=1e4, sigma_sim_noise=1e-5, sigma_recon_noise=1e-5):
    start_time = time.time()

    Ts = 8e-5       # Sampling period
    num_periods_generated = 10
    size = round(num_periods_generated/Ts)    # Number of samples in the simulation
    print("# Samples in the simulation: %s" % (size,))
    num_inputs = 1
    t = np.linspace(0,(size-1)*Ts, size)    # Time grid for control updates

    kappa = 1#np.sqrt(L)
    beta = 6250#1. / (Ts * (1/np.sqrt(M) + kappa))
    print("beta = {}".format(beta))
    

    print("Input Stability Margin: {}".format(1. - Ts*beta*(1/np.sqrt(M) + kappa)))
    print("Internal Stability Margin: {}".format(1 - Ts*beta*(np.sqrt(L) + kappa)))
    
    print("Input Stability criterion: {}".format(Ts*beta*(1/np.sqrt(M) + kappa) <= 1))
    print("Internal Stability criterion: {}".format(Ts*beta*(np.sqrt(L) + kappa) <= 1))

    H = hadamardMatrix(M)   
    

    A = np.zeros((N*M, N*M))
    MixingPi = np.empty((N-1,M,M))

    
    if L == 4:
        if N > 1:
            for k in range(N-1):
                MixingPi[k] = beta*(np.outer(H[:,0],H[:,0])*L)/np.sqrt(L)
                A[(k+1)*M:(k+2)*M, (k)*M:(k+1)*M] = MixingPi[k]
    else:
        if N > 1:
            for k in range(N-1):
                MixingPi[k] = beta*(sum(np.outer(H[:,i],H[:,i]) for i in range(M)))
                A[(k+1)*M:(k+2)*M, (k)*M:(k+1)*M] = MixingPi[k]

    nfft = (1<<16)
    selected_FFT_bin = 5
    nperseg = (1<<10)

    input_signals = []
    frequencies = []
    for i in range(num_inputs):
        amplitude = 0
        frequency = (i+1)*(selected_FFT_bin/(nperseg*Ts))
        print("Sinusoid frequency: {} Hz".format(frequency))
        # size = 
        frequencies.append(frequency)
        phase = 0
        vector = np.zeros(N*M)
        vector[0:M] = beta*(H[:,i])
        input_signals.append(system.Sin(Ts, amplitude=amplitude, frequency=frequency, phase=phase, steeringVector=vector))
    input_signals = tuple(input_signals)

    print("A = \n%s\nb = \n%s" % (A, vector))


    print("Gain factor: beta*L/sqrt(L) = {}".format(beta*np.sqrt(M)))
    print("Predicted SNR increase over Gain factor = beta: {} dB".format(10*np.log10(1./np.sqrt(L))*2*(N-1)))

    # Check Forward Covariance Matrices:
    # checkCovarianceMatrixConvergence(A,vector)
    # Check Backward Covariance Matrices:
    # checkCovarianceMatrixConvergence(-A,vector)

    
    # plt.figure()
    # plt.plot(t,np.sin(2.*np.pi*frequency*t))
    # plt.show()
    # exit()


    # for i in range(1,N+1):
    #     A[i*M:(i+1)*M,i*M:(i+1)*M] = MixingPi[i-1]

    
    

    # tmp_2 = np.hstack( (Pi_1, np.zeros( (Pi_1.shape[0], (K+1)*M - Pi_1.shape[1]))) )
    # tmp_3 = np.hstack( (np.zeros((M, Pi_1.shape[1])), Pi_2, np.zeros( (M, (K+1)*M - Pi_1.shape[1]-Pi_2.shape[1]) ) ) )
    # for j in range(K):
    #     tmp = np.zeros((M,(K+1)*M))
    #     # print("{}:{}".format(M*j,M*j+M))
    #     tmp[0:M, M*j:M*j + M] = Pi_1
    #     A = np.vstack((A, tmp))
    # A = np.vstack((tmp_1, tmp_2, tmp_3))

    # tmp_2_identity = np.hstack( (np.eye(Pi_1.shape[0])*beta, np.zeros( (Pi_1.shape[0], (K+1)*M - Pi_1.shape[1]))) )
    # tmp_3_identity = np.hstack( (np.zeros((M, Pi_1.shape[1])), np.eye(Pi_2.shape[0])*beta, np.zeros( (M, (K+1)*M - Pi_1.shape[1]-Pi_2.shape[1]) ) ) )
    # A_identity = np.vstack((tmp_1, tmp_2_identity, tmp_3_identity))
    
    # A_identity = np.zeros(((N+1)*M, (N+1)*M))
    # for j in range(K):
    #     tmp = np.zeros((M,(N+1)*M))
    #     tmp[0:M, M*j:M*j + M] = np.eye(M)
    #     A_identity = np.vstack((A_identity, tmp))


    c = np.eye(N*M)

    sys = system.System(A, c)
    

    # sys_identity = system.System(A_identity, c)

    mixingMatrix = - kappa * beta * np.eye(N*M)
    ctrl = system.Control(mixingMatrix, size)
    # ctrl_identity = system.Control(mixingMatrix, size)

    sim = simulator.Simulator(sys, ctrl, options={'stateBound': (Ts*beta*kappa)/(1. - (Ts*beta/np.sqrt(L))),
                                                  'stateBoundInputs': (Ts*beta*kappa)/(1. - (Ts*beta/np.sqrt(M))),
                                                  'num_parallel_converters':M,
                                                  'noise': [{'std':sigma_sim_noise, 'steeringVector': beta*np.eye(N*M)[:,i]}  for i in range(N*M)]})   # /np.sqrt((N+1)*M)


    # sim_identity = simulator.Simulator(sys_identity, ctrl_identity, options={'stateBound': (Ts*beta*kappa)/(1. - Ts*beta)+1.,
                                                  # 'noise': [{'std':1., 'steeringVector': np.ones((K+1)*M)}]})


    res = sim.simulate(t, input_signals)
    # res_identity = sim_identity.simulate(t, tuple(input_signals))

    fig, ax = plt.subplots()
    ax.plot(res['t'], res['output'])
    plt.legend(["%s" % x for x in range(res['output'].shape[0])])
    plt.title("Block diagonal Pi System")
    plt.savefig("./data/Tuesday 15.1.2019/SimulationOutput_M={}_N={}_L={}.png".format(M,N,L), dpi=300, format='png')    #M=1, N=1, L=1
    plt.close(fig)

    # exit()
    
    # plt.figure()
    # plt.plot(res_identity['t'], res_identity['output'])
    # plt.title("Block diagonal identity System")

    eta2 = np.ones(N*M) * eta2_magnitude

    # sigmaU2 = np.zeros((N+1)*M)
    # sigmaU2[0:M] = 1.
    options = {'eta2':eta2,
               'sigmaU2':[1.],
               'noise':[{'std': sigma_recon_noise, 'steeringVector': beta*np.eye(N*M)[:,i], 'name':'noise_{}'.format(i)} for i in range(N*M)]} # /np.sqrt((N+1)*M)

    recon = reconstruction.WienerFilter(t, sys, input_signals, options)
    input_estimates = recon.filter(ctrl)

    # recon_identity = reconstruction.WienerFilter(t, sys_identity, tuple(input_signals), options)
    # input_estimates_identity = recon.filter(ctrl_identity)

    print("Run Time: {} seconds".format(time.time()-start_time))

    spectrums = np.empty((num_inputs, nfft//2 + 1))
    for i in range(num_inputs):
        freq, spec = signal.welch(input_estimates[:,i], 1./Ts, window='hann', axis=0, nperseg = nperseg, nfft = nfft , scaling='density')
        spectrums[i,:] = spec
        # print(freq.shape, spec.shape)

        # plt.semilogx(freq, 10*np.log10(np.abs(spec)), label="$f = {:.2f}$Hz".format(frequencies[i]), alpha=0.7)
        # plt.legend()
        # plt.show()

    # specs_identity = np.empty((M, nfft//2 + 1))
    # for i in range(M):
    #     freq, spec = signal.welch(input_estimates[:,i], 1./Ts, axis=0, nperseg = (1<<12), nfft = nfft, scaling='density')
    #     specs_identity[i,:] = spec

        # plt.semilogx(freq, 10*np.log10(np.abs(spec)), label="$f = {:.2f}$Hz".format(frequencies[i]), alpha=0.7)
        # plt.legend()
        # plt.show()

    plt.figure()
    [plt.semilogx(freq, 10*np.log10(np.abs(spec)), label="") for spec in spectrums]
    plt.grid()
    # [plt.semilogx([f,f], [-120, 0]) for f in frequencies]
    plt.title("Block diagonal Pi System")

    return freq, spec




def plainVanilla():
    Ts = 80e-6
    beta = 6250
    order = 5
    size = 20000

    frequency = 80
    amplitude = 1.
    phase = pi * 7. / 8. # Rad
    vector = np.zeros(order)
    vector[0] = beta

    kappa = 1.
    t = np.linspace(0, (size-1)*Ts, size)

    stability = Ts*beta*(1 + kappa) <= 1
    print("Stability criterion: {}".format(stability))

    inp = system.Sin(Ts, amplitude=amplitude, frequency=frequency, phase=phase, steeringVector=vector)

    A = np.eye(order, k=-1)*beta
    c = np.eye(order)
    sys = system.System(A,c)

    checkCovarianceMatrixConvergence(A,vector)

    mixingMatrix = - kappa * beta * np.eye(order)
    ctrl = system.Control(mixingMatrix, size)

    sim = simulator.Simulator(sys, ctrl, options={'stateBound': (Ts*beta*kappa)/(1. - Ts*beta)+1.,
                                                  'noise': [{'std':1., 'steeringVector': np.ones(order)}]})
    res = sim.simulate(t, (inp,))

    eta2 = np.ones(order)
    options = {'eta2':eta2,
               'sigmaU2':[1.],
               'noise':[{'std': 1., 'steeringVector': np.ones(order), 'name':'Bastard'}]}

    recon = reconstruction.WienerFilter(t, sys, (inp,), options)
    input_estimates = recon.filter(ctrl)

    nfft = (1<<16)
    freq, spec = signal.welch(input_estimates, 1./Ts, axis=0, nperseg = (1<<12), nfft = nfft, scaling='density')

    plt.figure()
    plt.semilogx(freq, 10*np.log10(np.abs(spec)))
    plt.grid()
    plt.show()


if __name__ == "__main__":
    # arg_parser = argparse.ArgumentParser(description="")
    # arg_parser.add_argument("--jitterAsGaussianNoise", type=bool, default=False)
    # arg_parser.add_argument("--jitterAmount", type=float, default=0)
    # arg_parser.add_argument("--sigma_jitter", type=float, default=0)
    # arg_parser.add_argument("--sigma_noise", type=float, default=0)
    # arg_parser.add_argument("--boolSim", type=bool, default=False)
    # arg_parser.add_argument("--boolRec", type=bool, default=False)

    # args = vars(arg_parser.parse_args())


    # plt.figure()
    num_parallel_convs = [4]    # M
    specs = np.empty((len(num_parallel_convs),(1<<15)+1))

    # L = 1
    # N = 1

    sigma_sim_noise=1e-4
    sigma_recon_noise=1e-4
    eta2_magnitude = 1e1

    for N in [2]:
        for L in [1]:
            for i,M in enumerate(num_parallel_convs):
                if (M==1 and L!=1): continue
                print("L = {}\nN = {}\nM = {}".format(L,N,M))
                freq, spec = piBlockSystem(M=M,
                                           N=N,
                                           L=L,
                                           eta2_magnitude=eta2_magnitude,
                                           sigma_sim_noise=sigma_sim_noise,
                                           sigma_recon_noise=sigma_recon_noise)
                # freqs[i,:] = freq
                specs[i,:] = spec
                # plt.semilogx(freq, 10*np.log10(np.abs(spec)), label="M={}, N={}, L={}".format(M,N,L))


            df = pd.DataFrame({M: specs[i,:] for i,M in enumerate(num_parallel_convs)}, index=freq)
            df.to_csv("./data/Tuesday 15.1.2019/PSD_eta={}_M={}_N={}_beta=6250_L={}_forthtry.csv".format(eta2_magnitude,M,N,L))

    
    # plt.grid()
    # plt.legend()

    # multipleInputExperiment()
    # randomWalkPlusSinusoid()

    # main(**args)
=======

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(description="")
    arg_parser.add_argument("--jitterAsGaussianNoise", type=bool, default=False)
    arg_parser.add_argument("--jitterAmount", type=float, default=0)
    arg_parser.add_argument("--sigma_jitter", type=float, default=0)
    arg_parser.add_argument("--sigma_noise", type=float, default=0)
    arg_parser.add_argument("--boolSim", type=bool, default=False)
    arg_parser.add_argument("--boolRec", type=bool, default=False)

    args = vars(arg_parser.parse_args())

    randomWalkPlusSinusoid()
    # main(**args)
>>>>>>> Working files (i.e. not nice clean code) for simulations and data processing
