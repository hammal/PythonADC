###############################
###       ADC Packages      ###
###############################
import AnalogToDigital.system as system
import AnalogToDigital.simulator as simulator
import AnalogToDigital.reconstruction as reconstruction
import AnalogToDigital.filters as filters
import AnalogToDigital.evaluation as evaluation


import time
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from scipy import signal



def hadamardMatrix(n):
    return sp.linalg.hadamard(n)/np.sqrt(n)


np.seterr(all='raise')


def checkCovarianceMatrixConvergence(A,b,eta2=1):
    # Initialize V_frw:
    V_frw = np.eye(A.shape[0])
    V_tmp = np.zeros_like(V_frw)
    tau = 1e-7

    while not np.allclose(V_frw,V_tmp):
        V_tmp = V_frw
        try:
            V_frw = V_frw + tau * (np.dot(A,V_frw) + np.transpose(np.dot(A,V_frw)) + np.outer(b,b) - (1./eta2) * np.dot(V_frw, V_frw))
        except FloatingPointError:
            print("V_frw:\n{}\n".format(V_frw))
            print("V_frw.dot(V_frw):\n{}".format(np.dot(V_frw, V_frw)))
            return
    print("V_frw condition number: %s" % (np.linalg.cond(V_frw)))
    print("Covariance Matrices Converge for: \n%s" % V_frw)


def piBlockSystem1():
    start_time = time.time()

    size = 20000    # Number of samples in the simulation
    M = (1<<1)      # Number of parallel controlnverters
    N = 4           # Number of PI mixing submatrices
    Ts = 8e-5       # Sampling period
    num_inputs = 1
    t = np.linspace(0,(size-1)*Ts, size)    # Time grid for control updates

    beta = 6250
    kappa = 1

    stability = Ts*beta*(1/np.sqrt(M) + kappa) <= 1
    print("Stability margin: {}".format(1. - Ts*beta*(1/np.sqrt(M) + kappa)))
    print("Stability criterion: {}".format(stability))

    H = hadamardMatrix(M)
    L = 1
    
    
    A = np.zeros(((N+1)*M, (N+1)*M))
    MixingPi = np.empty((N,M,M))
    for k in range(N):
        MixingPi[k] = beta*np.outer(H[:,0],H[:,0])
        # (beta)*(0.2*np.outer(H[:,0],H[:,0])
        #                        + 0.1*np.outer(H[:,1],H[:,1])
        #                        + 0.3*np.outer(H[:,2],H[:,2])
        #                        + 0.4*np.outer(H[:,3],H[:,3]))  # Index set = {0,0}
        A[(k+1)*M:(k+2)*M, (k)*M:(k+1)*M] = MixingPi[k]

    nperseg = (1<<16)
    selected_FFT_bin = 250.

    input_signals = []
    frequencies = []
    # sins = np.empty((M,size))
    for i in range(num_inputs):
        amplitude = 1
        frequency = (i+1)*(selected_FFT_bin/(nperseg*Ts))
        print("{} Hz".format(frequency))
        # size = 
        frequencies.append(frequency)
        phase = 0
        vector = np.zeros((N+1)*M)
        vector[0:M] = beta * (H[:,i])
        input_signals.append(system.Sin(Ts, amplitude=amplitude, frequency=frequency, phase=phase, steeringVector=vector))

    input_signals = tuple(input_signals)

    print("A = \n%s\nb = \n%s" % (A, vector))
    checkCovarianceMatrixConvergence(A,vector)
    checkCovarianceMatrixConvergence(-A,vector)

    c = np.eye((N+1)*M)
    sys = system.System(A, c)
    

    mixingMatrix = - kappa * beta * np.eye((N+1)*M)
    ctrl = system.Control(mixingMatrix, size)

    sim = simulator.Simulator(sys, ctrl, options={'stateBound': (Ts*beta*kappa)/(1. - Ts*beta)+1.,
                                                  'noise': [{'std':1e-4, 'steeringVector': np.ones((N+1)*M)}]})


    res = sim.simulate(t, input_signals)

    # plt.figure()
    # plt.plot(res['t'], res['output'])
    # plt.legend(["%s" % x for x in range(res['output'].shape[0])])
    # plt.show()

    eta2 = np.ones((N+1)*M) * 1e5

    options = {'eta2':eta2,
               'sigmaU2':[1.],
               'noise':[{'std': 1e-4, 'steeringVector': np.ones((N+1)*M), 'name':'Bastard'}]}

    recon = reconstruction.WienerFilter(t, sys, input_signals, options)
    input_estimates = recon.filter(ctrl)


    print("Run Time: {} seconds".format(time.time()-start_time))

    nfft = (1<<16)
    spectrums = np.empty((num_inputs, nfft//2 + 1))
    for i in range(num_inputs):
        freq, spec = signal.welch(input_estimates[:,i], 1./Ts, axis=0, nperseg = nperseg, nfft = nfft, scaling='density')
        spectrums[i,:] = spec

    
    plt.figure()
    [plt.semilogx(freq, 10*np.log10(np.abs(spec)), label="") for spec in spectrums]
    plt.grid()
    plt.title("Block diagonal Pi System")

def piBlockSystem2():
    start_time = time.time()

    size = 20000    # Number of samples in the simulation
    M = (1<<1)      # Number of parallel controlnverters
    N = 2           # Number of PI mixing submatrices
    Ts = 8e-5       # Sampling period
    num_inputs = 2
    t = np.linspace(0,(size-1)*Ts, size)    # Time grid for control updates

    beta = 6250
    kappa = 1

    stability = Ts*beta*(1/np.sqrt(M) + kappa) <= 1
    print("Stability margin: {}".format(1. - Ts*beta*(1/np.sqrt(M) + kappa)))
    print("Stability criterion: {}".format(stability))

    H = hadamardMatrix(M)
    L = 1
    
    
    A = np.zeros(((N+1)*M, (N+1)*M))
    MixingPi = np.empty((N,M,M))
    for k in range(N):
        MixingPi[k] = (beta)*(0.2*np.outer(H[:,0],H[:,0])
                               + 0.1*np.outer(H[:,1],H[:,1]))
                            #    + 0.3*np.outer(H[:,2],H[:,2])
        #                        + 0.4*np.outer(H[:,3],H[:,3]))  # Index set = {0,0}
        A[(k+1)*M:(k+2)*M, (k)*M:(k+1)*M] = MixingPi[k]

    nperseg = (1<<16)
    selected_FFT_bin = 250.

    input_signals = []
    frequencies = []
    # sins = np.empty((M,size))
    for i in range(num_inputs):
        amplitude = 1
        frequency = (i+1)*(selected_FFT_bin/(nperseg*Ts))
        print("{} Hz".format(frequency))
        # size = 
        frequencies.append(frequency)
        phase = 0
        vector = np.zeros((N+1)*M)
        vector[0:M] = beta * (H[:,i])
        input_signals.append(system.Sin(Ts, amplitude=amplitude, frequency=frequency, phase=phase, steeringVector=vector))

    input_signals = tuple(input_signals)

    print("A = \n%s\nb = \n%s" % (A, vector))
    checkCovarianceMatrixConvergence(A,vector)
    checkCovarianceMatrixConvergence(-A,vector)

    c = np.eye((N+1)*M)
    sys = system.System(A, c)
    

    mixingMatrix = - kappa * beta * np.eye((N+1)*M)
    ctrl = system.Control(mixingMatrix, size)

    sim = simulator.Simulator(sys, ctrl, options={'stateBound': (Ts*beta*kappa)/(1. - Ts*beta)+1.,
                                                  'noise': [{'std':1e-4, 'steeringVector': np.ones((N+1)*M)}]})


    res = sim.simulate(t, input_signals)

    # plt.figure()
    # plt.plot(res['t'], res['output'])
    # plt.legend(["%s" % x for x in range(res['output'].shape[0])])
    # plt.show()

    eta2 = np.ones((N+1)*M) * 1e0

    options = {'eta2':eta2,
               'sigmaU2':[1., 1.],
               'noise':[{'std': 1e-4, 'steeringVector': np.ones((N+1)*M), 'name':'Bastard'}]}

    recon = reconstruction.WienerFilter(t, sys, input_signals, options)
    input_estimates = recon.filter(ctrl)


    print("Run Time: {} seconds".format(time.time()-start_time))

    nfft = (1<<16)
    spectrums = np.empty((num_inputs, nfft//2 + 1))
    for i in range(num_inputs):
        freq, spec = signal.welch(input_estimates[:,i], 1./Ts, axis=0, nperseg = nperseg, nfft = nfft, scaling='density')
        spectrums[i,:] = spec

    
    plt.figure()
    [plt.semilogx(freq, 10*np.log10(np.abs(spec)), label="") for spec in spectrums]
    plt.grid()
    plt.title("Block diagonal Pi System")

    
if __name__ == "__main__":
    piBlockSystem1()
    piBlockSystem2()
    plt.show()
