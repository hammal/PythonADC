import AnalogToDigital.system as system
import AnalogToDigital.simulator as simulator
import numpy as np
import scipy as sp
import time



def hadamardMatrix(n):
    return sp.linalg.hadamard(n)/np.sqrt(n)


def testRuntime():
    ### Set up a system with N=3, M=4, L=1
    N = 3
    M = 1
    bitsPerControl = M
    A = np.zeros((N*M, N*M))
    H = hadamardMatrix(M)
    beta = 6250
    sigma2_thermal = 1e-6
    

    for k in range(N-1):
        A[(k+1)*M:(k+2)*M, (k)*M:(k+1)*M] = beta * np.sqrt(M) * (np.outer(H[:,0],H[:,0]))

    sampling_period = 8e-5
    OSR = 16
    input_frequency = 1./(sampling_period * 2 * OSR)
    vector = np.zeros(M*N)
    vector[0:M] = H[:,0]

    input_signal = (system.Sin(sampling_period,
                               amplitude=1,
                               frequency=input_frequency,
                               phase=0,
                               steeringVector=vector),)

    c = np.eye(N*M)
    sys = system.System(A=A, c=c, b=vector)
    systemResponse = lambda f: np.dot(sys.frequencyResponse(f), sys.b)
    eta2_magnitude = np.max(np.abs(systemResponse(1./(2. * sampling_period * OSR)))**2)



    # controller == 'subspaceController':
    ctrlMixingMatrix = np.zeros((N*M, N))
    # if dither:
        # self.ctrlMixingMatrix =  (np.random.randint(2,size=(N*M , N))*2 - 1)*beta*1e-3
    for i in range(N):
      ctrlMixingMatrix[i*M:(i+1)*M,i] = - np.sqrt(M) * beta * H[:,0]

    ctrlOptions = {
            'bitsPerControl':bitsPerControl,
            'bound':1,
        }

    simulation_options = {'noise':[{'std':sigma2_thermal,
                          'steeringVector': beta*np.eye(N*M)[:,i]}  for i in range(M*N)],
                          'numberOfAdditionalPoints':0}    
    initalState = (2*np.random.rand(N*M) - np.ones(N*M))*1e-3

    simulation_runtimes = []
    sizes = np.linspace(1e3,1e5,5, dtype=int)
    for size in sizes:
        ctrl = system.Control(ctrlMixingMatrix, size, options=ctrlOptions)
        t = np.linspace(0,(size-1)*sampling_period, size)

        sim = simulator.Simulator(sys, ctrl, options=simulation_options, initalState=initalState)
        sim_start_time = time.time()
        result = sim.simulate(t, input_signal)
        current_runtime = time.time() - sim_start_time
        print("\n################\n")
        print("Number of samples: %i" % size)
        print("Runtime: %.3f seconds" % current_runtime)
        print("%.3f milliseconds/sample" % (1000*(current_runtime/size)))
        print("\n################")
        simulation_runtimes.append(current_runtime)

    # Fit the data to a straight line and print the coefficient
    # y = least_squares_coefficient * x
    least_squares_coefficient = np.dot(sizes, np.array(simulation_runtimes)) / np.dot(sizes, sizes)
    print("Least-Squares Coefficient: %f ms/sample" % (least_squares_coefficient*1000))

    # Compute the average square error of the least squares model
    mse = (sum((least_squares_coefficient*x - y)**2 for x,y in zip(sizes, simulation_runtimes)))/5
    print("MSE = %.2f" % mse)


if __name__ == "__main__":
    testRuntime()
