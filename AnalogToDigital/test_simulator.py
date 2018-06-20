import system
import simulator
import numpy as np
import matplotlib.pyplot as plt


def testSystemSetup():
    size = 1000
    Ts = 0.1
    coef = np.random.rand(size)
    vector = np.array([1., 0, 0])
    inp = system.Input(Ts, coefficients=coef, steeringVector=vector)

    A = np.random.rand(3,3)
    c = np.eye(3)

    sys = system.System(A, c)

    mixingMatrix = - np.eye(3)

    ctrl = system.Control(mixingMatrix, size)

    return {
        'size': size,
        'Ts': Ts,
        'inp': inp,
        'sys': sys,
        'ctrl': ctrl,
    }

def testSimulator():
    setup = testSystemSetup()
    sim = simulator.Simulator(setup['sys'], control=setup['ctrl'])
    t = np.linspace(0., 99., setup['size'])
    res = sim.simulate(t, (setup['inp'],))
    plt.figure()
    plt.plot(res['t'], res['output'])



def test_integratorChain():
    size = 1000
    Ts = 0.1
    coef = np.ones(size)
    vector = np.array([1., 0, 0])
    inp = system.Input(Ts, coefficients=coef, steeringVector=vector)

    A = np.eye(3, k=-1)
    c = np.eye(3)

    sys = system.System(A, c)

    mixingMatrix = - 1e1 * np.eye(3)

    ctrl = system.Control(mixingMatrix, size)

    sim = simulator.Simulator(sys, ctrl)
    t = np.linspace(0., 99., size)
    res = sim.simulate(t, (inp,))
    plt.figure()
    plt.plot(res['t'], res['output'])
    plt.legend(["%s" % x for x in range(3)])


def test_noisy_integratorChain():
    size = 1000
    Ts = 0.1
    order = 3
    coef = np.ones(size)
    vector = np.array([1., 0, 0])
    inp = system.Input(Ts, coefficients=coef, steeringVector=vector)

    A = np.eye(3, k=-1)
    c = np.eye(3)

    noiseVariance = np.ones(3) * 1e-4
    noiseSources = []
    for index in range(order):
        if noiseVariance[index]>0:
            vector = np.zeros(order)
            vector[index] = 1
            noiseSources.append(
            {
                "std": np.sqrt(noiseVariance[index]),
                "steeringVector": vector,
                "name": "Noise Source %s" % index
            }
            )


    options = {
        'noise': noiseSources
    }

    sys = system.System(A, c)

    mixingMatrix = - 1e1 * np.eye(3)

    ctrl = system.Control(mixingMatrix, size)

    sim = simulator.Simulator(sys, ctrl, options=options)
    t = np.linspace(0., 99., size)
    res = sim.simulate(t, (inp,))
    plt.figure()
    plt.plot(res['t'], res['output'])
    plt.legend(["%s" % x for x in range(3)])



if __name__ == "__main__":
    testSimulator()
    test_integratorChain()
    test_noisy_integratorChain()
    plt.show()
