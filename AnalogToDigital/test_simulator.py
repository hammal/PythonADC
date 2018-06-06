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
    coef = np.ones(size)
    vector = np.array([1., 0, 0])
    inp = system.Input(Ts, coefficients=coef, steeringVector=vector)

    A = np.eye(3, k=-1)
    c = np.eye(3)

    noise = {"standardDeviation": 1e-4}

    sys = system.System(A, c)

    mixingMatrix = - 1e1 * np.eye(3)

    ctrl = system.Control(mixingMatrix, size)

    sim = simulator.Simulator(sys, ctrl, options={"noise": noise})
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
