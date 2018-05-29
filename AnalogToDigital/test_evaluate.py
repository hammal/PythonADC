import system
import simulator
import reconstruction
import numpy as np
import matplotlib.pyplot as plt
import evaluation


def test_evaluate_PlotTransferFunctions():
    size = 10000
    order = 7
    Ts = 0.1
    coef = np.random.rand(size) * 2. - 1.
    vector = np.zeros(order)
    vector[0] = 2.
    inp = system.FirstOrderHold(Ts, coefficients=coef, steeringVector=vector)

    A = 2 * np.eye(order, k=-1)
    c = np.eye(order)

    sys = system.System(A, c)

    mixingMatrix = - 1e1 * np.eye(order)

    ctrl = system.Control(mixingMatrix, size)

    sim = simulator.Simulator(sys, ctrl)
    t = np.linspace(0., 99., size)
    res = sim.simulate(t, (inp,))
    plt.figure()
    plt.plot(res['t'], res['output'])
    plt.legend(["%s" % x for x in range(order)])


    recon = reconstruction.WienerFilter(t, sys, (inp,))

    u_hat = recon.filter(ctrl)

    ev = evaluation.Evaluation(sys, u_hat, (inp,))

    freqsLim = [1e-2, 1e2]

    figure1 = ev.PlotTransferFunctions(freqsLim)
    figure2 = ev.PlotPowerSpectralDensity(t)

if __name__ == "__main__":
    test_evaluate_PlotTransferFunctions()
    plt.show()
