import system
import simulator
import reconstruction
import numpy as np
import matplotlib.pyplot as plt
import evaluation
import filters


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


def test_evaluate_PlotTransferFunctions_For_PostFiltering():
    size = 10000
    order = 2
    postFilterOrder = 2
    Ts = 0.1
    eta2 = 1e-6
    filterEta2 = 1e-0


    amplitude = 1.
    frequency = 1e-1
    phase = np.pi * 7. / 8.
    vector = np.zeros(order)
    vector[0] = 1.
    inp = system.Sin(Ts, amplitude=amplitude, frequency=frequency, phase=phase, steeringVector=vector)


    # coef = np.random.rand(size) * 2. - 1.
    # vector = np.zeros(order)
    # vector[0] = 2.
    # inp = system.FirstOrderHold(Ts, coefficients=coef, steeringVector=vector)


    A = 2 * np.eye(order, k=-1, dtype=np.float)
    # make causal filter
    A +=  0. * np.eye(order, k=0)
    c = np.eye(order, dtype=np.float)

    sys = system.System(A, c)

    mixingMatrix = - 1e1 * np.eye(order)

    ctrl = system.Control(mixingMatrix, size)

    sim = simulator.Simulator(sys, ctrl)
    t = np.linspace(0., 99., size)
    res = sim.simulate(t, (inp,))
    plt.figure()
    plt.plot(res['t'], res['output'])
    plt.legend(["%s" % x for x in range(order)])

    # System 1
    recon = reconstruction.WienerFilter(t, sys, (inp,), {"eta2":eta2*np.ones(order)})

    u_hat = recon.filter(ctrl)

    ev = evaluation.Evaluation(sys, u_hat, (inp,))

    # Post Filtering
    bandwith = 2 * np.pi * 1e-1

    print("Bandwith = %s" % bandwith)

    postFilterTF = filters.TransferFunction()
    postFilterTF.butterWorth(postFilterOrder, (bandwith))
    postFilterElliptic = filters.TransferFunction()
    postFilterElliptic.iirFilter(postFilterOrder, (bandwith,), 'ellip')
    postFilterCheby1 = filters.TransferFunction()
    postFilterCheby1.iirFilter(postFilterOrder, (bandwith,), 'cheby1')
    postFilterCheby2 = filters.TransferFunction()
    postFilterCheby2.iirFilter(postFilterOrder, (bandwith,), 'cheby2')
    postFilterBessel = filters.TransferFunction()
    postFilterBessel.iirFilter(postFilterOrder, (bandwith,), 'bessel')


    plt.figure()
    w1, h1 = postFilterTF.frequencyResponse()
    w2, h2 = postFilterElliptic.frequencyResponse()
    w3, h3 = postFilterCheby1.frequencyResponse()
    w4, h4 = postFilterCheby2.frequencyResponse()
    w5, h5 = postFilterBessel.frequencyResponse()
    plt.semilogx(w1/np.pi/2., 20 * np.log10(abs(h1)), label="Butterworth")
    plt.semilogx(w2/np.pi/2., 20 * np.log10(abs(h2)), label="Elliptic")
    plt.semilogx(w3/np.pi/2., 20 * np.log10(abs(h3)), label="Cheby 1")
    plt.semilogx(w4/np.pi/2., 20 * np.log10(abs(h4)), label="Cheby 2")
    plt.semilogx(w5/np.pi/2., 20 * np.log10(abs(h5)), label="Bessel")
    plt.legend()
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Amplitude response [dB]')
    plt.grid()
    # plt.show()

    # print(postFilterTF.a, postFilterTF.b)
    postFilter1 = filters.Filter()
    postFilter2 = filters.Filter()
    postFilter3 = filters.Filter()
    postFilter4 = filters.Filter()
    postFilter5 = filters.Filter()
    postFilter1.tf2lssObservableForm(postFilterTF.b, postFilterTF.a)
    postFilter2.tf2lssObservableForm(postFilterElliptic.b, postFilterElliptic.a)
    postFilter3.tf2lssObservableForm(postFilterCheby1.b, postFilterCheby1.a)
    postFilter4.tf2lssObservableForm(postFilterCheby2.b, postFilterCheby2.a)
    postFilter5.tf2lssObservableForm(postFilterBessel.b, postFilterBessel.a)

    # eta2PostFilter = np.concatenate((eta2 * np.ones(order), filterEta2 * np.ones(postFilter.c.shape[1])))
    eta2PostFilter = eta2 * np.ones(order)
    reconstructionFiltered1 = reconstruction.WienerFilterWithPostFiltering(t, sys, (inp,), postFilter1, {"eta2":eta2PostFilter})
    reconstructionFiltered2 = reconstruction.WienerFilterWithPostFiltering(t, sys, (inp,), postFilter2, {"eta2":eta2PostFilter})
    reconstructionFiltered3 = reconstruction.WienerFilterWithPostFiltering(t, sys, (inp,), postFilter3, {"eta2":eta2PostFilter})
    reconstructionFiltered4 = reconstruction.WienerFilterWithPostFiltering(t, sys, (inp,), postFilter4, {"eta2":eta2PostFilter})
    reconstructionFiltered5 = reconstruction.WienerFilterWithPostFiltering(t, sys, (inp,), postFilter5, {"eta2":eta2PostFilter})

    print(ctrl.mixingMatrix)

    u_hatP_1 = reconstructionFiltered1.filter(ctrl)
    print(ctrl.mixingMatrix)

    print(postFilter2)
    u_hatP_2 = reconstructionFiltered2.filter(ctrl)
    u_hatP_3 = reconstructionFiltered3.filter(ctrl)
    u_hatP_4 = reconstructionFiltered4.filter(ctrl)
    u_hatP_5 = reconstructionFiltered5.filter(ctrl)

    evP1 = evaluation.Evaluation(sys, u_hatP_1, (inp,))
    evP2 = evaluation.Evaluation(sys, u_hatP_2, (inp,))
    evP3 = evaluation.Evaluation(sys, u_hatP_3, (inp,))
    evP4 = evaluation.Evaluation(sys, u_hatP_4, (inp,))
    evP5 = evaluation.Evaluation(sys, u_hatP_5, (inp,))

    freqsLim = [1e-2, 1e2]

    # figure1 = ev.PlotTransferFunctions(freqsLim)
    figure2 = ev.PlotPowerSpectralDensity(t)
    # figure3 = ev2.PlotTransferFunctions(freqsLim)
    figure4 = evP1.PlotPowerSpectralDensity(t)
    figure5 = evP2.PlotPowerSpectralDensity(t)
    figure6 = evP3.PlotPowerSpectralDensity(t)
    figure7 = evP4.PlotPowerSpectralDensity(t)
    figure8 = evP5.PlotPowerSpectralDensity(t)

    plt.figure()
    plt.plot(u_hat, label="u_hat")
    # plt.plot(u_hatP_1, label="Butter")
    # plt.plot(u_hatP_2, label="Elliptic")
    plt.plot(u_hatP_3, label="Chebyshev 1")
    # plt.plot(u_hatP_4, label="Chebychev 2")
    plt.plot(u_hatP_5, label="Bessel")
    # plt.plot(u_hat - u_hat2, label="diff")
    plt.legend()

if __name__ == "__main__":
    # test_evaluate_PlotTransferFunctions()
    test_evaluate_PlotTransferFunctions_For_PostFiltering()
    plt.show()
