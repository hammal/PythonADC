import AnalogToDigital.system as system
import AnalogToDigital.simulator as simulator
import AnalogToDigital.reconstruction as reconstruction
import numpy as np
import matplotlib.pyplot as plt
import AnalogToDigital.evaluation as evaluation
import AnalogToDigital.filters as filters
from scipy import signal
import AnalogToDigital as ADC
import AnalogToDigital.unstable_linear_filter as unstable_linear_filter

def test_evaluate_PlotTransferFunctions():
    size = 10000
    order = 7
    Ts = 0.1
    beta = 5.
    coef = np.random.rand(size) * 2. - 1.
    vector = np.zeros(order)
    vector[0] = 2.
    inp = system.FirstOrderHold(Ts, coefficients=coef, steeringVector=vector)
    T = 40
    t = 10 ** (T/10.)
    rho = np.power(t, 1./order)/beta * np.sqrt(2)

    A = beta * np.eye(order, k=-1) - rho * np.eye(order, k=0)
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

    u_hat, log = recon.filter(ctrl)

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

    u_hat, log = recon.filter(ctrl)

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

    u_hatP_1, log_1 = reconstructionFiltered1.filter(ctrl)
    print(ctrl.mixingMatrix)

    print(postFilter2)
    u_hatP_2, log_2 = reconstructionFiltered2.filter(ctrl)
    u_hatP_3, log_3 = reconstructionFiltered3.filter(ctrl)
    u_hatP_4, log_4 = reconstructionFiltered4.filter(ctrl)
    u_hatP_5, log_5 = reconstructionFiltered5.filter(ctrl)

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


def test_evaluate_WithNoise():
    size = 10000
    order = 5
    Ts = 0.0001
    eta2 = 1e-4
    beta = 1000.


    noiseVariance = np.ones(order) * 1e-12
    noiseSources = []
    for index in range(order):
        if noiseVariance[index]>0:
            vector = np.zeros(order)
            vector[index] = beta
            noiseSources.append(
            {
                "std": np.sqrt(noiseVariance[index]),
                "steeringVector": vector,
                "name": "Noise Source %s" % index
            }
            )


    options = {
        'noise': noiseSources,
        'eta2': eta2 * np.ones(order)
    }

    # options = {}

    # options2 = {
    #     # 'noise': {"standardDeviation": np.sqrt(noiseVariance) * np.array([1./(order - x)**3 for x in range(order)])},
    #     'noise': {"standardDeviation": 1e-4 * np.array([1. for x in range(order)])},
    #     'eta2': eta2 * np.ones(order)
    # }
    # options['noise']['standardDeviation'][5] = 1e-1

    amplitude = 1. * 1e-6
    frequency = 1e2 / (2. * np.pi)
    phase = np.pi * 7. / 8.
    vector = np.ones(order) * beta / 2.
    # vector = np.zeros(order)
    vector[0] = beta
    inp = system.Sin(Ts, amplitude=amplitude, frequency=frequency, phase=phase, steeringVector=vector)


    # coef = np.random.rand(size) * 2. - 1.
    # vector = np.zeros(order)
    # vector[0] = beta
    # inp = system.FirstOrderHold(Ts, coefficients=coef, steeringVector=vector)



    # T = 120
    # t = 10 ** (T/10.)
    # rho = np.power(t, 1./order)/beta * np.sqrt(2)
    rho = 0.

    A = beta/2. * np.eye(order, k=-1) - rho * np.eye(order, k=0)
    # A[0, :] =-np.ones(order) * beta/2.

    print(A)
    # A = 1e-2 * np.eye(order, k=-1) - rho * np.eye(order, k=0)
    # A = np.zeros((order, order))
    # A = np.random.randn(order, order)
    # A = np.dot(A, np.linalg.inv(np.diag(np.sum(np.abs(A), axis=1))))
    # print(A)

    # A = 9. * np.eye(order, k=-1, dtype=np.float)
    # make causal filter
    # A +=  0. * np.eye(order, k=0)
    c0 = np.zeros(order, dtype=np.float).reshape((order,1))
    c0[-1] = 1.
    c = np.eye(order, dtype=np.float)

    sys0 = system.System(A, c0)
    sys = system.System(A, c)

    mixingMatrix = - 10.5 * np.eye(order)

    ctrl = system.Control(mixingMatrix, size)
    ctrl0 = system.Control(mixingMatrix, size)


    sim0 = simulator.Simulator(sys0, ctrl0, options=options)
    sim = simulator.Simulator(sys, ctrl, options=options)
    t = np.linspace(0., Ts * (size - 1) , size)
    u = inp.scalarFunction(t)
    # res2 = sim0.simulate(t, (inp,))
    res = sim.simulate(t, (inp,))
    plt.figure()
    plt.plot(res['t'], res['output'])
    plt.legend(["%s" % x for x in range(order)])

    # System 1
    reconSISO = reconstruction.WienerFilter(t, sys0, (inp,), {"eta2":np.array(eta2)})
    reconSIMO = reconstruction.WienerFilter(t, sys, (inp,), {"eta2":eta2 * np.ones(order)})
    reconMIMO = reconstruction.WienerFilter(t, sys, [inp], options)

    u_hatSISO, log_SISO = reconSISO.filter(ctrl)
    u_hatSIMO, log_SIMO = reconSIMO.filter(ctrl)
    u_hatMIMO, log_MIMO = reconMIMO.filter(ctrl)

    print(reconSISO)


    ev_SISO = evaluation.Evaluation(sys, u_hatSISO, (inp,))
    ev_SIMO = evaluation.Evaluation(sys, u_hatSIMO, (inp,))
    ev_MIMO = evaluation.Evaluation(sys, u_hatMIMO, (inp,))


    freqsLim = [1e-2, 1e2]

    freq, SISOSPECTRUM, SIGNALSPECTRUM = ev_SISO.PowerSpectralDensity(t)
    _, SIMOSPECTRUM, _  = ev_SIMO.PowerSpectralDensity(t)
    _, MIMOSPECTRUM, _  = ev_MIMO.PowerSpectralDensity(t)

    plt.figure()
    plt.semilogx(freq, 10 * np.log10(np.abs(SIGNALSPECTRUM[0].flatten())), label="u_hatSISO")
    plt.semilogx(freq, 10 * np.log10(np.abs(SISOSPECTRUM.flatten())), label="u_hatSISO")
    plt.semilogx(freq, 10 * np.log10(np.abs(SIMOSPECTRUM.flatten())), label="u_hatSIMO")
    plt.semilogx(freq, 10 * np.log10(np.abs(MIMOSPECTRUM[:, 0].flatten())), label="u_hatMIMO")
    plt.legend()


    plt.figure()
    plt.plot(u, label="u")
    plt.plot(u_hatSISO, label="u_hatSISO")
    plt.plot(u_hatSIMO, label="u_hatSIMO")
    plt.plot(u_hatMIMO, label="u_hatMIMO")
    plt.legend()

    plt.figure()
    plt.plot(u_hatSISO.flatten() - u.flatten(), label="u_hatSISO")
    plt.plot(u_hatSIMO.flatten() - u.flatten(), label="u_hatSIMO")
    plt.plot(u_hatMIMO[:,0] - u.flatten(), label="u_hatMIMO")
    plt.legend()

    plt.figure()
    freq, u_hatSISOSP = signal.welch(u_hatSISO.flatten() - u.flatten(), 1./Ts)
    _, u_hatSIMOSP = signal.welch(u_hatSIMO.flatten() - u.flatten(), 1./Ts)
    _, u_hatMIMOSP = signal.welch(u_hatMIMO[:,0] - u.flatten(), 1./Ts)

    plt.semilogx(freq, 10 * np.log10(np.abs(u_hatSISOSP)), label="u_hatSISO")
    plt.semilogx(freq, 10 * np.log10(np.abs(u_hatSIMOSP)), label="u_hatSIMO")
    plt.semilogx(freq, 10 * np.log10(np.abs(u_hatMIMOSP)), label="u_hatMIMO")
    plt.legend()

    figure1 = ev_SISO.PlotTransferFunctions(freqsLim)

def test_evaluate_TextToFile():
    """
    System specification
    """
    order = 5
    frequency = 5e-3
    #### Suspicious
    # fs = 1./80e-6
    # Ts = 1./fs
    ####
    amplitude = 1.25
    phase = 0.


    R = 16e3
    RFB = 68e3
    C = 10e-9
    beta = -1. / (R*C)
    betaFB = -1. / (RFB * C)
    rho = 0.# needs to be in the left plane

    Ts = int(np.floor(1./(-beta - (beta + 4 * betaFB)) * 1e6)) * 1e-6
    print(Ts)
    fs = 1./Ts


    control_gain = beta #/np.sqrt(2.) * 10.
    control_frequency = np.copy(fs)
    bounds = 5 # 5V bound

    print("Zero Gain Frequency = %0.1f" % np.abs(beta/np.pi/2.))
    print("Expected SNR = %f" % (10 * np.log10((amplitude ** 2 / 2 * 12 / 1.25) * (2 * order + 1) * 32 ** (2 * order + 1) / ((2 * np.pi)**(2 * order)))))

    """
    System Model
    """
    A = np.eye(order, k=-1) * beta + np.eye(order, k=0) * rho
    c = np.eye(order)
    sys = ADC.system.System(A, c)

    """
    Input
    """
    b = np.zeros(order)
    b[0] = beta

    sys.b = b

    inp = ADC.system.Sin(Ts, amplitude=amplitude, frequency=frequency, phase=phase, steeringVector=b)

    mixingMatrix = np.eye(order) * control_gain * 1.25 # the 1.25 comes from the control voltage

    # New Feedback Structure
    mixingMatrix[0,1:] = np.ones(order - 1) * betaFB * 1.25 # the 1.25 comes from the control voltage

    data = "./AnalogToDigital/test/A0.05000-F625.adc"
    control_memory, _, _, _  = unstable_linear_filter.ascii_to_bitarray(data, order=order)
    size = control_memory.shape[0]
    t = np.arange(size) * Ts
    ctrl = ADC.system.Control(mixingMatrix, size, control_memory)
    eta2 = 1e0

    reconstruction = ADC.reconstruction.WienerFilter(t, sys, (inp,), {"eta2":np.ones(order) * eta2})
    u_hat, log = reconstruction.filter(ctrl)

    # if PLOTS:
    #     plt.figure()
    #     plt.title("Input estimations")
    #     plt.plot(t, u, label="u")
    #     plt.plot(t, u_hat, label="u_hat")
    #     plt.xlabel("t")
    #     plt.legend()


    """
    Evaluation
    """
    metrics = ADC.evaluation.SigmaDeltaPerformance(sys, u_hat)
    
    metrics.ToTextFile("test.csv")

if __name__ == "__main__":
    test_evaluate_TextToFile()
    test_evaluate_PlotTransferFunctions()
    # test_evaluate_PlotTransferFunctions_For_PostFiltering()
    test_evaluate_WithNoise()
    plt.show()
