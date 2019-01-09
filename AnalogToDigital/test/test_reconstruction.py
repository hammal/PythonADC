import AnalogToDigital.system as system
import AnalogToDigital.simulator as simulator
import AnalogToDigital.reconstruction as reconstruction
import numpy as np
import matplotlib.pyplot as plt
import AnalogToDigital.filters as filters


def test_for_constant_signal():
    size = 1000
    Ts = 0.1
    coef = np.ones(size) * 0.7
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


    recon = reconstruction.WienerFilter(t, sys, (inp,))

    u_hat = recon.filter(ctrl)

    plt.figure()
    plt.plot(t, coef, label="u")
    plt.plot(t, u_hat, label="u_hat")
    plt.legend()


def test_for_constant_signalParallel():
    size = 1000
    Ts = 0.1
    coef = np.ones(size) * 0.7
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


    recon = reconstruction.ParallelWienerFilter(t, sys, (inp,))

    u_hat = recon.filter(ctrl)

    plt.figure()
    plt.plot(t, coef, label="u")
    plt.plot(t, u_hat, label="u_hat")
    plt.legend()


def test_for_first_order_filter_signal():
    size = 10000
    order = 7
    Ts = 0.1
    coef = np.random.rand(size) * 2. - 1.
    vector = np.zeros(order)
    vector[0] = 1.
    inp = system.FirstOrderHold(Ts, coefficients=coef, steeringVector=vector)

    A = np.eye(order, k=-1)
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
    reconP = reconstruction.ParallelWienerFilter(t, sys, (inp,))

    u_hat = recon.filter(ctrl)
    u_hatP = reconP.filter(ctrl)

    plt.figure()
    plt.plot(t, coef, label="coef")
    plt.plot(t, inp.scalarFunction(t), label="u")
    plt.plot(t, u_hat, label="u_hat")
    plt.plot(t, u_hatP, label="u_hatP")
    plt.legend()


def test_for_sinusodial_signal():
    size = 1000
    order = 8
    Ts = 0.1
    amplitude = 0.35
    frequency = 1e-2
    phase = np.pi * 7. / 8.
    vector = np.zeros(order)
    vector[0] = 1.
    inp = system.Sin(Ts, amplitude=amplitude, frequency=frequency, phase=phase, steeringVector=vector)

    A = np.eye(order, k=-1)
    c = np.eye(order)

    sys = system.System(A, c)

    mixingMatrix = - 1e0 * np.eye(order)

    ctrl = system.Control(mixingMatrix, size)

    sim = simulator.Simulator(sys, ctrl)
    t = np.linspace(0., 99., size)
    res = sim.simulate(t, (inp,))
    plt.figure()
    plt.plot(res['t'], res['output'])
    plt.legend(["%s" % x for x in range(order)])


    recon = reconstruction.WienerFilter(t, sys, (inp,))

    u_hat = recon.filter(ctrl)

    plt.figure()
    plt.plot(t, inp.scalarFunction(t), label="u")
    plt.plot(t, u_hat, label="u_hat")
    plt.legend()

def test_lowering_order():
    size = 1000
    order = 8
    Ts = 0.1
    amplitude = 0.35
    frequency = 1e-2
    phase = np.pi * 7. / 8.
    vector = np.zeros(order)
    vector[0] = 1.
    inp = system.Sin(Ts, amplitude=amplitude, frequency=frequency, phase=phase, steeringVector=vector)

    A = np.eye(order, k=-1)
    c = np.eye(order)

    sys = system.System(A, c)

    mixingMatrix = - 1e0 * np.eye(order)

    ctrl = system.Control(mixingMatrix, size)

    sim = simulator.Simulator(sys, ctrl)
    t = np.linspace(0., 99., size)
    res = sim.simulate(t, (inp,))
    plt.figure()
    plt.plot(res['t'], res['output'])
    plt.legend(["%s" % x for x in range(order)])

    u_hats = []
    for index in range(1, order + 1):
        newSystem, newControl, newInput = system.LowerOrderSystem(sys, ctrl, inp, index)
        recon = reconstruction.WienerFilter(t, newSystem, (newInput,))
        u_hats.append(recon.filter(newControl))

    plt.figure()
    plt.plot(t, inp.scalarFunction(t), label="u")
    for index in range(order):
        plt.plot(t, u_hats[index], label="u_hat_%i" % (index + 1))
    plt.legend()


def test_postFiltering():
    size = 10000
    order = 4
    Ts = 0.1
    amplitude = 0.35
    frequency = 1e-2
    phase = np.pi * 7. / 8.
    vector = np.zeros(order)
    vector[0] = 1.
    inp = system.Sin(Ts, amplitude=amplitude, frequency=frequency, phase=phase, steeringVector=vector)

    A = np.eye(order, k=-1)
    c = np.eye(order)

    sys = system.System(A, c)

    mixingMatrix = - 1e0 * np.eye(order)

    ctrl = system.Control(mixingMatrix, size)

    sim = simulator.Simulator(sys, ctrl)
    t = np.linspace(0., Ts * (size - 1), size)
    res = sim.simulate(t, [inp])
    plt.figure()
    plt.plot(res['t'], res['output'])
    plt.legend(["%s" % x for x in range(order)])

    bandwith = 2 * np.pi * frequency * 1e-2

    print("Bandwith = %s" % bandwith)

    postFilterTF = filters.TransferFunction()
    postFilterTF.butterWorth(2, (bandwith,))
    print(postFilterTF.a, postFilterTF.b)
    postFilter = filters.Filter()
    postFilter.tf2lssObservableForm(postFilterTF.b, postFilterTF.a)
    print("Filter A = \n%s" % postFilter.A)
    print(postFilter.b)
    print(postFilter.c)

    recon = reconstruction.WienerFilterWithPostFiltering(t, sys, (inp,), postFilter)

    u_hat = recon.filter(ctrl)

    plt.figure()
    plt.plot(t, inp.scalarFunction(t), label="u")
    plt.plot(t, u_hat, label="u_hat")
    plt.legend()

def test_for_noise_simulation():
    size = 1000
    order = 8
    Ts = 0.1
    amplitude = 0.35
    frequency = 1e-2
    phase = np.pi * 7. / 8.
    vector = np.zeros(order)
    vector[0] = 1.
    inp = system.Sin(Ts, amplitude=amplitude, frequency=frequency, phase=phase, steeringVector=vector)

    A = np.eye(order, k=-1)
    c = np.eye(order)

    sys = system.System(A, c)

    mixingMatrix = - 1e0 * np.eye(order)

    ctrl = system.Control(mixingMatrix, size)

    noiseVariance = np.ones(order) * 1e-4
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

    sim = simulator.Simulator(sys, ctrl, options=options)
    t = np.linspace(0., 99., size)
    res = sim.simulate(t, [inp])
    plt.figure()
    plt.plot(res['t'], res['output'])
    plt.legend(["%s" % x for x in range(order)])


    recon = reconstruction.WienerFilter(t, sys, (inp,))
    recon_with_noise = reconstruction.WienerFilter(t, sys, [inp], options=options)
    recon_with_noise_Parallel = reconstruction.ParallelWienerFilter(t, sys, [inp], options=options)

    u_hat = recon.filter(ctrl)
    u_hat_with_noise = recon_with_noise.filter(ctrl)
    u_hat_with_noise_Parallel = recon_with_noise_Parallel.filter(ctrl)


    plt.figure()
    plt.plot(t, inp.scalarFunction(t), label="u")
    plt.plot(t, u_hat, label="u_hat")
    plt.plot(t, u_hat_with_noise, label="u_hat_with_noise")
    plt.plot(t, u_hat_with_noise_Parallel, label="u_hat_with_noise_Parallel")
    plt.legend()

if __name__ == "__main__":
    test_lowering_order()
    test_for_constant_signal()
    test_for_constant_signalParallel()
    test_for_first_order_filter_signal()
    test_for_sinusodial_signal()
    # test_postFiltering()
    test_for_noise_simulation()
    plt.show()
