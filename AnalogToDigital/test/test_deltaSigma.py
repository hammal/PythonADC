import AnalogToDigital.deltaSigma as deltaSigma
from AnalogToDigital.system import Sin
import numpy as np
import matplotlib.pyplot as plt

def test_Simulate():
    size = 500000
    Ts = 80e-6
    fp = 5e2
    order = 2
    OSR = 1. / Ts / 2. / fp
    ds = deltaSigma.DeltaSigma(OSR, order)
    amplitude = 1.
    frequency = fp * 2. / 3.
    phase = 0.
    steeringVector = np.zeros(order)
    signal = Sin(Ts, amplitude, frequency, phase, steeringVector)
    t = np.arange(size) * Ts
    v = ds.simulate(signal, t)
    plt.plot(v)
    plt.figure()
    plt.plot(ds.states.transpose())


def test_SimSpectrum():
    size = 50000
    Ts = 80e-6
    fp = 5e2
    order = 4
    OSR = 1. / Ts / 2. / fp
    ds = deltaSigma.DeltaSigma(OSR, order)
    amplitude = 1.
    frequency = fp * 2. / 3.
    phase = 0.
    steeringVector = np.zeros(order)
    signal = Sin(Ts, amplitude, frequency, phase, steeringVector)
    t = np.arange(size) * Ts
    ds.simSpectrum(signal, t)

def test_PredictSNR():
    order = 6
    Ts = 80e-6
    fp = 5e2
    OSR = 1. / Ts / 2. / fp
    ds = deltaSigma.DeltaSigma(OSR, order)
    ds.simSNR()

if __name__ == "__main__":
    test_Simulate()
    # test_PredictSNR()
    test_SimSpectrum()
    plt.show()
