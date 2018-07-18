import AnalogToDigital.deltaSigma as deltaSigma
from AnalogToDigital.system import Sin
import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(precision=1)
np.set_printoptions(edgeitems=15)

def test_Simulate():
    size = 8092
    Ts = 80e-6
    fp = 625.
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
    size = 8092
    Ts = 80e-6
    fp = 500.
    order = 8
    OSR = 1. / Ts / 2. / fp
    ds = deltaSigma.DeltaSigma(OSR, order)
    amplitude = 1.
    frequency = fp * 2. / 3.
    phase = 0.
    steeringVector = np.zeros(order)
    signal = Sin(Ts, amplitude, frequency, phase, steeringVector)
    t = np.arange(size) * Ts
    plt.figure()
    ds.simSpectrum(signal, t)

def test_PredictSNR():
    order = 6
    Ts = 80e-6
    fp = 625.
    OSR = 1. / Ts / 2. / fp
    ds = deltaSigma.DeltaSigma(OSR, order)
    ds.simSNR()

def test_decompose():
    ####
    # [0, 1, 2, 3],
    # [4, 5, 6, 7],
    # [8, 9, 10, 11],
    ABCD = np.arange(12).reshape((3, 4))
    A, B, b, C, d, D = deltaSigma.decompose(ABCD, 2)
    # print(A,B,b,C,D)
    np.testing.assert_array_equal(A, [[0, 1],[4,5]])
    np.testing.assert_array_equal(b, np.array([2, 6]).reshape((2,1)))
    np.testing.assert_array_equal(B, np.array([3, 7]).reshape((2,1)))
    np.testing.assert_array_equal(C, np.array([8, 9]).reshape((1,2)))
    np.testing.assert_array_equal(D, np.array([11]).reshape((1,1)))
    np.testing.assert_array_equal(d, np.array([10]).reshape((1,1)))


def test_checkScale():
    order = 6
    Ts = 80e-6
    fp = 625.
    OSR = 1. / Ts / 2. / fp
    ds = deltaSigma.DeltaSigma(OSR, order)
    print("ABCD before:\n%s" % ds.ABCD)
    ds.checkScale()
    print(ds.ABCD)
    print(ds.umax)
    print(ds.S)

def test_MASH_SIM_SPECTRUM():
    size = 8092
    Ts = 80e-6
    fp = 500.
    order = 1
    Number = 8
    OSR = 1. / Ts / 2. / fp
    ds = [deltaSigma.DeltaSigma(OSR, order) for x in range(Number)]
    mash = deltaSigma.MASH(OSR, ds)
    print("OSR = %s" % OSR)
    amplitude = 1.
    frequency = fp * 2. / 3.
    phase = 0.
    steeringVector = np.zeros(order)
    signal = Sin(Ts, amplitude, frequency, phase, steeringVector)
    t = np.arange(size) * Ts
    plt.figure()
    mash.simSpectrum(signal, t)

def test_MASH_RECONSTRUCTION():
    size = 8092
    Ts = 80e-6
    fp = 500.
    order = 1
    Number = 4
    OSR = 1. / Ts / 2. / fp
    ds = [deltaSigma.DeltaSigma(OSR, order) for x in range(Number)]
    mash = deltaSigma.MASH(OSR, ds)
    amplitude = 1.
    frequency = fp * 2. / 3.
    phase = 0.
    steeringVector = np.zeros(order)
    signal = Sin(Ts, amplitude, frequency, phase, steeringVector)
    t = np.arange(size) * Ts
    u = signal.scalarFunction(t)
    v = mash.simulate(signal, t)
    u_hat = mash.reconstruction(v)

    plt.figure()
    plt.plot(t, u, label="u")
    plt.plot(t, u_hat, label="u_hat")

if __name__ == "__main__":
    # test_Simulate()
    # test_PredictSNR()
    test_SimSpectrum()
    # test_decompose()
    # test_MASH_SIM_SPECTRUM()
    # test_MASH_RECONSTRUCTION()
    # test_checkScale()
    plt.show()
