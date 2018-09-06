"""
This package implements the design, simulation and reconstruction tools for classical
delta sigma converter based on the python port from Richard Schreiers book;
Understanding Sigma Delta Converters.
"""

from deltasigma import *
import matplotlib.pyplot as plt
from matplotlib.pyplot import plot, xlabel, ylabel, text
import numpy as np
from scipy.linalg import block_diag
import scipy.signal as signal
import AnalogToDigital as ADC

class DeltaSigma(object):
    """
    This is the delta sigma
    """

    def __init__(self, OSR, order, nlev = 2):
        self.nlev = [nlev]
        self.order = order
        self.OSR = OSR
        # self.form = "CIFB"
        self.form = "CRFB"
        self.ntf = synthesizeNTF(self.order, self.OSR, opt=0)
        a, g, b, c = realizeNTF(self.ntf, self.form)
        self.ABCD = stuffABCD(a, g, b, c, self.form)
        # print("ABCD:\n%s"% self.ABCD)

    def simSNR(self):
        snr_pred, amp_pred, k0, k1, se = predictSNR(self.ntf, self.OSR)
        snr_sim, amp_sim = simulateSNR(self.ntf, self.OSR)
        plt.figure()
        plot(amp_pred, snr_pred, '-', amp_sim, snr_sim, 'og-.')
        # figureMagic([-100, 0], 10, None, [0, 100], 10, None, (16, 6),'SQNR')
        xlabel('Input Level (dBFS)')
        ylabel('SQNR (dB)')
        pk_snr, pk_amp = peakSNR(snr_sim, amp_sim)
        text(-25, 85, 'peak SNR = %4.1fdB\n@ OSR = %d\n' % (pk_snr, self.OSR), horizontalalignment='right');

    def simSpectrum(self):
        plt.figure()
        N = 8092
        Ts = 1.
        scaling = self.nlev[0] / 2.
        amplitude = 1. * scaling
        frequency = 1./self.OSR / 2.
        frequencyIndex = np.int(1./Ts / frequency * N / 2)
        print("freqIndex", frequencyIndex)
        phase = 0.
        vector = np.zeros(self.order)
        inp = ADC.system.Sin(Ts, amplitude=amplitude, frequency=frequency, phase=phase, steeringVector=vector)
        t = np.arange(N) * Ts
        v = self.simulate(inp, t) / scaling
        ftest = frequencyIndex
        fB = int(np.ceil(N/(2. * self.OSR)))
        f = np.linspace(0, 0.5, N/2. + 1) / inp.Ts
        spec = np.fft.fft(v * ds_hann(N))/(N/4)
        plot(f, dbv(spec[:np.int(N/2) + 1]),'b', label='Simulation')
        # figureMagic([0, 0.5], 0.05, None, [-120, 0], 20, None, (16, 6), 'Output Spectrum')
        xlabel('Frequency')
        ylabel('dBFS')
        snr = calculateSNR(spec[2:fB+1], ftest - 2)
        text(0.05, -10, 'SNR = %4.1fdB @ OSR = %d' % (snr, self.OSR), verticalalignment='center')

    def checkScale(self):
        ABCD, umax, S = scaleABCD(self.ABCD, nlev = self.nlev[0])
        self.ABCD = ABCD
        self.umax = umax
        self.S = S

    def simulate(self, input, t):
        scaling = self.nlev[0] / 2.
        u = input.scalarFunction(t) * scaling
        [v, xn, xmax, y] = simulateDSM(u, self.ABCD, nlev=self.nlev, x0=0)
        self.observations = y
        self.states = xn
        self.maxObs = xmax
        self.ctrl = v
        return v / scaling


def decompose(ABCD, order):
    A = ABCD[:order, :order]
    b = ABCD[:order, order:order+1]
    B = ABCD[:order, (order + 1):]
    C = ABCD[order:, :order]
    d = ABCD[order:, order:order+1]
    D = ABCD[order:, (order + 1):]
    return A, B, b, C, d, D

class MASH(DeltaSigma):
    """
    This is essentially a standard delta sigma converter
    with the additional cancellation logic.
    """

    def __init__(self, OSR, deltaSigmas):
        self.OSR = OSR
        self.deltaSigmas = deltaSigmas
        self.order = len(deltaSigmas)
        self.computeCancellationLogic(deltaSigmas)
        self.filter = ADC.filters.TransferFunction()
        self.filter.butterWorth(self.order, 1./(2. * self.OSR))

    def simulate(self, input, t):
        u = input.scalarFunction(t)
        N = len(self.deltaSigmas)
        bits = np.zeros((u.size, N), dtype=np.int8)
        # Xmax = np.zeros(N)
        observations = np.zeros((u.size, N))

        for index in range(len(self.deltaSigmas)):
            [v, xn, xmax, y] = simulateDSM(u, self.deltaSigmas[index].ABCD, nlev=self.deltaSigmas[index].nlev, x0=0)
            bits[:, index] = v
            # print(xmax)
            # Xmax[index] = xmax
            observations[:, index] = y
            u = v - y
        return bits

    def computeCancellationLogic(self, deltaSigmas):
        STF = []
        NTF = []
        for df in deltaSigmas:
            # k is the gain of the system
            ntf, stf = calculateTF(df.ABCD, k=1.)
            stf = signal.ZerosPolesGain(*stf)
            ntf = signal.ZerosPolesGain(*ntf)

            STF.append(stf.to_tf())
            NTF.append(ntf.to_tf())

        # Create inverse filter
        # H1 = STF2
        # STF_N H_N = NTF_N-1 H_N-1
        for index in range(len(deltaSigmas)):
            if index == 0:
                self.H = [STF[1]]
            else:
                NTF_over_STF = self.inverseMultiple(NTF[index - 1], STF[index])
                self.H.append(self.concatenateSystems(NTF_over_STF, self.H[-1]))

    def reconstruction(self, v):
        out = np.zeros(v.shape[0])
        for index, filter in enumerate(self.H):
            print("For filter %i, with coefficents\n%s\n" % (index, filter))
            sos = signal.tf2sos(filter.num, filter.den)
            noise = signal.sosfilt(sos, v[:, index])
            out = out + (-1.)**(index) * noise
        return out

    def lowPassFilter(self, signal):
        return self.filter.filter(signal)


    def concatenateSystems(self,H1,H2):
        H = signal.TransferFunction(np.polymul(H1.num,H2.num),np.polymul(H1.den,H2.den))
        return H

    def inverseMultiple(self,H1,H2):
        H = signal.TransferFunction(np.polymul(H1.num,H2.den),np.polymul(H1.den,H2.num))
        return H

    def simSpectrum(self):
        plt.figure()
        N = 8092
        Ts = 1.
        scaling = self.deltaSigmas[0].nlev[0] / 2
        amplitude = 1. * scaling
        frequency = 1./self.OSR / 2.
        frequencyIndex = np.int(1./Ts / frequency * N)
        phase = 0.
        vector = np.zeros(5)
        inp = ADC.system.Sin(Ts, amplitude=amplitude, frequency=frequency, phase=phase, steeringVector=vector)
        t = np.arange(8092) * Ts
        v = v = self.reconstruction(self.simulate(inp, t)/scaling)

        ftest = frequencyIndex
        fB = int(np.ceil(N/(2. * self.OSR)))
        f = np.linspace(0, 0.5, N/2. + 1) / inp.Ts
        spec = np.fft.fft(v * ds_hann(N))/(N/4)
        plot(f, dbv(spec[:np.int(N/2) + 1]),'b', label='Simulation')
        # figureMagic([0, 0.5], 0.05, None, [-120, 0], 20, None, (16, 6), 'Output Spectrum')
        xlabel('Frequency')
        ylabel('dBFS')
        snr = calculateSNR(spec[2:fB+1], ftest - 2)
        text(0.05, -10, 'SNR = %4.1fdB @ OSR = %d' % (snr, self.OSR), verticalalignment='center')
