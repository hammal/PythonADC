"""
This package implements the design, simulation and reconstruction tools for classical
delta sigma converter based on the python port from Richard Schreiers book;
Understanding Sigma Delta Converters.
"""

from deltasigma import *
import matplotlib.pyplot as plt
from matplotlib.pyplot import plot, xlabel, ylabel, text
import numpy as np

class DeltaSigma(object):
    """
    This is the delta sigma
    """

    def __init__(self, OSR, order):
        self.order = order
        self.OSR = OSR
        self.form = "CRFB"
        self.ntf = synthesizeNTF(self.order, self.OSR, opt=1)
        a, g, b, c = realizeNTF(self.ntf, self.form)
        self.ABCD = stuffABCD(a, g, b, c, self.form)

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

    def simSpectrum(self, input, t):
        v = self.simulate(input, t)
        N = t.size
        ftest = input.frequency / input.Ts
        fB = int(np.ceil(N/(2. * self.OSR)))
        f = np.linspace(0, 0.5, N/2. + 1)
        spec = np.fft.fft(v * ds_hann(N))/(N/4)
        plot(f, dbv(spec[:N/2. + 1]),'b', label='Simulation')
        figureMagic([0, 0.5], 0.05, None, [-120, 0], 20, None, (16, 6), 'Output Spectrum')
        xlabel('Normalized Frequency')
        ylabel('dBFS')
        snr = calculateSNR(spec[2:fB+1], ftest - 2)
        text(0.05, -10, 'SNR = %4.1fdB @ OSR = %d' % (snr, self.OSR), verticalalignment='center')
        # NBW = 1.5/N
        # Sqq = 4*evalTF(H, np.exp(2j*np.pi*f)) ** 2/3.
        # hold(True)
        # plot(f, dbp(Sqq * NBW), 'm', linewidth=2, label='Expected PSD')
        # text(0.49, -90, 'NBW = %4.1E x $f_s$' % NBW, horizontalalignment='right')
        # legend(loc=4);

    def simulate(self, input, t):
        u = input.scalarFunction(t)
        [v, xn, xmax, y] = simulateDSM(u, self.ABCD, nlev=2, x0=0)
        self.observations = y
        self.states = xn
        self.maxObs = xmax
        self.ctrl = v
        return v
