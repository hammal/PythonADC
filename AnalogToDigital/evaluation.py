import numpy as np
from scipy import signal
import scipy
import matplotlib.pyplot as plt

NUMBER_OF_POINTS = 1000

class Evaluation(object):
    """
    This is a helper class for establishing concistent figures of merit.


    - Frequency spectrum
    - Least Square
    """

    def __init__(self, system, estimates=[], references=[]):
        self.system = system
        self.estimates = estimates
        self.references = references
        self.cmap = plt.get_cmap('jet_r')


    def AnalyticalTranferFunction(self, f, steeringVector):
        """
        Compute the transferfunction for an array of frequencies f
        """
        output = np.zeros((f.size, self.system.outputOrder), dtype=np.complex)

        for index, freq in enumerate(f):
            try:
                S = np.linalg.inv(2. * np.pi* np.complex(0.,freq) * np.eye(self.system.order, dtype=np.complex) - self.system.A)
            except:
                S = np.linalg.pinv(2. * np.pi* np.complex(0.,freq) * np.eye(self.system.order) - self.system.A)
            output[index,:] = np.dot(self.system.c.transpose(), np.dot(S, steeringVector))

        return output

    def PlotFFT(self, t):
        # N = (1 << 20)
        # N = (1 << 8)
        N = t.size
        Ts = t[1] - t[0]
        refSpec = [np.fft.fft(x.scalarFunction(t), n = N) for x in self.references]
        inputSpec = np.fft.fft(self.estimates, axis=0, n = N)
        freq = np.fft.fftfreq(inputSpec.shape[0], d=Ts)

        fig, ax = plt.subplots(nrows=2, ncols=1)
        index = 0
        colors = self.cmap(np.arange(self.estimates.shape[1] + len(self.references))/(self.estimates.shape[1] + len(self.references)))
        for ref in self.references:
            tf_abs = np.abs(refSpec[index])
            ax[0].plot(freq, tf_abs, label=ref.name, c=colors[index])
            ax[1].semilogx(freq, 20 * np.log10(tf_abs), label=ref.name, c=colors[index])
            index += 1

        for inp in range(inputSpec.shape[1]):
            tf_abs = np.abs(inputSpec[:, inp])
            ax[0].plot(freq, tf_abs, label="Input %s" % inp, c=colors[index])
            ax[1].semilogx(freq, 20 * np.log10(tf_abs), label="Input %s" % inp, c=colors[index])
            index += 1
        ax[0].legend()
        ax[0].set_xlabel("frequency")
        ax[0].set_ylabel("$|\cdot|$")
        ax[1].set_xlabel("frequency")
        ax[1].set_ylabel("dB")
        # ax[1].legend()

    def PlotPowerSpectralDensity(self, t):
        # N = (1 << 20)
        N = (1 << 16)
        # N = (1 << 8)
        # N = t.size
        Ts = t[1] - t[0]
        refSpec = [signal.welch(x.scalarFunction(t), 1./Ts, nperseg= N)[1] for x in self.references]
        freq, inputSpec = signal.welch(self.estimates, 1./Ts, axis=0, nperseg = N)
        # freq = np.fft.fftfreq(inputSpec.shape[0], d=Ts)

        max, min = self.findMaxAndMean(inputSpec)

        max = (np.ones_like(freq) * max).flatten()
        min = (np.ones_like(freq) * min).flatten()

        fig, ax = plt.subplots(nrows=2, ncols=1)
        index = 0
        colors = self.cmap(np.arange(self.estimates.shape[1] + len(self.references))/(self.estimates.shape[1] + len(self.references)))
        for ref in self.references:
            tf_abs = np.abs(refSpec[index])
            ax[0].plot(freq, tf_abs, label=ref.name, c=colors[index])
            ax[1].semilogx(freq, 10 * np.log10(tf_abs), label=ref.name, c=colors[index])
            index += 1

        for inp in range(inputSpec.shape[1]):
            tf_abs = np.abs(inputSpec[:, inp])
            ax[0].plot(freq, tf_abs, label="Input %s" % inp, c=colors[index])
            ax[1].semilogx(freq, 10 * np.log10(tf_abs), label="Input %s" % inp, c=colors[index])
            index += 1

        ax[0].plot(freq, min, label="Noise Floor")
        ax[1].semilogx(freq, 10 * np.log10(min), label="Noise Floor")
        ax[0].plot(freq, max, label="Signal Peak")
        ax[1].semilogx(freq, 10 * np.log10(max), label="Signal Peak")

        diff = max[0]/min[0]

        print(freq[freq.size/4], 0.01, "$\Delta = %0.1f$ dB" % (10 * np.log10(diff)))

        ax[0].text(freq[freq.size/4], 0.01, "$\Delta = %0.1f$ dB" % (10 * np.log10(diff)))
        ax[1].text(np.log(freq[freq.size/4]), 10 * np.log10(0.01), "$\Delta = %0.1f$ dB" % (10 * np.log10(diff)))

        ax[0].legend()
        ax[0].set_xlabel("frequency")
        ax[0].set_ylabel("$V^2$/Hz")
        ax[1].set_xlabel("frequency")
        ax[1].set_ylabel("$V^2$/Hz [dB]")
        # ax[1].legend()

    def findMaxAndMean(self, array):
        offset = 30
        indexMax = np.argmax(np.abs(array), axis = 0)
        mean = np.mean(array[5:indexMax[0] - offset, :], axis=0)
        mean = np.amax(array[5:indexMax[0] - offset, :], axis=0)
        return array[indexMax], mean

    def PlotTransferFunctions(self, freq):
        flin = np.linspace(freq[0], freq[1], NUMBER_OF_POINTS)
        flog = np.logspace(np.log(freq[0]), np.log(freq[1]), NUMBER_OF_POINTS)

        def angle(number):
            return np.arctan2(np.imag(number), np.real(number))

        fig, ax = plt.subplots(nrows=2, ncols=2)
        cmap = plt.get_cmap('jet_r')
        for index,ref in enumerate(self.references):
            tf = self.AnalyticalTranferFunction(flin, steeringVector=ref.steeringVector)
            tf_abs = np.abs(tf)
            tf_polar = angle(tf) * 180 / np.pi
            color = self.cmap(index/len(self.references))
            ax[0,0].plot(flin, tf_abs, label=ref.name, c=color)
            ax[1,0].semilogx(flog, 20 * np.log10(tf_abs), label=ref.name, c=color)
            ax[0,1].plot(flin, tf_polar, label=ref.name, c=color)
            ax[1,1].semilogx(flog, tf_polar, label=ref.name, c=color)

            ax[1,0].set_xlabel("frequency")
            ax[1,1].set_xlabel("frequency")
            ax[0,1].set_ylabel("deg")
            ax[1,1].set_ylabel("deg")
            ax[0,0].set_ylabel("$|\cdot|$")
            ax[1,0].set_ylabel("dB")

            # ax[0,0].legend()

            ax[1,1].legend()
        return fig


    def AdjustedMeanSquareError(self,t):
        TRUNCATION_SIZE = 1000
        ref = np.zeros_like(self.estimates)
        for index, signal in enumerate(self.references):
            ref[:, index] = signal.scalarFunction(t)

        estimates_Truncated = self.estimates[TRUNCATION_SIZE:-TRUNCATION_SIZE,:]
        references_Truncated = ref[TRUNCATION_SIZE:-TRUNCATION_SIZE,:]

        # compensation =

        ase = np.linalg.norm(estimates_Truncated - references_Truncated, axis=0)**2 / estimates_Truncated.shape[0]

        raise NotImplemented
        #TODO account for amplitude missmatch from ripple in filter
        return ase
