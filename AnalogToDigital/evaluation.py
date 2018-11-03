import numpy as np
from scipy import signal
import scipy
import matplotlib.pyplot as plt

NUMBER_OF_POINTS = 1000

class SigmaDeltaPerformance(object):
    """
    This is a helper class for establishing consistent figures of merit.
    - PSD
    - DR
    - SFDR
    - HD
    - THD
    - IPn (Interception point n, for n = 1,...,N
    """

    def __init__(self, system, estimate, fs = 1., osr = 32, fullScaleAmplitude = 1.):
        self.OSR = osr
        self.system = system
        self.fs = fs
        self.fullScale = fullScaleAmplitude
        self.estimate = estimate.flatten()
        self.estimate -= np.mean(self.estimate)
        self.freq, self.spec = self.powerSpectralDensity()
        self.fIndex = self.findMax(self.spec)
        self.f = self.fIndex * self.fs / (2. * self.N)
        print("f: %s" % self.f)
        self.harmonics, self.harmonicDistortion = self.computeHarmonics()
        # print(self.harmonics)

    def PlotPowerSpectralDensity(self, ax = None):
        if not ax:
            fig, ax = plt.subplots()
        
        noiseMask = np.ones_like(self.spec, dtype=bool)

        for h in self.harmonics:
            if h["power"] > 0:
                # print(h["support"])
                noiseMask[h["support"]] = False
        ax.semilogx(self.freq, 10 * np.log10(self.spec))
        ax.semilogx(self.freq[noiseMask], 10 * np.log10(self.spec[noiseMask]), color='b', label = "noise")
        for h in self.harmonics:
            ax.semilogx(self.freq[h['support']], 10 * np.log10(self.spec[h['support']]), "-*", color='r', label="h%i" % h['number'])
        return ax

    def findMax(self, arg):
        """
        findMax returns the index for the peak of arg
        """
        return np.argmax(arg)


    def PlotPerformanceOSR(self, ax=None, OSRMax = 256):
        if not ax:
            fig, ax = plt.subplots()
        # N = np.int(np.log2(OSRMax * 2))
        # osr = 2 ** np.arange(N)
        N = 100
        osr = np.logspace(0, np.log2(OSRMax), num=N, base=2.)
        # print(osr)
        dr = np.zeros(N)
        snr = np.zeros(N)
        esnr = np.zeros(N)
        thd = np.zeros(N)
        
        for n in range(N):
            dr[n], snr[n], thd[n], _, _ = self.Metrics(osr[n])
            esnr[n] = self.ExpectedSNR(osr[n])
        ax.plot(osr, dr, label="DR")
        ax.plot(osr, esnr, label="Expected SNR")
        ax.plot(osr, snr, label="SNR")
        ax.plot(osr, thd, label="THD")
        return ax
    
    def Metrics(self, OSR):
        epsilon = 1e-18
        if OSR < 1:
            raise "Non valid oversampling rate"
        fb = self.fs / np.float(2 * OSR)

        noiseMask = np.ones_like(self.spec, dtype=bool)


        THD = 0.
        signalPower = epsilon
        for h in self.harmonics:
            if h["power"] > 0 and h["f"] <= fb:
                noiseMask[h["support"]] = False
                if h["number"] == 1:
                    signalPower = h['power']
                else:
                    THD += h["power"]
        
        noise = self.spec[noiseMask]
        noisePower = np.sum(noise[:self.frequencyToIndex(fb)])
        noisePower += np.mean(noise[:self.frequencyToIndex(fb)]) * np.sum(noiseMask[:self.frequencyToIndex(fb)])

        DR = 10 * np.log10(1./noisePower)
        SNR = 10 * np.log10(signalPower) + DR
        THD = 10 * np.log10(THD/signalPower)
        THDN = 10 * np.log10((THD + noisePower)/signalPower)
        return DR, SNR, THD, THDN, self.ENOB(DR)

    def ENOB(self, DR):
        return (DR - 1.76) / 6.02

    def ExpectedSNR(self,OSR, N=1):
            DR = 10. * np.log10(3 * (2 ** N - 1)**2 * (2 * self.system.order + 1) * OSR ** (2 * self.system.order + 1) / (2 * np.pi ** (2 * self.system.order)))
            return DR
    def computeHarmonics(self):
        # print(f)
        fIndex = self.fIndex
        number = 1
        harmonics = []
        harmonicDistortion = []
        while fIndex < self.N / 2.:
            # print(f)
            harmonic, support = self.peakSupport(self.fIndex)
            if harmonic:
                power = np.sum(self.spec[support])
            else:
                power = 0.
                support = None
            harmonics.append(
                {   
                    "f": fIndex,
                    "number": number,
                    "power": power,
                    "support": support,
                }
            )

            if number == 1:
                harmonicDistortion.append(np.sqrt(power))
            else:
                harmonicDistortion.append(np.sqrt(power / harmonicDistortion[0]))
            number += 1
            fIndex *= 2.

        harmonicDistortion = np.array(harmonicDistortion)
        return harmonics, harmonicDistortion

    def powerSpectralDensity(self):
        """
        Compute Power-spectral density
        """
        self.N = 256 * self.OSR
        window = 'hanning'

        wind = np.hanning(self.N)

        FullScale = 1.25 # 2.5 Vpp

        w1 = np.linalg.norm(wind, 1)
        w2 = np.linalg.norm(wind, 2)

        NWB = w2 / w1
 

        # spectrum = np.abs(np.fft.fft(self.estimate[:self.N] * wind, axis=0, n = self.N)) ** 2 
        # spectrum = spectrum[:self.N/2]
        freq = np.fft.fftfreq(self.N)[:self.N/2]
        # spectrum /= (self.N / 2) * (self.fs / 2)
        freq, spectrum = scipy.signal.welch(self.estimate, fs=1.0, window=window, nperseg=self.N, noverlap=None, nfft=None, return_onesided=True, scaling='spectrum', axis=0)
        # spectrum /= (FullScale / 4. * w1) ** 2
        return freq, spectrum

    def frequencyToIndex(self, f):
        """
        Returns the index for frequency f
        """
        return np.int(f * 2 / self.fs * self.spec.size)

    def peakSupport(self, fIndex):
        """
        Checks if there is peak at f and returns its support
        """
        localPathSize = 200
        maxPeakNeighbor = 20
        midIndex = fIndex
        # print("Peak Index: %s" % midIndex)
        lowerIndexBound = np.minimum(localPathSize, midIndex)
        upperIndexBound = np.minimum(localPathSize, self.spec.size - midIndex - 1)
        tempSpec = self.spec[midIndex - lowerIndexBound:midIndex + upperIndexBound]
        # index = (upperIndexBound - lowerIndexBound) / 2
        index = lowerIndexBound - 1
        peakHeight = tempSpec[index]
        avgHeight = (np.mean(tempSpec[:index]) + np.mean(tempSpec[index+1:])) / 2.
        maxRange = {"range": np.array([midIndex]), "value": peakHeight / avgHeight }
        for offset in range(1, np.minimum(maxPeakNeighbor, np.minimum(lowerIndexBound, upperIndexBound))):
            # print(avgHeight, peakHeight)
            if peakHeight / avgHeight > maxRange["value"]:
                maxRange["range"] = np.arange( 1 + 2 * (offset - 1)) + midIndex - (offset - 1)
                maxRange["value"] = peakHeight / avgHeight 
                # print(maxRange["value"])
            peakHeight += tempSpec[index + offset] + tempSpec[index - offset] 
            avgHeight = (np.mean(tempSpec[index-offset:]) + np.mean(tempSpec[:index+offset+1])) / 2.

        if maxRange["value"] > 4:
            return True, maxRange["range"]
        else:
            return False, np.array([])


class Evaluation(object):
    """
    This is a helper class for establishing concistent figures of merit.


    - Frequency spectrum
    - Least Square
    """

    def __init__(self, system, estimates=[], references=[], signalBand=[0., 1.]):
        self.system = system
        self.estimates = estimates
        self.references = references
        self.cmap = plt.get_cmap('jet_r')
        self.signalBand = signalBand

    def ExpectedSNRForEquivalentIntegratorChain(self,OSR, N=1):
        DR = N * 6.02 + 1.76 + 10 * np.log10(2 * self.system.order + 1) - 20 * self.system.order * np.log10(np.pi) + 10 * (2 * self.system.order + 1) * np.log10(OSR)
        ENOB = (DR - 1.76) / 6.02
        return (DR, ENOB)


    def ExpectedQuantizationNoiseForEquivlanetIntegratorChainInDB(self, f, fs, N=1):
        res = np.zeros_like(f)
        for index, freq in enumerate(f):
            res[index] = -self.ExpectedSNRForEquivalentIntegratorChain(fs/(2. * freq), N=N)[0]
        return res

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
        N = (1 << 8)
        # N = t.size
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
        # N = (1 << 16)
        N = (1 << 8)
        # N = t.size
        Ts = t[1] - t[0]
        refSpec = [signal.welch(x.scalarFunction(t), 1./Ts, nperseg= N)[1] for x in self.references]
        freq, inputSpec = signal.welch(self.estimates, 1./Ts, axis=0, nperseg = N)
        # freq = np.fft.fftfreq(inputSpec.shape[0], d=Ts)

        # max, min = self.findMaxAndMean(inputSpec)
        # dnr = max/min
        # # dnr, max, min = self.DynamicRange(t, self.signalBand)
        #
        # max = (np.ones_like(freq) * max).flatten()
        # min = (np.ones_like(freq) * min).flatten()

        fig, ax = plt.subplots(nrows=2, ncols=1)
        index = 0
        colors = self.cmap(np.arange(self.estimates.shape[1] + len(self.references))/np.float(self.estimates.shape[1] + len(self.references)))
        # print(np.arange(self.estimates.shape[1] + 1)/np.float(self.estimates.shape[1] + len(self.references)))
        for ref in self.references:
            tf_abs = np.abs(refSpec[index])
            ax[0].plot(freq, tf_abs, label=ref.name, c=colors[index])
            ax[1].semilogx(freq, 10 * np.log10(tf_abs), label=ref.name, c=colors[index])
            index += 1

        for inp in range(inputSpec.shape[1]):
        # for inp in range(1):
            tf_abs = np.abs(inputSpec[:, inp])
            ax[0].plot(freq, tf_abs, label="Input %s" % inp, c=colors[index])
            ax[1].semilogx(freq, 10 * np.log10(tf_abs), label="Input %s" % inp, c=colors[index])
            index += 1

        # ax[0].plot(freq, min, label="Noise Floor")
        # ax[1].semilogx(freq, 10 * np.log10(min), label="Noise Floor")
        # ax[0].plot(freq, max, label="Signal Peak")
        # ax[1].semilogx(freq, 10 * np.log10(max), label="Signal Peak")

        # diff = dnr
        #
        # print(freq[freq.size/4], 0.01, "$\Delta = %0.1f$ dB" % (10 * np.log10(diff)))
        #
        # ax[0].text(freq[freq.size/4], 0.01, "$\Delta = %0.1f$ dB" % (10 * np.log10(diff)))
        # ax[1].text(np.log(freq[freq.size/4]), 10 * np.log10(0.01), "$\Delta = %0.1f$ dB" % (10 * np.log10(diff)))

        ax[0].legend()
        ax[0].set_xlabel("frequency")
        ax[0].set_ylabel("$V^2$/Hz")
        ax[1].set_xlabel("frequency")
        ax[1].set_ylabel("$V^2$/Hz [dB]")
        # ax[1].legend()

    def PowerSpectralDensity(self, t):
        # N = (1 << 20)
        N = (1 << 16)
        # N = (1 << 8)
        # N = t.size
        # 256 * OSR
        N = 256 * 32
        Ts = t[1] - t[0]
        refSpec = [signal.welch(x.scalarFunction(t), 1./Ts, nperseg= N)[1] for x in self.references]
        freq, inputSpec = signal.welch(self.estimates, 1./Ts, nperseg= N, axis=0)
        # freq = np.fft.fftfreq(inputSpec.shape[0], d=Ts)

        # max, min = self.findMaxAndMean(inputSpec)
        # dnr = max/min
        # # dnr, max, min = self.DynamicRange(t, self.signalBand)
        #
        # max = (np.ones_like(freq) * max).flatten()
        # min = (np.ones_like(freq) * min).flatten()

        return freq, inputSpec, refSpec

    def PowerSpectralDensityFFT(self, t):
        Ts = t[1] - t[0]
        fs = 1./Ts
        N = int(2 ** np.ceil(np.log2(t.size)))
        # N = (1 << 8)
        # window = np.ones(N)
        window = np.hanning(t.size)
        
        refSpec =  [np.abs(np.fft.fft(x.scalarFunction(t) * window, axis=0, n = N)) ** 2 / N for x in self.references]
        inputSpec = np.abs(np.fft.fft(self.estimates * window.reshape((window.size, 1)) * np.ones_like(self.estimates), axis=0, n = N)) ** 2 / ((N / 2) * (fs / 2)) 
        freq = np.fft.fftfreq(N, d=Ts)
        return freq[:N/2], inputSpec[:N/2], refSpec[:N/2]

    def findMaxAndMean(self, array):
        offset = 30
        indexMax = np.argmax(np.abs(array), axis = 0)
        mean = np.mean(array[5:indexMax[0] - offset, :], axis=0)
        mean = np.amax(array[5:indexMax[0] - offset, :], axis=0)
        return array[indexMax], mean

    def DynamicRange(self, t, signalBand):
        mask = 500
        f_l = signalBand[0]
        f_h = signalBand[1]
        # N = (1 << 20)
        N = (1 << 16)
        # N = (1 << 8)
        # N = t.size
        Ts = t[1] - t[0]
        freq, inputSpec = signal.welch(self.estimates[:,0], 1./Ts, nperseg = N)

        table = freq < f_h
        signalBand = inputSpec[table]
        table = freq[table] > f_l
        signalBand = signalBand[table]

        maxIndex = np.argmax(signalBand)

        signalDensity = signalBand[maxIndex]

        notConverged = True
        min = signalDensity

        # while(notConverged):
        #     if((signalDensity - min)

        # Mask surronding pixels
        signalBand[(maxIndex - mask/2): (maxIndex + mask/2)] = 0

        noiseDensity = np.amax(signalBand)

        return signalDensity/noiseDensity, signalDensity, noiseDensity

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
