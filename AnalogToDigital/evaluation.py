import numpy as np
from scipy import signal
import scipy
import matplotlib.pyplot as plt

class SNRvsAmplitude(object):
    """
    This is a helper class for plotting SNR vs input power
    """

    def __init__(self, system, estimates, OSR=32):
        self.estimates = []
        self.system = system
        self.OSR = OSR
        index = 0
        for est in estimates:
            self.estimates.append(
                {
                    "estimates": est,
                    "performance": SigmaDeltaPerformance(system, est),
                    "inputPower": np.var(est.flatten()) / (1.25 ** 2 / 2.)
                }
            )
            index += 1
        self.size = index
        self.sortAndMake(OSR)
    
    def sortAndMake(self, OSR):
        """
        Compute and make the SNR vs input amplitude plots
        """
        self.snrVsAmp = np.zeros((len(self.estimates), 5))
        for index, est in enumerate(self.estimates):
            # self.snrVsAmp[index,1] is SNR
            # self.snrVsAmp[index,2] is TSNR (Theoretical measured)
            # self.snrVsAmp[index,3] is Theoretical SNR computed
            # self.snrVsAmp[index,4] Total Harmonic distortion and noise
            DR, self.snrVsAmp[index,1], THD, self.snrVsAmp[index,4], ENOB, self.snrVsAmp[index,2] = est["performance"].Metrics(OSR)
            self.snrVsAmp[index, 0] = est["inputPower"]
            self.snrVsAmp[index, 3] = self.theoreticalPerformance(self.snrVsAmp[index,0] * (1.25 ** 2 / 2.))
        shuffleMask = np.argsort(self.snrVsAmp[:,0])
        self.snrVsAmp = self.snrVsAmp[shuffleMask,:]


    def ToTextFile(self, filename):
        description = ["IP", "SNR", "TMSNR", "TSNR", "THDN"]
        np.savetxt(filename, self.snrVsAmp, delimiter=', ', header=", ".join(description), comments='')

    def PlotInputPowerVsSNR(self, ax=False):
        # inputPower = np.zeros(self.size)
        # SNR = np.zeros_like(inputPower)
        # TSNR = np.zeros_like(inputPower)
        # SNRTheoretical = np.zeros_like(inputPower)
        # for index in range(self.size):
        #     inputPower[index] = self.estimates[index]["inputPower"]
        #     DR, SNR[index], THD, THDN, ENOB, TSNR[index] = self.estimates[index]["performance"].Metrics(OSR)
        #     SNRTheoretical[index] = self.theoreticalPerformance(inputPower[index] * (1.25 ** 2 / 2.), OSR=OSR)
        # sortMask = np.argsort(inputPower)
        if not ax:
                fig, ax = plt.subplots()
        ax.plot(10 * np.log10(self.snrVsAmp[:,0]), self.snrVsAmp[:,3], label="theoretical")
        ax.plot(10 * np.log10(self.snrVsAmp[:,0]), self.snrVsAmp[:,2], label="theoretical measured")
        ax.plot(10 * np.log10(self.snrVsAmp[:,0]), self.snrVsAmp[:,1], label="measured")
        ax.legend()
        ax.set_xlabel("Input Power dBFS")
        ax.set_ylabel("SNR dB")
        return ax

    def theoreticalPerformance(self,inputPower):
        return 10 * np.log10(inputPower * 12 * (2 * self.system.order + 1) * self.OSR ** (2 * self.system.order + 1) / ((2 * np.pi)**(2 * self.system.order)))



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
        self.theoreticalSpec = self.theoreticalNoiseTransferFunction(self.freq)
        self.fIndex = self.findMax(self.spec)
        self.f = self.fIndex * self.fs / (2. * self.N)
        # print("f: %s" % self.f)
        self.harmonics, self.harmonicDistortion = self.computeHarmonics()
        # print(self.harmonics)

    def PlotPowerSpectralDensity(self, ax = None, seperate=False, theoreticalComparision = False, label=""):

        if not ax:
            fig, ax = plt.subplots()
        if seperate:
            noiseMask = np.ones_like(self.spec, dtype=bool)

            for h in self.harmonics:
                if h["power"] > 0:
                    # print(h["support"])
                    noiseMask[h["support"]] = False
            ax.semilogx(self.freq[noiseMask], 10 * np.log10(self.spec[noiseMask]), label = label + "_noise")
            for h in self.harmonics:
                if h["power"] > 0:
                    ax.semilogx(self.freq[h['support']], 10 * np.log10(self.spec[h['support']]), "-*", label=label + "_h%i" % h['number'])
        else:
            # print(self.spec.shape)
            ax.semilogx(self.freq, 10 * np.log10(self.spec), label=label)

        # This is nonsense. The relation between these two should be different by one power of magnitude
        # since the SNR is the integral of the power spectral density.
        # if theoreticalComparision:
        #     steps = 2**(np.arange(np.log2(self.OSR)))
        #     values = np.zeros(steps.size)
        #     for i,osr in enumerate(steps):
        #         values[i] = self.ExpectedSNR(osr)
        #     ax.semilogx(steps, values, "+-", label="Theoretical Reference")
        return ax

    def theoreticalNoiseTransferFunction(self, freqs):
        """
        Reference Transferfunction
        """
        systemResponse = lambda f: np.dot(self.system.frequencyResponse(f), self.system.b)
        Tf = lambda f: np.dot(np.conj(np.transpose(systemResponse(f))), np.linalg.pinv(np.outer(systemResponse(f), systemResponse(f).conj())))
        noisePowerSpectralDensity = np.zeros_like(freqs)
        for i,f in enumerate(freqs):
            noisePowerSpectralDensity[i] = np.sum(np.abs(Tf(f)))**2
        return noisePowerSpectralDensity

    def findMax(self, arg):
        """
        findMax returns the index for the peak of arg
        """
        range = np.int(self.fs / np.float(2 * self.OSR) * self.N)
        return np.argmax(arg[:range+5])

    def PlotPerformance(self, ax=None, SNR=True, DR=False, ESNR=False, THD=False):
        if not ax:
            fig, ax = plt.subplots()
        # N = np.int(np.log2(OSRMax * 2))
        # osr = 2 ** np.arange(N)
        N = 500
        f = np.logspace(-3,np.log10(0.5), N)
        dr = np.zeros(N)
        snr = np.zeros(N)
        esnr = np.zeros(N)
        thd = np.zeros(N)

        for n in range(N):
            dr[n], snr[n], thd[n], _, _, _ = self.Metrics(0.5/f[n])
            esnr[n] = self.ExpectedSNR(0.5/f[n])
        
        if SNR:
            ax.semilogx(f, snr, label="SNR")
        if DR:
            ax.semilogx(f, dr, label="DR")
        if ESNR:
            ax.semilogx(f, esnr, label="Expected SNR")
        if THD:
            ax.semilogx(f, thd, label="THD")
        ax.set_xlabel("Normalized Frequency")
        ax.set_ylabel("dB")
        return ax

    def PlotPerformanceOSR(self, ax=None, OSRMax = 256, SNR=True, DR=False, ESNR=False, THD=False):
        if not ax:
            fig, ax = plt.subplots()
        # N = np.int(np.log2(OSRMax * 2))
        # osr = 2 ** np.arange(N)
        N = 500
        osr = np.logspace(0, np.log2(OSRMax), num=N, base=2.)
        # print(osr)
        dr = np.zeros(N)
        snr = np.zeros(N)
        esnr = np.zeros(N)
        thd = np.zeros(N)
        
        for n in range(N):
            dr[n], snr[n], thd[n], _, _, _ = self.Metrics(osr[n])
            esnr[n] = self.ExpectedSNR(osr[n])
        if SNR:
            ax.plot(osr, snr, label="SNR")
        if DR: 
            ax.plot(osr, dr, label="DR")
        if ESNR:
            ax.plot(osr, esnr, label="Expected SNR")
        if THD:
            ax.plot(osr, thd, label="THD")
        ax.set_ylabel("dB")
        ax.set_xlabel("OSR")
        return ax
    
    def Metrics(self, OSR):
        epsilon = 1e-18
        if OSR < 1:
            raise "Non valid oversampling rate"
        fb = np.int(self.fs / np.float(2 * OSR) * self.N)
      
        noiseMask = np.ones_like(self.spec, dtype=bool)
        self.harmonicMask = np.zeros_like(noiseMask, dtype=bool)
        self.signalMask = np.zeros_like(self.harmonicMask, dtype=bool)


        thd = 0.
        signalPower = epsilon
        support = 0 
        for h in self.harmonics:
            # print(h["fIndex"] , fb)
            if h["power"] > 0 and (h["fIndex"]) <= fb :
                noiseMask[h["support"]] = False
                if h["number"] == 1:
                    signalPower = h['power']
                    support = len(h['support'])
                    self.signalMask[h["support"]] = True
                else:
                    thd += h["power"]
                    self.harmonicMask[h["support"]] = True
        
        noise = self.spec[noiseMask]
        self.noiseMask = noiseMask
        # print(noise.size, fb)
        startOffset = 5
        noisePower = np.sum(noise[startOffset:fb])
        # noisePower = np.sum(noise[startOffset:fb - support/2])
        noisePower += np.mean(noise[startOffset:fb]) * (support + startOffset)
        # noisePower = np.mean(noise[startOffset:fb]) * (fb)

        # print(signalPower, noisePower, OSR)
        DR = 10 * np.log10(1./noisePower)
        SNR = 10 * np.log10(signalPower) + DR
        THD = 10 * np.log10(thd / signalPower)
        THDN = 10 * np.log10((thd + noisePower) / signalPower)

        theoreticalSNR = 10 * np.log10(signalPower/np.sum(self.theoreticalSpec[startOffset:fb])) - 363
        return DR, SNR, THD, THDN, self.ENOB(DR), theoreticalSNR

    def ENOB(self, DR):
        return (DR - 1.76) / 6.02

    def ExpectedSNR(self,OSR, N=1):
            DR = 10. * np.log10(3 * (2 ** N - 1)**2 * (2 * self.system.order + 1) * OSR ** (2 * self.system.order + 1) / (2 * np.pi ** (2 * self.system.order)))
            return DR
    def computeHarmonics(self):
        # print(f)
        fIndex = self.fIndex
        if fIndex == 0:
            fIndex = 1
        number = 1
        harmonics = []
        harmonicDistortion = []
        # print(self.N)
        while fIndex < self.N / 2.:
            # print(f)
            harmonic, support = self.peakSupport(fIndex)
            if harmonic:
                power = np.sum(self.spec[support])
            else:
                power = 0.
                support = None
            harmonics.append(
                {   
                    "fIndex": fIndex,
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
            fIndex *= 2
        # print("Harmonics:")
        # print(harmonics)

        harmonicDistortion = np.array(harmonicDistortion)
        return harmonics, harmonicDistortion

    def powerSpectralDensity(self):
        """
        Compute Power-spectral density
        """
        self.N = min([256 * 1 * self.OSR, self.estimate.shape[0]])
        window = 'hanning'

        wind = np.hanning(self.N)

        FullScale = 1.25 # 2.5 Vpp

        w1 = np.linalg.norm(wind, 1)
        w2 = np.linalg.norm(wind, 2)

        NWB = w2 / w1
 

        # spectrum = np.abs(np.fft.fft(self.estimate[:self.N] * wind, axis=0, n = self.N)) ** 2 
        # spectrum = spectrum[:self.N/2]
        # freq = np.fft.fftfreq(self.N)[:self.N/2]
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
        # print("PeakSupport:")
        localPathSize = np.int(20)
        maxPeakNeighbor = np.int(5)     
        midIndex = np.int(fIndex)
        # print("Peak Index: %s" % midIndex)
        lowerIndexBound = np.minimum(localPathSize, midIndex)
        upperIndexBound = np.minimum(localPathSize, self.spec.size - midIndex - 1)
        # print(lowerIndexBound, upperIndexBound, midIndex)
        tempSpec = self.spec[midIndex - lowerIndexBound:midIndex + upperIndexBound]
        # index = (upperIndexBound - lowerIndexBound) / 2
        index = lowerIndexBound - 1
        # print(tempSpec)
        peakHeight = tempSpec[index]
        avgHeight = (np.mean(tempSpec[:index]) + np.mean(tempSpec[index+1:])) / 2.
        maxRange = {"range": np.array([midIndex]), "value": peakHeight / avgHeight }
        
        print("Mark")
        for offset in range(1,  np.minimum(maxPeakNeighbor, np.minimum(lowerIndexBound, upperIndexBound))):
            peakChange = tempSpec[index + offset] + tempSpec[index - offset]
            peakHeight += peakChange
            # avgHeight = (np.mean(tempSpec[:index-offset]) + np.mean(tempSpec[index+offset+1:])) / 2.
            avgHeight = (np.mean(tempSpec[index - 2 * offset:index-offset]))# + np.mean(tempSpec[index+offset+1:])) / 2.
            diff = np.abs(peakChange/2. - avgHeight)/avgHeight
            ratio = peakChange/(2 * avgHeight)
            print(diff, ratio, offset, peakChange / 2, avgHeight)
            if ratio < 1.:
                print("Break")
                break
            maxRange["range"] = np.arange( 1 + 2 * (offset - 1)) + midIndex - (offset - 1)
            maxRange["value"] = peakHeight / avgHeight 
        return True, maxRange["range"]
        
        # maxRange = {"range": np.array([midIndex]), "value": peakHeight / avgHeight }
        # diff = 1
        # for offset in range(1, np.minimum(maxPeakNeighbor, np.minimum(lowerIndexBound, upperIndexBound))):
        #     # print(avgHeight, peakHeight)
        #     # if peakHeight / avgHeight > 4 * maxRange["value"]:
        #     # print(diff)
        #     if diff > 1e-9:
        #         maxRange["range"] = np.arange( 1 + 2 * (offset - 1)) + midIndex - (offset - 1)
        #         maxRange["value"] = peakHeight / avgHeight 
        #         # print(maxRange["value"])
        #     diff = np.abs(tempSpec[index + offset] + tempSpec[index - offset])
        #     # print(diff)
        #     peakHeight += tempSpec[index + offset] + tempSpec[index - offset]
        #     # avgHeight = (np.mean(tempSpec[index-offset:]) + np.mean(tempSpec[:index+offset+1])) / 2.

        # if maxRange["value"] > 4:
        #     return True, maxRange["range"]
        # else:
        #     return False, np.array([])
        
    def ToTextFile(self, filename, OSR = 32):
        _, _, _, _, _, _ = self.Metrics(OSR)
        signal = np.ones((self.freq.size)) * 1e-10
        harmonics = np.copy(signal)
        noise = np.copy(signal)

        signal[self.signalMask] = self.spec[self.signalMask]
        harmonics[self.harmonicMask] = self.spec[self.harmonicMask]
        noise[self.noiseMask] = self.spec[self.noiseMask]

        data = np.zeros((self.freq.size, 5))

        description = ["f", "Sx", "Su", "Sh", "Sn"]

        # Frequencies  
        data[:, 0] = self.freq
        # Power Spectral Density
        data[:, 1] = 10 * np.log10(self.spec)
        # Signal
        data[:, 2] = 10 * np.log10(signal)
        # Harmonics
        data[:, 3] = 10 * np.log10(harmonics)
        # Noise
        data[:, 4] = 10 * np.log10(noise)

        data = np.nan_to_num(data)

        np.savetxt(filename, data, delimiter=', ', header=", ".join(description), comments='')
        

class Evaluation(object):
    """
    This is a helper class for establishing consistent figures of merit.


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
        to = int(N/2)
        return freq[:to], inputSpec[:to], refSpec[:to]

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

        # raise NotImplemented
        #TODO account for amplitude missmatch from ripple in filter
        return ase
