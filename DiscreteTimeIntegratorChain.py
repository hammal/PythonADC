import AnalogToDigital as ADC
import matplotlib.pyplot as plt
import numpy as np

defaultSystems = ADC.DefaultSystems()
order = 4
model = defaultSystems.discreteIntegratorChain(order)

OSR = 15
SNR = 1 * 6.02 * order + 1.76 + 10 * np.log10(2 * order + 1) - 20 * order * np.log10(np.pi) + 10 * (2 * order + 1) * np.log10(OSR) - 10 *(2 *order + 1)*np.log10(2)
print("Expected Noise Power = -%s [dB]" % SNR )

fsignal = 0.1
asignal = 0.3
# fsignal = fp
print("For order = %s\nOSR = %s" % (order, OSR))
print("fsignal = %s" % fsignal)

SIM_OSR = 1
Tc = 1.
Ts= Tc / SIM_OSR
length = 1e3 #2e6

import scipy.signal as signal
import scipy

print(model)

# w, mag, phase = signal.bode((model.A, model.b, model.c.transpose(), np.array([0])))
# plt.figure()
# plt.semilogx(w/(np.pi * 2.), mag)    # Bode magnitude plot
# plt.title("BodeMagnitude Plot")
# plt.xlabel('f Hz')

# plt.figure()
# plt.semilogx(w, phase)  # Bode phase plot
# plt.title("BodePhase Plot")
# plt.show()

# plt.ylim([0, 200])
# plt.xlim([1e3, 1e7])


"""
Generate Signal
"""
t = np.arange(length) * Ts
# u = np.sin(2. * np.pi * t)
# u[-5000:] = 0
fspace = np.array([fsignal])
# fspace = np.linspace(0, fsignal, 13)
u = np.zeros_like(t)
for f in fspace:
    u += asignal / fspace.size * np.sin(2 * np.pi * f * t)


# """
# Random smooth signal
# """
# OSR = 30
# SEED = 42
# np.random.seed(SEED)
# SIZE = int(length / OSR)
# randomSignal = np.random.randn(int(SIZE))
# smoothedRandomSignal = np.zeros(SIZE * OSR)
# smoothedRandomSignal = signal.resample(randomSignal, int(length), window=signal.get_window('hann', SIZE))
#
# u = smoothedRandomSignal

simulator = ADC.Simulator(model, 1./Ts, 1./Tc, u)
controller, y = simulator.Run()

plt.figure()

for index in range(order):
    plt.plot(controller[index, :],label="Control = %s" % index)
plt.plot(y, label="y")
plt.title("Controlls")
plt.legend()

# plt.figure()
# plt.title("FFT of error signal")
# plt.xlabel("freq")
# plt.ylabel("$| \cdot| $")
# plt.loglog(np.fft.fftfreq(y.size, d=Ts), np.abs(np.fft.fft(y)))

plt.figure()
plt.title("Power Spectral Density")
f, Pxx_den = signal.welch(y.flatten(), 1./Ts, nperseg=2**14)
plt.semilogx(f, 20 * np.log10(Pxx_den), label="y")
plt.xlabel("f Hz")
plt.ylabel("PSD [dB/Hz]")

## Downsample
controller.subsample()
t = t[::SIM_OSR]
u = u[::SIM_OSR]


T = np.zeros((f.size,order))

for index, freq in enumerate(f):
    T[index,:] = model.discreteTimeFrequencyResponse(freq + 1e-12).flatten()


plt.figure()
plt.title("transferfunction")
plt.semilogx(f, 20 * np.log10(np.abs(T)))
plt.xlabel("f Hz")
plt.ylabel("dB")
plt.legend(["State %i" % (i+1) for i in range(order)])

print(model)
# plt.show()
# eta2 = [np.abs(gm1/C/(1j * 2 * np.pi * fp))]
# for i in range(order - 1):
#     eta2.append(eta2[-1]**2)
# eta2 = np.diag(eta2)
# # eta2 = np.abs((gm1/C /(1j * 2 * np.pi * fp))**order) ** 2
# # eta2 = 1e11
# print("$\eta^2$ = %s" % eta2)
# filterSpec = {
#     'eta2': eta2,
#     'model': model,
#     'Tc': Tc
# }
#
# filter = ADC.WienerFilter(**filterSpec)
#
# u_hat = np.roll(filter.filter(controller), 0)
# # stf, ntf = filter.frequencyResponse(fspace)
#
#
# size = int(u.size)
# u_middle = u[int(size/4):-int(size/4)]
# u_hat_middle = u_hat[int(size/4):-int(size/4)]
# # stf = [1./(np.dot(u_middle, u_hat_middle) / np.linalg.norm(u_hat_middle)**2)]
# stf = [1.]
# print("stf", stf)
#
#
# plt.figure()
# t = np.arange(t.size)
# plt.plot(t, u, label="u")
# plt.plot(t, u_hat/ stf[0], label="u_hat")
# # plt.plot(t, u, label="u amp-corrected")
# plt.title("Input Reconstruction")
# plt.legend()
# freq = np.logspace(np.log10(fsignal) - 5, np.log10(fsignal) + 5)
#
# # STF, NTF = filter.frequencyResponse(freq)
# # fig = plt.figure()
# # ax1 = fig.add_subplot(2, 1, 1)
# # ax2 = fig.add_subplot(2, 1, 2)
# # ax1.semilogx(freq, 20 * np.log10(STF), label="STF")
# # ax1.semilogx(freq, 20 * np.log10(np.abs(NTF)), label="NTF")
# # ax2.semilogx(freq, np.angle(STF, deg=True))
# # ax2.semilogx(freq, np.angle(NTF, deg=True))
# # ax1.legend()
# #
# # ax1.set_ylim(-100, 10)
#
# error = u_middle  - u_hat_middle / stf[0]
#
# plt.figure()
# plt.title("Error Power Spectral Density")
# # f, Pxx_den = signal.welch(error.flatten(), 1./Tc, nperseg=2**16)
# Pxx_den = np.abs(scipy.fftpack.fft(error.flatten()))**2 / error.size * Tc
# f = scipy.fftpack.fftfreq(error.size, d = Tc)
# plt.semilogx(f, 20 * np.log10(Pxx_den ), label="error")
# # f, Pxx_den = signal.welch(u_hat.flatten(), 1./Tc, nperseg=2**20)
# f = scipy.fftpack.fftfreq(u_hat.size, d = Tc)
# Pxx_den = np.abs(scipy.fftpack.fft(u_hat.flatten()))**2 / u_hat.size * Tc
# plt.semilogx(f, 20 * np.log10(Pxx_den), label="u_hat")
#
# Pxx_den = np.abs(scipy.fftpack.fft(u.flatten()))**2 / u.size * Tc
# plt.semilogx(f, 20 * np.log10(Pxx_den), label="u")
#
# plt.legend()
# plt.xlabel("f Hz")
# plt.ylabel("PSD [dB/Hz]")
#
# # print("Mean Square Errror = %0.18e" % (np.linalg.norm(error)**2/error.size))
# errorMS = np.linalg.norm(error)**2/error.size
# signalMS = np.linalg.norm(u_middle)**2/u_middle.size
# snr = signalMS / errorMS
#
# # snr = 1./errorMS / 2.
# print("Average squared error = %0.1d dB" % (10 * np.log10(errorMS)))
# print("SNR = %0.1d dB" % (10 * np.log10(snr+1e-100)))
# print("Last stage energy %d dB" % (10 * np.log10(np.var(y))))
# # print(model)
# plt.show()
