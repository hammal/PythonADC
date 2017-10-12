import AnalogToDigital as ADC
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal
import scipy
import scipy.fftpack

order = 4
OSR = 16
SNR = 1 * 6.02 * order + 1.76 + 10 * np.log10(2 * order + 1) - 20 * order * np.log10(np.pi) + 10 * (2 * order + 1) * np.log10(OSR) - 10 *(2 *order + 1)*np.log10(2)
print("Expected Noise Power = -%s [dB]" % SNR )

# gm1 = 1. / 16e3  # 1/16kOhm
gm1 = OSR * 1e3 * 1e-8 / 2 # 1/16kOhm
# gm2 = 1e-7
C   = 1e-8 # 10nF

defaultSystems = ADC.DefaultSystems()
# model = defaultSystems.gmCChain(order, gm1, gm2, C)
model1 = defaultSystems.gmCIntergratorChain(order, gm1, C)
model2 = defaultSystems.gmCIntergratorChain(order, gm1, C)

# model.B = np.zeros_like(model.A)
# model.B[-1, -1] = - gm1 / C
# model.B = - np.eye(order) * gm1 / C * 1.0
model1.B = - np.eye(order) * gm1 / C * 1.
model2.B = - np.eye(order) * gm1 / C * 1.
model1.c = np.zeros((order, 1))
model1.c[-1] = 1
model2.c = np.eye(order)

# model2.b = np.ones((order,1)) * gm1/C
# model1.b = model2.b
print(model1)

SIM_OSR = 1000
Tc = 1./(OSR * 1e3)
Ts= Tc / SIM_OSR
length = 2e6

fb = 1. / (Tc * OSR * 2.)

fp = fb / 2
# fsignal = 1e2
asignal = 1.0
# fsignal = 0.00001/Ts
# fsignal = fp # No difference
fsignal = fp/3 # 2 dB improvment
# fsignal = fp/4 # 0 db improvment
# fsignal << fp # improvment due to border effects
print("For order = %s\nOSR = %s" % (order, OSR))
print("fsignal = %s\nfb = %s" % (fsignal,fb))


# import scipy.signal as signal
#
# # w, mag, phase = signal.bode((model.A, model.b, model.c.transpose(), np.array([0])))
# # plt.figure()
# # plt.semilogx(w/(np.pi * 2.), mag)    # Bode magnitude plot
# # plt.title("BodeMagnitude Plot")
# # plt.xlabel('f Hz')
#
# # plt.figure()
# # plt.semilogx(w, phase)  # Bode phase plot
# # plt.title("BodePhase Plot")
# # plt.show()
#
# # plt.ylim([0, 200])
# # plt.xlim([1e3, 1e7])
#
#
# """
# Generate Signal
# """
t = np.arange(length) * Ts
# u = np.sin(2. * np.pi * t)
# u[-5000:] = 0
fspace = np.array([fsignal])
# fspace = np.linspace(0, fsignal, 13)
u = np.zeros_like(t)
for f in fspace:
    u += asignal / fspace.size * np.sin(2 * np.pi * f * t)


# # """
# # Random smooth signal
# # """
# # OSR = 30
# # SEED = 42
# # np.random.seed(SEED)
# # SIZE = int(length / OSR)
# # randomSignal = np.random.randn(int(SIZE))
# # smoothedRandomSignal = np.zeros(SIZE * OSR)
# # smoothedRandomSignal = signal.resample(randomSignal, int(length), window=signal.get_window('hann', SIZE))
# #
# # u = smoothedRandomSignal
#
simulator1 = ADC.Simulator(model1, 1./Ts, 1./Tc, u)
controller1, y = simulator1.Run()

simulator2 = ADC.Simulator(model2, 1./Ts, 1./Tc, u)
controller2, y = simulator2.Run()

#
# plt.figure()
#
# for index in range(order):
#     plt.plot(controller[index, :],label="Control = %s" % index)
# plt.plot(y, label="y")
# plt.title("Controlls")
# plt.legend()
#
# # plt.figure()
# # plt.title("FFT of error signal")
# # plt.xlabel("freq")
# # plt.ylabel("$| \cdot| $")
# # plt.loglog(np.fft.fftfreq(y.size, d=Ts), np.abs(np.fft.fft(y)))
#
# plt.figure()
# plt.title("Power Spectral Density")
# f, Pxx_den = signal.welch(y.flatten(), 1./Ts, nperseg=2**14)
# plt.semilogx(f, 20 * np.log10(Pxx_den), label="y")
# plt.xlabel("f Hz")
# plt.ylabel("PSD [dB/Hz]")
#
# ## Downsample
controller1.subsample()
controller2.subsample()
t = t[::SIM_OSR]
u = u[::SIM_OSR]
#
# eta12 = (gm1/C / (2 * np.pi * fb))**(2*order)
eta12 = (np.abs(model1.frequncyResponse(fb))**2)[0]

# eta12 = 3
# eta12 = 10 ** (40/10)
# eta22 = np.zeros(order)
# eta22[0] = (np.abs(model2.frequncyResponse(fb))**1)[0]
# eta22 = [eta12]
# for i in range(1,order):
#     # eta22[i] = eta22[i-1] * eta22[0]
#     eta22.append(eta12)
#     print(eta22)
# eta22 = np.diag(eta22)
eta22 = np.eye(order) * eta12
# # eta2 = 1e11
print("$\eta^2$ old = ", 20 * np.log10(eta12))
print("$\eta^2$ new = ", 20 * np.log10(np.diag(eta22)))
filterSpec1 = {
    'eta2': eta12,
    'model': model1,
    'Tc': Tc
}
filterSpec2 = {
    'eta2': eta22,
    'model': model2,
    'Tc': Tc
}
#
filter1 = ADC.WienerFilter(**filterSpec1)
filter2 = ADC.WienerFilter(**filterSpec2)
#
u_hat1 = filter1.filter(controller1)
u_hat2 = filter2.filter(controller2)

# u_hat2 = np.roll(u_hat2, -1)

## Postprocessing
# taps = 5000
# epsilon = 1e-7
# freqs = [0, fp * Tc * 2 - epsilon, fp * Tc * 2 + epsilon, 1]
# gains = [1, 1, 0, 0]
# print([f / Tc / 2 for f in freqs])
# postFilter = scipy.signal.firwin2(taps,freqs, gains)
b,a = scipy.signal.butter(13, fb * Tc * 2)
for time in range(3):
    u_hat2 = scipy.signal.filtfilt(b,a, u_hat2)
    u_hat1 = scipy.signal.filtfilt(b,a, u_hat1)

size = int(u.size)
u_middle = u[int(size/4):-int(size/4)]
u_hat_middle1 = u_hat1[int(size/4):-int(size/4)]
u_hat_middle2 = u_hat2[int(size/4):-int(size/4)]
stf1 = [1./(np.dot(u_middle, u_hat_middle1) / np.linalg.norm(u_hat_middle1)**2)]
stf2 = [1./(np.dot(u_middle, u_hat_middle2) / np.linalg.norm(u_hat_middle2)**2)]
# stf2 = [1.]
# stf1 = [1.]
# print("stf", stf)
#
#
plt.figure()
# t = np.arange(t.size)
plt.plot(t, u, label="u")
plt.plot(t, u_hat1/ stf1[0], label="u_hat_old")
plt.plot(t, u_hat2/ stf2[0], label="u_hat_full")
# # plt.plot(t, u, label="u amp-corrected")
plt.title("Input Reconstruction")
plt.legend()

freq = np.logspace(np.log10(fsignal) - 5, np.log10(fsignal) + 5)

STF1, NTF1 = filter1.frequencyResponse(freq)
STF2, NTF2 = filter2.frequencyResponse(freq)
fig = plt.figure()
ax1 = fig.add_subplot(2, 1, 1)
ax2 = fig.add_subplot(2, 1, 2)
ax1.semilogx(freq, 20 * np.log10(STF1), label="STFOld")
ax1.semilogx(freq, 20 * np.log10(np.abs(NTF1)), label="NTFOld")
ax2.semilogx(freq, 20 * np.log10(STF2), label="STFNew")
ax2.semilogx(freq, 20 * np.log10(np.abs(NTF2)), label="NTFNew")
ax2.semilogx(freq, 20 * np.log10(np.abs(np.sum(NTF2,axis=1))), label="NTFNewCombined")
# ax2.semilogx(freq, np.angle(STF1, deg=True))
# ax2.semilogx(freq, np.angle(NTF, deg=True))
ax1.legend()

# ax1.set_ylim(-100, 10)

error1 = u_middle  - u_hat_middle1 / stf1[0]
error2 = u_middle  - u_hat_middle2 / stf2[0]
#
plt.figure()
plt.title("Simultated Power Spectral Density")
error1_den = np.abs(scipy.fftpack.fft(error1.flatten()))**2 / error1.size * Tc
error2_den = np.abs(scipy.fftpack.fft(error2.flatten()))**2 / error2.size * Tc
f = scipy.fftpack.fftfreq(error1.size, d = Tc)
plt.semilogx(f, 10 * np.log10(error1_den ), label="error_den_old")
plt.semilogx(f, 10 * np.log10(error2_den ), label="error_den_full")
# # f, Pxx_den = signal.welch(u_hat.flatten(), 1./Tc, nperseg=2**20)
# f = scipy.fftpack.fftfreq(u_hat.size, d = Tc)
u_hat1_den = np.abs(scipy.fftpack.fft(u_hat_middle1.flatten()))**2 / u_hat_middle1.size * Tc
u_hat2_den = np.abs(scipy.fftpack.fft(u_hat_middle2.flatten()))**2 / u_hat_middle2.size * Tc
plt.semilogx(f, 20 * np.log10(u_hat1_den), label="u_hat_den_old")
plt.semilogx(f, 20 * np.log10(u_hat2_den), label="u_hat_den_full")
input_den = np.abs(scipy.fftpack.fft(u.flatten()))**2 / u.size * Ts
f = scipy.fftpack.fftfreq(u.size, d = Tc)
plt.semilogx(f, 10 * np.log10(input_den ), label="u")

plt.legend()
plt.xlabel("f Hz")
plt.ylabel("PSD [dB/Hz]")
#
# # print("Mean Square Errror = %0.18e" % (np.linalg.norm(error)**2/error.size))
signalMS = np.linalg.norm(u_middle)**2/u_middle.size

errorMS1 = np.linalg.norm(error1)**2/error1.size
snr1 = signalMS / errorMS1
errorMS2 = np.linalg.norm(error2)**2/error2.size
snr2 = signalMS / errorMS2

#
# # snr = 1./errorMS / 2.
print("Average squared error:\nOld = %0.1d dB\nNew = %0.1d dB" % (10 * np.log10(errorMS1), 10 * np.log10(errorMS2)))
print("SNR Old = %0.1d dB\nSNR New = %0.1d dB" % (10 * np.log10(snr1+1e-100), 10 * np.log10(snr2+1e-100)))
# print("Last stage energy %d dB" % (10 * np.log10(np.var(y))))
# # print(model)
plt.show()
