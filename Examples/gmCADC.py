import AnalogToDigital as ADC
import matplotlib.pyplot as plt
import numpy as np


"""
 What is a fair comparision with the pure integrator chain (gm2 = 0)?
 Since adding gm2 != 0 changes the control requirements additively we
 can formulate the following two conditions namely:

 gm_1 + gm_2 = gm_integrator

 Furthermore, since we have a formula relating gm_1 and gm_2 for a given
 w_p

 gm2 = 1/4 * (w_p * C)^2 / gm_1

 so finding gm_1 is similair to finding the positve root of the polynomial

 gm_1 ^ 2 - gm_1 gm_integrator + 1/4 * (wp * C)^2
"""
gm_integrator = 1./16e3
# gm_integrator = 15e3 * 1e-8 / 2. # 1/16kOhm

C = 1e-8 # 10nF
order = 5
# wp = 2 * np.pi * 500

# poly = [1, -gm_integrator, 1./4. * (wp * C) ** 2]

# gm1 = np.max(np.roots(poly))
#  Overwrite previous comparision gm_1 = gm_integrator
gm1 = gm_integrator

# OSR = 16
OSR = 15e3/(5e2 * 2)
# OSR = 50
SNR = 1 * 6.02 + 1.76 + 10 * np.log10(2 * order + 1) - 20 * order * np.log10(np.pi) + 10 * (2 * order + 1) * np.log10(OSR)
print("Expected SNR = %s" % SNR)

fp = 15e3 / OSR / np.sqrt(2) / 2.
fp = 15e3 / OSR / np.sqrt(2) / 2. / 10.
# fp = 15e3 / 4. / 2.
# fsignal = 1e1
wp = np.pi * fp * 2.


gm2 = (wp * C) ** 2 / gm1 / 4
gm2 = 526.37890139143245967117285332673 * C
# wp = 2 * np.sqrt(gm1 * gm2 / C ** 2)
# fp = (wp / (2. * np.pi))
# dcGain = (gm1/gm2)**(order/2.)


defaultSystems = ADC.DefaultSystems()
model, steeringVector = defaultSystems.gmCChain(order, gm1, gm2, C)
# model = defaultSystems.gmCIntergratorChain(order, gm1, C)

mixingMatrix = - np.eye(order) * gm_integrator / C * 1.0
# model.B = - np.eye(order) * gm_integrator / C * 1.25
# model.c = np.zeros((order, 1))
# model.c[-1] = 1
# model.c = np.eye(order)

def TOscillator(s):
    return 1./ np.linalg.det( s * np.eye(order) - model.A) * (gm1/C) ** order

print(model)

# fsignal = 1e1
fsignal = fp * np.sqrt(2)

# SIM_OSR = 1000
Tc = 1./15e3
# Ts= Tc / SIM_OSR
length = int(1e6)
length = int(1e5)


ctrl = ADC.Control(mixingMatrix, length)

import scipy.signal as signal

# w, mag, phase = signal.bode((model.A, model.b, model.c.transpose(), np.array([0])))
# plt.figure()
# plt.semilogx(w/(np.pi * 2.), mag)    # Bode magnitude plot
# plt.title("BodeMagnitude Plot")
# plt.xlabel('f Hz')
# # plt.figure()
# # plt.semilogx(w, phase)  # Bode phase plot
# # plt.title("BodePhase Plot")
# # plt.show()
#
# plt.ylim([0, 200])
# plt.xlim([1e3, 1e7])

t = np.arange(length) * Tc
# u = np.sin(2. * np.pi * t)
# u[-5000:] = 0
# fspace = np.array([fsignal])
# fspace = np.linspace(0, fsignal, 13)
# u = np.zeros_like(t)
# for f in fspace:
#     u += 1./fspace.size * np.sin(2 * np.pi * f * t)

input = ADC.Sin(Tc, 1., fsignal, 0., steeringVector)

simulator = ADC.Simulator(model, ctrl)
simRes = simulator.simulate(t, inputs=(input,))


plt.figure()

y = simRes['output']
for index in range(order):
    pass
plt.plot(y[:, -1],label="Output at last stage ")
# plt.plot(y, label="y")
plt.title("Controlls")
plt.legend()

# plt.figure()
# plt.title("FFT of error signal")
# plt.xlabel("freq")
# plt.ylabel("$| \cdot| $")
# plt.loglog(np.fft.fftfreq(y.size, d=Ts), np.abs(np.fft.fft(y)))

plt.figure()
plt.title("Power Spectral Density")
f, Pxx_den = signal.welch(y[:,-1].flatten(), 1./Tc, nperseg=2**14)
plt.semilogx(f, 20 * np.log10(Pxx_den))
plt.xlabel("f Hz")
plt.ylabel("PSD [dB/Hz]")

## Downsample
# controller.subsample()
# t = t[::SIM_OSR]
# u = u[::SIM_OSR]

eta2 = np.abs(TOscillator(1j * wp)) ** 2

print("eta2 = %s" % eta2)

eta2 = 114593.20985705861473520190874423
eta2 = 1.
# eta2 = np.eye(order) * eta2
filterSpec = {
    # 'eta2': 2.762478e+04 * 2 ** 4,
    # 'eta2': 2.453811e+02 * 2,
    "eta2": np.array([eta2]),
    'model': model,
    'Tc': Tc
}

filter = ADC.WienerFilter(t, model, inputs=(input,), options=filterSpec)

u_hat, logstr = filter.filter(ctrl)
# stf, ntf = filter.frequencyResponse(fspace)
# import scipy.signal
# b,a = scipy.signal.butter(13, fp * Tc * 2)
# u_hat = scipy.signal.filtfilt(b,a, u_hat)


evaluation = ADC.Evaluation(model, u_hat, (input,))

fig1 = evaluation.PlotTransferFunctions((1e-1, 15e3))
fig2 = evaluation.PlotPowerSpectralDensity(t)
# size = int(u.size)
size = length
# u_middle = u[int(size/4):-int(size/4)]
# u_hat_middle = u_hat[int(size/4):-int(size/4)]
# stf = [1./(np.dot(u_middle, u_hat_middle) / np.linalg.norm(u_hat_middle)**2)]

# print(stf)

plt.figure()
plt.plot(t, input.scalarFunction(t), label="u")
plt.plot(t, u_hat, label="u_hat")
# plt.plot(t, u_hat/stf[0], label="u_hat")
# plt.plot(t, u * stf[0], label="u amp-corrected")
plt.title("Input Reconstruction")
plt.legend()
# freq = np.logspace(np.log10(fsignal) - 2, np.log10(fsignal) + 3)


# STF, NTF = filter.frequencyResponse(freq)
# fig = plt.figure()
# ax1 = fig.add_subplot(2, 1, 1)
# ax2 = fig.add_subplot(2, 1, 2)
# ax1.semilogx(freq, 20 * np.log10(STF), label="STF")
# ax1.semilogx(freq, 20 * np.log10(np.abs(NTF)), label="NTF")
# ax2.semilogx(freq, np.angle(STF, deg=True))
# ax2.semilogx(freq, np.angle(NTF, deg=True))
# ax1.legend()
#
# ax1.set_ylim(-100, 10)


# error = u_middle - u_hat_middle / stf[0]
#
plt.figure()
plt.title("Power Spectral Density")
f, Pxx_den = signal.welch(u_hat.flatten(), 1./Tc, nperseg=2**16)
plt.semilogx(f, 10 * np.log10(Pxx_den))
plt.xlabel("f Hz")
plt.ylabel("PSD [dB/Hz]")


# print("Mean Square Errror = %0.18e" % (np.linalg.norm(error)**2/error.size))
# ase = np.linalg.norm(error)**2/error.size
# mss = np.linalg.norm(u_middle)**2/u_middle.size
# snr = mss / ase
# print("Average squared error = %0.18e" % (ase))
# print("SNR = %0.1d dB" % (10 * np.log10(snr)))
# print("Last stage energy %d dB" % (10 * np.log10(np.var(y))))
# print(model)
plt.show()
