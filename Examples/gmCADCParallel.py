import AnalogToDigital as ADC
# import matplotlib.pyplot as plt
import numpy as np
import sys

gm_integrator = 1./16e3
C   = 1e-8 # 10nF
order = 5
wp = 2 * np.pi * 500

# poly = [1, -gm_integrator, 1./4. * (wp * C) ** 2]

# gm1 = np.max(np.roots(poly))
#  Overwrite previous comparision gm_1 = gm_integrator
gm1 = gm_integrator


gm2 = (wp * C) ** 2 / gm1 / 4
gm2 = 526.37890139143245967117285332673 * C
# wp = 2 * np.sqrt(gm1 * gm2 / C ** 2)
fp = (wp / (2. * np.pi))
# dcGain = (gm1/gm2)**(order/2.)

NUMBER_OF_JOBS = int(sys.argv[2])
LENGTH = 1e7
ETA2INDEX = int(sys.argv[3])



eta2 = np.array([
     1e0,
     1e1,
     1e2,
     1e-2,
     1e-1
])


defaultSystems = ADC.DefaultSystems()
model = defaultSystems.gmCChain(order, gm1, gm2, C)
# model = defaultSystems.gmCIntergratorChain(order, gm1, C)

model.B = - np.eye(order) * gm1 / C
model.c = np.zeros((order, 1))
model.c[-1] = 1

# eta2 *= np.abs(1./ np.linalg.det( 1j * wp * np.eye(order) - model.A) * (gm1/C) ** order)**2
eta2 *= 114593.20985705861473520190874423

frange = np.logspace(0, 4, NUMBER_OF_JOBS)
fsignal = frange[int(sys.argv[1])]

SIM_OSR = 1000
Tc = 1./15e3
Ts= Tc / SIM_OSR
length = 1e7

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

t = np.arange(LENGTH) * Ts
# u = np.sin(2. * np.pi * t)
# u[-5000:] = 0
fspace = np.array([fsignal])
# fspace = np.linspace(0, fsignal, 13)
u = np.zeros_like(t)
for f in fspace:
    u += 1./fspace.size * np.sin(2 * np.pi * f * t)
simulator = ADC.Simulator(model, 1./Ts, 1./Tc, u)
controller, y = simulator.Run()

# plt.figure()
#
# for index in range(order):
#     plt.plot(controller[index, :],label="Control = %s" % index)
# plt.plot(y, label="y")
# plt.title("Controlls")
# plt.legend()

# plt.figure()
# plt.title("FFT of error signal")
# plt.xlabel("freq")
# plt.ylabel("$| \cdot| $")
# plt.loglog(np.fft.fftfreq(y.size, d=Ts), np.abs(np.fft.fft(y)))

# plt.figure()
# plt.title("Power Spectral Density")
# f, Pxx_den = signal.welch(y.flatten(), 1./Ts, nperseg=2**14)
# plt.loglog(f, Pxx_den)

## Downsample
controller.subsample()
t = t[::SIM_OSR]
u = u[::SIM_OSR]

filterSpec = {
    'eta2': eta2[ETA2INDEX],
    'model': model,
    'Tc': Tc
}

filter = ADC.WienerFilter(**filterSpec)

u_hat = filter.filter(controller)
stf, ntf = filter.frequencyResponse(fspace)

size = int(u.size)
u_middle = u[int(size/4):-int(size/4)]
u_hat_middle = u_hat[int(size/4):-int(size/4)]
# stf = [1./(np.dot(u_middle, u_hat_middle) / np.linalg.norm(u_hat_middle)**2)]

# print(stf)

# plt.figure()
# plt.plot(t, u, label="u")
# plt.plot(t, u_hat, label="u_hat")
# plt.plot(t, u_hat / stf, label="u_hat amp-corrected")
# plt.title("Input Reconstruction")
# plt.legend()
# freq = np.logspace(np.log10(fsignal) - 1, np.log10(fsignal) + 1)
# STF, NTF = filter.frequencyResponse(freq)
#
# fig = plt.figure()
# ax1 = fig.add_subplot(2, 1, 1)
# ax2 = fig.add_subplot(2, 1, 2)
# ax1.semilogx(freq, 20 * np.log10(STF), label="STF")
# ax1.semilogx(freq, 20 * np.log10(np.abs(NTF)), label="NTF")
# ax2.semilogx(freq, np.angle(STF, deg=True))
# ax2.semilogx(freq, np.angle(NTF, deg=True))
# ax1.legend()

# ax1.set_ylim(-100, 10)

error = u_middle - u_hat_middle / stf[0]

# print("Mean Square Errror = %0.18e" % (np.linalg.norm(error)**2/error.size))
ase = np.linalg.norm(error)**2/error.size
mss = np.linalg.norm(u_middle)**2/u_middle.size
snr = mss / ase

# plt.figure()
# plt.title("Error Power Spectral Density")
# f, Pxx_den = signal.welch(error.flatten(), 1./Ts, nperseg=2**14)
# plt.loglog(f, Pxx_den)

# print("Mean Square Errror = %0.18e" % (np.linalg.norm(error)**2/error.size))
print("%0.18e, %0.18e, %0.18e" % (fsignal, ase, snr))

# plt.show()
