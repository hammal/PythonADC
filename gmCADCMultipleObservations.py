import AnalogToDigital as ADC
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as signal

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
C   = 1e-8 # 10nF
order = 5
wp = 2 * np.pi * 500

poly = [1, -gm_integrator, 1./4. * (wp * C) ** 2]

gm1 = np.max(np.roots(poly))
#  Overwrite previous comparision gm_1 = gm_integrator
gm1 = gm_integrator


gm2 = (wp * C) ** 2 / gm1 / 4
# wp = 2 * np.sqrt(gm1 * gm2 / C ** 2)
fp = (wp / (2. * np.pi))
# dcGain = (gm1/gm2)**(order/2.)


defaultSystems = ADC.DefaultSystems()
model = defaultSystems.gmCChain(order, gm1, gm2, C)
# model = defaultSystems.gmCIntergratorChain(order, gm1, C)

# model.B = - np.eye(order) * gm_integrator / C
model.B = - np.eye(order) * gm_integrator / C * 1.25
# model.c = np.zeros((order, 1))
# model.c[-1] = 1
model.c = np.eye(order)
model.c = np.zeros_like(model.c)
model.c[4, 4] = 1

fsignal = 1e2

Tc = 1./15e3
Ts= Tc / 100.# 1000.
length = 1e5

t = np.arange(length) * Ts
# u = np.sin(2. * np.pi * t)
# u[-5000:] = 0
fspace = np.array([fsignal])
# fspace = np.linspace(0, fsignal, 13)
u = np.zeros_like(t)
for f in fspace:
    u += 1./fspace.size * np.sin(2 * np.pi * f * t)
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
plt.title("Intermediate signal power spectral density")
f, Pxx_den = signal.welch(y.transpose(), 1./Ts, nperseg=2**14)
plt.semilogx(f, 20 * np.log10(Pxx_den.transpose()), label="output")
plt.xlabel("f Hz")
plt.ylabel("PSD [dB/Hz]")

filterSpec = {
    'eta2': 2.762478e+04,
    # 'eta2': 2.453811e+02 * 2,
    'model': model,
    'Ts': Ts
}

filter = ADC.WienerFilter(**filterSpec)

u_hat = filter.filter(controller)
# stf, ntf = filter.frequencyResponse(fspace)

size = int(u.size)
u_middle = u[int(size/4):-int(size/4)]
u_hat_middle = u_hat[int(size/4):-int(size/4)]
stf = [1./(np.dot(u_middle, u_hat_middle) / np.linalg.norm(u_hat_middle)**2)]

# print(stf)

plt.figure()
plt.plot(t, u, label="u")
plt.plot(t, u_hat, label="u_hat")
plt.plot(t, u * stf, label="u amp-corrected")
plt.title("Input Reconstruction")
plt.legend()

freq = np.logspace(np.log10(fsignal) - 2, np.log10(fsignal) + 3, 200)
STF = np.zeros((freq.shape[0], order))
NTF = np.zeros_like(STF)
STF, NTF = filter.frequencyResponse(freq)
fig = plt.figure()
ax1 = fig.add_subplot(2, 1, 1)
ax2 = fig.add_subplot(2, 1, 2)
ax1.semilogx(freq, 20 * np.log10(STF), label="STF")

ax1.set_color_cycle(None)

ax1.semilogx(freq, 20 * np.log10(np.abs(NTF)), '--', label="NTF")
ax2.semilogx(freq, np.angle(STF, deg=True))
ax2.semilogx(freq, np.angle(NTF, deg=True))
ax1.legend()

# ax1.set_ylim(-100, 10)


error = u_middle * stf[0] - u_hat_middle

plt.figure()
plt.title("Error Power Spectral Density")
f, Pxx_den = signal.welch(error.flatten(), 1./Ts, nperseg=2**14)
plt.semilogx(f, 20 * np.log10(Pxx_den ))
plt.xlabel("f Hz")
plt.ylabel("PSD[dB/Hz]")

# print("Mean Square Errror = %0.18e" % (np.linalg.norm(error)**2/error.size))
ase = np.linalg.norm(error)**2/error.size
print("Average squared error = %0.18e" % (ase))
print("SNR = %0.1d dB" % (-10 * np.log10(ase)))
print("Last stage energy %d dB" % (10 * np.log10(np.var(y))))
# print(model)
plt.show()
