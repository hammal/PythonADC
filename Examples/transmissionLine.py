import AnalogToDigital as ADC
import matplotlib.pyplot as plt
import numpy as np

order = 6
Rs = -1
Ls = 1e-2
Gp = 0
Cp = 1e-2


defaultSystems = ADC.DefaultSystems()
model = defaultSystems.transmissionLine(order, Rs, Ls, Gp, Cp)

print(model)

Ts = 1e-3
Tc = 1e-3

import scipy.signal as signal

w, mag, phase = signal.bode((model.A, model.b, model.c.transpose(), np.array([0])))
plt.figure()
plt.semilogx(w, mag)    # Bode magnitude plot
plt.figure()
plt.semilogx(w, phase)  # Bode phase plot
plt.show()


t = np.arange(100000) * Ts
# u = np.sin(2. * np.pi * t)
# u[-5000:] = 0
u = np.sin(2. * np.pi * (t**2)*1e-1)
simulator = ADC.Simulator(model, 1./Ts, 1./Tc, u)
controller, y = simulator.Run()


for index in range(order):
    plt.plot(controller[index, :],label="Control = %s" % index)
plt.plot(y, label="y")
plt.title("Controlls")
plt.legend()

filterSpec = {
    'eta2': 1e2,
    'model': model,
    'Ts': Ts
}

filter = ADC.WienerFilter(filterSpec)

u_hat = filter.filter(controller)

plt.figure()
plt.plot(t, u, label="u")
plt.plot(t, u_hat, label="u_hat")
plt.title("Input Reconstruction")
plt.legend()
freq = np.logspace(-2, 4)
STF, NTF = filter.frequencyResponse(freq)

fig = plt.figure()
ax1 = fig.add_subplot(2, 1, 1)
ax2 = fig.add_subplot(2, 1, 2)
ax1.semilogx(freq, 20 * np.log10(STF), label="STF")
ax1.semilogx(freq, 20 * np.log10(np.abs(NTF)), label="NTF")
ax2.semilogx(freq, np.angle(STF, deg=True))
ax2.semilogx(freq, np.angle(NTF, deg=True))
ax1.legend()
print("Mean Square Errror = %0.18f" % (np.linalg.norm(u - u_hat)**2 / np.double(t.size)))

plt.show()
