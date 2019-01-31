import AnalogToDigital as ADC
import matplotlib.pyplot as plt


import numpy as np

A = [[0, 0, 0], [10, 0, 0], [0, 10, 0]]
B = [[-1, 0, 0], [0, -1, 0], [0, 0, -1]]
b = [1, 0, 0]
c = [0, 0, 1]

Ts = 1e-2
Tc = 1e-2


t = np.arange(10000) * Ts

model = ADC.Model(A, B, b, c)


u = np.sin(2. * np.pi * 1e-1 * t)
simulator = ADC.Simulator(model, 1./Ts, 1./Tc, u)
controller, y =simulator.Run()

print(model)

plt.plot(controller[0, :])
plt.plot(controller[1, :])
plt.plot(controller[2, :])
plt.plot(y)
plt.show()

filterSpec = {
    'eta2': 1e2,
    'model': model,
    'Ts': Ts
}

filter = ADC.WienerFilter(filterSpec)

u_hat, logstr = filter.filter(controller)

plt.figure()
plt.plot(t, u)
plt.plot(t, u_hat)
plt.show()

freq = np.logspace(-2, 4)
STF, NTF = filter.frequencyResponse(freq)

fig = plt.figure()
ax1 = fig.add_subplot(2, 1, 1)
ax2 = fig.add_subplot(2, 1, 2)
ax1.semilogx(freq, 20 * np.log10(STF))
ax1.semilogx(freq, 20 * np.log10(np.abs(NTF)))
ax2.semilogx(freq, np.angle(STF, deg=True))
ax2.semilogx(freq, np.angle(NTF, deg=True))

print(np.linalg.norm(u - u_hat)/np.double(t.size))

plt.show()
