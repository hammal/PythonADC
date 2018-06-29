import AnalogToDigital as ADC

import numpy as np

A = [[0, 0, 0], [1, 0, 0], [0, 1, 0]]
B = [[-1, 0, 0], [0, -1, 0], [0, 0, -1]]
b = [1, 0, 0]
c = [0, 0, 1]

Ts = 1e-3
Tc = 1e-1


model = ADC.Model(A, B, b, c)
u = np.sin(2. * np.pi * np.arange(10000) * Ts)
simulator = ADC.Simulator(model, 1./Ts, 1./Tc, u)
controls, y = simulator.Run()

import matplotlib.pyplot as plt

plt.plot(controls[0, :])
plt.plot(controls[1, :])
plt.plot(controls[2, :])
plt.plot(y)
plt.show()

print(controls)
