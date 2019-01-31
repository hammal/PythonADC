import numpy as np
import AnalogToDigital as ADC
import matplotlib.pyplot as plt


systemOrder = 8
gm = 1e-3
C = 1e-5

defaultSystems = ADC.DefaultSystems()
model = defaultSystems.gmCChain(systemOrder, gm, gm * 1e-1, C)
model.B = - np.eye(systemOrder)  * gm / C

print(model)

Ts = 1e-4

t = np.arange(1000) * Ts

u = np.sin(2. * np.pi * t**3 * 10 + np.pi/2.)
# u = np.zeros_like(t)
# u[0]= 10.
# u[-5000:] = 0

simulator = ADC.autoControlSimulator(model, 1./Ts)
states, observations = simulator.Step(u)

plt.figure()
plt.semilogy(t, np.abs(u), label="u")
for index in range(systemOrder):
    plt.semilogy(t, np.abs(states[:,index]), label="state %s" % (index))
plt.legend()

plt.figure()
plt.plot(t, (u), label="u")
for index in range(systemOrder):
    plt.plot(t, (states[:,index]), label="state %s" % (index))
plt.legend()



filterSpec = {
    'eta2': 1e2,
    'model': model,
    'Ts': Ts
}

filter = ADC.WienerFilterAutomaticSystem(filterSpec)

u_hat, logstr = filter.filter(states)

print(u_hat)

plt.figure()
plt.plot(t, u, label="u")
plt.plot(t, u_hat, label="u_hat")
plt.title("Input Reconstruction")
plt.legend()

print("Mean Square Errror = %f" % (np.linalg.norm(u - u_hat)**2 / np.double(t.size)))
plt.show()
