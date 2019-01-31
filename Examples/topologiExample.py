import AnalogToDigital as ADC
import matplotlib.pyplot as plt
import copy
import numpy as np

A = [[0, 0, 0, 0], [10, 0, 0, 0], [0, 10, 0, 0], [0, 0, 10, 0]]
B = [[-10, 0, 0, 0], [0, -10, 0, 0], [0, 0, -10, 0], [0, 0, 0, -10]]
b = [[1], [0], [0], [0]]
c = [[0], [0], [0], [1]]

Ts = 1e-2
Tc = 1e-2


t = np.arange(10000) * Ts

model = ADC.Model(A, B, b, c)
model.discretize(Ts)
filterSpec = {
    'eta2': 1e2,
    'model': model,
    'Ts': Ts,
    'order': 128,
    'f3dB': 1e2
}

simulationSetups = {'original':filterSpec}
# filterSpec = topology.DiagonaliseSteadyStateMatrix()
print("Original")
print(simulationSetups['original']['model'])
topology1 = ADC.Topology(copy.deepcopy(filterSpec))

simulationSetups['diagonalised'] = topology1.DiagonaliseSteadyStateMatrix()
print("Diagonalised")
print(simulationSetups['diagonalised']['model'])

topology2 = ADC.Topology(copy.deepcopy(filterSpec))
simulationSetups['ButterWorth'] = topology2.ButterWorthSystem()
print("Butterworth")
print(simulationSetups['ButterWorth']['model'])

topology3 = ADC.Topology(copy.deepcopy(filterSpec))
simulationSetups['Controllable'] = topology3.ControllableForm()
print("Controllable")
print(simulationSetups['Controllable']['model'])

topology4 = ADC.Topology(copy.deepcopy(filterSpec))
simulationSetups['Observable'] = topology4.ObservableForm()
print("Observable")
print(simulationSetups['Observable']['model'])


plt.figure(1)

results = {}
# u = np.sin(2. * np.pi * 1e-1 * t)
from AnalogToDigital.defaultSystems import DefaultSystems
dfs = DefaultSystems()
u = dfs.integratorChain(5).filter(dfs.whiteNoise(t.size))

plt.plot(u, 'k', label='reference')
for key, value in simulationSetups.items():
    print(key)
    name = key + "w/oMC"
    # Without minimal control
    model = ADC.Model.checkTypes(value['model'])
    sim = ADC.Simulator(model, 1. / Ts, 1. / Tc, u)
    controller, y = sim.Run()
    filter = ADC.WienerFilter(value)
    u_hat, logstr = filter.filter(controller)
    results[name] = np.linalg.norm(u - u_hat) / np.double(t.size)

    plt.figure(2)
    plt.plot(controller[0, :], label=name)
    plt.figure(3)
    plt.plot(controller[1, :], label=name)
    plt.figure(4)
    plt.plot(controller[2, :], label=name)
    plt.figure(1)
    plt.plot(u_hat, label=name)
    plt.figure(5)
    plt.plot(y, label=name)
    # With minimal control
    name = key + "wMC"
    model = topology1.computeSmallestControlStrength(model, Ts, 1e0)
    sim = ADC.Simulator(model, 1. / Ts, 1. / Tc, u)
    controller, y = sim.Run()
    filter = ADC.WienerFilter(value)
    u_hat, logstr = filter.filter(controller)
    results[name] = np.linalg.norm(u - u_hat) / np.double(t.size)

    plt.figure(2)
    plt.plot(controller[0, :], label=name)
    plt.figure(3)
    plt.plot(controller[1, :], label=name)
    plt.figure(4)
    plt.plot(controller[2, :], label=name)
    plt.figure(1)
    plt.plot(u_hat, label=name)


    plt.figure(5)
    plt.plot(y, label=name)
# filterSpec['model'] = topology.computeSmallestControlStrength(filterSpec['model'], Ts, 1e0)


# simulator = Simulator(filterSpec['model'], 1./Ts, 1./Tc, u)
# controller, y = simulator.Run()
#
#
#
# plt.plot(controller[0, :])
# plt.plot(controller[1, :])
# plt.plot(controller[2, :])
print(results)
for i in range(1,6):
    plt.figure(i)
    plt.legend()
# plt.plot(y)
plt.show()
#
#
# filter = WienerFilter(filterSpec)
#
# u_hat, logstr = filter.filter(controller)
#
# plt.figure()
# plt.plot(t, u)
# plt.plot(t, u_hat)
# plt.show()
#
# freq = np.logspace(-2, 4)
# STF, NTF = filter.frequencyResponse(freq)
#
# fig = plt.figure()
# ax1 = fig.add_subplot(2, 1, 1)
# ax2 = fig.add_subplot(2, 1, 2)
# ax1.semilogx(freq, 20 * np.log10(STF))
# ax1.semilogx(freq, 20 * np.log10(np.abs(NTF)))
# ax2.semilogx(freq, np.angle(STF, deg=True))
# ax2.semilogx(freq, np.angle(NTF, deg=True))
#
# print(np.linalg.norm(u - u_hat)/np.double(t.size))
#
# plt.show()
