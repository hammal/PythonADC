import AnalogToDigital as ADC
import matplotlib.pyplot as plt
import numpy as np


def quantizer(x):
    return np.digitize(x, bins=np.array([-1, 0]))

# gm_integrator = 1./16e3
gm_integrator = 15e3 * 1e-8 / 2. # 1/16kOhm

C   = 1e-8 # 10nF
order = 2
# wp = 2 * np.pi * 500

# poly = [1, -gm_integrator, 1./4. * (wp * C) ** 2]

# gm1 = np.max(np.roots(poly))
#  Overwrite previous comparision gm_1 = gm_integrator
gm1 = gm_integrator

OSR = 16
fp = 15e3 / OSR / 2.
# fsignal = 1e1
fsignal = fp
wp = np.pi * fp * 2.


gm2 = (wp * C) ** 2 / gm1 / 4
# wp = 2 * np.sqrt(gm1 * gm2 / C ** 2)
# fp = (wp / (2. * np.pi))
# dcGain = (gm1/gm2)**(order/2.)


defaultSystems = ADC.DefaultSystems()
# model = defaultSystems.gmCChain(order, gm1, gm2, C)
model = defaultSystems.gmCIntergratorChain(order, gm1, C)

model.B = - np.eye(order) * gm_integrator / C
# model.B = - np.eye(order) * gm_integrator / C * 1.25
# model.c = np.zeros((order, 1))
# model.c[-1] = 1



import scipy.signal as signal

SIM_OSR = 1000
Tc = 1./15e3
Ts= Tc / SIM_OSR
length = 2e6

t = np.arange(length) * Ts
# u = signal.resample(np.random.randn(int(length/SIM_OSR)), int(length))
u = signal.resample(2 * (np.random.rand(int(length/SIM_OSR))) - 1, int(length))
u = u/np.max(np.abs(u))
plt.plot(t, u)

simulator = ADC.Simulator(model, 1./Ts, 1./Tc, u)
controller, y = simulator.Run()

print(model)


plt.figure()
y_d = 2*(quantizer(y)-1.5)
e = y - y_d
plt.plot(t, y)
plt.plot(t, y_d)
plt.plot(t, e)

plt.figure()
plt.hist(e, bins=1000)


y2 = model.filter(u)
y2_s = model.filter(y_d.flatten())
e2 = y2 - y2_s

plt.figure()
plt.hist(e2,bins=1000)
plt.show()

## Downsample
controller.subsample()
t = t[::SIM_OSR]
u = u[::SIM_OSR]


## Store dataset
data = {
    "name":"GaussianIntegrator",
    "y":u,
    "x":controller.control_sequence,
    "Tc":Tc,
    "Ts":Ts,
    "length":length
}

np.save("data",data)
