import AnalogToDigital as ADC
import matplotlib.pyplot as plt
import numpy as np

# gm_integrator = 1./16e3
gm_integrator = 15e3 * 1e-8 / 2. # 1/16kOhm

C   = 1e-8 # 10nF
order = 5
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

# print(model)


import scipy.signal as signal

SIM_OSR = 1000
Tc = 1./15e3
Ts= Tc / SIM_OSR
length = 2e6

t = np.arange(length) * Ts
u = signal.resample(np.random.randn(int(length/SIM_OSR)), int(length))
simulator = ADC.Simulator(model, 1./Ts, 1./Tc, u)
controller, y = simulator.Run()

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
