import numpy as np
import AnalogToDigital as ADC
import matplotlib.pyplot as plt
import scipy.signal
np.random.seed(1287)

bound = 1.
order = 1

SIM_OSR = 1000
Tc = 1./15e3
Ts= Tc / SIM_OSR
length = 1e6

fsignal = 7e1


A = - np.eye(order) * 0.0001

c = np.eye(order)
B = - np.eye(order) / Tc /2
b = np.ones((order, 1))
# b[0] = 1 * norm[0,0];
b = -np.dot(B,b)

model = ADC.Model(A, B, b, c)
model.discretize(Tc)
print(model)
frequency = np.logspace(-1,3, 1000)

amplitudeResponse = np.zeros((frequency.size, order))

for index, freq in enumerate(frequency):
    amplitudeResponse[index, :] = 20 * np.log10(np.abs(model.frequncyResponse(freq).flatten()))

plt.semilogx(frequency, amplitudeResponse)

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
plt.semilogy(t,np.abs(y))
plt.xlabel('time s')
plt.ylabel('amplitude')

plt.figure()

for index in range(order):
    plt.plot(controller[index, :],label="Control = %s" % index)
plt.title("Controlls")
plt.legend()


## Downsample
controller.subsample()
t = t[::SIM_OSR]
u = u[::SIM_OSR]

# eta2 = 114593.20985705861473520190874423
eta2 = np.eye(order) * 1000.
filterSpec = {
    # 'eta2': 2.762478e+04 * 2 ** 4,
    # 'eta2': 2.453811e+02 * 2,
    "eta2": eta2,
    'model': model,
    'Tc': Tc
}

filter = ADC.WienerFilter(**filterSpec)

u_hat, logstr = filter.filter(controller)
# stf, ntf = filter.frequencyResponse(fspace)
# import scipy.signal
b,a = scipy.signal.butter(5, fsignal * Tc * 2 )
# u_hat = scipy.signal.filtfilt(b,a, u_hat)
# u_hat = scipy.signal.filtfilt(b,a, u_hat)
# u_hat = scipy.signal.filtfilt(b,a, u_hat)


size = int(u.size)
u_middle = u[int(size/4):-int(size/4)]
u_hat_middle = u_hat[int(size/4):-int(size/4)]
stf = [1./(np.dot(u_middle, u_hat_middle) / np.linalg.norm(u_hat_middle)**2)]

# print(stf)

plt.figure()
plt.plot(t, u, label="u")
plt.plot(t, u_hat/stf[0], label="u_hat")
# plt.plot(t, u * stf[0], label="u amp-corrected")
plt.title("Input Reconstruction")
plt.legend()
freq = np.logspace(np.log10(fsignal) - 1, np.log10(fsignal) + 1)


quantaisationNoise = np.var(y,axis=0) * Tc

STF, NTF = filter.frequencyResponse(freq)
fig = plt.figure()
ax1 = fig.add_subplot(2, 1, 1)
ax2 = fig.add_subplot(2, 1, 2)
ax2.semilogx(freq, 20 * np.log10(np.abs(NTF)), label="NTF")
ax2.semilogx(freq, 10 * np.log10(np.dot(np.abs(NTF)**2, quantaisationNoise)), label="NTFCombined")
ax1.semilogx(freq, 20 * np.log10(STF), label="STF")
# ax2.semilogx(freq, np.angle(STF, deg=True))
# ax2.semilogx(freq, np.angle(NTF, deg=True))
ax1.legend()
ax2.legend()
ax1.set_ylim(-100, 10)


error = u_middle - u_hat_middle/stf[0]


# print("Mean Square Errror = %0.18e" % (np.linalg.norm(error)**2/error.size))
ase = np.linalg.norm(error)**2/error.size
mss = np.linalg.norm(u_middle)**2/u_middle.size
snr = mss / ase
print("Average squared error = %0.18e" % (ase))
print("SNR = %0.1d dB" % (10 * np.log10(snr)))
print(y.shape)
print("Last stage energy %s dB" % (10 * np.log10(np.var(y,axis=0))))
# print(model)
plt.show()
