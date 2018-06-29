import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

Tc =  1. / 15e3
OSR = 15
fp = 1/(Tc * OSR * 2.)

gm1 = 1. / 16e3 # 1/16kOhm
gm1 = 15e3 * 1e-8 / 2. # 1/16kOhm
# gm2 = 1e-7
C   = 1e-8 # 10nF
order = 5
wp = 2 * np.pi * fp
# This is a hack
gm2 = (wp * C) ** 2 / gm1 / 3.9
gm2 = 438.6490844928604  * C
# gm2 = (wp * C) ** 2 / gm1 / np.pi
# gm2 /= 5.
# wp = 2 * np.sqrt(gm1 * gm2 / C ** 2)
dcGain = (gm1/gm2)**(order/2.)

AI = np.eye(order, k=-1) * gm1 / C
AO = AI - np.eye(order, k=1) * gm2 / C

b = np.zeros((order,1))
b[0, 0] = gm1
c = np.zeros_like(b)
c[-int(order/2)] = 1.



# SNR = lambda order, OSR: 1 * 6.02 + 1.76 +  10 * np.log10(2 * order + 1) - 20 * order * np.log10(np.pi) + 10 * (2 * order + 1) * np.log10(OSR)
SNR = lambda order, OSR: 1 * 6.02 + 1.76 +  10 * np.log10(2 * order + 1)- 10 * (2 * order + 1)*np.log10(2) - 20 * order * np.log10(np.pi) + 10 * (2 * order + 1) * np.log10(OSR)
for n in range(1, order+1):
    print("Expected SNR = %s\nFor OSR = %s, N = %s" % (SNR(n, OSR),OSR,n))

# print("AI", AI, "AO", AO)

def TIntegrator(s):
    return 1./ np.linalg.det( s * np.eye(order) - AI) * (gm1/C) ** order

def TOscillator(s):
    return 1./ np.linalg.det( s * np.eye(order) - AO) * (gm1/C) ** order

## Compute frequency
# tempPoly = np.poly(AO)
# tempPoly[-1] -= (gm1/C) ** order / np.abs(TOscillator(1j * wp / 2.))
# wp_new = np.max(np.abs(np.roots(tempPoly)))
# print(wp, wp_new)
# # wp = wp_new
wp_new = wp


## compute 3dB points
tempPoly = np.poly(AO)
# tempPoly[-1] -= ((gm1/C) ** order) / dcGain * np.sqrt(2)
poly = np.poly1d(tempPoly)

w3dB = np.imag(poly.r)
# w3dB = (poly.r)

epsilon = 1e-1
x0 = 1j * np.argmax(np.imag(poly.r))

def func(x):
    return -np.abs(poly(x))

res = minimize(func,x0)

print(res)


TI = np.vectorize(TIntegrator)
TO = np.vectorize(TOscillator)

Tp = np.abs(TO(1j * w3dB))
print(Tp)

f = np.logspace(np.log10(fp)-1, np.log10(fp)+1, 1000)
#
plt.figure()
plt.loglog(f, np.abs(poly(2 * np.pi * f * 1j)))

IGain = np.abs(TIntegrator(1j * wp))
OGain = dcGain
OGain = np.abs(TOscillator(1j * wp))


plt.figure()
plt.title("Transferfunction Comparision")
plt.semilogx(f, 20 * np.log10(np.abs(TI( 2 * np.pi * f * 1j))), label='Integrator')
plt.semilogx(f, 20 * np.log10(np.abs(TO( 2 * np.pi * f * 1j))), label='Oscillator')
plt.semilogx([fp, fp/2.], 20 * np.log10(np.abs(TO( 2 * np.pi * np.array([fp, fp/2.]) * 1j))), "+", label='fp')
plt.semilogx([wp_new/(2. * np.pi)], 20 * np.log10(np.abs(TO(wp_new * 1j))), "+", label='fp_new')
plt.semilogx(w3dB/(2. * np.pi), 20 * np.log10(Tp), '*', label="3dB points")
plt.semilogx(f, np.ones_like(f) * (20 * np.log10( np.abs(TOscillator(1j * wp)))), label="dcGain")


plt.xlabel('$f$ Hz')
plt.ylabel('dB')

plt.legend()

##### Roots

pcoeffO = np.poly(AO)
prootsO = np.roots(pcoeffO)


pcoeffI = np.poly(AI)
prootsI = np.roots(pcoeffI)


plt.figure()
plt.title("Transferfunction Poles")
plt.xlabel("Real axis")
plt.ylabel("Imaginary axis")
plt.plot(np.real(prootsO), np.imag(prootsO), 'x', label="Oscillator")
plt.plot(np.real(prootsI), np.imag(prootsI), 'x', label="Integrator")
plt.plot([0, 0],[wp, - wp],'+', label="wp")
plt.axis('equal')
plt.legend()

## Continue with reconstruction for different eta2

def NTFIntegrator(s, eta2):
    return np.conj(TIntegrator(s)) / (np.abs(TIntegrator(s))**2 + eta2)

def STFIntegrator(s, eta2):
    return NTFIntegrator(s, eta2) * TIntegrator(s)

def NTFOscillator(s, eta2):
    return np.conj(TOscillator(s)) / (np.abs(TOscillator(s))**2 + eta2)

def STFOscillator(s, eta2):
    return NTFOscillator(s, eta2) * TOscillator(s)


NTFI = np.vectorize(NTFIntegrator)
STFI = np.vectorize(STFIntegrator)
NTFO = np.vectorize(NTFOscillator)
STFO = np.vectorize(STFOscillator)

plt.figure()
plt.subplot(211)
plt.title("Signal Transferfunction")
# plt.xlabel("$f$ Hz")
plt.ylabel("dB")

plt.subplot(212)
plt.title("Noise Transferfunction")
plt.xlabel("$f$ Hz")
plt.ylabel("dB")


eta2 = np.array([IGain])
print("eta2 integrator = %e" % eta2 ** 2)
# eta2O = np.logspace(2 * np.log10(dcGain), 2 * np.log10(dcGain) - 8, 5)
# eta2O = np.logspace(2 * np.log10(np.abs(TO(wp_new * 1j))) + 2, 2 * np.log10(np.abs(TO(wp_new * 1j))) - 8, 6)
# eta2O = np.logspace( 2 * np.log10(np.abs(IGain)) + 4, 2* np.log10(np.abs(IGain)), 5)
eta2O = np.array([np.abs(OGain ** 2)])
print("eta2 Oscillator = %e" % np.abs(OGain))
for e in eta2:
    plt.subplot(211)
    # plt.semilogx(f, 20 * np.log10(np.abs(STFI(1j * 2 * np.pi * f, e))), "--", label=("Integrator, eta2 = %1.e" % (e)))
    plt.semilogx(f, 20 * np.log10(np.abs(STFI(1j * 2 * np.pi * f, e))), "--")
    plt.subplot(212)
    plt.semilogx(f, 20 * np.log10(np.abs(NTFI(1j * 2 * np.pi * f, e))), "--", label=("I, $\eta^2=$%1.e" % (e)))
    plt.semilogx(f, 20 * np.log10(1. - np.abs(STFI(1j * 2 * np.pi * f, e))), "--", label=("I, $\eta^2=$%1.e" % (e)))
    # plt.semilogx(f, 20 * np.log10(np.abs(NTFI(1j * 2 * np.pi * f, e))), "--")
    plt.semilogx(f, - 20 * np.log10(e) * np.ones_like(f), "--" , label="dB = %0.1e" % e)

# Reset color cycle
plt.subplot(211)
plt.gca().set_prop_cycle(None)
plt.subplot(212)
plt.gca().set_prop_cycle(None)

for e in eta2O:
    plt.subplot(211)
    # plt.semilogx(f, 20 * np.log10(np.abs(STFO(1j * 2 * np.pi * f, e))), "-", label=("Oscillator, eta2 = %1.e" % (e)))
    plt.semilogx(f, 20 * np.log10(np.abs(STFO(1j * 2 * np.pi * f, e))), "-")
    plt.subplot(212)
    # plt.semilogx(f, 20 * np.log10(np.abs(NTFO(1j * 2 * np.pi * f, e))), "-", label=("Oscillator, eta2 = %1.e" % (e)))
    plt.semilogx(f, 20 * np.log10(np.abs(NTFO(1j * 2 * np.pi * f, e))), "-", label=("$O, \eta^2=$%1.e" % (e)))
    plt.semilogx(f, 20 * np.log10(1. - np.abs(STFO(1j * 2 * np.pi * f, e))), label=("$O, \eta^2=$%1.e" % (e)))
    plt.semilogx(f, 20 * np.log10(np.abs(STFO(1j * 2 * np.pi * f, e))) - np.log10(np.abs(NTFO(1j * 2 * np.pi * f, e))), label="STF/NTF")
    plt.semilogx(f, - 20 * np.log10(e) * np.ones_like(f), label="dB = %0.1e" % e)

plt.legend()

plt.figure()
for e in eta2:
    plt.semilogx(f, 20 * np.log10(1. - np.abs(STFI(1j * 2 * np.pi * f, e))), label=("I, $\eta^2=$%1.e" % (e)))
for e in eta2O:
    plt.semilogx(f, 20 * np.log10(1. - np.abs(STFO(1j * 2 * np.pi * f, e))), label=("$O, \eta^2=$%1.e" % (e)))

plt.legend()


plt.figure()
plt.semilogx(f, 20 * np.log10(np.abs(TI( 2 * np.pi * f * 1j))), label='$|G|$')
plt.semilogx(f, 20 * np.log10(np.abs(1./(order + 1)/(1 + np.abs(TI(2 * np.pi * fp * 1j))))) * np.ones_like(f), label="$NTF$")
plt.semilogx(f, -SNR(order, OSR) * np.ones_like(f), label="PSD")
plt.semilogx(f, 40 * np.log10(np.abs(TI( 2 * np.pi * f * 1j))), label='$|G|^2$')
# plt.semilogx(f, 20 * np.log10(np.abs(TI( 2 * np.pi * f * 1j))) - 40 * np.log10(eta2) , label='Integrator')
plt.semilogx(f, -20 * np.log10(eta2)* np.ones_like(f), label = "$\eta^{-2}$")
plt.semilogx(f, 20 * np.log10(eta2)* np.ones_like(f), label = "$\eta^{2}$")
plt.semilogx(f, 20 * np.log10(np.abs(STFI(1j * 2 * np.pi * f, eta2))), "--")
plt.semilogx(f, 20 * np.log10(np.abs(NTFI(1j * 2 * np.pi * f, eta2))), "--", label=("I, $\eta^2=$%1.e" % (eta2)))
plt.semilogx(f, 20 * np.log10(1. - np.abs(STFI(1j * 2 * np.pi * f, eta2))), "--", label=("I, $\eta^2=$%1.e" % (eta2)))
plt.legend()


plt.figure()
plt.semilogx(f, -20 * np.log10(eta2O)* np.ones_like(f), label = "eta2")
plt.semilogx(f, 20 * np.log10(np.abs(TO( 2 * np.pi * f * 1j))), label='Oscillator')
# plt.semilogx(f, 20 * np.log10(np.abs(TO( 2 * np.pi * f * 1j))) - 40 * np.log10(eta2O), label='Oscillator')
plt.semilogx(f, 20 * np.log10(np.abs(STFO(1j * 2 * np.pi * f, eta2O))), "--")
plt.semilogx(f, 20 * np.log10(np.abs(NTFO(1j * 2 * np.pi * f, eta2O))), "--", label=("I, $\eta^2=$%1.e" % (eta2O)))
plt.semilogx(f, 20 * np.log10(1. - np.abs(STFO(1j * 2 * np.pi * f, eta2O))), "--", label=("I, $\eta^2=$%1.e" % (eta2O)))
plt.legend()

plt.show()
