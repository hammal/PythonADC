import matplotlib.pyplot as plt
import numpy as np

data = np.load("sgeData.npz")

def T(s, A, gain):
    return  gain / np.linalg.det( s * np.eye(A.shape[0]) - A)

def generateSTFNTFANDTF(f, eta2, A, gain):
    def T2(s):
        return  gain / np.linalg.det( s * np.eye(A.shape[0]) - A)

    T2 = np.vectorize(T2)

    def NTF(s, eta2):
        return np.conj(T2(s)) / (np.abs(T2(s))**2 + eta2)

    def STF(s, eta2):
        return NTF(s, eta2) * T2(s)

    NTF = np.vectorize(NTF)
    STF = np.vectorize(STF)

    ntf = []
    stf = []

    for e in eta2:
        ntf.append(np.abs(NTF(2. * np.pi * f * 1j, e)))
        stf.append(np.abs(STF(2. * np.pi * f * 1j, e)))

    return np.array(ntf).transpose(), np.array(stf).transpose(), np.abs(T2(2. * np.pi * f * 1j))

gm1 = 1. / 16e3 # 1/16kOhm
C   = 1e-8 # 10nF
order = 5
wp = 2 * np.pi * 500
gm2 = (wp * C) ** 2 / gm1 / 4

AI = np.eye(order, k=-1) * gm1 / C
AO = AI - np.eye(order, k=1) * gm2 / C

eta2 = np.abs(T(1j*wp, AI, (gm1/C) ** order)) ** 2
eta24 = np.abs(T(1j*wp, AO, (gm1/C) ** order)) ** 2

print(eta2)

simulations = [
    "Integrator",
    "Oscillator0",
    # "Oscillator1",
    # "Oscillator2",
    "Oscillator3",
    "Oscillator4",
    # "OscillatorNumberComparissonRef",
    # "OscillatorNumberComparisson0",
    # "OscillatorNumberComparisson1",
    # "OscillatorNumberComparisson2",
    # "OscillatorNumberComparisson3",
    # "OscillatorNumberComparisson4",
    # "OscillatorCompensated0",
    # "OscillatorCompensated1",
    # "OscillatorCompensated2",
    # "OscillatorCompensated3",
    # "OscillatorCompensated4"
]

# simulations = [
#     "Oscillator0",
#     "Oscillator1",
#     "Oscillator2",
#     "Oscillator3",
#     "Oscillator4",
#     "Integrator"
# ]
#
# numberComparisson = [
#     "OscillatorNumberComparisson0",
#     "OscillatorNumberComparisson1",
#     "OscillatorNumberComparisson2",
#     "OscillatorNumberComparisson3",
#     "OscillatorNumberComparisson4",
#     "OscillatorNumberComparissonRef"
# ]
#
# gainCompensated = [
#     "OscillatorCompensated0",
#     "OscillatorCompensated1",
#     "OscillatorCompensated2",
#     "OscillatorCompensated3",
#     "OscillatorCompensated4"
# ]

LABELS = {
    "OscillatorCompensated0":"Oscillator, eta2 = %0.1e" %(1e0 * eta2),
    "OscillatorCompensated1":"Oscillator, eta2 = %0.1e" %(1e1 * eta2),
    "OscillatorCompensated2":"Oscillator, eta2 = %0.1e" %(1e2 * eta2),
    "OscillatorCompensated3":"Oscillator, eta2 = %0.1e" %(1e3 * eta2),
    "OscillatorCompensated4":"Oscillator, eta2 = %0.1e" %(1e4 * eta2),
    "OscillatorNumberComparissonRef":"Integrator eta2 = %0.1e" % eta24,
    "OscillatorNumberComparisson0":"Oscillator, eta2 = %0.1e" %(1e-2 * eta24),
    "OscillatorNumberComparisson1":"Oscillator, eta2 = %0.1e" %(1e-1 * eta24),
    "OscillatorNumberComparisson2":"Oscillator, eta2 = %0.1e" %(1e0 * eta24),
    "OscillatorNumberComparisson3":"Oscillator, eta2 = %0.1e" %(1e1 * eta24),
    "OscillatorNumberComparisson4":"Oscillator, eta2 = %0.1e" %(1e2 * eta24),
    "Oscillator0":"Oscillator, eta2 = %0.1e" % (1e0 * eta24),
    "Oscillator1":"Oscillator, eta2 = %0.1e" % (1e1 * eta24),
    "Oscillator2":"Oscillator, eta2 = %0.1e" % (1e2 * eta24),
    "Oscillator3":"Oscillator, eta2 = %0.1e" % (1e-2 * eta24),
    "Oscillator4":"Oscillator, eta2 = %0.1e" % (1e-1 * eta24),
    "Integrator" : "Integrator, eta2 = %0.1e" % eta2
}

ETA2 = [
(1e0 * eta2),
(1e1 * eta2),
(1e2 * eta2),
(1e3 * eta2),
(1e4 * eta2),
]

ETA24 =[
    (1e0 * eta24),
    (1e1 * eta24),
    (1e2 * eta24),
    (1e-2 * eta24),
    (1e-1 * eta24)
]

frequency = data[simulations[0]][:,0]



stfntftfIntegrator = generateSTFNTFANDTF(frequency, [eta2], AI, (gm1/C) ** order)
stfntftfOscillator = generateSTFNTFANDTF(frequency, ETA2, AO, (gm1/C) ** order)

f, axarr = plt.subplots(2, 2)
axarr[1,1].set_title("Simulated SINADR")
for sim in simulations:
    axarr[1,1].semilogx(data[sim][:,0], -10 * np.log10(np.abs(data[sim][:,1])), label=LABELS[sim] + " ASE")
    # axarr[1,1].semilogx(data[sim][:,0], 10 * np.log10(np.abs(data[sim][:,2])), label=LABELS[sim] + " SNR")
# axarr[1,1].set_ylim([60,120])
axarr[1,1].set_xlabel("f Hz")
axarr[1,1].set_ylabel("-ASE dB")
axarr[1,1].legend()

axarr[0,1].set_title("Noise transfer function")
axarr[0,1].semilogx(frequency, 20 * np.log10(stfntftfOscillator[0]), label="Oscillator")
axarr[0,1].semilogx(frequency, 20 * np.log10(stfntftfIntegrator[0]), label="Integrator")
axarr[0,1].set_xlabel("f Hz")
axarr[0,1].set_ylabel("dB")
axarr[0,1].legend()

axarr[0,0].set_title("Transfer function")
axarr[0,0].semilogx(frequency, 20 * np.log10(stfntftfOscillator[2]), label="Oscillator")
axarr[0,0].semilogx(frequency, 20 * np.log10(stfntftfIntegrator[2]), label="Integrator")
axarr[0,0].set_xlabel("f Hz")
axarr[0,0].set_ylabel("dB")
axarr[0,0].legend()

axarr[1,0].set_title("Signal transfer function")
axarr[1,0].semilogx(frequency, 20 * np.log10(stfntftfOscillator[1]), label="Oscillator")
axarr[1,0].semilogx(frequency, 20 * np.log10(stfntftfIntegrator[1]), label="Integrator")
axarr[1,0].set_xlabel("f Hz")
axarr[1,0].set_ylabel("dB")
axarr[1,0].legend()

# ###################### Figure 2
#
# wp = 2 * np.pi * 500
# poly = [1, -gm1, 1./4. * (wp * C) ** 2]
# gm1GC = np.max(np.roots(poly))
# gm2GC = (wp * C) ** 2 / gm1GC / 4.
#
# AIGC = np.eye(order, k=-1) * gm1GC / C
# AOGC = AI - np.eye(order, k=1) * gm2GC / C
#
# stfntftfOscillatorGC = generateSTFNTFANDTF(frequency, ETA2, AO, (gm1GC/C) ** order)
#
# f, axarr = plt.subplots(2, 2)
# axarr[1,1].set_title("Simulated SINADR")
# for sim in simulations:
#     axarr[1,1].semilogx(data[sim][:,0], -10 * np.log10(np.abs(data[sim][:,1])), label=LABELS[sim])
# axarr[1,1].set_color_cycle(None)
# for sim in gainCompensated:
#     axarr[1,1].semilogx(data[sim][:,0], -10 * np.log10(np.abs(data[sim][:,1])), "--", label=LABELS[sim])
# axarr[1,1].set_ylim([60,120])
# axarr[1,1].set_xlabel("f Hz")
# axarr[1,1].set_ylabel("SINADR dB")
# axarr[1,1].legend()
#
# axarr[0,1].set_title("Noise transfer function")
# axarr[0,1].semilogx(frequency, 20 * np.log10(stfntftfOscillator[0]), label="Oscillator")
# axarr[0,1].set_color_cycle(None)
# axarr[0,1].semilogx(frequency, 20 * np.log10(stfntftfOscillatorGC[0]), "--", label="Oscillator Gain Compenstated")
# axarr[0,1].set_xlabel("f Hz")
# axarr[0,1].set_ylabel("dB")
# axarr[0,1].legend()
#
# axarr[0,0].set_title("Transfer function")
# axarr[0,0].semilogx(frequency, 20 * np.log10(stfntftfOscillator[2]), label="Oscillator")
# axarr[0,0].set_color_cycle(None)
# axarr[0,0].semilogx(frequency, 20 * np.log10(stfntftfOscillatorGC[2]), "--", label="Oscillator Gain Compenstated")
# axarr[0,0].set_xlabel("f Hz")
# axarr[0,0].set_ylabel("dB")
# axarr[0,0].legend()
#
# axarr[1,0].set_title("Signal transfer function")
# axarr[1,0].semilogx(frequency, 20 * np.log10(stfntftfOscillator[1]), label="Oscillator")
# axarr[1,0].set_color_cycle(None)
# axarr[1,0].semilogx(frequency, 20 * np.log10(stfntftfOscillatorGC[1]), "--", label="Oscillator Gain Compenstated")
# axarr[1,0].set_xlabel("f Hz")
# axarr[1,0].set_ylabel("dB")
# axarr[1,0].legend()
#
#
# ####### Figuer 3
#
# gm1 = 1. / 16e3 # 1/16kOhm
# C   = 1e-8 # 10nF
# order = 4
# order2 = 2
# wp = 2 * np.pi * 500
# gm2 = (wp * C) ** 2 / gm1 / 4
#
# AI = np.eye(order, k=-1) * gm1 / C
# AI2 = np.eye(order2, k=-1) * gm1 / C
# AO = AI2 - np.eye(order2, k=1) * gm2 / C
#
# stfntftfIntegrator = generateSTFNTFANDTF(frequency, [eta24], AI, (gm1/C) ** order)
# stfntftfOscillator = generateSTFNTFANDTF(frequency, ETA24, AO, (gm1/C) ** (order2))
#
# f, axarr = plt.subplots(2, 2)
# axarr[1,1].set_title("Simulated SINADR")
# for sim in numberComparisson:
#     axarr[1,1].semilogx(data[sim][:,0], -10 * np.log10(np.abs(data[sim][:,1])), label=LABELS[sim])
# axarr[1,1].set_ylim([20,100])
# axarr[1,1].set_xlabel("f Hz")
# axarr[1,1].set_ylabel("SINADR dB")
# axarr[1,1].legend()
#
# axarr[0,1].set_title("Noise transfer function")
# axarr[0,1].semilogx(frequency, 20 * np.log10(stfntftfOscillator[0]), label="Oscillator")
# axarr[0,1].semilogx(frequency, 20 * np.log10(stfntftfIntegrator[0]), label="Integrator")
# axarr[0,1].set_xlabel("f Hz")
# axarr[0,1].set_ylabel("dB")
# axarr[0,1].legend()
#
# axarr[0,0].set_title("Transfer function")
# axarr[0,0].semilogx(frequency, 20 * np.log10(stfntftfOscillator[2]), label="Oscillator")
# axarr[0,0].semilogx(frequency, 20 * np.log10(stfntftfIntegrator[2]), label="Integrator")
# axarr[0,0].set_xlabel("f Hz")
# axarr[0,0].set_ylabel("dB")
# axarr[0,0].legend()
#
# axarr[1,0].set_title("Signal transfer function")
# axarr[1,0].semilogx(frequency, 20 * np.log10(stfntftfOscillator[1]), label="Oscillator")
# axarr[1,0].semilogx(frequency, 20 * np.log10(stfntftfIntegrator[1]), label="Integrator")
# axarr[1,0].set_xlabel("f Hz")
# axarr[1,0].set_ylabel("dB")
# axarr[1,0].legend()
#
#
# # plt.figure()
# plt.title("2 stage Oscillator vs 4 stage integrator")
plt.show()
