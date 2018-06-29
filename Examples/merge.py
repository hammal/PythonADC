import subprocess
import numpy as np
import re

def sort(dataset):
    temp = np.array(dataset)
    perm = np.argsort(temp[:, 0])
    temp[:, 0] = temp[perm, 0]
    temp[:, 1] = temp[perm, 1]
    return temp

# simulations = [
#     "Integrator",
#     "Oscillator0",
#     "Oscillator1",
#     "Oscillator2",
#     "Oscillator3",
#     "Oscillator4",
#     "OscillatorNumberComparissonRef",
#     "OscillatorNumberComparisson0",
#     "OscillatorNumberComparisson1",
#     "OscillatorNumberComparisson2",
#     "OscillatorNumberComparisson3",
#     "OscillatorNumberComparisson4",
#     "OscillatorCompensated0",
#     "OscillatorCompensated1",
#     "OscillatorCompensated2",
#     "OscillatorCompensated3",
#     "OscillatorCompensated4"
#     ]

simulations = [
    "Integrator",
    "Oscillator0",
    "Oscillator1",
    "Oscillator2",
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


types = {}
data = {}

for simType in simulations:
    types[simType] = re.compile(simType + '.[0-9]+.out')
    data[simType] = []

files = subprocess.check_output(["ls","sqe"]).decode().split("\n")

issues = 0

for filename in files:
    for simType in types.keys():
        if types[simType].match(filename):
            print(filename)
            with open("sqe/" + filename,'r') as file:
                temp = file.read().replace("\n", "").replace(" ", "").split(",")
                try:
                    data[simType].append([np.double(temp[0]), np.double(temp[1]), np.double(temp[2])])
                except:
                    issues += 1
            # os.remove("sqe/" + filename)

for simType in types.keys():
    print(simType)
    data[simType] = sort(data[simType])

print("Number of reading issues was %s" %issues)

np.savez_compressed("sgeData.npz", **data)
