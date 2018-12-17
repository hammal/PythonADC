import numpy as np
import bitarray
import re

frequencyPattern = re.compile(r".*Frequency: (\d*\.?\d*)")
amplitudePattern = re.compile(r".*Amplitude: (\d*\.?\d*)")
TsPattern = re.compile(r".*Ts = (\d+)uS")

def scrapHeader(file):
    f = None
    a = None
    Ts = None
    x = file.tell()
    line = file.readline()
    while(line[0] == "#"):
        x = file.tell()
        match = frequencyPattern.match(line)
        # print(line)
        if match:
            # print(match)
            f = np.float(match.group(1))

        match = amplitudePattern.match(line)
        if match:
            # print(match)
            a = np.float(match.group(1))

        match = TsPattern.match(line)
        if match:
            # print(match)
            Ts = np.float(match.group(1)) * 1e-6

        line = file.readline()
    file.seek(x)
    return a, f, Ts

def ascii_to_bitarray(file_name, order=5):
    bitstring = bitarray.bitarray()
    with open(file_name,'rb') as file:
        a, f, Ts = scrapHeader(file)
        bitstring.fromfile(file)
    size = int(len(bitstring)/8)
    bitlist = bitstring.tolist()
    numpyArray = np.zeros((size, order))
    for index in range(size):
        start_index = index * 8
        numpyArray[index, :] = bitlist[start_index + 8 - order: start_index + 8]
        # Reverse order of bits
        numpyArray[index, :] = numpyArray[index, :][::-1] * 2 - 1
    # print(numpyArray)
    return numpyArray, a, f, Ts

def acsii_to_controls(file_name, control):
    bitstring = bitarray.bitarray()
    file = open(file_name)
    bitstring.fromfile(file)

    length = int(bitstring.length())
    for index in range(length/8):
        start_index = index * 8
        bitstringlist = bitstring.tolist()
        mean = np.array(bitstringlist[start_index+3:start_index+8], dtype=float)[::-1].reshape((5,1))
        control.decide_control_bit_hardware(mean, index)
    return bitstring



def acsii_to_controls2(file_name, controls):
    bitstring = bitarray.bitarray()
    file = open(file_name)
    bitstring.fromfile(file)

    length = int(bitstring.length())
    for index in range(length/8):
        start_index = index * 8
        bitstringlist = bitstring.tolist()
        mean = np.array(bitstringlist[start_index+3:start_index+8], dtype=float)[::-1].reshape((5,1))
        for index2, control in enumerate(controls):
            #print mean
            control.decide_control_bit_hardware(mean[0:index2+1].reshape((index2+1,1)), index)
    return bitstring


def Offline_Matrix(Beta, additional_folder=None):
    MM = {}
    af = ''
    if additional_folder:
        af = '%s/'%(additional_folder)
    for beta in Beta:
        NN = []
        for index in range(5):
            index += 1

            Af_temp = np.matrix(np.loadtxt(open("./offline/"+af+beta+"/AfH_real%i.csv"%(index),"rb"), delimiter=",")) + 1j * np.matrix(np.loadtxt(open("./offline/"+af+beta+"/AfH_imag%i.csv"%(index),"rb"), delimiter=","))

            Bf_temp = np.matrix(np.loadtxt(open("./offline/"+af+beta+"/BfH_real%i.csv"%(index),"r"), delimiter=",")) + 1j * np.matrix(np.loadtxt(open("./offline/"+af+beta+"/BfH_imag%i.csv"%(index),"r"), delimiter=","))

            Ab_temp = np.matrix(np.loadtxt(open("./offline/"+af+beta+"/AbH_real%i.csv"%(index),"rb"), delimiter=",")) + 1j * np.matrix(np.loadtxt(open("./offline/"+af+beta+"/AbH_imag%i.csv"%(index),"rb"), delimiter=","))

            Bb_temp = np.matrix(np.loadtxt(open("./offline/"+af+beta+"/BbH_real%i.csv"%(index),"r"), delimiter=",")) + 1j * np.matrix(np.loadtxt(open("./offline/"+af+beta+"/BbH_imag%i.csv"%(index),"r"), delimiter=","))

            w_temp = np.loadtxt(open("./offline/"+af+beta+"/wH%i.csv"%(index), "r"), delimiter=",")

            Dic = {}
            Dic['Af'] = Af_temp
            Dic['Bf'] = Bf_temp
            Dic['Ab'] = Ab_temp
            Dic['Bb'] = Bb_temp
            Dic['w'] = w_temp

            NN.append(Dic)

        MM[beta] = NN

    return MM
