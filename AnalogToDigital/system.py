""" This file contains various objects for describing the system """
import numpy as np
import matplotlib.pyplot as plt
import sys

if sys.version_info >= (3,0):
    import pickle as cPickle
else:
    import cPickle

class Input(object):
    """
    This is the default input class which is a PAM type encoding with zero order
    hold filter
    """

    def __str__(self):
        return "Steering vector = %s, scalar function f(1.25) %s" % (self.steeringVector, self.scalarFunction(1.25))

    def __init__(self, Ts, coefficients, steeringVector, name="Zero Order Hold"):
        self.Ts = Ts
        self.u = coefficients
        self.steeringVector = steeringVector
        self.name = name

    def fun(self, t):
        return self.steeringVector * self.scalarFunction(t)


    def scalarFunction(self, t):
        """
        This is the main function used by the simulation tool to determine the
        input signal at some input time t.
        """
        # Determine which time coefficient to apply
        index = np.array(t/self.Ts, dtype=np.int)
        # return the input
        return self.u[index]

class FirstOrderHold(Input):

    def triangle(self, t):
        return np.maximum(0.,1. - np.abs(t))

    def fun(self, t):
        return self.scalarFunction(t) * self.steeringVector

    def scalarFunction(self, t):
        # Determine which time coefficient to apply
        index = np.array(t/self.Ts, dtype=np.int)
        res = self.u[index] * self.triangle((t - index * self.Ts)/self.Ts)
        # Check such that left index is defined
        res += self.u[index - 1] * self.triangle((t - (index - 1) * self.Ts)/self.Ts)
        res += self.u[(index + 1) % self.u.size] * self.triangle((t - (index + 1) * self.Ts)/self.Ts)
        return res

class Sin(Input):
    """
    A pure sinusodial is only parametericed by amplitude frequency and phase.
    """
    def __init__(self, Ts, amplitude, frequency, phase, steeringVector, name="Sinusodial", offset = 0.):
        self.Ts = Ts
        self.amplitude = amplitude
        self.frequency = frequency
        self.phase = phase
        self.offset = offset
        self.steeringVector = steeringVector
        self.name = name

    def fun(self, t):
        return self.steeringVector * self.scalarFunction(t) 

    def scalarFunction(self, t):
        return self.amplitude * np.sin(2. * np.pi * self.frequency * t + self.phase) + self.offset


class Noise(Input):
    """
    Gaussian white noise
    """
    def __init__(self, standardDeviation, steeringVector, name="Noise"):
        self.std = standardDeviation
        self.steeringVector = steeringVector
        self.name = name

    def fun(self, t):
        return self.steeringVector * self.scalarFunction(t)

    def scalarFunction(self, t):
        return  self.std * np.random.randn(t.size)

class System(object):

    def __init__(self, A, c):
        """
        Initialize the Model.
        """
        self.A = np.array(A, dtype=np.float64)
        self.c = np.array(c, dtype=np.float64)
        # self.b = np.zeros(self.A.shape[0])
        # self.b[0] = 1.

        self.order = self.A.shape[0]
        self.outputOrder = self.c.shape[0]

    def __str__(self):
        return "A = \n %s \n\nc = \n%s\n" % (self.A, self.c)

    def frequencyResponse(self, frequency):
        s = np.complex(0, 2. * np.pi * frequency)
        try:
            return np.dot(self.c, np.linalg.inv(s * np.eye(self.order) - self.A))
        except np.linalg.LinAlgError:
            return np.dot(self.c, np.linalg.pinv(s * np.eye(self.order) - self.A))

    def output(self, state):
        """
        Computes the output from a given stateself.

        Here Quantization could be implemented!
        """
        return np.dot(self.c.transpose(), state).flatten()

class Control(object):
    """
    Analog Switch Control is the default control behavior.

    It keeps the:
    - Mixingmatrix
    - The control decisions
    - The type
    which are all needed for reconstruction
    """
    def __init__(self, mixingMatrix, size, memory=np.array([]), options = {}):
        self.type = 'analog switch'
        # The internal memory
        self.mixingMatrix = mixingMatrix
        self.leftpadd = 8 - (mixingMatrix.shape[0] )
        self.leftpadding = np.zeros(self.leftpadd, dtype=np.int)
 
        # Check if passed memory
        if memory.shape[0] == size:
            self.memory = memory
        else:
            self.memory = np.zeros((size, mixingMatrix.shape[1]), dtype=np.int8)
        self.size = size
        self.memory_Pointer = 0

        if 'scalingSequence' in options:
            print("Scaling Sequence Active")
            self.scalingSequence = options['scalingSequence']
        else:
            self.scalingSequence = None

        if 'references' in options:
            print("Using references: %s" % options['references'])
            self.references = options['references']
        else:
            self.references = np.zeros(mixingMatrix.shape[1])

        if 'offsets' in options:
            print("Using offsets: %s" % options['offsets'])
            self.offsets = options['offsets']
        else:
            self.offsets = np.zeros(mixingMatrix.shape[1])
        
        if 'name' in options:
            print("Using name: %s, for storage" % options['name'])
            self.name = options['name']
        else:
            self.name = 'default'

    def __getitem__(self, item):
        """
        this function is for retriving control decisions from memory at index k
        as:
        control[k]
        """
        if self.scalingSequence:
            return np.dot(self.scalingSequence(item), self.memory[item])

        return self.memory[item]

    def getIndex(self, item):
        """
        Transform output to unique index
        """
        return self.packbits(np.array((self.memory[item] + 1) / 2, dtype=np.int))

    def packbits(self, bitarray):
        """
        unpacks the bitarray into its integer representation
        """
        # print(np.concatenate((self.leftpadding, bitarray.flatten())))
        return np.packbits(np.concatenate((self.leftpadding, bitarray.flatten())))[0]

    def unpackbits(self, value):
        """
        packs the value into an bitarray
        """
        return np.unpackbits(np.array([value], dtype=np.uint8))[self.leftpadd:]

    def update(self, state):
        """
        This is a function that sets the next control decisions
        """
        self.memory[self.memory_Pointer, :] = (state > self.references).flatten() * 2 - 1
        self.memory_Pointer += 1

    def fun(self, t):
        """
        This is the control function evaluated at time t
        """
        return np.dot(self.mixingMatrix, self[self.memory_Pointer - 1].reshape((self.mixingMatrix.shape[1],1))).flatten()

    def save(self):
        """save class as self.name.txt"""
        file = open(self.name+'.txt','wb')
        file.write(cPickle.dumps(self.__dict__))
        file.close()

    def load(self):
        """try load self.name.txt"""
        file = open(self.name+'.txt','rb')
        dataPickle = file.read()
        file.close()

        self.__dict__ = cPickle.loads(dataPickle)

# class Controller(object):
#     def __init__(self, model, fs, fc, size):
#         self.model = model
#         if fs >= fc and float(fs / fc).is_integer():
#             self.osr = int(fs / fc)
#         else:
#             print("The sampling rate fc has to be a multiple of fs.")
#             raise
#         self.control_sequence = np.zeros((self.model.B.shape[0], size), dtype=np.int8)
#         self.size = size
#
#     def subsample(self):
#         """Subsamble the controls"""
#         self.control_sequence = self.control_sequence[:,::self.osr]
#         self.size = int(self.size/self.osr)
#
#     def computeControls(self, state):
#         """return a {1, -1} array depending on state > 0"""
#         return np.array((state > 0) * 2 - 1, dtype=np.int8)
#
#
#     def controlUpdate(self, index):
#         """
#         Returns True if the index is at an oversampling instance
#         :param index:
#         :return:
#         """
#         return index % self.osr == 0
#
#     def __getitem__(self, item):
#         return self.control_sequence[item]
#
#     def __setitem__(self, key, value):
#         self.control_sequence[key] = value
