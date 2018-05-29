""" This file contains various objects for describing the system """
import numpy as np
import matplotlib.pyplot as plt

class Input(object):
    """
    This is the default input class which is a PAM type encoding with zero order
    hold filter
    """

    def __init__(self, Ts, coefficients, steeringVector, name="Zero Order Hold"):
        self.Ts = Ts
        self.u = coefficients
        self.steeringVector = steeringVector
        self.name = name

    def fun(self, t):
        return self.steeringVector * self.scalarFunction(t)


    def c(self, t):
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
    A pure sinussodial is only parametericed by amplitude frequency and phase.
    """
    def __init__(self, Ts, amplitude, frequency, phase, steeringVector, name="Sinusodial"):
        self.Ts = Ts
        self.amplitude = amplitude
        self.frequency = frequency
        self.phase = phase
        self.steeringVector = steeringVector
        self.name = name

    def fun(self, t):
        return self.steeringVector * self.amplitude * np.sin(2. * np.pi * self.frequency * t + self.phase)

    def scalarFunction(self, t):
        return self.amplitude * np.sin(2. * np.pi * self.frequency * t + self.phase)

class System(object):

    def __init__(self, A, c):
        """
        Initalize the Model.
        """
        self.A = np.array(A, dtype=np.double)
        self.c = np.array(c, dtype=np.double)

        self.order = self.A.shape[0]
        self.outputOrder = self.c.shape[0]

    def __str__(self):
        return "A = \n %s \n\nc = \n%s\n" % (self.A, self.c)

    def frequencyResponse(self, frequency):
        s = np.complex(0, 2. * np.pi * frequency)
        return np.linalg.inv(s * np.eye(self.order) - self.A)

    def output(self, state):
        """
        Computes the output from a given stateself.

        Here Quantization could be implemented!
        """
        return np.dot(self.c.transpose(), state).flatten()

class Control(object):
    """
    Analog Switch Control is the default control behaviour.

    It keeps the:
    - Mixingmatrix
    - The control desisions
    - The type
    which are all needed for reconstruction
    """
    def __init__(self, mixingMatrix, size):
        self.type = 'analog switch'
        # The internal memory
        self.mixingMatrix = mixingMatrix
        self.memory = np.zeros((size, mixingMatrix.shape[1]), dtype=np.int8)
        self.size = size
        self.memory_Pointer = 0

    def __getitem__(self, item):
        """
        this function is for retriving control descisions from memory at index k
        as:
        control[k]
        """
        return self.memory[item]

    def update(self, state):
        """
        This is a function that sets the next control descisions
        """
        self.memory[self.memory_Pointer, :] = (state > 0).flatten() * 2 - 1
        self.memory_Pointer += 1

    def fun(self, t):
        """
        This is the control function evaluated at time t
        """
        return np.dot(self.mixingMatrix, self.memory[self.memory_Pointer - 1, :].reshape((self.mixingMatrix.shape[1],1))).flatten()


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
