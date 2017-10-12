""" This file contains various objects for describing the system """
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg

MAX_NUMBER_OF_ITERATIONS = 30

class Model(object):

    def __init__(self, A, B, b, c):
        """
        Initalize the Model.
        """
        self.A = np.array(A, dtype=np.double)
        self.B = np.array(B, dtype=np.double)
        self.b = np.array(b, dtype=np.double)
        self.c = np.array(c, dtype=np.double)

        self.T = np.eye(self.A.shape[0])
        self.Tinv = self.T

        self.order = self.A.shape[0]

        self.Ad = np.zeros_like(self.A)
        self.Bd = np.zeros_like(self.B)
        self.bd = np.zeros_like(self.b)

    def __str__(self):
        return "A = \n %s \n\nB = \n%s\n\nb = \n%s\n\nc = \n%s\n\nAd = \n%s\n\nBd = \n%s\n\n" % (self.A, self.B, self.b, self.c, self.Ad, self.Bd)

    def checkTypes(model):
        m, n = np.shape(model.A)
        model.A = model.A.reshape((m, n))
        model.B = model.B.reshape((m, n))
        model.c = model.c.reshape((m))
        model.b = model.b.reshape((n))
        return model

    def discretize(self, Ts):
        # Discretize state space matrix
        self.fs = 1./Ts
        self.Ad = scipy.linalg.expm(self.A * Ts)
        # Compute zero order hold steps.
        temp = self.zeroOrderHold(self.A, Ts)
        self.Bd = np.dot(temp, self.B)
        self.bd = np.dot(temp, self.b)

    def discretize_full(Ts):
        self.discretize(Ts)
        self.Adb = scipy.linalg.expm(-self.A * Ts)
        temp = self.zeroOrderHold(-self.A, Ts)
        self.Bdb = -np.dot(temp, self.B)
        self.bdb = -np.dot(temp, self.b)

    def zeroOrderHold(self, M, Ts):
        Md = scipy.linalg.expm(M * Ts)
        try:
            temp = np.dot(np.linalg.inv(M), (Md - np.eye(M.shape[0])))
        except np.linalg.LinAlgError:
            # print("this")
            temp = np.eye(M.shape[0]) * Ts
            tempA = M
            index = 2
            while tempA.any() and index < MAX_NUMBER_OF_ITERATIONS:
                temp += tempA * (Ts ** index) / np.float(index - 1.)
                tempA = np.dot(M, tempA)
                index += 1
        if np.isnan(temp).any():
            raise "Can't compute zero order hold matrix"
        return temp

    def frequncyResponse(self, frequency):
        s = np.complex(0, 2. * np.pi * frequency)
        return np.dot(self.c.transpose(), np.dot(self.systemInverse(s), self.b))

    def systemInverse(self, s):
        return np.linalg.inv(s * np.eye(self.order) - self.A)

    def discreteTimeFrequencyResponse(self, frequency):
        z = np.exp(np.complex(0, 2. * np.pi * frequency / self.fs))
        systemInverse = z * np.eye(self.order) - self.Ad
        return np.dot(self.c, np.dot(np.linalg.pinv(systemInverse), self.bd))

    def discreteImpulseResponse(self, numberOfPoints):
        state = self.bd
        impulseResponse = np.zeros(numberOfPoints)
        for index in range(numberOfPoints):
            impulseResponse[index] = np.dot(self.c, state)
            state = np.dot(self.Ad, state)
        return impulseResponse

    def plotImpulseResponse(self, numberOfPoints=100000):
        fig = plt.figure()
        ax1 = fig.add_subplot(1, 1, 1)
        T = np.arange(numberOfPoints) / self.fs
        h = self.discreteImpulseResponse(numberOfPoints)
        ax1.plot(T, h)
        plt.show()

    def plotFrequencyResponse(self, numberOfPoints=1000):
        fig = plt.figure()
        ax1 = fig.add_subplot(4, 1, 1)
        ax2 = fig.add_subplot(4, 1, 2)
        ax3 = fig.add_subplot(4, 1, 3)
        ax4 = fig.add_subplot(4, 1, 4)

        fresp = np.zeros(numberOfPoints, dtype=np.complex64)
        f = np.logspace(-2, np.log10(self.fs), num=numberOfPoints)
        dfresp = np.zeros_like(fresp)
        for point in range(numberOfPoints):
            fresp[point] = self.frequncyResponse(f[point])
            dfresp[point] = self.discreteTimeFrequencyResponse(f[point])
        # print(np.abs(fresp))
        ax1.semilogx(f, 20 * np.log10(np.abs(fresp)))
        ax2.semilogx(f, np.angle(fresp, deg=True))
        ax3.semilogx(f, 20 * np.log10(np.abs(dfresp)))
        ax4.semilogx(f, np.angle(dfresp, deg=True))
        plt.show()

    def filter(self, signal):
        state = np.zeros((self.A.shape[0], 1))
        y = np.zeros_like(signal)
        for index, value in enumerate(signal):
            state = np.dot(self.Ad, state) + np.dot(self.bd, value)
            y[index] = np.dot(self.c.transpose(), state)
        return y

class Controller(object):
    def __init__(self, model, fs, fc, size):
        self.model = model
        if fs >= fc and float(fs / fc).is_integer():
            self.osr = int(fs / fc)
        else:
            print("The sampling rate fc has to be a multiple of fs.")
            raise
        self.control_sequence = np.zeros((self.model.B.shape[0], size), dtype=np.int8)
        self.size = size

    def subsample(self):
        """Subsamble the controls"""
        self.control_sequence = self.control_sequence[:,::self.osr]
        self.size = int(self.size/self.osr)

    def computeControls(self, state):
        """return a {1, -1} array depending on state > 0"""
        return np.array((state > 0) * 2 - 1, dtype=np.int8)


    def controlUpdate(self, index):
        """
        Returns True if the index is at an oversampling instance
        :param index:
        :return:
        """
        return index % self.osr == 0

    def __getitem__(self, item):
        return self.control_sequence[item]

    def __setitem__(self, key, value):
        self.control_sequence[key] = value
