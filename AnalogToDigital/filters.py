"""
This file implements various standard filters and useful utilitiy functions
"""
import numpy as np
from scipy import signal
from .topologiGenerator import SystemTransformations

class TransferFunction(object):
    """
    The transfer function object is a wrapper function for creating classical
    filters according to specifications
    """

    def __init__(self):
        self.b = np.array([])
        self.a = np.array([])

    def fixNumeratorDenominator(self):
        bnew = self.b
        if self.b.size < self.a.size:
            bnew = np.zeros_like(self.a)
            for index in range(self.b.size):
                bnew[-index - 1] = self.b[-index - 1]
        self.b = bnew



    def butterWorth(self, order, criticalFrequencies, type='lowpass'):
        """
        Wrapper for scipy butterworth filter design.
        """
        self.b, self.a = signal.iirfilter(order, criticalFrequencies, btype=type, analog=True, rs=60, ftype='butter')
        # Fix such that a and b are always of the same length.
        self.fixNumeratorDenominator()

    def iirFilter(self, order, criticalFrequencies, filterClass, filterType="lowpass", rippleStopBand=60, ripplePassBand=1 ):
        """
        Filterclasses:
        - butter
        - cheby1
        - cheby2
        - ellip
        - bessel
        """
        self.b, self.a = signal.iirfilter(order, criticalFrequencies, btype=filterType, analog=True, rp=ripplePassBand, rs=rippleStopBand, ftype=filterClass)
        self.fixNumeratorDenominator()




    def frequencyResponse(self):
        """
        frequencyResponse of transfer function.
        Returns w, h which is frequency range in rad/s and frequency response h
        """
        w, h = signal.freqs(self.b, self.a, 1000)
        return w, h


class Filter(object):
    """
    The filter class creates linear state space models from given tfs
    """

    def __str__(self):
        return "A = \n%s\nB = \n%s\nC = \n%s\n" % (self.A, self.b, self.c)

    def __init__(self, A=np.array([]), b=np.array([]), c=np.array([])):
        self.A = A
        self.b = b
        self.c = c
        self.order = 0

    def tf2lssControllableForm(self, b, a):
        """
        Convert transferfunction to Linear state space model (controllable form)
        """
        order = a.size - 1
        A = np.zeros((order, order))
        # normalize the first coeffiecent
        if a[0] != 1:
            a /= a[0]

        A[0, :] = - a[1:]
        A[1:, 0:order -1] = np.eye(order - 1)
        B = np.zeros((order, 1))
        B[0] = 1.
        if b[0] != 0:
            b = b/b[0]


        C = b[1:].reshape((order, 1))

        self.A = A
        self.b = B
        self.c = C
        self.order = A.shape[1]

    def tf2lssObservableForm(self, b, a):
        """
        Convert transferfunction to Linear state space model (controllable form)
        """
        order = a.size - 1
        A = np.zeros((order, order))
        # normalize the first coeffiecent
        if a[0] != 1:
            a /= a[0]

        A[:, 0] = - a[1:]
        A[:-1, 1:] = np.eye(order - 1)
        C = np.zeros((order,1))
        C[0] = 1.
        if b[0] != 0:
            b = b/b[0]


        B = b[1:].reshape((order, 1))

        self.A = A
        self.b = B
        self.c = C
        self.order = A.shape[1]
