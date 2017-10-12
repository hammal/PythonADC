""" This file contains various reconstruction implementatons """

import numpy as np
import scipy.linalg
from cvxopt import matrix, solvers
import scipy.linalg
import copy
from scipy import integrate
CHAIN_MODEL = True


def controlabillityGramian(deltaT, count, model, covariance, sigmaZ, sigmaU):
    if count > 0:
        A = model.A
        b = model.b
        c = model.c
        covariance += deltaT * (
                        np.dot(A, covariance) \
                        + np.dot(covariance, A.transpose()) \
                        - np.dot(covariance, np.dot(c, np.dot(c.transpose(), covariance))) / sigmaZ \
                        + sigmaU * np.dot(b, b.transpose())
        )
        return controlabillityGramian(deltaT, count - 1, model, covariance, sigmaZ, sigmaU)
    else:
        return covariance


class SigmaDeltaApproach(object):

    def __init__(self, filterSettings):
        self.filterSettings = filterSettings
        model = filterSettings['model']
        Ts = filterSettings['Ts']
        f3dB = filterSettings['f3dB']
        order = filterSettings['order']
        self.cl = self.cancellationLogic(model)
        self.fir = self.filterdesign(f3dB, order)

    def filterdesign(self, f3dB, order):
        return 0

    def cancellationLogic(self, model):
        """

        :param model:
        :return:
        """
        Ad = np.eye(model.order)
        # Ad = model.Ad
        # This should work but naturally singular
        # Ad = np.linalg.inv(np.eye(model.order) - model.Ad)
        Bd = model.Bd
        bd = model.bd
        bd = bd.reshape((bd.shape[0], 1))
        if CHAIN_MODEL:
            # Create constrain matrix by forcing all controls but the last to be zero and the signal path to be one.
            constraintMatrix = np.dot(Ad, np.concatenate((bd, Bd[:, :-1]), axis=1))
            constraints = np.zeros(constraintMatrix.shape[0])
            constraints[0] = 1.

            sol = np.linalg.solve(constraintMatrix, constraints)
        else:
            """
            Solve the optimization problem:
            arg min ||Ad Bd x||^2
                 x
                s.t
                (Ad bd)^T x = 1

            i.e. find the smallest least squares contribution where the signal transfer function is still constrained 1.
            """
            AB = np.dot(Ad, Bd)
            P = 2. * matrix(np.dot(AB, AB.transpose()), tc='d')
            q = matrix(np.zeros((AB.shape[0], 1)), tc='d')
            G = matrix(np.zeros((1, model.order)), tc='d')
            h = matrix(np.array([0]), tc='d')
            A = matrix(np.dot(Ad, bd).transpose(), tc='d')
            b = matrix(np.array([1]), tc='d')

            sol = solvers.qp(P, q, G, h, A, b)
            sol = np.array(sol['x'])
        return np.dot(sol.transpose(), Bd)

    def filter(self, controller):
        u = np.zeros(controller.size)
        bitStream = np.zeros_like(u)

        for index in range(controller.size):
            bitStream[index] = np.dot(self.cl, controller[:, index - 1])

        return bitStream


    def frequencyResponse(self, frequency):
        self.NTF = np.zeros_like(frequency, dtype=np.complex64)
        self.STF = np.zeros_like(frequency, dtype=np.complex64)
        model = self.filterSettings['model']
        for index, f in enumerate(frequency):
            s = np.complex(0, 2. * np.pi * f)
            systemInverse = model.systemInverse(s)
            temp = np.dot(self.cl, systemInverse)
            self.NTF[index] = np.dot(temp, np.diag(model.Bd))
            self.STF[index] = np.dot(temp, model.bd)

        return self.STF, self.NTF

class WienerFilter(object):
    """
    This is the Wiener filter which is the standard implementation of
    [On sparsity by NUV-EM, Gaussian message passing, and Kalman smoothing](http://ieeexplore.ieee.org/document/7888168/).
    Note that this method assumes steady state conditions and is therefore prune
    to initialisation effects.

    For a given model:
    >>> import defaultSystems as df
    >>> defaultSystems = df.defaultSystems()
    >>> model = defaultSystems.gmCChain(order, 1e-3, 1e-3, 1e-6)

    Initalize by:
    >>> WienerFilter(model=model, eta2=1e2, Ts=1e-1)
    """
    def __init__(self, **kwargs):
        """
        This constructor requries:
        - model _a model object from Model.py_
        - eta2 _The perfomance parameter of the filter_
        - Ts _The period sampling time_
        """
        self.filterSettings = kwargs
        self.model = kwargs['model']
        eta2 = np.array(kwargs['eta2'])
        Ts = kwargs['Tc']
        Vf, Vb = self.care(eta2, self.model)

        ## Testing changing the interpolation midpoint
        Adb = scipy.linalg.expm(-self.model.A * Ts/2.)
        Af = scipy.linalg.expm(self.model.A * Ts/2.)
        # Vf = np.dot(Af, np.dot(Vf, Af.transpose()))
        # Vb = np.dot(Adb, np.dot(Vb, Adb.transpose()))
        # Vf += self.controlabillityGramian(Ts, self.model.A, self.model.b)
        # shift = scipy.linalg.expm(self.model.A * Ts / 2.)
        shift = np.eye(self.model.order)
        # extra = Ts * (self.controlabillityGramian(Ts, self.model.A, self.model.b) - np.outer(self.model.b, self.model.b))
        extra = 0 * np.eye(self.model.order)

        if eta2.ndim < 2:
            tempAf = (self.model.A - np.dot(Vf, np.dot(self.model.c, self.model.c.transpose())) / eta2)
            tempAb = (self.model.A + np.dot(Vb, np.dot(self.model.c, self.model.c.transpose())) / eta2)
        else:
            eta2inv = np.linalg.inv(eta2)
            tempAf = (self.model.A - np.dot(Vf, np.dot(self.model.c, np.dot(eta2inv, self.model.c.transpose()))))
            tempAb = (self.model.A + np.dot(Vb, np.dot(self.model.c, np.dot(eta2inv, self.model.c.transpose()))))
        self.Af = scipy.linalg.expm(tempAf * Ts)
        self.Bf = np.dot(self.model.zeroOrderHold(tempAf, Ts), self.model.B)

        self.Ab = scipy.linalg.expm(-tempAb * Ts)
        self.Bb = -np.dot(self.model.zeroOrderHold(-tempAb, Ts), self.model.B)

        self.w = np.linalg.solve(Vf + Vb + extra, np.dot(shift, self.model.b))

    def care(self, eta2, model):
        """
        This function solves the forward and backward continuous Riccati equation.
        """
        Vf = scipy.linalg.solve_continuous_are(model.A.transpose(), model.c,
                                               np.outer(model.b, model.b), np.eye(model.c.shape[1]) * eta2)
        Vb = scipy.linalg.solve_continuous_are(-model.A.transpose(), model.c,
                                               np.outer(model.b, model.b), np.eye(model.c.shape[1]) * eta2)
        return Vf, Vb

    def filter(self, controller):
        """
        This is the acctual filter operation. The controller needs to be a
        Controller class instance from system.py.
        """

        u = np.zeros(controller.size)
        mf = np.zeros((self.model.b.size, controller.size))
        mb = np.zeros_like(mf)
        for index in range(1, controller.size):
            mf[:, index] = np.dot(self.Af, mf[:, index - 1]) + np.dot(self.Bf, controller[:, index - 1])
        for index in range(controller.size - 2, 1, -1):
            mb[:, index] = np.dot(self.Ab, mb[:, index + 1]) + np.dot(self.Bb, controller[:, index])
            u[index + 0] = np.dot(self.w.transpose(), mb[:, index] - mf[:, index])
        return u


    # def frequencyResponse(self, frequencies):
    #     """
    #     This function computes the signal and noise transfer function under steady
    #     state assumptions.
    #     """
    #     self.NTF = np.zeros_like(frequencies, dtype=np.complex64)
    #     self.STF = np.zeros_like(frequencies)
    #     model = self.filterSettings['model']
    #     eta2 = self.filterSettings['eta2']
    #     for index,f in enumerate(frequencies):
    #         G = model.frequncyResponse(f)
    #         den = np.abs(G) ** 2 + eta2
    #         self.NTF[index] = np.conj(G) / den
    #         self.STF[index] = np.abs(G) ** 2 / den
    #
    #     return self.STF, self.NTF

    def frequencyResponse(self, frequencies):
        """
        This function computes the signal and noise transfer function under steady
        state assumptions.
        """
        model = self.filterSettings['model']
        eta2 = self.filterSettings['eta2']
        self.NTF = np.zeros((frequencies.size, model.c.shape[1]), dtype=np.complex64)
        self.STF = np.zeros(frequencies.size, dtype=np.complex64)
        for index,f in enumerate(frequencies):
            G = model.frequncyResponse(f)
            den = np.linalg.pinv(np.outer(G,np.transpose(np.conj(G))) + eta2)
            self.NTF[index, :] = np.dot(np.transpose(np.conj(G)),den).flatten()
            self.STF[index] = np.dot(self.NTF[index, :], G)

        return self.STF, self.NTF

class WienerFilterWithObservations(WienerFilter):
    """ This is an attempt on integrating the known
    boundary and quantasation information at the driscrete updates.
    """

    def __init__(self, **kwargs):
        """
        This constructor requries:
        - model _a model object from Model.py_
        - eta2 _The perfomance parameter of the filter_
        - Ts _The period sampling time_
        """
        self.filterSettings = kwargs
        self.model = kwargs['model']
        eta2 = kwargs['eta2']
        Ts = kwargs['Ts']
        Vf, Vb = self.care(eta2, self.model)

        Gf = np.linalg.inv(eta2 + np.dot(self.model.c.transpose(), np.dot(Vf, self.model.c)))
        Gb = np.linalg.inv(eta2 + np.dot(self.model.c.transpose(), np.dot(Vb, self.model.c)))
        self.Ff = np.eye(self.model.order) - np.dot(Vf, np.dot(self.model.c, np.dot(Gf, self.model.c.transpose())))
        self.Fb = np.eye(self.model.order) - np.dot(Vb, np.dot(self.model.c, np.dot(Gb, self.model.c.transpose())))


        bound = self.model.A[1, 0] * self.filterSettings["Ts"] / 2.
        bound = 0.


        self.Yf = np.dot(np.dot(Vf, np.dot(self.model.c, Gf)), np.dot(self.model.c.transpose(), bound * np.eye(self.model.order)))
        self.Yb = np.dot(np.dot(Vb, np.dot(self.model.c, Gb)), np.dot(self.model.c.transpose(), bound * np.eye(self.model.order)))


        ## Testing changing the interpolation midpoint
        Adb = scipy.linalg.expm(-self.model.A * Ts/2.)
        Af = scipy.linalg.expm(self.model.A * Ts/2.)
        # Vf = np.dot(Af, np.dot(Vf, Af.transpose()))
        # Vb = np.dot(Adb, np.dot(Vb, Adb.transpose()))
        # Vf += self.controlabillityGramian(Ts, self.model.A, self.model.b)
        shift = scipy.linalg.expm(self.model.A * Ts / 2.)
        # shift = np.eye(self.model.order)
        # extra = Ts * (self.controlabillityGramian(Ts, self.model.A, self.model.b) - np.outer(self.model.b, self.model.b))
        extra = np.eye(self.model.order)

        tempAf = (self.model.A - np.dot(Vf, np.dot(self.model.c, self.model.c.transpose())) / eta2)
        tempAb = (self.model.A + np.dot(Vb, np.dot(self.model.c, self.model.c.transpose())) / eta2)
        self.Af = scipy.linalg.expm(tempAf * Ts)
        self.Bf = np.dot(self.model.zeroOrderHold(tempAf, Ts), self.model.B)

        self.Ab = scipy.linalg.expm(-tempAb * Ts)
        self.Bb = -np.dot(self.model.zeroOrderHold(-tempAb, Ts), self.model.B)

        self.w = np.linalg.solve(Vf + Vb + extra, np.dot(shift, self.model.b))


    def filter(self, controller):
        """
        This is the acctual filter operation. The controller needs to be a
        Controller class instance from system.py.
        """

        u = np.zeros(controller.size)
        mf = np.zeros((self.model.b.size, controller.size))
        mb = np.zeros_like(mf)
        for index in range(1, controller.size):
            mf[:, index] = np.dot(self.Af, mf[:, index - 1]) + np.dot(self.Bf, controller[:, index - 1])
            mf[:, index] = np.dot(self.Ff, mf[:, index]) + np.dot(self.Yf, controller[:, index])
        for index in range(controller.size - 2, 1, -1):
            mb[:, index] = np.dot(self.Ab, mb[:, index + 1]) + np.dot(self.Bb, controller[:, index])
            mb[:, index] = np.dot(self.Fb, mb[:, index + 1]) + np.dot(self.Yb, controller[:, index])
            u[index + 0] = np.dot(self.w.transpose(), mb[:, index] - mf[:, index])
        return u



class KalmanFilter(object):
    """
    This is the Kalman filter where additionally to the Wiener filter,
    the Covariance matrices are tracked.
    """
    def __init__(self, **kwargs):
        """
        The constructor requires the following arguments:
        - model _a model object from Model.py_
        - eta2 _The perfomance parameter of the filter_
        - Ts _The period sampling time_

        Additionally, these can be supplied
        - Vf inital forward covariance
        - Vb intal backward covariance
        - mf inital forward mean vector
        - mb inital backward mean vector
        """

        self.model = kwargs['model']
        self.eta2 = kwargs['eta2']
        self.Ts = kwargs['Ts']

        # Import Vf
        if "Vf" in kwargs:
            self.Vf = [kwargs["Vf"]]
        else:
            self.Vf = [np.zeros_like(self.model.A)]

        # Import Vb
        if "Vb" in kwargs:
            self.Vb = [kwargs["Vb"]]
        else:
            self.Vb = [np.zeros_like(self.model.A)]

        # Import mf
        if "mf" in kwargs:
            self.mf = [kwargs["mf"]]
        else:
            self.mf = [np.zeros_like(self.model.b)]

        # Import mb
        if "mb" in kwargs:
            self.mb = [kwargs["mb"]]
        else:
            self.mb = [np.zeros_like(self.model.b)]

    def initalizeModel(self, model):

        if model.continuous:
            self.model.discretize_full(self.Ts)
            self.Af = model.Ad
            self.Ab = model.Adb
            self.bf = model.bd
            self.bb = model.bdb
            self.Bf = model.Bd
            self.Bb = model.Bdb
            self.Vfb = integrate.quad(self.controlabillityGramian, 0., self.Ts, args=(self.model.A, self.model.b))
            self.Vbb = integrate.quad(self.controlabillityGramian, 0., self.Ts, args=(-self.model.A, self.model.b))
        else:
            self.Af = model.A
            self.Ab = np.linalg.inv(model.A)
            self.bf = model.b
            self.bb = None
            self.Bf = model.B
            self.Bb = None


    def filter(self, controller):
        """
        This is the acctual filter operation. The controller needs to be a
        Controller class instance from system.py.
        """
        u = np.zeros(controller.size)


        for index in range(1, controller.size):
            Vftemp = np.dot(self.model.Ad, np.dot(Vf[index - 1], self.model.Ad.transpose())) + self.Ts * np.outer(self.model.b, self.model.b)
            F = np.eye(self.model.order) - np.dot(Vftemp, np.dot(self.model.c, np.dot( np.linalg.inv( np.eye(self.model.c.shape[1]) * self.eta2 / self.Ts + np.dot(self.model.c.transpose(), np.dot(Vftemp, self.model.c))), self.model.c.transpose())))
            Vf[index] = np.dot(F, Vftemp)
            mf[index] = np.dot(F, np.dot(self.model.Af, mf[index - 1]) + np.dot(self.Bf, controller[:, index - 1]))
        for index in range(controller.size - 2, 1, -1):
            # Vbtemp =
            mb[:, index] = np.dot(self.Ab, mb[:, index + 1]) + np.dot(self.Bb, controller[:, index])
            u[index + 0] = np.dot(self.w.transpose(), mb[:, index] - mf[:, index])
        return u




from .simulator import autoControlSimulator
from .system import Controller

class WienerFilterAutomaticSystem(WienerFilter):
    def __init__(self, filterSettings):
        self.filterSettings = filterSettings
        self.eta2 = filterSettings['eta2']
        self.Ts = filterSettings['Ts']
        self.model = filterSettings['model']
        self.autoSystem = autoControlSimulator(self.model, 1./self.Ts)

    def preCompute(self, states, Vf, Vb, size):
        Vff = [Vf] * size
        Vbb = [Vb] * size
        for index in range(size):
            model = self.autoSystem.setSystem(states[index, :])
            expAT = scipy.linalg.expm(model.A * self.Ts)
            Vf = np.dot(expAT, np.dot(Vf, expAT.transpose())) + self.Ts * np.outer(model.b, model.b)
            Vf = np.dot(Vf, np.eye(model.order) - np.dot(model.c, np.dot( np.linalg.inv( np.eye(model.c.shape[1]) * self.eta2 / self.Ts + np.dot(model.c.transpose(), np.dot(Vf, model.c))), np.dot(model.c.transpose(), Vf))))
            # Vf = Vf + self.Ts * (np.dot(model.A, Vf) + np.dot(Vf, model.A.transpose()) + np.outer(model.b, model.b) - 1./self.eta2 * np.dot(Vf, np.dot(model.c, np.dot(model.c.transpose(), Vf))))
            Vb = Vb - self.Ts * (np.dot(model.A, Vb) + np.dot(Vb, model.A.transpose()) - np.outer(model.b, model.b) + 1./self.eta2 * np.dot(Vb, np.dot(model.c, np.dot(model.c.transpose(), Vb))))
            tempAf = (model.A - np.dot(Vf, np.dot(model.c, model.c.transpose())) / self.eta2)
            # print("Vf = ", Vf)
            # print("Vb = ", Vb)
            # print("tempAf = ", tempAf)
            self.Af[index] = scipy.linalg.expm(tempAf * self.Ts)
            self.Bf[index] = np.dot(model.zeroOrderHold(tempAf, self.Ts), model.B)

            tempAb = (model.A + np.dot(Vb, np.dot(model.c, model.c.transpose())) / self.eta2)
            self.Ab[- (index + 1)] = scipy.linalg.expm(-tempAb * self.Ts)
            self.Bb[-(index + 1)] = -np.dot(model.zeroOrderHold(-tempAb, self.Ts), model.B)

            Vff[index] = Vf
            Vbb[index] = Vb

        for index in range(size):
            try:
                self.w[index] = np.linalg.solve(Vff[index] + Vbb[index], model.b).reshape(model.b.shape)
            except np.linalg.linalg.LinAlgError:
                self.w[index] = np.zeros_like(model.b)


    def computeSystemFilter(self, model):
        Vf, Vb = self.care(self.eta2, self.model)
        tempAf = (model.A - np.dot(Vf, np.outer(model.c, model.c)) / self.eta2)
        self.Af = scipy.linalg.expm(tempAf * self.Ts)
        self.Bf = np.dot(model.zeroOrderHold(tempAf, self.Ts), model.B)
        tempAb = (model.A + np.dot(Vb, np.outer(model.c, model.c)) / self.eta2)
        self.Ab = scipy.linalg.expm(-tempAb * self.Ts)
        self.Bb = -np.dot(model.zeroOrderHold(-tempAb, self.Ts), model.B)

        self.w = np.linalg.solve(Vf + Vb, model.b).reshape((self.Af.shape[0], 1))


    def filter(self, states):
        self.w = [np.zeros_like(self.model.b)] * states.shape[0]
        self.Af = [np.zeros_like(self.model.A)] * states.shape[0]
        self.Bf = [np.zeros_like(self.model.B)] * states.shape[0]
        self.Ab = [np.zeros_like(self.model.A)] * states.shape[0]
        self.Bb = [np.zeros_like(self.model.B)] * states.shape[0]
        Vf = np.ones((self.model.order, self.model.order)) * 0
        Vb = np.ones_like(Vf) * 0
        self.preCompute(states, Vf, Vb, states.shape[0])

        control = np.zeros_like(states.transpose())
        for index in range(1, states.shape[1]):
            control[:, index] = self.autoSystem.computeControls(states[index, :]).flatten()
        u = np.zeros(states.shape[0])
        mf = np.zeros((self.filterSettings['model'].b.size, states.shape[0]))
        mb = np.zeros_like(mf)
        mf[:, 0] = self.autoSystem.initalState.flatten()


        for index in range(1, states.shape[0]):
            # Vf, Vb = self.computeSystemFilter(self.autoSystem.setSystem(states[index - 1, :]), Vf, Vb)
            # mf[:, index] = (np.dot(self.Af, mf[:, index - 1].reshape(self.Bf.shape)) + np.dot(self.Bf, self.autoSystem.knownSignal(index - 1))).flatten()
            mf[:, index] = (np.dot(self.Af[index - 1], mf[:, index - 1].reshape((self.model.order, 1))) + np.dot(self.Bf[index - 1], control[:, index - 1].reshape((self.model.order, 1)))).flatten()

        for index in range(states.shape[0] - 2, 1, -1):
            # Vf, Vb = self.computeSystemFilter(self.autoSystem.setSystem(states[index, :]), Vf, Vb)
            # mb[:, index] = (np.dot(self.Ab, mb[:, index + 1].reshape(self.Bb.shape)) + np.dot(self.Bb, self.autoSystem.knownSignal(index))).flatten()
            mb[:, index] = (np.dot(self.Ab[index + 1], mb[:, index + 1].reshape((self.model.order, 1))) + np.dot(self.Bb[index + 1], control[:, index].reshape((self.model.order, 1)))).flatten()
            u[index + 0] = np.dot(self.w[index].transpose(), mb[:, index] - mf[:, index])
        return u

class LeastMeanSquare(object):

    def __str__(self):
        return "The current FIR filter is %s" % self.w

    def __init__(self):
        self.w = None

    def train(self, data):
        x = data["x"]
        y = data["y"]
        self.w = np.linalg.solve(np.dot(x.transpose(), x), np.dot(x.transpose(), y))

    def predict(self, dataPoint):
        return np.dot(self.w, dataPoint)



if __name__ == "__main__":
    import doctest
    doctest.testmod()
