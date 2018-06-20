""" This file contains various reconstruction implementatons """

import numpy as np
import scipy.linalg
from topologiGenerator import SystemTransformations
from system import Noise
# from cvxopt import matrix, solvers
# import copy
from scipy.integrate import odeint
CHAIN_system = True

def care(A, B, Q, R):
    """
    This function solves the forward and backward continuous Riccati equation.
    """
    # print("Solve CARE for:")
    # print(A)
    # print(B)
    # print(Q)
    # print(R)
    Vf = scipy.linalg.solve_continuous_are(A, B, Q, R)
    Vb = scipy.linalg.solve_continuous_are(-A, B, Q, R)
    # A^TX + X A - X B (R)^(-1) B^T X + Q = 0
    # res1 = np.dot(A.transpose(), Vf) + np.dot(Vf, A) - np.dot(Vf, np.dot(B, np.dot(np.lina)))
    return Vf, Vb
#
#
# def controlabillityGramian(deltaT, count, system, covariance, sigmaZ, sigmaU):
#     if count > 0:
#         A = system.A
#         b = system.b
#         c = system.c
#         covariance += deltaT * (
#                         np.dot(A, covariance) \
#                         + np.dot(covariance, A.transpose()) \
#                         - np.dot(covariance, np.dot(c, np.dot(c.transpose(), covariance))) / sigmaZ \
#                         + sigmaU * np.dot(b, b.transpose())
#         )
#         return controlabillityGramian(deltaT, count - 1, system, covariance, sigmaZ, sigmaU)
#     else:
#         return covariance
#
#
# class SigmaDeltaApproach(object):
#
#     def __init__(self, filterSettings):
#         self.filterSettings = filterSettings
#         system = filterSettings['system']
#         Ts = filterSettings['Ts']
#         f3dB = filterSettings['f3dB']
#         order = filterSettings['order']
#         self.cl = self.cancellationLogic(system)
#         self.fir = self.filterdesign(f3dB, order)
#
#     def filterdesign(self, f3dB, order):
#         return 0
#
#     def cancellationLogic(self, system):
#         """
#
#         :param system:
#         :return:
#         """
#         Ad = np.eye(system.order)
#         # Ad = system.Ad
#         # This should work but naturally singular
#         # Ad = np.linalg.inv(np.eye(system.order) - system.Ad)
#         Bd = system.Bd
#         bd = system.bd
#         bd = bd.reshape((bd.shape[0], 1))
#         if CHAIN_system:
#             # Create constrain matrix by forcing all controls but the last to be zero and the signal path to be one.
#             constraintMatrix = np.dot(Ad, np.concatenate((bd, Bd[:, :-1]), axis=1))
#             constraints = np.zeros(constraintMatrix.shape[0])
#             constraints[0] = 1.
#
#             sol = np.linalg.solve(constraintMatrix, constraints)
#         else:
#             """
#             Solve the optimization problem:
#             arg min ||Ad Bd x||^2
#                  x
#                 s.t
#                 (Ad bd)^T x = 1
#
#             i.e. find the smallest least squares contribution where the signal transfer function is still constrained 1.
#             """
#             AB = np.dot(Ad, Bd)
#             P = 2. * matrix(np.dot(AB, AB.transpose()), tc='d')
#             q = matrix(np.zeros((AB.shape[0], 1)), tc='d')
#             G = matrix(np.zeros((1, system.order)), tc='d')
#             h = matrix(np.array([0]), tc='d')
#             A = matrix(np.dot(Ad, bd).transpose(), tc='d')
#             b = matrix(np.array([1]), tc='d')
#
#             sol = solvers.qp(P, q, G, h, A, b)
#             sol = np.array(sol['x'])
#         return np.dot(sol.transpose(), Bd)
#
#     def filter(self, controller):
#         u = np.zeros(controller.size)
#         bitStream = np.zeros_like(u)
#
#         for index in range(controller.size):
#             bitStream[index] = np.dot(self.cl, controller[:, index - 1])
#
#         return bitStream
#
#
#     def frequencyResponse(self, frequency):
#         self.NTF = np.zeros_like(frequency, dtype=np.complex64)
#         self.STF = np.zeros_like(frequency, dtype=np.complex64)
#         system = self.filterSettings['system']
#         for index, f in enumerate(frequency):
#             s = np.complex(0, 2. * np.pi * f)
#             systemInverse = system.systemInverse(s)
#             temp = np.dot(self.cl, systemInverse)
#             self.NTF[index] = np.dot(temp, np.diag(system.Bd))
#             self.STF[index] = np.dot(temp, system.bd)
#
#         return self.STF, self.NTF
#
class WienerFilter(object):
    """
    This is the Wiener filter which is the standard implementation of
    [On sparsity by NUV-EM, Gaussian message passing, and Kalman smoothing](http://ieeexplore.ieee.org/document/7888168/).
    Note that this method assumes steady state conditions and is therefore prune
    to initialisation effects.
    """
    def __init__(self, t, system, inputs, options={}):
        """
        This constructor requries:
        - t which are the times to reconstruct at (assumed to be uniformly spaced)
        - a system model
        - inputs an iterable of inputs to be estimatedself.
        - options
        """
        self.Ts = t[1] - t[0]
        self.system = system
        self.inputs = inputs
        self.order = self.system.order
        self.noise = []

        if 'eta2' in options:
            self.eta2 = options['eta2']
        else:
            self.eta2 = np.ones(self.order)

        if 'noise' in options:
            for source in options['noise']:
                self.noise.append(Noise(standardDeviation=source['std'], steeringVector=source['steeringVector'], name=source["name"]))


        if 'sigmaU2' in options:
            self.sigmaU2 = options['sigmaU2']
        else:
            self.sigmaU2 = np.ones(len(inputs))



        # Solve care
        # A^TX + X A - X B (R)^(-1) B^T X + Q = 0
        A = self.system.A.transpose()
        B = self.system.c
        Q = np.zeros((self.order, self.order))
        for index, input in enumerate(inputs):
            Q += self.sigmaU2[index] * np.outer(input.steeringVector,input.steeringVector)

        print("Q before ", Q)
        for noiseSource in self.noise:
            Q += (noiseSource.std) ** 2 * np.outer(noiseSource.steeringVector, noiseSource.steeringVector)
        print("Q after ", Q)
        if self.eta2.size > 1:
            R = np.diag(self.eta2)
        else:
            R = self.eta2.reshape((1,1))
        # print(A,B,Q,R)
        Vf, Vb = care(A, B, Q, R)
        print("Vf, Vb\n",Vf + Vb)

        if self.eta2.ndim < 2:
            self.tempAf = (self.system.A - np.dot(Vf, np.dot(self.system.c, self.system.c.transpose())) / self.eta2)
            self.tempAb = (self.system.A + np.dot(Vb, np.dot(self.system.c, self.system.c.transpose())) / self.eta2)
        else:
            eta2inv = np.linalg.inv(np.diag(self.eta2))
            self.tempAf = (self.system.A - np.dot(Vf, np.dot(self.system.c, np.dot(eta2inv, self.system.c.transpose()))))
            self.tempAb = (self.system.A + np.dot(Vb, np.dot(self.system.c, np.dot(eta2inv, self.system.c.transpose()))))

        self.Af = scipy.linalg.expm(self.tempAf * self.Ts)

        self.Ab = scipy.linalg.expm(-self.tempAb * self.Ts)

        B = np.zeros((self.order, len(inputs)))
        for index, input in enumerate(inputs):
            B[:, index] = input.steeringVector

        self.w = np.linalg.solve(Vf + Vb, B)


    def __str__(self):
        return "Af = \n%s\nBf = \n%s\nAb = \n%s\nBb = \n%s\nw = \n%s\n" % (self.Af, self.Bf, self.Ab, self.Bb, self.w)

    def computeControlTrajectories(self, control):
        if control.type == 'analog switch':
            self.Bf = np.zeros((self.order, self.order))
            self.Bb = np.zeros((self.order, self.order))
            for controlIndex in range(self.order):
                def ForwardDerivative(x, t):
                    hom = np.dot(self.tempAf, x.reshape((self.order,1))).flatten()
                    control = np.zeros(self.order)
                    control[controlIndex] = 1
                    return hom + control

                def BackwardDerivative(x, t):
                    hom = - np.dot(self.tempAb, x.reshape((self.order,1))).flatten()
                    control = np.zeros(self.order)
                    control[controlIndex] = 1
                    return hom + control

                # self.Bf = np.dot(self.system.zeroOrderHold(tempAf, Ts), self.system.B)
                self.Bf[:, controlIndex] = odeint(ForwardDerivative, np.zeros(self.order), np.array([0., self.Ts]))[-1,:]
                # self.Bb = -np.dot(self.system.zeroOrderHold(-tempAb, Ts), self.system.B)
                self.Bb[:, controlIndex] = - odeint(BackwardDerivative, np.zeros(self.order), np.array([0., self.Ts]))[-1,:]

            self.Bf = np.dot(self.Bf, self.mixingMatrix)
            self.Bb = np.dot(self.Bb, self.mixingMatrix)
        else:
            raise NotImplemented



    def filter(self, control):
        """
        This is the actual filter operation. The controller needs to be a
        Controller class instance from system.py.
        """
        # Compute Bf and Bb for this type of control
        self.mixingMatrix = control.mixingMatrix
        self.computeControlTrajectories(control)

        # Initalise memory
        u = np.zeros((control.size, len(self.inputs)))
        mf = np.zeros((self.order, control.size))
        mb = np.zeros_like(mf)

        for index in range(1, control.size):
            mf[:, index] = np.dot(self.Af, mf[:, index - 1]) + np.dot(self.Bf, control[index - 1])
        for index in range(control.size - 2, 1, -1):
            mb[:, index] = np.dot(self.Ab, mb[:, index + 1]) + np.dot(self.Bb, control[index])
            u[index] = np.dot(self.w.transpose(), mb[:, index] - mf[:, index])
        return u



class WienerFilterWithPostFiltering(WienerFilter):
    """
    This filter implements the automatic post filtering as described in Lukas
    Bruderer's thesis, Theorem 5.4. The main idea is to expand the system model
    as:

    A = [[A_system, 0],[0,A_filter]]
    B = [[B_system],[B_filter]]
    C = [[C_system],[0]]

    then instead of computing the system input

    u_hat = - B(Vf + Vb)^(-1)(mf - mb)

    we instead compute the output of the post filter which is a mapping from the
    state estimate.

    u_hat = C_filter^(T) (Wf + Wb)^(-1)(Wf mf + Wb mb)

    To achive this we additionally make the following changes:
    - Vf and Vb needs to be recomputed for the new larger system
    - Af and Ab gets reparameterised as Af_new = Wf Af Vf and Ab_new = Wf Ab Vb
    - w is computed as solve(Wf + Wb, C_filter)
    - Bf and Bb are reparameterised as Bf_new = Wf Bf and Bb_new = Wb Bb
    """

    def __init__(self, t, system, inputs, postFilteringSystem, options={}):
        """
        Additonally to the init function in Wiener Filter a postFilteringSystem
        needs to be provided. This postfilteringSystem must provide a MIMO filter
        that has the same number of inputs as the the number of inputs to the
        system.
        """


        self.Ts = t[1] - t[0]
        self.system = system
        self.inputs = inputs
        self.postFilteringSystem = postFilteringSystem

        if 'eta2' in options:
            self.eta2 = options['eta2']
        else:
            self.eta2 = np.ones(self.system.order)
            # self.eta2 = np.ones(self.system.order + self.postFilteringSystem.c.shape[1])

        if 'sigmaU2' in options:
            self.sigmaU2 = options['sigmaU2']
        else:
            self.sigmaU2 = np.ones(len(inputs))

        # First check that there is one postfilter for each input
        if len(inputs) != self.postFilteringSystem.b.shape[1]:
            print("Not the same number of postfilters and inputs")
            raise NotImplemented
        # Combine system and postfilering into one system
        Anew = scipy.linalg.block_diag(self.system.A, self.postFilteringSystem.A)

        cnew = np.vstack((self.system.c, np.zeros((self.postFilteringSystem.c.shape[0], self.system.c.shape[1]))))
        # cnew = np.hstack((cnew, np.concatenate((np.zeros(self.system.order), self.postFilteringSystem.c.flatten())).reshape((cnew.shape[0], 1))))
        # print(cnew)

        postFilterC = np.vstack((np.zeros((self.system.c.shape[0], self.postFilteringSystem.c.shape[1])), self.postFilteringSystem.c))
        # bnew = np.vstack((np.zeros((self.system.order, self.postFilteringSystem.b.shape[1])), self.postFilteringSystem.b))

        self.order = Anew.shape[1]


        # Solve care
        # A^TX + X A - X B (R)^(-1) B^T X + Q = 0
        A = Anew.transpose()
        B = cnew
        # Q = np.outer(bnew, bnew)
        bSystemTemp = np.zeros((self.system.order, len(self.inputs)))
        for index, input in enumerate(inputs):
            bSystemTemp[:, index] = input.steeringVector * np.sqrt(self.sigmaU2[index])
            # newb = np.hstack((input.steeringVector, self.postFilteringSystem.b))
            # Q += self.sigmaU2[index] * np.outer(newb, newb)
        newb = np.vstack((bSystemTemp, self.postFilteringSystem.b))
        Q = np.outer(newb, newb)
        R = np.diag(self.eta2)
        Vf, Vb = care(A, B, Q, R)
        Wf = np.linalg.inv(Vf)
        Wb = np.linalg.inv(Vb)

        print("Vf, Vb, Wf, Wb")
        print(Vf)
        print(Vb)
        print(Wf)
        print(Wb)
        self.covariances = {
        "Vf": Vf,
        "Vb": Vb,
        "Wf": Wf,
        "Wb": Wb
        }

        if self.eta2.size < 2:
            self.tempAf = (Anew - np.dot(Vf, np.dot(cnew, cnew.transpose())) / self.eta2)
            self.tempAb = (Anew + np.dot(Vb, np.dot(cnew, cnew.transpose())) / self.eta2)
        else:
            eta2inv = np.linalg.inv(np.diag(self.eta2))
            self.tempAf = (Anew - np.dot(Vf, np.dot(cnew, np.dot(eta2inv, cnew.transpose()))))
            self.tempAb = (Anew + np.dot(Vb, np.dot(cnew, np.dot(eta2inv, cnew.transpose()))))

        self.Af = np.dot(Wf, np.dot(scipy.linalg.expm(self.tempAf * self.Ts), Vf))

        self.Ab = np.dot(Wb, np.dot(scipy.linalg.expm(-self.tempAb * self.Ts), Vb))

        self.w = np.linalg.solve(Wf + Wb, postFilterC)


        print("Afc")
        print(self.tempAf)
        print("Abc")
        print(self.tempAb)
        print("Af")
        print(self.Af)
        print("Ab")
        print(self.Ab)
        print("w")
        print(self.w)


    def filter(self, control):
        """
        This is the filtering operation. The only adjustment here is:
        - redefine controlMixing matrix such that it follows the new sizes
        - pre multiply Bf and Bb according to the simalirity transformation that
        was previously applied to the state
        - adjust the memory size of the mf and mb such that they accomondate the
        new system size.
        """
        # Exand control mixing matrix
        self.mixingMatrix = np.vstack((control.mixingMatrix, np.zeros((self.postFilteringSystem.A.shape[0], control.mixingMatrix.shape[1]))))
        # Compute Bf and Bb for this type of control
        self.computeControlTrajectories(control)
        # Correct for the message passing variables Wf mf and Wb mb
        self.Bf = np.dot(self.covariances["Wf"], self.Bf)
        self.Bb = np.dot(self.covariances["Wb"], self.Bb)

        # Initalise memory
        u = np.zeros((control.size, len(self.inputs)))
        mf = np.zeros((self.order, control.size))
        mb = np.zeros_like(mf)

        for index in range(1, control.size):
            mf[:, index] = np.dot(self.Af, mf[:, index - 1]) + np.dot(self.Bf, control[index - 1])
            # print(mf[:, index])
        for index in range(control.size - 2, 1, -1):
            mb[:, index] = np.dot(self.Ab, mb[:, index + 1]) + np.dot(self.Bb, control[index])
            u[index] = np.dot(self.w.transpose(), mf[:, index] + mb[:, index])

            # These are for debugging
            # u[index] = np.dot(self.w.transpose(), mf[:, index] + 0 * mb[:, index])
            # u[index] = np.dot(self.w.transpose(), 0 *mf[:, index] + mb[:, index])
        return u


class DiscreteTimeKalmanFilter(object):
    """
    This is the Kalman filter approach to the reconstruction problem where
    Covariance is not assumed fixed and control sequences are treated as both
    observations and input.
    """
    def __init__(self, t, system, inputs, options={}):
        """
        This constructor requries:
        - t which are the times to reconstruct at (assumed to be uniformly spaced)
        - a system model
        - inputs an iterable of inputs to be estimated
        - options
        """
        self.Ts = t[1] - t[0]
        self.system = system
        self.inputs = inputs
        self.order = self.system.order
        self.noise = []

        if 'eta2' in options:
            self.eta2 = options['eta2']
        else:
            self.eta2 = np.ones(self.order)

        if 'noise' in options:
            for source in options['noise']:
                self.noise.append(Noise(standardDeviation=source['std'], steeringVector=source['steeringVector'], name=source["name"]))


        if 'sigmaU2' in options:
            self.sigmaU2 = options['sigmaU2']
        else:
            self.sigmaU2 = np.ones(len(inputs))

        self.Vb, self.Ad = self.discretize(self.inputs[0].steeringVector, self.Ts)



    def discretize(self, b, deltaT):
        Ad = scipy.linalg.expm(self.system.A * deltaT)
        def derivative(x, t):
            Ad = scipy.linalg.expm(self.system.A * t)
            return np.dot(Ad, np.dot(b, np.dot(b.transpose(), Ad.transpose()))).flatten()
        Vinp = odeint(derivative, np.zeros(self.system.order ** 2), np.array([0., deltaT]))[-1,:].reshape(self.system.A.shape)
        return Vinp, Ad


    def __str__(self):
        return "Ad = \n%s\nBf = \n%s\n" % (self.Af, self.Bf)

    def computeControlTrajectories(self, control):
        if control.type == 'analog switch':
            self.Bf = np.zeros((self.order, self.order))
            for controlIndex in range(self.order):
                def ForwardDerivative(x, t):
                    hom = np.dot(self.system.A, x.reshape((self.order,1))).flatten()
                    control = np.zeros(self.order)
                    control[controlIndex] = 1
                    return hom + control

                self.Bf[:, controlIndex] = odeint(ForwardDerivative, np.zeros(self.order), np.array([0., self.Ts]))[-1,:]

            self.Bf = np.dot(self.Bf, self.mixingMatrix)
            self.w = self.inputs[0].steeringVector.transpose() * self.sigmaU2
        else:
            raise NotImplemented



    def filter(self, control):
        """
        This is the actual filter operation. The controller needs to be a
        Controller class instance from system.py.
        """
        # Compute Bf and Bb for this type of control
        self.mixingMatrix = control.mixingMatrix
        self.computeControlTrajectories(control)

        outputGain = 1.

        # Initalise memory
        u = np.zeros((control.size, len(self.inputs)))
        mf = np.zeros((self.order, control.size))
        Vf = [np.eye(self.order) * 1e15] * control.size
        G = [np.eye(self.system.outputOrder)] * control.size
        F = [np.zeros((self.order, self.order))]* control.size
        xi = np.zeros_like(mf)
        W_tilde = [np.zeros((self.order, self.order))] * control.size
        observation = [np.zeros(self.system.outputOrder)] * control.size
        for index in range(control.size):
            observation[index] = outputGain * control[index]

        for index in range(1, control.size):
            G[index - 1] = np.linalg.inv(self.eta2 * self.sigmaU2 * np.eye(self.system.outputOrder)  + np.dot(self.system.c.transpose(), np.dot(Vf[index - 1], self.system.c)))
            F[index - 1] = np.eye(self.order) - np.dot(Vf[index - 1], np.dot(self.system.c, np.dot(G[index - 1], self.system.c.transpose())))
            Vf[index] = np.dot(self.Ad, np.dot(F[index - 1], np.dot(Vf[index - 1], self.Ad.transpose()))) + self.Vb
            print(np.dot(self.Ad, np.dot(F[index - 1], mf[:, index - 1]) + np.dot(Vf[index - 1], np.dot(self.system.c, np.dot(G[index - 1], observation[index - 1])))))
            mf[:, index] = np.dot(self.Ad, np.dot(F[index - 1], mf[:, index - 1]) + np.dot(Vf[index - 1], np.dot(self.system.c, np.dot(G[index - 1], observation[index - 1])))) + np.dot(self.Bf, control[index])
        for index in range(control.size - 2, 1, -1):
            temp = np.dot(F[index].transpose(),self.Ad.transpose())
            W_tilde[index] = np.dot(temp, np.dot(W_tilde[index + 1], temp.transpose())) + np.dot(self.system.c, np.dot(G[index], self.system.c.transpose()))
            xi[:, index] = np.dot(temp, xi[:, index + 1]) + np.dot(self.system.c, np.dot(G[index], np.dot(self.system.c.transpose(), mf[:, index]) - observation[index]))
            u[index] = np.dot(self.w, xi[:, index])
        return u

class Calibration(object):
    """
    This calibration rutine estimates the parameters of an ADC
    """

    def __init__(self, system):
        self.system = system

    def calibrate(self, control, inputs):
        pass

    def costFunction(self, theta):
        pass


# class WienerFilterWithObservations(WienerFilter):
#     """ This is an attempt on integrating the known
#     boundary and quantasation information at the driscrete updates.
#     """
#
#     def __init__(self, **kwargs):
#         """
#         This constructor requries:
#         - system _a system object from system.py_
#         - eta2 _The perfomance parameter of the filter_
#         - Ts _The period sampling time_
#         """
#         self.filterSettings = kwargs
#         self.system = kwargs['system']
#         eta2 = kwargs['eta2']
#         Ts = kwargs['Ts']
#         Vf, Vb = self.care(eta2, self.system)
#
#         Gf = np.linalg.inv(eta2 + np.dot(self.system.c.transpose(), np.dot(Vf, self.system.c)))
#         Gb = np.linalg.inv(eta2 + np.dot(self.system.c.transpose(), np.dot(Vb, self.system.c)))
#         self.Ff = np.eye(self.system.order) - np.dot(Vf, np.dot(self.system.c, np.dot(Gf, self.system.c.transpose())))
#         self.Fb = np.eye(self.system.order) - np.dot(Vb, np.dot(self.system.c, np.dot(Gb, self.system.c.transpose())))
#
#
#         bound = self.system.A[1, 0] * self.filterSettings["Ts"] / 2.
#         bound = 0.
#
#
#         self.Yf = np.dot(np.dot(Vf, np.dot(self.system.c, Gf)), np.dot(self.system.c.transpose(), bound * np.eye(self.system.order)))
#         self.Yb = np.dot(np.dot(Vb, np.dot(self.system.c, Gb)), np.dot(self.system.c.transpose(), bound * np.eye(self.system.order)))
#
#
#         ## Testing changing the interpolation midpoint
#         Adb = scipy.linalg.expm(-self.system.A * Ts/2.)
#         Af = scipy.linalg.expm(self.system.A * Ts/2.)
#         # Vf = np.dot(Af, np.dot(Vf, Af.transpose()))
#         # Vb = np.dot(Adb, np.dot(Vb, Adb.transpose()))
#         # Vf += self.controlabillityGramian(Ts, self.system.A, self.system.b)
#         shift = scipy.linalg.expm(self.system.A * Ts / 2.)
#         # shift = np.eye(self.system.order)
#         # extra = Ts * (self.controlabillityGramian(Ts, self.system.A, self.system.b) - np.outer(self.system.b, self.system.b))
#         extra = np.eye(self.system.order)
#
#         tempAf = (self.system.A - np.dot(Vf, np.dot(self.system.c, self.system.c.transpose())) / eta2)
#         tempAb = (self.system.A + np.dot(Vb, np.dot(self.system.c, self.system.c.transpose())) / eta2)
#         self.Af = scipy.linalg.expm(tempAf * Ts)
#         self.Bf = np.dot(self.system.zeroOrderHold(tempAf, Ts), self.system.B)
#
#         self.Ab = scipy.linalg.expm(-tempAb * Ts)
#         self.Bb = -np.dot(self.system.zeroOrderHold(-tempAb, Ts), self.system.B)
#
#         self.w = np.linalg.solve(Vf + Vb + extra, np.dot(shift, self.system.b))
#
#
#     def filter(self, controller):
#         """
#         This is the acctual filter operation. The controller needs to be a
#         Controller class instance from system.py.
#         """
#
#         u = np.zeros(controller.size)
#         mf = np.zeros((self.system.b.size, controller.size))
#         mb = np.zeros_like(mf)
#         for index in range(1, controller.size):
#             mf[:, index] = np.dot(self.Af, mf[:, index - 1]) + np.dot(self.Bf, controller[:, index - 1])
#             mf[:, index] = np.dot(self.Ff, mf[:, index]) + np.dot(self.Yf, controller[:, index])
#         for index in range(controller.size - 2, 1, -1):
#             mb[:, index] = np.dot(self.Ab, mb[:, index + 1]) + np.dot(self.Bb, controller[:, index])
#             mb[:, index] = np.dot(self.Fb, mb[:, index + 1]) + np.dot(self.Yb, controller[:, index])
#             u[index + 0] = np.dot(self.w.transpose(), mb[:, index] - mf[:, index])
#         return u
#
#
#
# class KalmanFilter(object):
#     """
#     This is the Kalman filter where additionally to the Wiener filter,
#     the Covariance matrices are tracked.
#     """
#     def __init__(self, **kwargs):
#         """
#         The constructor requires the following arguments:
#         - system _a system object from system.py_
#         - eta2 _The perfomance parameter of the filter_
#         - Ts _The period sampling time_
#
#         Additionally, these can be supplied
#         - Vf inital forward covariance
#         - Vb intal backward covariance
#         - mf inital forward mean vector
#         - mb inital backward mean vector
#         """
#
#         self.system = kwargs['system']
#         self.eta2 = kwargs['eta2']
#         self.Ts = kwargs['Ts']
#
#         # Import Vf
#         if "Vf" in kwargs:
#             self.Vf = [kwargs["Vf"]]
#         else:
#             self.Vf = [np.zeros_like(self.system.A)]
#
#         # Import Vb
#         if "Vb" in kwargs:
#             self.Vb = [kwargs["Vb"]]
#         else:
#             self.Vb = [np.zeros_like(self.system.A)]
#
#         # Import mf
#         if "mf" in kwargs:
#             self.mf = [kwargs["mf"]]
#         else:
#             self.mf = [np.zeros_like(self.system.b)]
#
#         # Import mb
#         if "mb" in kwargs:
#             self.mb = [kwargs["mb"]]
#         else:
#             self.mb = [np.zeros_like(self.system.b)]
#
#     def initalizesystem(self, system):
#
#         if system.continuous:
#             self.system.discretize_full(self.Ts)
#             self.Af = system.Ad
#             self.Ab = system.Adb
#             self.bf = system.bd
#             self.bb = system.bdb
#             self.Bf = system.Bd
#             self.Bb = system.Bdb
#             self.Vfb = integrate.quad(self.controlabillityGramian, 0., self.Ts, args=(self.system.A, self.system.b))
#             self.Vbb = integrate.quad(self.controlabillityGramian, 0., self.Ts, args=(-self.system.A, self.system.b))
#         else:
#             self.Af = system.A
#             self.Ab = np.linalg.inv(system.A)
#             self.bf = system.b
#             self.bb = None
#             self.Bf = system.B
#             self.Bb = None
#
#
#     def filter(self, controller):
#         """
#         This is the acctual filter operation. The controller needs to be a
#         Controller class instance from system.py.
#         """
#         u = np.zeros(controller.size)
#
#
#         for index in range(1, controller.size):
#             Vftemp = np.dot(self.system.Ad, np.dot(Vf[index - 1], self.system.Ad.transpose())) + self.Ts * np.outer(self.system.b, self.system.b)
#             F = np.eye(self.system.order) - np.dot(Vftemp, np.dot(self.system.c, np.dot( np.linalg.inv( np.eye(self.system.c.shape[1]) * self.eta2 / self.Ts + np.dot(self.system.c.transpose(), np.dot(Vftemp, self.system.c))), self.system.c.transpose())))
#             Vf[index] = np.dot(F, Vftemp)
#             mf[index] = np.dot(F, np.dot(self.system.Af, mf[index - 1]) + np.dot(self.Bf, controller[:, index - 1]))
#         for index in range(controller.size - 2, 1, -1):
#             # Vbtemp =
#             mb[:, index] = np.dot(self.Ab, mb[:, index + 1]) + np.dot(self.Bb, controller[:, index])
#             u[index + 0] = np.dot(self.w.transpose(), mb[:, index] - mf[:, index])
#         return u
#
#
#
#
# from .simulator import autoControlSimulator
# from .system import Controller
#
# class WienerFilterAutomaticSystem(WienerFilter):
#     def __init__(self, filterSettings):
#         self.filterSettings = filterSettings
#         self.eta2 = filterSettings['eta2']
#         self.Ts = filterSettings['Ts']
#         self.system = filterSettings['system']
#         self.autoSystem = autoControlSimulator(self.system, 1./self.Ts)
#
#     def preCompute(self, states, Vf, Vb, size):
#         Vff = [Vf] * size
#         Vbb = [Vb] * size
#         for index in range(size):
#             system = self.autoSystem.setSystem(states[index, :])
#             expAT = scipy.linalg.expm(system.A * self.Ts)
#             Vf = np.dot(expAT, np.dot(Vf, expAT.transpose())) + self.Ts * np.outer(system.b, system.b)
#             Vf = np.dot(Vf, np.eye(system.order) - np.dot(system.c, np.dot( np.linalg.inv( np.eye(system.c.shape[1]) * self.eta2 / self.Ts + np.dot(system.c.transpose(), np.dot(Vf, system.c))), np.dot(system.c.transpose(), Vf))))
#             # Vf = Vf + self.Ts * (np.dot(system.A, Vf) + np.dot(Vf, system.A.transpose()) + np.outer(system.b, system.b) - 1./self.eta2 * np.dot(Vf, np.dot(system.c, np.dot(system.c.transpose(), Vf))))
#             Vb = Vb - self.Ts * (np.dot(system.A, Vb) + np.dot(Vb, system.A.transpose()) - np.outer(system.b, system.b) + 1./self.eta2 * np.dot(Vb, np.dot(system.c, np.dot(system.c.transpose(), Vb))))
#             tempAf = (system.A - np.dot(Vf, np.dot(system.c, system.c.transpose())) / self.eta2)
#             # print("Vf = ", Vf)
#             # print("Vb = ", Vb)
#             # print("tempAf = ", tempAf)
#             self.Af[index] = scipy.linalg.expm(tempAf * self.Ts)
#             self.Bf[index] = np.dot(system.zeroOrderHold(tempAf, self.Ts), system.B)
#
#             tempAb = (system.A + np.dot(Vb, np.dot(system.c, system.c.transpose())) / self.eta2)
#             self.Ab[- (index + 1)] = scipy.linalg.expm(-tempAb * self.Ts)
#             self.Bb[-(index + 1)] = -np.dot(system.zeroOrderHold(-tempAb, self.Ts), system.B)
#
#             Vff[index] = Vf
#             Vbb[index] = Vb
#
#         for index in range(size):
#             try:
#                 self.w[index] = np.linalg.solve(Vff[index] + Vbb[index], system.b).reshape(system.b.shape)
#             except np.linalg.linalg.LinAlgError:
#                 self.w[index] = np.zeros_like(system.b)
#
#
#     def computeSystemFilter(self, system):
#         Vf, Vb = self.care(self.eta2, self.system)
#         tempAf = (system.A - np.dot(Vf, np.outer(system.c, system.c)) / self.eta2)
#         self.Af = scipy.linalg.expm(tempAf * self.Ts)
#         self.Bf = np.dot(system.zeroOrderHold(tempAf, self.Ts), system.B)
#         tempAb = (system.A + np.dot(Vb, np.outer(system.c, system.c)) / self.eta2)
#         self.Ab = scipy.linalg.expm(-tempAb * self.Ts)
#         self.Bb = -np.dot(system.zeroOrderHold(-tempAb, self.Ts), system.B)
#
#         self.w = np.linalg.solve(Vf + Vb, system.b).reshape((self.Af.shape[0], 1))
#
#
#     def filter(self, states):
#         self.w = [np.zeros_like(self.system.b)] * states.shape[0]
#         self.Af = [np.zeros_like(self.system.A)] * states.shape[0]
#         self.Bf = [np.zeros_like(self.system.B)] * states.shape[0]
#         self.Ab = [np.zeros_like(self.system.A)] * states.shape[0]
#         self.Bb = [np.zeros_like(self.system.B)] * states.shape[0]
#         Vf = np.ones((self.system.order, self.system.order)) * 0
#         Vb = np.ones_like(Vf) * 0
#         self.preCompute(states, Vf, Vb, states.shape[0])
#
#         control = np.zeros_like(states.transpose())
#         for index in range(1, states.shape[1]):
#             control[:, index] = self.autoSystem.computeControls(states[index, :]).flatten()
#         u = np.zeros(states.shape[0])
#         mf = np.zeros((self.filterSettings['system'].b.size, states.shape[0]))
#         mb = np.zeros_like(mf)
#         mf[:, 0] = self.autoSystem.initalState.flatten()
#
#
#         for index in range(1, states.shape[0]):
#             # Vf, Vb = self.computeSystemFilter(self.autoSystem.setSystem(states[index - 1, :]), Vf, Vb)
#             # mf[:, index] = (np.dot(self.Af, mf[:, index - 1].reshape(self.Bf.shape)) + np.dot(self.Bf, self.autoSystem.knownSignal(index - 1))).flatten()
#             mf[:, index] = (np.dot(self.Af[index - 1], mf[:, index - 1].reshape((self.system.order, 1))) + np.dot(self.Bf[index - 1], control[:, index - 1].reshape((self.system.order, 1)))).flatten()
#
#         for index in range(states.shape[0] - 2, 1, -1):
#             # Vf, Vb = self.computeSystemFilter(self.autoSystem.setSystem(states[index, :]), Vf, Vb)
#             # mb[:, index] = (np.dot(self.Ab, mb[:, index + 1].reshape(self.Bb.shape)) + np.dot(self.Bb, self.autoSystem.knownSignal(index))).flatten()
#             mb[:, index] = (np.dot(self.Ab[index + 1], mb[:, index + 1].reshape((self.system.order, 1))) + np.dot(self.Bb[index + 1], control[:, index].reshape((self.system.order, 1)))).flatten()
#             u[index + 0] = np.dot(self.w[index].transpose(), mb[:, index] - mf[:, index])
#         return u
#
# class LeastMeanSquare(object):
#
#     def __str__(self):
#         return "The current FIR filter is %s" % self.w
#
#     def __init__(self):
#         self.w = None
#
#     def train(self, data):
#         x = data["x"]
#         y = data["y"]
#         self.w = np.linalg.solve(np.dot(x.transpose(), x), np.dot(x.transpose(), y))
#
#     def predict(self, dataPoint):
#         return np.dot(self.w, dataPoint)
#
#
#
# if __name__ == "__main__":
#     import doctest
#     doctest.testmod()
