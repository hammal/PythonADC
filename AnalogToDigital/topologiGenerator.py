import numpy as np
# from .system import *
# from .reconstruction import WienerFilter
# from cvxopt import matrix, solvers

class SystemTransformations(object):
    """
    This class transforms general linear state space models into standard forms
    for SISO filters
    """

    def __init__(self, A, b, c):

        # Check that it is a SISO filter
        if b.shape[1] > 1 or c.shape[1] > 1:
            raise "Not a SISO filter for A=%s, b=%s, c=%s" % (A, b, c)

        self.A = A
        self.b = b
        self.c = c

        self.order = self.A.shape[1]

    def similarityTransform(self, T):
        """
        Perform the similarity transform and return the resulting system
        Anew = T A T^(-1)
        Bnew = T B
        Cnew = (C^T T^(-1))^T
        """
        Tinv = np.linalg.inv(T)
        Anew = np.dot(T, self.A, Tinv)
        bnew = np.dot(T, self.b)
        cnew = np.dot(self.c.transpose(), Tinv).transpose()
        return Anew, bnew, cnew

    def controllabilityMatrix(self, A, b):
        """
        Computes the controllability matrix:
        [b, Ab, ..., A^(n-1)b]
        where n is the order of A. Additionally it validates that the
        controllability matrix has full rank.
        """
        controllabilityMatrixSystem = np.zeros_like(A)
        for column in range(A.shape[1]):
            if column == 0:
                controllabilityMatrixSystem[:, 0] = b.flatten()
            else:
                controllabilityMatrixSystem[:, column] = np.dot(A, controllabilityMatrixSystem[:, column - 1])

        return controllabilityMatrixSystem, np.linalg.det(controllabilityMatrixSystem) > 0.

    def observabilityMatrix(self, A, c):
        """
        Computes the observability matrix and validates that it has full rank.
        """
        observabilityMatrix = np.zeros_like(A)
        for row in range(A.shape[0]):
            if row == 0:
                observabilityMatrix[0, :] = c.flatten()
            else:
                observabilityMatrix[row, :] = np.dot(observabilityMatrix[row - 1, :].transpose(), A)
        return observabilityMatrix, np.linalg.det(observabilityMatrix) > 0.

    def controllabilityForm(self):
        """
        Computes the contrallability form using the fact that:
        1. T(s) = C^T * 1/det(sI - A) * Adj(A) * B. That is the poles of the system are determined by
        the eigenvalues and thus the respective characterisitc polynomial coefficients can be computed using poly.
        2. The controllabilityMatrix exposes the transformation between the two systems. That is:
        [b_1, A_1 b_1 , ..., A_1^(n-1)b_1] = T[b_2, A_2 b_2, ..., A_2^(n-1)b_2]
        where n is the system order. And the relation between the two system states is:

        x_2 = T x_1

        thus,

        A_2 = T A_1 T^(-1)
        b_2 = T b_1
        c_2 = (c_1^T T^(-1))^T

        In summary: The controllability form has a known A_2 and b_2 (from its definition and the knowledge of A_1).
        Thererfore, c_2 can also be computed by using the transformation as shown above.
        """

        controllabilityMatrixSystem1, fullRank = self.controllabilityMatrix(self.A, self.b)
        if not fullRank:
            print(controllabilityMatrixSystem1)
            raise "Controllability matrix not full rank."

        Anew = np.zeros_like(self.A)
        Anew[0,:] = - np.poly(self.A)[1:]
        Anew[1:, 0: self.order-1] = np.eye(self.order - 1)

        bnew = np.zeros((self.order,1))
        bnew[0] = 1.

        controllabilityMatrixSystem2, _ = self.controllabilityMatrix(Anew, bnew)

        T = np.dot(controllabilityMatrixSystem2, np.linalg.inv(controllabilityMatrixSystem1))

        cnew = np.dot(self.c.transpose(), np.linalg.inv(T)).transpose()

        return Anew, bnew, cnew, T


    def observableForm(self):
        """
        Computes the observableform of the matrix. For a more in detail explination see
        controllableform.

        [
        [C_1^T],
        [C_1^T A_1],
        .
        .
        .
        [C_1^T A_1^(n-1)]
        ] = [
        [C_2^T],
        [C_2^T A_2],
        .
        .
        .
        [C_2^T A_2^(n-1)]
        ] T^(-1)

        """
        observabilityMatrix1, fullRank = self.observabilityMatrix(self.A, self.b)
        if not fullRank:
            print(observabilityMatrix1)
            raise "Observability matrix not full rank"

        Anew = np.zeros_like(self.A)
        Anew[:, 0] = -np.poly(self.A)[1:]
        Anew[:-1, 1:] = np.eye(self.order - 1)

        cnew = np.zeros((self.order,1))
        cnew[0] = 1.

        observabilityMatrix2, _ = self.observabilityMatrix(Anew, cnew)
        # T = np.dot(np.linalg.inv(observabilityMatrix2), observabilityMatrix1)
        T = np.linalg.solve(observabilityMatrix2, observabilityMatrix1)
        # T = np.linalg.pinv(Tinv)
        bnew = np.dot(T, self.b)
        # I don't quite agree with this to the best of my understanding
        # T should be Tinv...
        return Anew, bnew, cnew, T

    def modalForm(self):
        raise NotImplemented

    def blockJordanForm(self):
        raise NotImplemented

# class Topology(object):
#
#     def __init__(self, filterSettings):
#         self.filterSettings = filterSettings
#         self.model = filterSettings['model']
#         """
#         TODO here implement different chain models
#         """
#
#     def systemTransf(self, T):
#         iT = np.linalg.inv(T)
#         model = self.model
#         model.A = np.dot(T, np.dot(self.model.A, iT))
#         # model.B = np.dot(T, self.model.B)
#         model.b = np.dot(T, self.model.b)
#         model.c = np.dot(iT.transpose().conj(), self.model.c)
#         model.T = T
#         model.Tinv = iT
#         return model
#
#     def DiagonaliseSteadyStateMatrix(self):
#         wienerfilter = WienerFilter(self.filterSettings)
#         Vf, Vb = wienerfilter.care(self.filterSettings['eta2'], self.model)
#         V = Vf + Vb
#         eigVal, T = np.linalg.eig(V)
#         # Due to Vf, Vb both being symmetric real matrices. The Eigen vectors Q^{-1} = Q^T
#         # so Q Lambda Q^T = Vf + Vb
#         #
#         # for more info see pp 20-25 in Lukas Bruderes thesis.
#         transformation = np.dot(T.transpose().conj(), np.diag(np.sqrt(eigVal)))
#         # transformation = T.transpose().conj()
#         filterSettings = self.filterSettings
#         filterSettings['model'] = self.systemTransf(transformation)
#         return filterSettings
#
#     def computeSmallestControlStrength(self, model, Ts, bound):
#         Aint = model.zeroOrderHold(model.A, Ts)
#         bd = np.dot(Aint, model.b)
#         vec = (Aint.shape[0], 1)
#         one = np.ones(vec)
#         Identity = np.eye(model.order)
#         if (np.dot(Aint, np.dot(Aint, one) + bd) <= one - bd).any():
#             P = 2. * matrix(np.eye(model.order), tc='d')
#             q = matrix(np.zeros((model.order, 1)), tc='d')
#             # G = matrix(-np.eye(model.order), tc='d')
#             # h = matrix(np.zeros_like(vec), tc='d')
#             G = np.concatenate((Aint, -Aint, -Aint, -Aint), axis=0)
#
#             vec1 = one - np.abs(bd.reshape(vec))
#             vec2 = np.dot(Aint, one) + np.dot(Aint,model.b.reshape(model.order, 1))
#             h = bound * np.concatenate((vec1, vec1, -vec2, vec2))
#
#             # for index in range(model.order):
#             #     selector = np.zeros((model.order, model.order))
#             #     selector[index, index] = 1
#             #     G = np.concatenate((G, -np.dot(Aint, selector)), axis=0)
#             #
#             # h = np.vstack((h, np.zeros((model.order**2, 1))))
#
#             G = matrix(G, tc='d')
#             # h = bound * np.concatenate((one - bd.reshape(vec), -(np.dot(Aint, one) + bd.reshape(vec))), axis=0)
#             h = matrix(h, tc='d')
#
#             # sol = solvers.lp(matrix(one, tc='d'), A, b)
#             ## QP currently not working
#             sol = solvers.qp(P, q, G, h)
#             sol = np.array(sol['x'])
#         else:
#             print("Lower Ts. Can't uphold stability constraints")
#             raise
#         model.B = -np.diag(np.array(sol, dtype=np.double).reshape(model.order))
#         for index in range(model.order):
#             model.B[index, index] = model.B[index, index]# * np.sign(Aint[index, index])
#         print(model.B)
#         return model
#
#     def ButterWorthSystem(self):
#         wfd = WienerFilterDesign(self.filterSettings)
#         order = self.filterSettings['model'].order
#         f3dB = self.filterSettings['f3dB']
#         A, B, C = wfd.genButter(f3dB, np.int(order/2))
#         filterSettings = self.filterSettings
#         filterSettings['model'].A = A
#         filterSettings['model'].b = B
#         filterSettings['model'].c = C
#         return filterSettings
#
#     def ControllableForm(self):
#         wfd = WienerFilterDesign(self.filterSettings)
#         return wfd.controllableForm()
#
#     def ObservableForm(self):
#         wfd = WienerFilterDesign(self.filterSettings)
#         return wfd.observableForm()
#
#
# class WienerFilterDesign(object):
#
#     def __init__(self, filterSettings):
#         self.filterSettings = filterSettings
#         self.model = filterSettings['model']
#
#     def ntf(self, frequency):
#         eta2 = self.filterSettings['eta2']
#         G = self.model.frequncyResponse(frequency)
#         den = np.abs(G) ** 2 + eta2
#         return np.conj(G) / den
#
#     def stf(self, frequency):
#         eta2 = self.filterSettings['eta2']
#         G = self.model.frequncyResponse(frequency)
#         den = np.abs(G) ** 2 + eta2
#         return np.abs(G) ** 2 / den
#
#     def controllableForm(self):
#         """ This is the trafo function from Lukas Bruderers python library
#
#         A = [[a_11, a_12],[a_21, a_22]]
#         b = [[b_1],[b_2]]
#         c = [c_1, c_2]
#
#         Anew = [[-p2, 1],[p3, 0]]
#         bnew = [[1],[0]]
#         cnew =
#
#
#         """
#         A = self.filterSettings['model'].A
#         b = self.filterSettings['model'].b
#         c = self.filterSettings['model'].c
#         (m, n) = np.shape(A)
#         S = np.zeros_like(A, dtype=np.complex)
#         char_poly = np.poly(A)
#         for j in range(m):
#             if j == 0:
#                 S[:, j:j + 1] = b
#             else:
#                 S[:, j:j + 1] = np.dot(char_poly[j], b) + np.dot(A, S[:, j - 1:j])
#         Tinv = S
#         T = np.linalg.inv(Tinv)
#
#         Aa = np.hstack((np.eye(n - 1), np.zeros((n - 1, 1))))
#         Aa = np.vstack((-char_poly[1:], Aa))
#
#         ba = np.zeros((n, 1))
#         ba[0, 0] = 1.0
#
#         self.filterSettings['model'].A = Aa.real
#         self.filterSettings['model'].b = ba.real
#         self.filterSettings['model'].c = np.dot(c.transpose(), Tinv).real.transpose()
#         return self.filterSettings
#
#     def observableForm(self):
#         self.filterSettings = self.controllableForm()
#         self.filterSettings['model'].A = self.filterSettings['model'].A.transpose()
#         tempb = self.filterSettings['model'].b.transpose()
#         self.filterSettings['model'].b = self.filterSettings['model'].c.transpose()
#         self.filterSettings['model'].c = tempb
#         return self.filterSettings
#
#     def genSOS(self, p):
#         """ Generate a contious-time second-order filter with two conjugate complex poles """
#         A = np.array([[np.real(p), -np.imag(p)], [np.imag(p), np.real(p)]])
#         B = np.array([[1.0], [-1.0]]) * -abs(p) ** 2 / np.imag(p)
#         C = np.array([[0.5], [0.5]])
#         return (A, B, C)
#
#     def mergeSys(self, sections):
#         """ Calculate the total system if multiple sections are put in series """
#         if len(sections) > 1:
#             (A1, B1, C1) = sections.pop()
#             (A0, B0, C0) = sections.pop()
#             A = np.vstack((np.hstack((A0, np.zeros((len(A0), len(A1))))), np.hstack((np.outer(B1, C0), A1))))
#             B = np.vstack((B0, np.zeros(B1.shape)))
#             C = np.vstack((np.zeros(C0.shape), C1))
#
#             sections.append((A, B, C))
#             return self.mergeSys(sections)
#         else:
#             return sections[0]
#
#     def genButter(self, fc, numSec):
#         r""" Generate a Butterworth Filter
#
#         COMPUTATIONS
#         ------------
#
#         Generates a low-pass butterworth filter with order N anc cut-off frequency fc
#
#         .. math::
#            :nowrap:
#
#            \begin{equation}
#            H(s) = \frac{1}{\prod_k^N (1-s/p_k)}, \quad p_k= w_c\exp \left(i (\pi)/(2N)(2k+N-1)\right) \nonumber
#            \end{equation}
#         """
#         sections = []
#
#         phi = np.pi / (4 * numSec) * np.array([2 * k + 2 * numSec - 1. for k in range(1, numSec + 1)])
#         p = 2 * np.pi * np.exp(phi * 1j)[::-1]
#
#         for k in range(numSec):
#             sections.append(self.genSOS(fc * p[k]))
#         butSys = self.mergeSys(sections)
#         ABt = np.array(butSys[0])
#         bBt = np.array(butSys[1])
#         cBt = np.array(butSys[2])
#         return (ABt, bBt, cBt)
