import numpy as np
from .system import Model
from .system import DiscreteModel

class DefaultSystems(object):
    """
    This class is made such that standard systems and input sequences
    can quickly be generated and reproduced.
    """

    def __init__(self):
        pass

    def whiteNoise(self, length):
        """
        This function returns a Gaussian white noise vecor as

        Example:
        >>> whiteNoiseVector = whiteNoise(100)
        >>> whiteNoiseVector.size
        100

        """
        return np.random.randn(length)


    def integratorChain(self, size, rho=0.):
        """
        This function generates a block Jordan SSM which corresponds to a
        integrator chain with transferfunction
                    1
        H(s) = -----------
                (s - rho)^size

        """
        A = rho * np.eye(size, k=0) + np.eye(size, k=-1)
        b = np.zeros((size, 1))
        c = np.zeros_like(b)
        b[0] = 1.
        c[-1] = 1.

        return Model(A, np.zeros_like(A), b, c)

    def gmCIntergratorChain(self, size, gm, C):
        """
        this generates a gmC chain as:
        A = [
                [0, 0, 0, ...],
                [gm/C, 0, 0, ...],
                [0, gm/C, 0, ...] ...
            ]
        b = [gm/C, 0, ...]^T
        c = [0,...,0, 1]^T
        """
        stageAmplification = gm/np.float(C)
        A = stageAmplification * np.eye(size, k=-1)
        # A += np.eye(size) * 0.1
        b = np.zeros((size, 1))
        b[0] = stageAmplification

        #
        c = np.zeros_like(b)
        c[-1] = 1.

        # Make all states observable
        # c = np.eye(size)
        return Model(A, np.zeros_like(A), b, c)

    def gmCChain(self, size, gm1, gm2, C):
        """
        this generates a gmC chain as:
        A = [
                [0, -gm2/C, 0, ...],
                [gm1/C, 0, -gm2/C, ...],
                [0, gm1/C, 0, ...] ...
            ]
        b = [gm1/C, 0, ...]^T
        c = [0,...,0, 1]^T
        """
        stageAmplification1 = gm1/np.float(C)
        stageAmplification2 = gm2/np.float(C)
        A = stageAmplification1 * np.eye(size, k=-1) - stageAmplification2 * np.eye(size, k=1)
        # temp = np.zeros(size)
        # temp[0:size:2] = 1
        # A += np.diag(stageAmplification * temp)
        b = np.zeros((size, 1))
        b[0] = stageAmplification1

        #
        c = np.zeros_like(b)
        c[-1] = 1.



        # Make all states observable
        # c = np.eye(size)
        return Model(A, np.zeros_like(A), b, c)

    def transmissionLine(self, size, Rs = 0., L = 1e-6, Gp = 0., C = 1e-7):
        """
        """
        if (size % 2) != 0:
            print("Number of transmission line segments must be a multiple of 2")
            raise("Value Error")

        numberOfStages = size >> 1

        stages = []

        # Create each transmission link
        for stage in range(numberOfStages):
            A = np.array([
                [-Gp/C, 1/C],
                [-1/L,   -Rs/L]
            ])
            b = np.array([
                [0, 1/C],
                [1/L, 0]
            ])
            c = np.eye(2)
            stages.append([A, b, c])
        stage = stages.pop(0)
        A = stage[0]
        b = np.zeros((size, 1))
        b[0:2, 0] = stage[1][:,0]
        c = np.zeros_like(b)

        # Cascade each stage
        for stage in stages:
            if A.size > 4:
                A = np.hstack((A, np.vstack((np.zeros((A.shape[0] -2, 2)), -np.outer(stage[1][:, 1], stage[2][:, 1])))))
                temp = np.hstack((np.zeros((2, A.shape[1] - 4)), np.hstack((np.outer(stage[1][:, 0], stage[2][:, 0]), stage[0]))))
            else:
                A = np.hstack((A, -np.outer(stage[1][:, 1], stage[2][:, 1])))
                temp = np.hstack((np.outer(stage[1][:, 0], stage[2][:, 0]), stage[0]))
            A = np.vstack((A,temp))

        c[-2:size, 0] = stage[2][:,0]

        #  Control Current at each
        B = np.zeros(size)
        B[0:size:2] = - 1 / C
        B = np.diag(B)
        return Model(A, B, b, c)


    def discreteIntegratorChain(self, size):
        A = 1. * np.eye(size) + 1./2.*np.eye(size, k=-1)
        b = np.zeros((size, 1))
        b[0,0] = 1./2.
        B = -1./2. * np.eye(size)
        c = np.eye(size)
        return DiscreteModel(A,B,b,c)

if __name__ == "__main__":
    import doctest
    doctest.testmod()
