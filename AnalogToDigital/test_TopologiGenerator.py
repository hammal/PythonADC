import numpy as np
import topologiGenerator
np.set_printoptions(precision=2)

n = 5
A = np.random.randn(n, n)
b = np.random.randn(n,1)
c = np.random.randn(n,1)


A = np.array([
    [30, 0, 0 ,0 ,0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0]
    ]) + np.diag(np.arange(5))
print(A)
b = np.array([1, 2, 3, 4, 5]).reshape((5,1))
c = np.array([1, 1, 1, 1, 1]).reshape((5,1))


def testConstructor():
    return topologiGenerator.SystemTransformations(A, b, c)

def testControllableForm():
    system = testConstructor()
    Anew, bnew, cnew, T = system.controllabilityForm()
    Tinv = np.linalg.inv(T)
    print(Anew)
    print(np.dot(T, np.dot(A, Tinv)))
    print(np.dot(c.transpose(), Tinv).transpose())
    assert np.allclose(Anew, np.dot(T, np.dot(A, Tinv)))
    assert np.allclose(bnew, np.dot(T, b))
    assert np.allclose(cnew, np.dot(c.transpose(), Tinv).transpose())

def testObservabilityForm():
    system = testConstructor()
    Anew, bnew, cnew, T = system.observableForm()
    Tinv = np.linalg.inv(T)
    print(Anew)
    print(np.dot(T, np.dot(A, Tinv)))
    print(np.dot(c.transpose(), Tinv).transpose())
    assert np.allclose(Anew, np.dot(T, np.dot(A, Tinv)))
    assert np.allclose(bnew, np.dot(T, b))
    assert np.allclose(cnew, np.dot(c.transpose(), Tinv).transpose())

if __name__ == "__main__":
    testConstructor()
    testControllableForm()
    testObservabilityForm()
