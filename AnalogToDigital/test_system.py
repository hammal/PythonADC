import system
import numpy as np

def testInputs():
    t = np.linspace(0, 100., 1000)
    coef = np.random.rand(1000)
    vector = np.array([1.])

    zeroOrderHold1 = system.Input(t[1]- t[0], coefficients=coef, steeringVector=vector)
    firstOrderHold1 = system.FirstOrderHold(t[1]- t[0], coefficients=coef, steeringVector=vector)
    error1 = np.linalg.norm(zeroOrderHold1.fun(t) - firstOrderHold1.fun(t))
    zeroOrderHold2 = system.Input(t[2]- t[0], coefficients=coef, steeringVector=vector)
    firstOrderHold2 = system.FirstOrderHold(t[2]- t[0], coefficients=coef, steeringVector=vector)
    error2 = np.linalg.norm(zeroOrderHold2.fun(t) - firstOrderHold2.fun(t))

    # Error2 should be larger since it has a larger step size
    assert(error2 > error1)


def testSystem():
    A = np.array([[1., 0],[2.,1.]])
    c = np.array([[1., 0],[0,1]])

    sys = system.System(A, c)
    # Test str function
    print(sys)

    f = sys.frequencyResponse(0.)

    state = np.array([3, 2])

    output = sys.output(state)

    np.testing.assert_array_equal(state, output)


def testControl():
    mixingMatrix = - np.eye(3) * 24.
    state = np.random.rand(3)
    size = 100
    Ts = 0.1
    cntrl = system.Control(mixingMatrix, size)
    for index in range(size):
        cntrl.update(state)
        print(cntrl[index])
        state += cntrl.fun(index * Ts)
        print(state)



if __name__ == "__main__":
    testInputs()
    testSystem()
    testControl()
