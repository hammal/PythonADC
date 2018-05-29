""" This file contains different ODE implementations for simulating the
analog-to-digital conversion hardware. """

import numpy as np
import copy
import scipy
from scipy.optimize import minimize
from scipy.integrate import odeint


class Simulator(object):
    """
    The Simulator is an object that is defined by a system, a control and can
    handle one or multiple inputs for simulating.
    """
    def __init__(self, system, control=None, initalState=None,options=None):
        self.system = system
        self.control = control

        # Set the inital state
        if not initalState:
            self.state = np.zeros(system.order)
        else:
            self.state = initalState.reshape((system.order,))

        self.options = options


    def simulate(self, t, inputs=None):
        """
        The simulate function takes no, one or multiple inputs and simulates the
        system from the current state for all time marks in t. At these time the
        control is also updated!
        """
        outputDimension = self.system.outputOrder
        if outputDimension:
            output = np.zeros((t.size, outputDimension))

        t0 = t[0]
        index = 0
        for timeInstance in t[1:]:
            # Store observations
            if outputDimension:
                output[index, :] = self.system.output(self.state)
                index += 1

            def derivate(x, t):
                """
                Compute the system derivative considering state control and input.
                """
                hom = np.dot(self.system.A, x.reshape((self.system.order,1))).flatten()
                control = self.control.fun(t)
                input = np.zeros_like(hom)
                if inputs:
                    for signal in inputs:
                        input += signal.fun(timeInstance)
                return hom + control + input
            # Solve ordinary differential equation
            self.state = odeint(derivate, self.state, np.array([t0, timeInstance]))[-1, :]
            # Increase time
            t0 = timeInstance
            # Update control descisions
            self.control.update(self.state)

        # Return simulation object
        return {
            't': t,
            'control': self.control,
            'output': output,
            'system': self.system,
            'state': self.state,
            'options': self.options,
        }


# class autoControlSimulator(object):
#     """
#     This is the automatic controll simulation Example
#     """
#
#     def __init__(self, model, fs):
#         self.model = model
#         self.model.discretize(1./fs)
#         self.fs = fs
#         self.initalState = np.zeros((self.model.A.shape[0], 1))
#         self.state = copy.deepcopy(self.initalState)
#
#     def knownSignal(self, index):
#         return np.sin(2. * np.pi * 1 / 1000. * index)
#
#     def computeControls(self, state):
#         """return a {1, -1} array depending on state > 0"""
#         return np.array((state > 0) * 2 - 1, dtype=np.int8)
#
#
#     def Step(self, data):
#         """
#         This function enables a more controlled execution of the simulator
#         such that alternative controll structures can be tested the default is
#         however the Run() function.
#         """
#         states = np.zeros((data.size, self.model.order))
#         self.control = np.zeros((self.model.order, 1))
#         observation = np.zeros((data.size, self.model.c.shape[1]))
#         for index, dataPoint in enumerate(data):
#             # Generate a system according to the current states
#             # tempModel = self.setSystem(self.state )
#             tempModel = self.model
#             print("A = %s\nState = %s" % (tempModel.A, self.state))
#
#             # State transistion equation
#             self.state = np.dot(tempModel.Ad, self.state) + np.dot(tempModel.bd, dataPoint) \
#                         + np.dot(tempModel.Bd, self.control)
#                         # + np.dot(tempModel.Bd, self.knownSignal(index - 1)) \
#
#             # Note that the controll part is intentionally missing from the
#             # previous control setup (the system is controlled by the system
#             # matrix itself.)
#
#             # Store the state
#             states[index, :] = self.state.flatten()
#             self.control = self.computeControls(self.state)
#
#             # Generate system obsevation
#             observation[index, :] = np.dot(tempModel.c.transpose(), self.state).flatten()
#         # Return the observed state
#         return states, observation
#
#     def setSystem(self, state):
#         # Size of system used for iteration length
#         size = np.size(state)
#         # Convert the state into a bitrepresentation with the first state at the
#         # most left representation
#         bitRepresentation = np.packbits(state.flatten()>0)
#         # Compute XOR between each state and its neightbouring state
#         # This will be the forward paths.
#         forward = np.unpackbits(bitRepresentation ^ (bitRepresentation << 1))
#         # print("Forward: %s" % forward)
#
#         # We now create a mask to block the valid paths
#         mask = np.ones_like(self.model.A)
#         # Fill out the paths
#         for gate in range(size - 1):
#             if forward[gate]:
#                 # if forward then mask backward and vice versa
#                 mask[gate, gate + 1] = 0.
#             # Then the gate is opened in the backward direction
#             else:
#                 mask[gate + 1, gate] = 0.
#
#         # Update the system paths
#         tempModel = copy.deepcopy(self.model)
#         # Apply mask
#         tempModel.A = self.model.A * mask
#         # Discretize the system
#         tempModel.discretize(1./self.fs)
#
#         # Return the new system
#         # print("NewModel", tempModel)
#         return tempModel
