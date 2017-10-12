""" This file contains different ODE implementations for simulating the
analog-to-digital conversion hardware. """

import numpy as np
import copy
from .system import Controller


class Simulator(object):
    def __init__(self, model, fs, fc, data):
        self.model = model
        self.fs = fs
        self.fc = fc
        self.model.discretize(1. / self.fs)
        self.state = np.zeros((self.model.A.shape[0], 1))
        self.controller = Controller(self.model, self.fs, self.fc, data.size)
        self.data = data
        # print(model)

    def Run(self):
        """Main run function"""
        y = np.zeros((self.data.size, self.model.c.shape[1]))
        for index, u in enumerate(self.data):
            self.state = np.dot(self.model.Ad, self.state) + np.dot(self.model.Bd,
                                                                    self.controller[:, index - 1].reshape(self.model.b.shape)) + np.dot(
                self.model.bd, u)
            y[index, :] = (np.dot(self.model.c.transpose(), self.state)).flatten()
            if self.controller.controlUpdate(index):
                self.controller[:, index] = self.controller.computeControls(self.state).flatten()
                # self.controller[:, index] = self.controller.computeControls(np.dot(self.model.Tinv, self.state))
            else:
                self.controller[:, index] = self.controller[:, index - 1]

        return self.controller, y

    # def KeepRunning(self, data, state):
    #     self.state = state
    #     self.controller.size += data.size
    #     self.controller.control_sequence = np.concatenate((self.controller.control_sequence, np.zeros((self.controller.control_sequence.shape[0], data.size))), axis=1)
    #     y = np.zeros(data.size)


class autoControlSimulator(object):
    """
    This is the automatic controll simulation Example
    """

    def __init__(self, model, fs):
        self.model = model
        self.model.discretize(1./fs)
        self.fs = fs
        self.initalState = np.zeros((self.model.A.shape[0], 1))
        self.state = copy.deepcopy(self.initalState)

    def knownSignal(self, index):
        return np.sin(2. * np.pi * 1 / 1000. * index)

    def computeControls(self, state):
        """return a {1, -1} array depending on state > 0"""
        return np.array((state > 0) * 2 - 1, dtype=np.int8)


    def Step(self, data):
        """
        This function enables a more controlled execution of the simulator
        such that alternative controll structures can be tested the default is
        however the Run() function.
        """
        states = np.zeros((data.size, self.model.order))
        self.control = np.zeros((self.model.order, 1))
        observation = np.zeros((data.size, self.model.c.shape[1]))
        for index, dataPoint in enumerate(data):
            # Generate a system according to the current states
            # tempModel = self.setSystem(self.state )
            tempModel = self.model
            print("A = %s\nState = %s" % (tempModel.A, self.state))

            # State transistion equation
            self.state = np.dot(tempModel.Ad, self.state) + np.dot(tempModel.bd, dataPoint) \
                        + np.dot(tempModel.Bd, self.control)
                        # + np.dot(tempModel.Bd, self.knownSignal(index - 1)) \

            # Note that the controll part is intentionally missing from the
            # previous control setup (the system is controlled by the system
            # matrix itself.)

            # Store the state
            states[index, :] = self.state.flatten()
            self.control = self.computeControls(self.state)

            # Generate system obsevation
            observation[index, :] = np.dot(tempModel.c.transpose(), self.state).flatten()
        # Return the observed state
        return states, observation

    def setSystem(self, state):
        # Size of system used for iteration length
        size = np.size(state)
        # Convert the state into a bitrepresentation with the first state at the
        # most left representation
        bitRepresentation = np.packbits(state.flatten()>0)
        # Compute XOR between each state and its neightbouring state
        # This will be the forward paths.
        forward = np.unpackbits(bitRepresentation ^ (bitRepresentation << 1))
        # print("Forward: %s" % forward)

        # We now create a mask to block the valid paths
        mask = np.ones_like(self.model.A)
        # Fill out the paths
        for gate in range(size - 1):
            if forward[gate]:
                # if forward then mask backward and vice versa
                mask[gate, gate + 1] = 0.
            # Then the gate is opened in the backward direction
            else:
                mask[gate + 1, gate] = 0.

        # Update the system paths
        tempModel = copy.deepcopy(self.model)
        # Apply mask
        tempModel.A = self.model.A * mask
        # Discretize the system
        tempModel.discretize(1./self.fs)

        # Return the new system
        # print("NewModel", tempModel)
        return tempModel
