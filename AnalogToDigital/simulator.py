""" This file contains different ODE implementations for simulating the
analog-to-digital conversion hardware. """

import numpy as np
import copy
import scipy
from scipy.optimize import minimize
from scipy.integrate import odeint
import sdeint
import time


class Simulator(object):
    """
    The Simulator is an object that is defined by a system, a control and can
    handle one or multiple inputs for simulating.
    """
    def __init__(self, system, control=None, initalState=None, options={}):
        self.system = system
        self.control = control

        # Set the inital state
        if type(initalState) is np.ndarray:
            self.state = initalState.reshape((system.order,))
        else:
            # self.state = np.zeros(system.order)
            self.state = np.random.randint(2, size=system.order) * 2. - 1.
            print("Initial state set: %s" % self.state)

        self.options = options
        self.logstr = ""
        self.log("Simulation started!")
        self.num_oob = 0


    def log(self,message=""):
        tmp = "{}: {}\n".format(time.strftime("%d/%m/%Y %H:%M:%S"), message)
        self.logstr += tmp


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
        tnew = t
        current_sample = 0

        if 'jitter' in self.options:
            jitter_range = self.options['jitter']['range']
            if jitter_range > (t[1] - t[0])/2.:
                raise "Too large jitter range. Time steps could change order"
            tnew = t + (np.random.rand(t.size) - 0.5) * jitter_range
            print("With Jitter!")
            # print(t)
            # print(tnew)

        for timeInstance in tnew[1:]:
            # Store observations
            if outputDimension:
                output[index, :] = self.system.output(self.state)
                index += 1

            def f(x, t):
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
            if "noise" in self.options:
                 # Shock Noise
                noise_state = np.zeros((self.system.order, len(self.options["noise"])))
                for i, noiseSource in enumerate(self.options['noise']):
                    # print(noiseSource['std'])
                    if noiseSource['std'] > 0:
                        # std = np.sqrt(noiseSource["std"] ** 2 * (timeInstance - t0)) * noiseSource["steeringVector"]
                        std = noiseSource["std"] * noiseSource["steeringVector"]
                        noise_state[:, i]= std
                        # noise_state += (np.random.rand() - 0.5 ) * 2 * std   
                def g(x, t):
                    return noise_state
                   
            else:
                def g(x,t):
                    return np.zeros((self.system.order, 1))

            # Solve ordinary differential equation
            # self.state = odeint(derivate, self.state, np.array([t0, timeInstance]), mxstep=100, rtol=1e-13, hmin=1e-12)[-1, :]
            tspace = np.linspace(t0, timeInstance, 10)
            self.state = sdeint.itoint(f, g, self.state, tspace)[-1, :]
            # If thermal noise should be simulated
            # if "noise" in self.options:

                # # Shock Noise
                # noise_state = np.zeros(self.system.order)
                # for noiseSource in self.options['noise']:
                #     # print(noiseSource['std'])
                #     if noiseSource['std'] > 0:
                #         # std = np.sqrt(noiseSource["std"] ** 2 * (timeInstance - t0)) * noiseSource["steeringVector"]
                #         std = noiseSource["std"] * noiseSource["steeringVector"]
                #         noise_state += np.random.randn() * std
                #         # noise_state += (np.random.rand() - 0.5 ) * 2 * std
                # self.state += noise_state

                # # Thermal Noise Simulation
                # for noiseSource in self.options['noise']:
                #     if noiseSource['std'] > 0:
                #         def noiseDerivative(x, t):
                #             hom = np.dot(self.system.A, x.reshape((self.system.order,1))).flatten()
                #             noise = np.random.randn() * noiseSource["std"] * noiseSource["steeringVector"]
                #             return hom + noise
                #         noise_state = odeint(noiseDerivative, np.zeros_like(self.state), np.array([t0, timeInstance]))[-1, :]
                #         # print("Noise state %s" %noise_state)
                #         # print("state before ", self.state)
                #         self.state += noise_state
                #         # print("noise ", noise_state)
                #         # print("state after ", self.state)

            # Increase time
            t0 = timeInstance
            # Update control descisions
            # print(self.state)

            # Clip if state is out of bound
            if True:
                bound = 1.
                above = self.state > bound
                below = self.state < -bound

                oob_states = np.arange(self.system.order)[np.logical_or(above,below)]
                if any(oob_states):
                    self.log("STATE BOUND EXCEEDED! Sample #: {}".format(current_sample))
                    self.log("X_{} = {}".format(oob_states, self.state[oob_states]))
                    self.num_oob += 1
                    #self.state[above] = bound
                    #self.state[below] = -bound

            # print(self.state)
            current_sample += 1
            self.control.update(self.state)

        # Return simulation object
        return {
            't': t,
            'control': self.control,
            'output': output,
            'system': self.system,
            'state': self.state,
            'options': self.options,
            'log': self.logstr,
            'num_oob': self.num_oob
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
