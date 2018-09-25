# Analog to digital converter
This python package serves as a experimental package for A/D conversion
using the principles of [Control-based analog-to-digital conversion without sampling and quantization](http://ieeexplore.ieee.org/document/7308975/) and beyond.

## Overview
This package is divided into 3 main layers
1. Input layer:
	- Input models
	- Control models
	- System models
2. Simulation layer:
	- Simulation engine that can simulate and evaluate the system at different time
	steps.
3. Reconstruction layer:
	- Wiener Filter reconstruction
	- Sigma Delta reconstruction


## Input & System Layer
The input layer currently holds three input models
- FirstOrderHold
- Sin
- Noise

and some default system models.
The purpose of the input models is to specify a continuous time signal
that later can be used in simulation. Typically a input and system is specified as this
```python
# system order
order = 2
# sample period
Ts = 1e3
# signal amplitude
amplitude = 1.
# signal frequency
frequency = 5e2
# signal phase
phase = 3./4.*np.pi
# steering vector
b = np.random.randn(order)
# initialise the input
input = system.Sin(Ts, amplitude, frequency, phase, steeringVector)

# Define system model
A = np.array([[1., 0],[2.,1.]])
c = np.array([[1., 0],[0,1]])
# initialise system
sys = system.System(A, c)
```
note that the steering vector in the input model specifies how the input enters the system.

### Control
Additionally, any ADC simulating system needs a control system. A control is defined as
```python
# define the length of the simulation sequence
size = 1000
# Define the mixing matrix
mixingMatrix = - np.eye(3)
# initialise the control
ctrl = system.Control(mixingMatrix, size)
```
Here the mixing matrix determines how each control
policy get fed into the system. Currently only
analog switches are implemented.


## Simulation layer
Here we construct a simulator from an system model and control model. Then by
feeding an input and evaluating the system at different time steps the system can
be simulated. The output of this stage is a simulation object that contains the
controls, the system outputs and other valuable information. An example of how this can be done is
as follows:
```python
# setup simulation object
sim = simulator.Simulator(sys, control=ctrl, initalState=np.ones(order), options={})
# define the simulation time steps
t = np.linspace(0., 99., size)
# feed the input and simulate the system
res = sim.simulate(t, (input,))
```

## Reconstruction layer
Finally, we pertain our input estimate by filter the
control polices, embedded in the "ctrl" object.
```python
# setup wiener filter based on system and inputs
recon = reconstruction.WienerFilter(t, sys, (inp,))
# finally the filtering is done as
u_hat = recon.filter(ctrl)
```
note that it might seem as cheating by passing the input directly into our estimation algorithm. The
purpose of this is to extract the steering vector
of the input which is an essential part of the Wiener filter.
