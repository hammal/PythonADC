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


## Input Layer

## Simulation layer
Here we construct a simulator from an system model and control model. Then by
feeding an input and evaluating the system at different time steps the system can
be simulated. The output of this stage is a simulation object that contains the
controls, the system outputs and other valuable information.

## Reconstruction layer
