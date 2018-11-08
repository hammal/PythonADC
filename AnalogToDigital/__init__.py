""" This package contains experimental tools for examining the analog-to-digital
converter"""

import sys
import matplotlib
if sys.platform == "darwin":
    print("System: %s" % sys.platform)
else:
    print("Matplotlib set to PS for plattform: %s" % sys.platform)
    matplotlib.use('PS')

# from .reconstruction import LeastMeanSquare
from AnalogToDigital.reconstruction import WienerFilter
# from .reconstruction import WienerFilterAutomaticSystem
# from .reconstruction import WienerFilterWithObservations
# from .reconstruction import SigmaDeltaApproach
from AnalogToDigital.simulator import Simulator
from AnalogToDigital.system import System, Control, Input, FirstOrderHold, Sin
# from .topologiGenerator import Topology
from AnalogToDigital.defaultSystems import DefaultSystems
from AnalogToDigital.evaluation import Evaluation
import AnalogToDigital.system as system
import AnalogToDigital.simulator as simulator
import AnalogToDigital.reconstruction as reconstruction
import AnalogToDigital.evaluation as evalutation
import AnalogToDigital.filters as filters

import AnalogToDigital.saveFigs as saveFigs
