#!/bin/bash
xterm -e "python integratorVsFeedback.py" &
xterm -e "python gmCADC.py" &
xterm -e "python gmCIntegratorChain.py" &
