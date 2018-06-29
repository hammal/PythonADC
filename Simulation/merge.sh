#!/bin/bash
rm sqeData.npz
ssh isitux "cd PythonLibrary; python merge.py"
rsync -auv isitux:~/PythonLibrary/sgeData.npz .
