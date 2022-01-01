import numpy as np
import matplotlib
import time
#########################################################################################
# GLOBAL PARAMS | INITIAL PARAMS
TIMER = 0
SETPOINT = 0
TIME_STEP = 0.001

"""

SETPOINT = 
KP = 
KI = 
KD = 

# Zeigler - Nichols Method
KU = # KP when the error stably oscillate
TU = # Period of the stable oscillate error
KP = 0.6 * KU
TI = TU / 2
TD = TU / 8
KI = (1.2 * KU) / TU
KD = (3 * KU * TU) / 40

"""
#########################################################################################
class PID(object):
    def __init__(self, KP, KI, KD, setPoint, maxValue):
        self.KP = KP
        self.KI = KI
        self.KD = KD
        self.setPoint = setPoint
        self.maxValue = maxValue
        self.error = 0
        self.lastError = 0
        self.integralError = 0
        self.derivativeError = 0
        self.output = 0

    def compute(self, currentValue):
        self.error = self.setPoint - currentValue # How far away are we from out set point
        self.integralError += self.error * TIME_STEP # Add the error with respect to time
        self.derivativeError = (self.error - self.lastError) / TIME_STEP # The change of error over time
        self.lastError = self.error # Set the current error to the last error for the after loop
        self.output = (self.KP * self.error) + (self.KI * self.integralError) + (self.KD * self.derivativeError)

        if (self.output >= self.maxValue): # If saturation
            self.output = self.maxValue


        return self.output