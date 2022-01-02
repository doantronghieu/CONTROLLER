import numpy as np
import matplotlib
import time
#########################################################################################
# GLOBAL PARAMS | INITIAL PARAMS
"""

set_point = 
max_value = 
sample_time = 
current_time = 
windup_guard = 
Kp = 
Ki = 
Kd = 

# Zeigler - Nichols Method
KU = # Kp when the error stably oscillate
TU = # Period of the stable oscillate error
Kp = 0.6 * KU
TI = TU / 2
TD = TU / 8
Ki = (1.2 * KU) / TU
Kd = (3 * KU * TU) / 40

"""
#########################################################################################
class PID(object):
    def __init__(self, Kp, Ki, Kd, max_value, set_point = 0.0, current_time = None, windup_guard = None):
        self.Kp, self.Ki, self.Kd = Kp, Ki, Kd
        self.P_term, self.I_term, self.D_term = 0.0, 0.0, 0.0

        self.sample_time = 0.0
        self.current_time = current_time if current_time is not None else time.time()
        self.last_time = self.current_time

        self.set_point = set_point
        self.max_value = max_value

        self.error, self.last_error, self.integral_error, self.derivative_error = 0.0, 0.0, 0.0, 0.0

        self.output = 0.0

        # Windup Guard
        self.windup_guard = windup_guard

    def set_parameter(self, proportional_gain = None, integral_gain = None, derivative_gain = None):
        if (proportional_gain is not None):
            # Determines how aggressively the PID reacts to the current error with setting Proportional Gain
            self.Kp = proportional_gain

        if (integral_gain is not None):
            # Determines how aggressively the PID reacts to the current error with setting Integral Gain
            self.Ki = integral_gain

        if (derivative_gain is not None):
            # Determines how aggressively the PID reacts to the current error with setting Derivative Gain
            self.Kd = derivative_gain

    def set_sample_time(self, sample_time):
        """
        PID should be updated at a regular interval.
        Based on a pre-determined sampe time, the PID decides if it should compute or return immediately.
        """
        self.sample_time = sample_time

    def set_windup(self, windup_para):
        """
        Integral (reset) windup, the situation where a large change in setpoint  
        occurs (positive change) and the integral terms accumulates a significant 
        error during the rise (windup), thus overshooting and continuing to
        increase as this accumulated error is unwound
        (offset by errors in the other direction), excess overshooting.
        """
        self.windup_guard = windup_para

    def update(self, feedback_value, current_time = None):
        # Calculates PID value for given reference feedback
        # u(t) = K_p e(t) + K_i \int_{0}^{t} e(t)dt + K_d {de}/{dt}

        self.error = self.set_point - feedback_value # How far away are we from out set point
        delta_error = self.error - self.last_error

        self.current_time = current_time if current_time is not None else time.time()
        delta_time = self.current_time - self.last_time

        if (delta_time >= self.sample_time):
            self.P_term = self.Kp * self.error
            
            self.integral_error += self.error * delta_time # Add the error with respect to time
            if (self.integral_error < -self.windup_guard):
                self.integral_error = -self.windup_guard
            elif (self.integral_error > self.windup_guard):
                self.integral_error = self.windup_guard
            self.I_term = self.Ki * self.integral_error
            
            self.derivative_error = delta_error / delta_time # The change of error over time
            self.D_term = self.Kd * self.derivative_error
        
        # Remember last time and last error for next calculation
        self.last_time = self.current_time
        self.last_error = self.error

        self.output = self.P_term + self.I_term + self.D_term

        if (self.output >= self.max_value): # If saturation
            self.output = self.max_value

        return self.output