import numpy as np
import matplotlib.pyplot as plt
import time
import warnings
#########################################################################################
# GLOBAL PARAMS | INITIAL PARAMS
"""

set_point = 
sample_time = 
current_time = 
output_limits = (, )
Kp = 
Ki = 
Kd = 

# Zeigler - Nichols Method
Ku = # Kp when the error stably oscillate
Tu = # Period of the stable oscillate error
Kp = 0.6 * KU
Ti = TU / 2
Td = TU / 8
Ki = (1.2 * KU) / TU
Kd = (3 * KU * TU) / 40

"""
#########################################################################################
def clamp(value, limits):
    lower_limit, upper_limit = limits

    if (value is None):
        return None
    elif (upper_limit is not None) and (value > upper_limit):
        return upper_limit
    elif (lower_limit is not None) and (value < lower_limit):
        return lower_limit

    return value

#########################################################################################
class PID(object):
    """
    PID controller with integral-windup & derivative-kick prevention and bumpless
        manual-to-auto-mode transfer

    Simple implementation of a PID controller for a closed loop control system.
        As of self.now_time by setting either Ki or Kd equal to zero you can use, respectively
        a PD or a PI controller. If you set all three the parameters then you'll be 
        using a PID.
    
    The PID is provided with a raw (to be improved) anti-windup system.

    Proportional term: Related to system response, Proportional gain, Kp: determines the response speed of the system
    Integral term:     Accumulate errors to make the output closer to target. 
    Derivative term:   By derivative errors, the vibration of response is reduced.
    """
    def __init__(self, Kp = 1.0, Ki = 0.0, Kd = 0.0, set_point = 0.0, sample_time = 0.01, 
                 output_limits = (None, None), proportional_on_measurement = False):
        
        """
        Initialize a new PID controller.
        :param Kp, Ki, Kd: The value for the proportional gain Kp, integral gain Ki, derivative gain Kd
        :param set_point: The initial setpoint that the PID will try to achieve.
            Constant value to be reached by the measured variable y
            Set point is set to zero by default
        :param sample_time: The time in seconds which the controller should wait before generating
            a new output value. The PID works best when it is constantly called (eg. during a
            loop), but with a sample time set so that the time difference between each update is
            (close to) constant. If set to None, the PID will compute a new output value every time
            it is called.
        :param output_limit: The initial output limits to use, given as an iterable with 2
            elements, for example: (lower, upper). The output will never go below the lower limit
            or above the upper limit. Either of the limits can also be set to None to have no limit
            in that direction. Setting output limits also avoids integral windup, since the
            integral term will never be allowed to grow outside of the limits.
        :param auto_mode: Whether the controller should be enabled (auto mode) or not (manual mode)
        :param proportional_on_measurement: Whether the proportional term should be calculated on
            the input directly rather than on the error (which is the traditional way). Using
            proportional-on-measurement avoids overshoot for some types of systems.
        :param error_map: Function to transform the error value in another constrained value.

        Set PID parameters. By setting some parameters to zero we can get
            a P or PI controller (or even a pure integrator if we wish)
        """
        
        # PID parameters
        self.Kp, self.Ki, self.Kd = Kp, Ki, Kd
        # PID terms
        self.P_term, self.I_term, self.D_term = 0.0, 0.0, 0.0

        # Set point is set to zero by default
        self.set_point = set_point

        # Sampling time
        self.sample_time = sample_time
        self.old_time = None

        self.current_error, self.last_error, self.integral_error, self.derivative_error = 0.0, 0.0, 0.0, 0.0

        # Anti windup
        self.min_output, self.max_output = None, None
        self.output_limits = output_limits

        # PID last output value returned
        self.output = 0.0 
        self.last_output = 0.0
        
        # Last y measured
        self.last_feedback_value = 0.0

        # Timing
        self.start_time = time.monotonic()
        self.now_time = time.monotonic()
        self.old_time = self.now_time

        self.proportional_on_measurement = proportional_on_measurement

        # !
        self.set_point_array, self.feedback_value_array, self.time_duration_array = [], [], []

        self.reset()
    
    def reset(self):
        """
        Reset the PID controller internals.
        This sets each term to 0 as well as clearing the integral, the last output and the last
            input (derivative calculation).
        """

        self.P_term, self.I_term, self.D_term = 0.0, 0.0, 0.0
        self.I_term = clamp(value=self.I_term, limits=self.output_limits)
        self.old_time = time.monotonic()
        self.last_output = 0.0
        self.last_feedback_value = 0.0

    def set_sample_time(self, sample_time):
        """
        PID should be updated at a regular interval.
        Based on a pre-determined sampe time, the PID decides if it should compute or return immediately.
        """
        self.sample_time = sample_time

    def update(self, feedback_value, dt = None):

        """
            Update the PID controller. Calculates PID value for given reference feedback
        Call the PID controller with *feedback_value* and calculate and return a control output if
        sample_time seconds has passed since the last update. If no new output is calculated,
        return the previous output instead (or None if no value has been calculated yet).
        :param dt: If set, uses this value for timestep instead of real time.

        u(t)        = K_p * e(t) + K_i \int_{0}^{t} e(t) dt        + K_d {de} / {dt}
        outValue(t) = K_p * e(t) + K_i * cumulative_sum(e(t) * dt) + K_d * de(t) / dt

        Note: If the set point is constant, then de(t) / dt = -dy / dt
            this little tweak helps us avoiding impulsive terms (spikes) due to 
            the derivative of the error (since the error changes instantly when
            switching the set point, its derivative ends up being infinite).   
        """
        # Get elapsed time. Get monotonic time to ensure that time deltas are always positive
        self.now_time = time.monotonic()
        if (dt is None):
            dt = (self.now_time - self.old_time) if (self.now_time > self.old_time) else 1e-16
        elif (dt <= 0):
            raise ValueError('dt has negative value {}, must be positive'.format(dt))
        delta_time = dt

        # Only update every sample_time seconds
        if (self.sample_time is not None) and (delta_time < self.sample_time) and (self.last_output is not None):
            return self.last_output

        # Compute the current Error terms. How far away are we from out set point
        self.current_error = self.set_point - feedback_value
        delta_feedback_value = self.last_feedback_value - (feedback_value if (self.last_feedback_value is not None) else feedback_value)

        # Calculate output
        if (delta_time >= self.sample_time):
            # Compute the proportional term 
            if (not self.proportional_on_measurement):
                # Regular proportional-on-error, simply set the proportional term
                self.P_term = self.Kp * self.current_error
            else:
                # Add the proportional error on measurement to error_sum
                self.P_term += self.Kp * delta_feedback_value
            
            # Compute the integral term. Add the error with respect to time
            self.integral_error = self.current_error * delta_time 
            self.I_term += self.Ki * self.integral_error
            """ Avoid integral windup. Prevents the I-Controller from diverging
            Integral (reset) windup, the situation where a large change in setpoint  
            occurs (positive change) and the integral terms accumulates a significant 
            error during the rise (windup), thus overshooting and continuing to
            increase as this accumulated error is unwound
            (offset by errors in the other direction), excess overshooting.
            """
            self.I_term = clamp(value=self.I_term, limits=self.output_limits)

            # Compute the derivative term
            delta_error = self.last_error - self.current_error
            self.derivative_error = delta_feedback_value / delta_time # The change of error over time
            self.D_term = self.Kd * self.derivative_error
        
        # Compute final output
        self.output = self.P_term + self.I_term + self.D_term
        self.output = clamp(value=self.output, limits=self.output_limits)

        # Arrays for Visualization of Output Results
        self.time_duration_array += [self.now_time - self.start_time]
        self.feedback_value_array += [feedback_value]
        self.set_point_array += [self.set_point]

        # Keep track of state. Remember last time and last error for next calculation. Update variables
        self.old_time = self.now_time
        self.last_error = self.current_error
        self.last_output = self.output
        self.last_feedback_value = feedback_value

        return self.output
    
    def show_response(self):
        plt.plot(self.time_duration_array, self.set_point_array, label = 'Set Point')
        plt.plot(self.time_duration_array, self.feedback_value_array, label = 'PID Controller')
        plt.xlabel('Time')
        plt.ylabel("System")
        plt.pause(self.sample_time)

    # RETURNING VALUE FUNCTIONS
    def get_info(self, get_error = False, get_I_term = False, get_D_term = False):
        if (get_error == True):
            return self.error
        if (get_I_term == True):
            return self.I_term
        if (get_D_term == True):
            return self.D_term
    
    @property
    def PID_terms(self):
        """
        The P-, I- and D-terms from the last computation as separate components as a tuple. Useful
        for visualizing what the controller is doing or when tuning hard-to-tune systems.
        """
        return self.P_term, self.I_term, self.D_term
    
    @property
    def tune_parameters(self):
        # The tunings used by the controller as a tuple: (Kp, Ki, Kd)
        return self.Kp, self.Ki, self.Kd
    
    @tune_parameters.setter
    def tunings(self, proportional_gain, integral_gain, derivative_gain):
        # Set the PID tunings
        # Determines how aggressively the PID reacts to the current error with setting Proportional Gain
        self.Kp = proportional_gain
        # Determines how aggressively the PID reacts to the current error with setting Integral Gain
        self.Ki = integral_gain
        # Determines how aggressively the PID reacts to the current error with setting Derivative Gain
        self.Kd = derivative_gain

    @property
    def output_limits(self):
        # The current output limits as a 2-tuple: (lower, upper)
        return self.min_output, self.max_output
    
    @output_limits.setter
    def output_limits(self, limits):
        # Set the output limit
        if (limits is None):
            self.min_output, self.max_output = None, None
            return

        min_output, max_output = limits
        
        if (None not in limits) and (max_output < min_output):
            raise ValueError('Lower limit must be less than upper limit')

        self.min_output = min_output
        self.max_output = max_output

        self.I_term = clamp(value=self.I_term, limits=self.output_limits)
        # self.last_output = clamp(value=self.last_output, limits=self.output_limits)
#########################################################################################
# Example Heating a room
class Heater:
    """
    The variable `temp` is the output of the whole system, it reflects the situation of the
        variable `power`, which is a control variable of PID controller. `dt` is a time
        variable
    """
    def __init__(self):
        self.temp = 25
    
    def update(self, power, dt):
        if (power > 0):
            # Variation of room temperature with power and time variable dt during heating
            self.temp += 2 * power * dt

        # Indicates heat loss in a room
        self.temp -= 0.5 * dt

        return self.temp
#########################################################################################
def main():
    # Write the created model into the main function
    my_heater = Heater()
    temp = my_heater.temp
    sample_time = 0.01

    # Set the 03 parameters of PID and limit output
    my_PID = PID(Kp=2, Ki=0.01, Kd=0.1, set_point=temp, output_limits=(0, None), sample_time=sample_time)

    # Used to set time parameters
    start_time = time.time()
    last_time = start_time

    # Set System Runtime
    while (time.time() - start_time < 10):
        # Setting the time variable `dt`
        current_time = time.time()
        dt = current_time - last_time

        """
        The variable `temp` is used as the output in the whole system
        The difference between the variable `temp` and the ideal value is used as the input
            in the feedback loop to adjust the change of the variable `power`
        """
        power = my_PID.update(feedback_value=temp)
        temp = my_heater.update(power=power, dt=dt)

        # Used for initial value assignment of variable `temp`
        if (current_time - start_time > 0):
            my_PID.set_point = 100
        
        last_time = current_time

        # Visualization of Output Results
        my_PID.show_response()

    plt.show() # For reviewing the plot after `while True`
    
    
if (__name__ == '__main__'):
    main()