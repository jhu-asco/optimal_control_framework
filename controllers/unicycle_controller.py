from optimal_control_framework.controllers import AbstractController
import numpy as np

class UnicycleController(AbstractController):
    # Goal
    _xd = np.array([0, 0])
    # kp, ktheta
    _gains = np.array([1,1])

    def setGoal(self, xd):
        """
        Set goal x, y positions
        """
        self._xd = xd[:2]

    def setGains(self, gains):
        """
        set gain array np.array([kp, ktheta])
        """
        self._gains = gains

    def deterministic_control(self, i, x):
        """
        Compute the velocity and angular velocity
        to get to a desired goal
        """
        e = (x[:2] - self._xd)
        edot_desired = -1*self._gains[0]*e
        current_dirxn = np.array([np.cos(x[2]), np.sin(x[2])])
        v = np.dot(edot_desired, current_dirxn)
        theta_desired = np.arctan2(edot_desired[1], edot_desired[0])
        theta_corrected = np.remainder(x[2], 2*np.pi)
        e_theta = (theta_corrected - theta_desired)
        if e_theta > np.pi:
            e_theta = e_theta - 2*np.pi
        elif e_theta < -np.pi:
            e_theta = e_theta + 2*np.pi
        thetadot = -1*self._gains[1]*e_theta
        return np.array([v, thetadot])
