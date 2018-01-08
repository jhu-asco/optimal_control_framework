from optimal_control_framework.controllers import AbstractController
import numpy as np
import sys

class AccUnicycleController(AbstractController):
    # Goal
    _xd = np.array([0, 0, 0, 0])
    # kp, kd
    _gains = np.array([1,1])
    max_thetadot = np.pi/2

    def setGoal(self, xd):
        """
        Set goal x, y positions
        """
        self._xd = xd[:4]

    def setGains(self, gains):
        """
        set gain array np.array([kp, ktheta])
        """
        self._gains = gains

    def getVelocity(self, x):
        v = x[3]
        theta = x[2]
        return np.array([v*np.cos(theta), v*np.sin(theta)])

    def getRotation(self, x):
        theta = x[2]
        return np.array([[np.cos(theta), -np.sin(theta)],
                         [np.sin(theta), np.cos(theta)]])

    def deterministic_control(self, i, x):
        """
        Compute the velocity and angular velocity
        to get to a desired goal
        """
        e_p = (x[:2] - self._xd[:2])
        e_v = (self.getVelocity(x) - self.getVelocity(self._xd))
        eddot_desired = -1*self._gains[0]*e_p - 1*self._gains[1]*e_v
        #print "eddot_desired: ", eddot_desired
        rotation = self.getRotation(x)
        #print "rotation: ", rotation
        u = np.dot(rotation.T, eddot_desired)
        #print "u: ", u
        # Convert v thetadot to thetadot
        v = x[3]
        #print "v: ", v
        #sys.exit(-1)
        #if np.abs(v) < u[1]/self.max_thetadot:
        #    u[1] = np.sign(v)*self.max_thetadot
        #else:
        u_prev = u[1]
        u[1] = u[1]/v
        #print "e_v: ", e_v, "e_p: ", e_p, "ed: ", eddot_desired, "u: ", u, "v: ", v, "u_p: ", u_prev
        return u
