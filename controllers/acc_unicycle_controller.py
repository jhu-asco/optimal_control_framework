from optimal_control_framework.controllers import AbstractController
import numpy as np
import sys

class AccUnicycleController(AbstractController):
    # Goal (p, pdot, pddot)
    _xd = np.array([0, 0, 0, 0, 0, 0])
    # kp, kd
    _gains = np.array([1,1])
    _tol = 1e-4

    def __init__(self, dynamics):
      super(AccUnicycleController, self).__init__(dynamics)
      self.m = 2  # Feedforward trajectory pddot
      self.max_thetadot = np.pi/2

    def setGoal(self, xd, pddot=np.zeros(2)):
        """
        Set goal x, y positions
        """
        p = xd[:2]
        pdot = self.getVelocity(xd)
        self._xd = np.hstack([p, pdot, pddot])

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
    def inverseSpeed(self, x):
        v = x[3]
        return v/(v*v + self._tol)

    def deterministic_control(self, i, x):
        """
        Compute the velocity and angular velocity
        to get to a desired goal
        """
        e_p = (x[:2] - self._xd[:2])
        e_v = (self.getVelocity(x) - self._xd[2:4])
        eddot_desired = self._xd[4:]-1*self._gains[0]*e_p - 1*self._gains[1]*e_v
        #print "eddot_desired: ", eddot_desired
        rotation = self.getRotation(x)
        #print "rotation: ", rotation
        u = np.dot(rotation.T, eddot_desired)
        #print "u: ", u
        # Convert v thetadot to thetadot
        u[1] = u[1]*self.inverseSpeed(x)
        if np.abs(u[1]) > self.max_thetadot:
            u[1] = np.sign(u[1])*self.max_thetadot
        #print "e_v: ", e_v, "e_p: ", e_p, "ed: ", eddot_desired, "u: ", u, "v: ", v, "u_p: ", u_prev
        return u
