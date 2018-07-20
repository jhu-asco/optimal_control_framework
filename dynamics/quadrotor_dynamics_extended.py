#!/usr/bin/env python2
from optimal_control_framework.dynamics import QuadrotorDynamics
import numpy as np
from transforms3d.euler import euler2mat

class QuadrotorDynamicsExt(QuadrotorDynamics):
    """
    Simple car dynamics with controls as acceleration,
    steering rate
    """
    def __init__(self, mass = 1.0, J=np.eye(3), g=[0,0,-1]):
        """
        Constructor that initializes the order
        """
        super(QuadrotorDynamicsExt, self).__init__(mass, J, g)
        self.n = 14

    def xdot(self, t, x, u, w):
        """
        X = [x, y, z, vx, vy, vz, r,p,y, omegax, omegay, omegaz, thrust, thrust_dot]
        u = [thrust_ddot, taux, tauy, tauz]
        """
        ubar = np.hstack((x[12], u[1:]))
        xdotbar = super(QuadrotorDynamicsExt, self).xdot(t, x[:12], ubar, w[:12])
        return np.hstack((xdotbar, x[13]+w[12], u[0]+w[13]))

    def jacobian(self, t, x, u, w):
        """
        Find jacobian of xdot wrt x, u, w
        """
        raise Exception('Jacobian not implemented yet')
