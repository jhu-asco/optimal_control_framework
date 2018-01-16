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
        tau = u[1:]
        thrust_ddot= u[0]
        e3 = np.array([0, 0, 1])
        v = x[3:6]
        rpy = x[6:9]
        R = euler2mat(rpy[0], rpy[1], rpy[2], 'rzyx')
        omega = x[9:12]
        thrust = x[12]
        thrust_dot = x[13]
        J_omega_dot = np.cross(np.dot(self.J, omega), omega) + tau
        thrust_vec = np.dot(R, e3)
        acceleration = (1.0/self.mass)*(thrust_vec*thrust + self.g)
        omega_dot = np.linalg.solve(self.J, J_omega_dot)
        rpy_dot = self.omegaToRpyDot(rpy, omega)
        return np.hstack([v, acceleration, rpy_dot, omega_dot, thrust_dot, thrust_ddot]) + w

    def jacobian(self, t, x, u, w):
        """
        Find jacobian of xdot wrt x, u, w
        """
        raise Exception('Jacobian not implemented yet')
