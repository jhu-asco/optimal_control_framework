#!/usr/bin/env python2
from optimal_control_framework.dynamics import AbstractDynamicSystem
import numpy as np
from transforms3d.euler import euler2mat

class QuadrotorDynamics(AbstractDynamicSystem):
    """
    Simple car dynamics with controls as acceleration,
    steering rate
    """
    def __init__(self, mass = 1.0, J=np.eye(3), g=[0,0,-1]):
        """
        Constructor that initializes the order
        """
        self.n = 12
        self.m = 4
        self.mass = mass
        self.J = J
        self.g = g

    def omegaToRpyDot(self, rpy, omega):
        secant_pitch= (1.0/np.cos(rpy[1]))
        c_roll = np.cos(rpy[0])
        s_roll = np.sqrt(1.0 - c_roll**2)
        tan_pitch = np.sqrt(secant_pitch**2 - 1)
        Momega_to_rpydot = np.array([[1, s_roll*tan_pitch, c_roll*tan_pitch],
                                    [0, c_roll,           -s_roll],
                                    [0, s_roll*secant_pitch, c_roll*secant_pitch]]);
        rpy_dot = np.dot(Momega_to_rpydot, omega)
        return rpy_dot

    def xdot(self, t, x, u, w):
        """
        X = [x, y, z, vx, vy, vz, r,p,y, omegax, omegay, omegaz]
        u = [thrust, taux, tauy, tauz]
        """
        tau = u[1:]
        thrust= u[0]
        e3 = np.array([0, 0, 1])
        v = x[3:6]
        rpy = x[6:9]
        R = euler2mat(rpy[0], rpy[1], rpy[2], 'rzyx')
        omega = x[9:12]
        J_omega_dot = np.cross(np.dot(self.J, omega), omega) + tau
        thrust_vec = np.dot(R, e3)
        acceleration = (1.0/self.mass)*(thrust_vec*thrust + self.g)
        omega_dot = np.linalg.solve(self.J, J_omega_dot)
        rpy_dot = self.omegaToRpyDot(rpy, omega)
        return np.hstack([v, acceleration, rpy_dot, omega_dot]) + w

    def jacobian(self, t, x, u, w):
        """
        Find jacobian of xdot wrt x, u, w
        """
        raise Exception('Jacobian not implemented yet')
