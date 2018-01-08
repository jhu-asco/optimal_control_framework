#!/usr/bin/env python2
from optimal_control_framework.dynamics import AbstractDynamicSystem
import numpy as np

class AccUnicycleDynamics(AbstractDynamicSystem):
    """
    Zero order dynamics with n = m = 1
    """
    def __init__(self):
        """
        Constructor that initializes the order
        """
        self.n = 4
        self.m = 2

    def xdot(self, t, x, u, w):
        """
        Basic dynamics of xdot = u + w
        X = [x, y, theta, v]
        U = [a, thetadot]
        """
        a= u[0]
        thetadot = u[1]
        theta = x[2]
        v = x[3]
        return np.array([v*np.cos(theta), v*np.sin(theta), thetadot, a]) + w

    def jacobian(self, t, x, u, w):
        """
        Find jacobian of xdot wrt x, u, w
        """
        a= u[0]
        theta = x[2]
        v = x[3]
        fx = np.array([[0, 0, 0, 0],
                       [0, 0, 0, 0],
                       [-v*np.sin(theta), v*np.cos(theta), 0, 0],
                       [np.cos(theta), np.sin(theta), 0, 0]])
        fu = np.array([[0, 0, 0, 1],
                       [0, 0, 1, 0]])
        fw = np.eye(4)
        return [fx, fu, fw]



