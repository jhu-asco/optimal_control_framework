#!/usr/bin/env python2
from optimal_control_framework.dynamics import AbstractDynamicSystem
import numpy as np

class SimpleCarDynamics(AbstractDynamicSystem):
    """
    Simple car dynamics with controls as acceleration,
    steering rate
    """
    def __init__(self, L = 1.0):
        """
        Constructor that initializes the order
        """
        self.n = 5
        self.m = 2
        self.L = L

    def xdot(self, t, x, u, w):
        """
        x = [x, y, theta, v, phi]
        Basic dynamics of xdot = [v c_theta, v sin_theta, v tan_phi/L, a, phidot]
        u = [a, phidot]
        """
        a= u[0]
        phidot = u[1]
        theta = x[2]
        v = x[3]
        phi = x[4]
        L = self.L
        return np.array([v*np.cos(theta), v*np.sin(theta), v*np.tan(phi)/L, a, phidot]) + w

    def jacobian(self, t, x, u, w):
        """
        Find jacobian of xdot wrt x, u, w
        """
        a= u[0]
        phidot = u[1]
        theta = x[2]
        v = x[3]
        phi = x[4]
        L = self.L
        fx = np.array([[0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0],
                       [-v*np.sin(theta), v*np.cos(theta), 0, 0, 0],
                       [np.cos(theta), np.sin(theta), np.tan(phi)/L, 0, 0],
                       [0, 0, v/(L*(np.cos(phi)**2)), 0, 0]]).T
        fu = np.array([[0, 0, 0, 1, 0],
                       [0, 0, 0, 0, 1]]).T
        fw = np.eye(5)
        return [fx, fu, fw]



