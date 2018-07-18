#!/usr/bin/env python2
from optimal_control_framework.dynamics import AbstractDynamicSystem
import numpy as np


class UnicycleDynamics(AbstractDynamicSystem):
    """
    Zero order dynamics with n = m = 1
    """

    def __init__(self):
        """
        Constructor that initializes the order
        """
        self.n = 3
        self.m = 2

    def xdot(self, t, x, u, w):
        """
        Basic dynamics of xdot = u + w
        """
        v = u[0]
        thetadot = u[1]
        theta = x[2]
        return np.array([v * np.cos(theta), v * np.sin(theta), thetadot]) + w

    def jacobian(self, t, x, u, w):
        """
        Find jacobian of xdot wrt x, u, w
        """
        v = u[0]
        theta = x[2]
        c_theta = np.cos(theta)
        s_theta = np.sin(theta)
        fx = np.array([[0, 0, 0],
                       [0, 0, 0],
                       [-v * s_theta, v * c_theta, 0]]).T
        fu = np.array([[c_theta, s_theta, 0],
                       [0, 0, 1]]).T
        fw = np.eye(3)
        return [fx, fu, fw]
