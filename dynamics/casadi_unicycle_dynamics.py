#!/usr/bin/env python2
from optimal_control_framework.dynamics import AbstractCasadiSystem
import casadi as cs

class CasadiUnicycleDynamics(AbstractCasadiSystem):
    """
    Zero order dynamics with n = m = 1
    """

    def __init__(self):
        """
        Constructor that initializes the order
        """
        super(CasadiUnicycleDynamics, self).__init__(3, 2)

    def casadi_ode(self, t, x, u, w):
        """
        Basic dynamics of xdot for unicycle
        """
        v = u[0]
        thetadot = u[1]
        theta = x[2]
        return cs.vertcat(v * cs.cos(theta), v * cs.sin(theta), thetadot) + w
