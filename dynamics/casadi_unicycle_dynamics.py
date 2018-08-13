#!/usr/bin/env python2
from optimal_control_framework.dynamics import AbstractCasadiSystem
import casadi as cs

class CasadiUnicycleDynamics(AbstractCasadiSystem):
    """
    Zero order dynamics with n = m = 1
    """

    def __init__(self, use_nonlinear_noise_model=False, scale=[1,1,1]):
        """
        Constructor that initializes the order
        """
        self.use_nonlinear_noise_model=use_nonlinear_noise_model
        self.scale = scale
        super(CasadiUnicycleDynamics, self).__init__(3, 2)

    def casadi_ode(self, t, x, u, w):
        """
        Basic dynamics of xdot for unicycle
        """
        v = u[0]
        thetadot = u[1]
        theta = x[2]
        f = cs.vertcat(v * cs.cos(theta), v * cs.sin(theta), thetadot)
        if self.use_nonlinear_noise_model:
            w_vec = cs.vertcat(self.scale[0]*(cs.cos(theta) * w[0] - cs.sin(theta) * w[1]),
                               self.scale[1]*(cs.sin(theta) * w[0] + cs.cos(theta) * w[1]),
                               self.scale[2]*(v * w[2]))
        else:
            w_vec = w
        return f + w_vec
