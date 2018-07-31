#!/usr/bin/env python

from optimal_control_framework.discrete_integrators import AbstractCasadiIntegrator
import numpy as np
import casadi as cs

class SemiImplicitCarIntegrator(AbstractCasadiIntegrator):
    def __init__(self, car_dynamics):
        """
        Constructor that stores dynamics
        """
        super(SemiImplicitCarIntegrator, self).__init__(car_dynamics)

    def step_sym(self, t, h, x, u, w):
        thetadot = u[1]
        v = u[0]
        theta_out = x[2] + thetadot*h + w[2]
        theta_average = 0.5*(theta_out + x[2])
        x_out = x[0] + h*v*cs.cos(theta_average) + w[0]
        y_out = x[1] + h*v*cs.sin(theta_average) + w[1]
        return cs.vertcat(x_out, y_out, theta_out)
