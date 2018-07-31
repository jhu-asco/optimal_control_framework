#!/usr/bin/env python

from optimal_control_framework.discrete_integrators import AbstractIntegrator
import numpy as np

class EulerIntegrator(AbstractIntegrator):

    def step(self, t, h, x, u, w):
        xout = x + h*self.dynamics.xdot(t, x, u, w)
        return xout

    def jacobian(self, t, h, x, u, w):
        fx, fu, fw = self.dynamics.jacobian(t, x, u, w)
        fx_out = np.eye(x.size) + h*fx
        fu_out = fu*h
        fw_out = fw*h
        return fx_out, fu_out, fw_out
