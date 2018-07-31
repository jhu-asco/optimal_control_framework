#!/usr/bin/env python

from optimal_control_framework.discrete_integrators import AbstractCasadiIntegrator
import casadi as cs

class SemiImplicitQuadIntegrator(AbstractCasadiIntegrator):
    def __init__(self, quad_dynamics):
        """
        Constructor that stores dynamics
        """
        super(SemiImplicitQuadIntegrator, self).__init__(quad_dynamics)

    def step_sym(self, t, h, x, u, w):
        f = self.dynamics.xdot_fcn(t, x, u, w)
        v = f[:3]
        a = f[3:6] 
        rpydot = f[6:9]
        omega_dot = f[9:12]

        p = x[:3]
        rpy = x[6:9]
        omega = x[9:12]

        vout = x[3:6] + h*a
        pout = p + (0.5*h)*(v + vout)
        omega_out = omega + h*omega_dot
        rpy_out = rpy + h*rpydot
        rpydot_out = self.dynamics.omegaToRpyDot(rpy_out, omega_out)
        rpy_out = rpy + 0.5*h*(rpydot + rpydot_out)
        xout = cs.vertcat(pout, vout, rpy_out, omega_out)
        return xout
