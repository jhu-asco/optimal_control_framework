#!/usr/bin/env python

from abc import ABCMeta, abstractmethod
from optimal_control_framework.dynamics import AbstractDynamicSystem, AbstractCasadiSystem
import casadi as cs

class AbstractIntegrator(object):
    __metaclass__ = ABCMeta

    def __init__(self, dynamics):
        """
        Constructor that stores dynamics
        Parameters:
        dynamics -- Should be a subclass of
                    AbstractDynamicSystem
        """
        self.dynamics = dynamics
        self.n = self.dynamics.n
        self.m = self.dynamics.m
        assert(isinstance(self.dynamics, AbstractDynamicSystem))
  
    @abstractmethod
    def step(self, t, h, x, u, w):
        """
        Integrate the dynamics assuming constant control for h
        Parameters:
        t - Current time
        h - time step
        x - State
        u - Control
        w - noise
        Output:
        xout - State at (t + h)
        """
        pass

    @abstractmethod
    def jacobian(self, t, h, x, u, w):
        """
        Find the discrete jacobian of xout wrt x, u, w
        """
        pass

class AbstractCasadiIntegrator(AbstractIntegrator):
    __metaclass__ = ABCMeta
    def __init__(self, casadi_dynamics):
        """
        Constructor that stores dynamics
        """
        super(AbstractCasadiIntegrator, self).__init__(casadi_dynamics)
        assert(isinstance(self.dynamics, AbstractCasadiSystem))
        t = cs.MX.sym('t', 1)
        h = cs.MX.sym('h', 1)
        x = cs.MX.sym('x', self.n)
        u = cs.MX.sym('u', self.m)
        w = cs.MX.sym('w', self.n)
        self.xout_sym = self.step_sym(t, h, x, u, w)
        jac = [cs.jacobian(self.xout_sym, var) for var in [x, u, w]]
        self.jac_fcn = cs.Function('discrete_jacobian', [t, h, x, u, w], jac)
        self.xout_fcn = cs.Function('xout', [t, h, x, u, w], [self.xout_sym])

    @abstractmethod
    def step_sym(self, t, h, x, u, w):
        pass

    def step(self, t, h, x, u, w):
        return self.xout_fcn(t, h, x, u, w).full().ravel()

    def jacobian(self, t, h, x, u, w):
        return [C.full() for C in self.jac_fcn(t, h, x, u, w)]
