#!/usr/bin/env python

from abc import ABCMeta, abstractmethod
from abstract_dynamics import AbstractDynamicSystem
import casadi as cs

class AbstractCasadiSystem(AbstractDynamicSystem):
  __metaclass__ = ABCMeta

  def __init__(self, n, m):
      self.n = n
      self.m = m
      x = cs.MX.sym('x', n)
      u = cs.MX.sym('u', m)
      w = cs.MX.sym('w', n)
      t = cs.MX.sym('t', 1)
      self.xdot_sym = self.casadi_ode(t, x, u, w)
      jac = [cs.jacobian(self.xdot_sym, var) for var in [x, u, w]]
      self.jac_fcn = cs.Function('jacobian', [t, x, u, w], jac)
      self.xdot_fcn = cs.Function('xdot', [t, x, u, w], [self.xdot_sym])

  @abstractmethod
  def casadi_ode(self, t, x, u, w):
      """
      Assume inputs are casadi entitites,
      find xdot
      """
      pass

  def jacobian(self, t, x, u, wbar):
      """
      Return jacobian for the dynamics
      function. The ODE: xdot = f_theta(t,x,u,w)
      Parameters:
        t    -- Time in seconds
        x    -- Current state
        u    -- Control
        wbar -- Mean Parameters that change along trajectory
      Returns:
        [f_x, f_u, f_w]
      """
      return [C.full() for C in self.jac_fcn(t, x, u, wbar)]

  def xdot(self, t, x, u, w):
      """
      The dynamic function f_theta(t,x,u)
      that specifies the ODE: xdot = f_theta(t,x,u,p)
      Parameters:
        t   -- Time in seconds
        x   -- Current state
        u   -- Control
        w   -- Parameters that change along trajectory
      """
      return self.xdot_fcn(t, x, u, w).full().ravel()
