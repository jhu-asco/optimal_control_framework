#!/usr/bin/env python

from abc import ABCMeta, abstractmethod

class AbstractDynamicSystem(object):
  __metaclass__ = ABCMeta
  @abstractmethod
  def __init__(self, theta):
    """
    Constructor. Should define
    the size of state and control
    parameters. It also takes in
    the system parameters that
    are constant over the entire
    trajectory.
    """
    self.n = 1
    self.m = 1

  @abstractmethod
  def jacobian(t, x, u, wbar):
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
    pass

  @abstractmethod
  def xdot(t, x, u, w):
    """
    The dynamic function f_theta(t,x,u)
    that specifies the ODE: xdot = f_theta(t,x,u,p)
    Parameters:
      t   -- Time in seconds
      x   -- Current state
      u   -- Control
      w   -- Parameters that change along trajectory
    """
    pass
