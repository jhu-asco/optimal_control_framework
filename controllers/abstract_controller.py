#!/usr/bin/env python

from abc import ABCMeta, abstractmethod
from dynamics.abstract_dynamics import AbstractDynamicSystem

class AbstractController(object):
  __metaclass__ = ABCMeta

  def __init__(self, dynamics):
    """
    Constructor that stores dynamics
    Parameters:
    dynamics -- Should be a subclass of
                AbstractDynamicSystem
    """
    self.dynamics = dynamics
    assert(isinstance(self.dynamics, AbstractDynamicSystem))

  def control(self, i, t, x):
    """
    Compute the control to be sent based on inputs and
    a noise sample (sampled internally)
    Parameters:
    i   -- Index along trajectory for discrete controllers
    t   -- Time along trajectory for continous controllers
    x   -- Current state
    """
    return self.deterministic_control(i,t,x)

  @abstractmethod
  def deterministic_control(self, i, t, x):
    """
    Compute a deterministic control purely based on inputs
    Parameters:
    i   -- Index along trajectory for discrete controllers
    t   -- Time along trajectory for continous controllers
    x   -- Current state
    """
    pass
