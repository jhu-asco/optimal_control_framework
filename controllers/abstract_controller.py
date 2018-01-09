#!/usr/bin/env python

from abc import ABCMeta, abstractmethod
from optimal_control_framework.dynamics import AbstractDynamicSystem

class AbstractController(object):
  __metaclass__ = ABCMeta

  def __init__(self, dynamics, discrete=True):
    """
    Constructor that stores dynamics
    Parameters:
    dynamics -- Should be a subclass of
                AbstractDynamicSystem
    discrete -- If the controller is discrete or continous
    """
    self.dynamics = dynamics
    self.discrete = discrete
    self.m = self.dynamics.m
    assert(isinstance(dynamics, AbstractDynamicSystem))

  def control(self, i, x):
    """
    Compute the control to be sent based on inputs and
    a noise sample (sampled internally)
    Parameters:
    i   -- Index along trajectory for discrete controllers
           It can also be  time along trajectory for continous
           controllers
    x   -- Current state
    """
    return self.deterministic_control(i,x)

  @abstractmethod
  def deterministic_control(self, i, x):
    """
    Compute a deterministic control purely based on inputs
    Parameters:
    i   -- Index along trajectory for discrete controllers
           It can also be  time along trajectory for continous
           controllers
    x   -- Current state
    """
    pass
