#!/usr/bin/env python

from abc import ABCMeta, abstractmethod
from dynamics.abstract_dynamics import AbstractDynamicSystem

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
    assert(isinstance(self.dynamics, AbstractDynamicSystem))
  
  @abstractmethod
  def integrate(self, x0, ts, controller, ws_sample_fun):
    """
    Integrate the dynamics and provide the states at ts
    Parameters:
    x0            -- Initial State
    ts            -- array of times where the controls are applied
    controller    -- controller applied to the system
    ws_sample_fun -- Sampling function for time varying parameters
    Output:
    xs       -- Output states at the times specified as an array
    us       -- Controls used at the times
    """
    pass
