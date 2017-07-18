#!/usr/bin/env python

from optimal_control_framework.controllers import AbstractController
import numpy as np

class FeedforwardController(AbstractController):
  def __init__(self, us, Sigmaus=None):
    """
    Constructor that stores dynamics
    Parameters:
    us       -- Feedforward controls as [Nxm] matrix
    Sigmaus  -- Covariance of the controls
    """
    self.us = us
    self.discrete = True
    self.Sigmaus = Sigmaus
    if len(us.shape) == 1:
        self.m = 1
    else:
        self.m = us.shape[1]

  def deterministic_control(self, i, x):
    """
    Compute the control to be sent based on inputs
    Parameters:
    i   -- Index along trajectory for discrete controllers
    t   -- Time along trajectory for continous controllers
    x   -- Current state
    """
    return self.us[i]

  def control(self, i, x):
    if self.Sigmaus is None:
      noise = 0
    else:
      noise = np.random.multivariate_normal(np.zeros(self.m),
                                            np.diag(self.Sigmaus[i]))
    return self.deterministic_control(i,x) + noise
