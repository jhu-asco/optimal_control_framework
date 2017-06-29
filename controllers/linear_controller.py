#!/usr/bin/env python

from optimal_control_framework.controllers import AbstractController
import numpy as np

class LinearController(AbstractController):
  def __init__(self, dynamics, Ks, ks, Sigmas):
    """
    Constructor that stores dynamics
    Parameters:
    dynamics -- Should be a subclass of
                AbstractDynamicSystem
    Ks       -- Feedback gains as a [Nxnxm] matrix
    ks       -- Feedforward controls as [Nxm] matrix
    Sigmas   -- Covariance at each stage as [Nxm] matrix
    """
    super(LinearController, self).__init__(dynamics)
    self.Ks = Ks
    self.ks = ks
    self.Sigmas = Sigmas

  def deterministic_control(self, i, x):
    """
    Compute the control to be sent based on inputs
    Parameters:
    i   -- Index along trajectory for discrete controllers
    t   -- Time along trajectory for continous controllers
    x   -- Current state
    """
    return np.matmul(x, self.Ks[i]) + self.ks[i]

  def control(self, i, x):
    noise = np.random.multivariate_normal(np.zeros(self.dynamics.m),
                                          np.diag(self.Sigmas[i]))
    return self.deterministic_control(i,x) + noise
