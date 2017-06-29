#!/usr/bin/env python

from optimal_control_framework.controllers import AbstractController
import numpy as np

class FeedforwardController(AbstractController):
  def __init__(self, us):
    """
    Constructor that stores dynamics
    Parameters:
    Ks       -- Feedback gains as a [Nxnxm] matrix
    ks       -- Feedforward controls as [Nxm] matrix
    """
    self.us = us
    self.discrete = True

  def deterministic_control(self, i, x):
    """
    Compute the control to be sent based on inputs
    Parameters:
    i   -- Index along trajectory for discrete controllers
    t   -- Time along trajectory for continous controllers
    x   -- Current state
    """
    return self.us[i]
