#!/usr/bin/env python

from abstract_controller import AbstractController
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

  def deterministic_control(self, i, t, x):
    """
    Compute the control to be sent based on inputs
    Parameters:
    i   -- Index along trajectory for discrete controllers
    t   -- Time along trajectory for continous controllers
    x   -- Current state
    """
    return self.us[i]
