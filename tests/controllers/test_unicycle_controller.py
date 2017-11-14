#!/usr/bin/env python

from optimal_control_framework.controllers import UnicycleController
from optimal_control_framework.dynamics import UnicycleDynamics
import unittest
import numpy as np
import numpy.testing as np_testing

class TestUnicycleController(unittest.TestCase):
  def setUp(self):
    self.unicycle_dynamics = UnicycleDynamics()
    self.n = self.unicycle_dynamics.n
    self.m = self.unicycle_dynamics.m
    self.controller = UnicycleController(self.unicycle_dynamics)

  def testControl(self):
    x0 = -1*np.ones(self.n)
    max_iters = 1000
    iters = 0
    converged = False
    x = x0
    dt = 0.1
    t = 0
    while iters < max_iters:
        iters = iters+1
        u = self.controller.control(t, x)
        xdot = self.unicycle_dynamics.xdot(t, x, u, np.zeros(self.n))
        x = x + xdot*dt
        t = t + dt
        error = np.linalg.norm(x[:2])
        if error < 1e-2:
            converged = True
            break
    self.assertTrue(converged)
