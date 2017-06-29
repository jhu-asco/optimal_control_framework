#!/usr/bin/env python

from optimal_control_framework.controllers import LinearController
from optimal_control_framework.dynamics import LinearDynamics
import unittest
import numpy as np
import numpy.testing as np_testing

class TestLinearController(unittest.TestCase):
  def setUp(self):
    self.n = 5
    self.m = 2
    N = 10
    A = np.ones([self.n, self.n])
    B = np.ones([self.n, self.m])
    self.dynamics = LinearDynamics([A,B])
    Ks = np.ones([N,self.n, self.m])
    ks = np.ones([N, self.m])
    Sigmas = 1e-3*np.ones([N, self.m])
    self.controller = LinearController(self.dynamics, Ks, ks, Sigmas)

  def testControl(self):
    x = np.ones(self.n)
    u = self.controller.deterministic_control(0, x)
    self.assertEqual(len(u), self.m)
    np_testing.assert_allclose(u, 6.0*np.ones(self.m),atol=1e-8)
    u_s = self.controller.control(0, x)
    np_testing.assert_allclose(u, u_s,atol=0.1)
