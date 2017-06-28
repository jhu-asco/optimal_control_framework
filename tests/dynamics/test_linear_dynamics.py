#!/usr/bin/env python

from dynamics.linear_dynamics import LinearDynamics
import unittest
import numpy as np
import numpy.testing as np_testing

class TestLinearDynamics(unittest.TestCase):
  def setUp(self):
    self.n = 5
    self.m = 2
    A = np.ones([self.n, self.n])
    B = np.ones([self.n, self.m])
    self.dynamics = LinearDynamics([A,B])
  def testProperties(self):
    self.assertEqual(self.dynamics.n, self.n)
    self.assertEqual(self.dynamics.m, self.m)

  def testXdot(self):
    x = np.ones(self.n)
    u = np.ones(self.m)
    w = np.zeros(self.n)
    xdot = self.dynamics.xdot(1.0, x, u, w)
    # print xdot
    self.assertEqual(len(xdot), self.n)
    np_testing.assert_allclose(xdot, 7.0*np.ones(self.n),atol=1e-8)

  def testJacobian(self):
    x = np.ones(self.n)
    u = np.ones(self.m)
    w = np.zeros(self.n)
    jac = self.dynamics.jacobian(1.0, x, u, w)
    self.assertEqual(jac[0].shape, (self.n, self.n))
    self.assertEqual(jac[1].shape, (self.n, self.m))
    self.assertEqual(jac[2].shape, (self.n, self.n))
    
