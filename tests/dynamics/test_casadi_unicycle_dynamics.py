#!/usr/bin/env python

from optimal_control_framework.dynamics import CasadiUnicycleDynamics
import unittest
import numpy as np
import numpy.testing as np_testing

class TestLinearDynamics(unittest.TestCase):
  def setUp(self):
    self.dynamics = CasadiUnicycleDynamics()

  def testProperties(self):
    self.assertEqual(self.dynamics.n, 3)
    self.assertEqual(self.dynamics.m, 2)

  def testXdot(self):
    x = np.zeros(3)
    u = np.array([2, 1])
    w = np.zeros(3)
    xdot = self.dynamics.xdot(1.0, x, u, w)
    self.assertEqual(len(xdot), 3)
    np_testing.assert_allclose(xdot, np.array([2, 0, 1]))

  def testJacobian(self):
    x = np.zeros(3)
    u = np.array([2, 0])
    w = np.zeros(3)
    jac = self.dynamics.jacobian(1.0, x, u, w)
    np_testing.assert_allclose(jac[0][:,0], np.zeros(3))
    np_testing.assert_allclose(jac[0][:,1], np.zeros(3))
    np_testing.assert_allclose(jac[0][:,2], np.array([0, 2, 0]))
    np_testing.assert_allclose(jac[1][:,0], np.array([1, 0, 0]))
    np_testing.assert_allclose(jac[1][:,1], np.array([0, 0, 1]))
    np_testing.assert_allclose(jac[2], np.eye(3))

if __name__ == "__main__":
    unittest.main()
