#!/usr/bin/env python

from optimal_control_framework.dynamics import CasadiQuadrotorDynamics
from transforms3d.euler import euler2mat
import casadi as cs
import unittest
import numpy as np
import numpy.testing as np_testing

class TestCasadiQuadrotorDynamics(unittest.TestCase):
    def setUp(self):
        self.dynamics = CasadiQuadrotorDynamics()

    def testProperties(self):
        self.assertEqual(self.dynamics.n, 12)
        self.assertEqual(self.dynamics.m, 4)

    def testRotation(self):
        rpy = np.random.sample(3)
        R = euler2mat(rpy[2], rpy[1], rpy[0], axes='rzyx')
        R_cs = self.dynamics.euler2mat(cs.DM(rpy)).full()
        np_testing.assert_almost_equal(R, R_cs)

    def testOmegaToRpydot(self):
        omega = np.array([2, 0, 0])
        rpydot = self.dynamics.omegaToRpyDot(cs.DM(np.random.sample(3)),
                                             cs.DM(omega)).full().ravel()
        np_testing.assert_almost_equal(rpydot, omega)

    def testXdot(self):
        x = np.zeros(12)
        u = np.array([10, 0, 0, 0])
        w = np.zeros_like(x)
        xdot = self.dynamics.xdot(1.0, x, u, w)
        xdot_expected = np.zeros_like(xdot)
        xdot_expected[5] = 10 - 1 # Thrust - gravity
        self.assertEqual(len(xdot), 12)
        np_testing.assert_almost_equal(xdot, xdot_expected)

    def testJacobian(self):
        x = np.zeros(12)
        u = np.array([10, 0, 0, 0])
        w = np.zeros_like(x)
        jac = self.dynamics.jacobian(1.0, x, u, w)
        np_testing.assert_allclose(jac[2], np.eye(12))

if __name__ == "__main__":
    unittest.main()
