#!/usr/bin/env python

from optimal_control_framework.dynamics import LinearDynamics, UnicycleDynamics
from optimal_control_framework.mpc_solvers import Ddp
from optimal_control_framework.costs import LQRCost
import unittest
import numpy as np
import numpy.testing as np_testing

class TestDDPLinearDynamics(unittest.TestCase):
    def setUp(self):
        self.n = 2
        self.m = 1
        A = np.zeros([self.n, self.n])
        A[0, 1] = 1
        B = np.zeros([self.n, self.m])
        B[1, 0] = 1
        self.dynamics = LinearDynamics([A,B])
        # Trajectory info
        self.dt = 0.1
        self.N = 10
        Q = self.dt*np.zeros(self.n)
        R = 0.01*self.dt*np.eye(self.m)
        Qf = 100*np.eye(self.n)
        self.xd = np.array([1, 0])
        self.cost = LQRCost(self.N, Q, R, Qf, self.xd)
        self.max_step = 0.5  # Allowed step for control

    def testConstructor(self):
        x0 = np.array([0, 0])
        us0 = np.zeros([self.N, self.m])
        ddp = Ddp(self.dynamics, self.cost, us0, x0, self.dt, self.max_step)
        self.assertGreater(ddp.V, 0)
        self.assertLess(ddp.V, np.Inf)

    def testOptimization(self):
        x0 = np.array([0, 0])
        us0 = np.zeros([self.N, self.m])
        ddp =Ddp(self.dynamics, self.cost, us0, x0, self.dt, self.max_step)
        V = ddp.V
        for i in range(50):
            ddp.iterate()
            self.assertLessEqual(ddp.V, V)
            V = ddp.V
            # print("xn: ", ddp.xs[-1])
        np_testing.assert_almost_equal(ddp.xs[-1], self.xd, decimal=2)

class TestDDPUnicycleDynamics(unittest.TestCase):
    def setUp(self):
        self.dynamics = UnicycleDynamics()
        # Trajectory info
        self.dt = 0.05
        self.N = 20
        Q = self.dt*np.zeros(self.dynamics.n)
        R = 0.01*self.dt*np.eye(self.dynamics.m)
        Qf = 100*np.eye(self.dynamics.n)
        self.xd = np.array([1, 0.5, np.pi/2])
        self.cost = LQRCost(self.N, Q, R, Qf, self.xd)
        self.max_step = 0.5  # Allowed step for control

    def testConstructor(self):
        x0 = np.array([0, 0, 0])
        us0 = np.zeros([self.N, self.dynamics.m])
        ddp = Ddp(self.dynamics, self.cost, us0, x0, self.dt, self.max_step)
        self.assertGreater(ddp.V, 0)
        self.assertLess(ddp.V, np.Inf)

    def testOptimization(self):
        x0 = np.array([0, 0, 0])
        us0 = np.zeros([self.N, self.dynamics.m])
        ddp =Ddp(self.dynamics, self.cost, us0, x0, self.dt, self.max_step)
        V = ddp.V
        for i in range(50):
            ddp.iterate()
            self.assertLessEqual(ddp.V, V)
            V = ddp.V
            # print("xn: ", ddp.xs[-1])
        np_testing.assert_almost_equal(ddp.xs[-1], self.xd, decimal=2)

if __name__ == "__main__":
    unittest.main()
