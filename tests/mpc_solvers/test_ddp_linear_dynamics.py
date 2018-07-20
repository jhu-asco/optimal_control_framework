#!/usr/bin/env python

from optimal_control_framework.dynamics import LinearDynamics
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
        R = self.dt*np.eye(self.m)
        Qf = 10*np.eye(self.n)
        xd = np.array([1, 0])
        self.cost = LQRCost(self.N, Q, R, Qf, xd)
        self.max_step = 0.1  # Allowed step for control

    def testConstructor(self):
        x0 = np.array([0, 0])
        us0 = np.zeros([self.N, self.m])
        ddp = Ddp(self.dynamics, self.cost, us0, x0, self.dt, self.max_step)
