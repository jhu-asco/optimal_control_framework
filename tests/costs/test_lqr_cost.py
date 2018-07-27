#!/usr/bin/env python

from optimal_control_framework.costs import LQRCost
import unittest
import numpy as np
import numpy.testing as np_testing


class TestLQRCost(unittest.TestCase):
    def setUp(self):
        self.N = 5  # Length of traj
        self.n = 2  # lenght of state
        self.m = 1  # Length of control
        self.Q = np.ones(self.n)
        self.Qf = 5 * np.ones(self.n)
        self.R = 2 * np.ones(self.m)
        self.cost = LQRCost(self.N, self.Q, self.R, self.Qf)

    def test_stagewise_cost(self):
        self.assertEqual(self.cost.stagewise_cost(
            0, np.ones(self.n), np.ones(self.m)), 2)

    def test_terminal_cost(self):
        self.assertEqual(self.cost.terminal_cost(np.ones(self.n)), 5.0)

    def test_cumulative_cost(self):
        Js = self.cost.cumulative_cost(
            np.ones([self.N + 1, self.n]), np.ones([self.N, self.m]))
        self.assertEqual(len(Js), self.N)
        self.assertEqual(Js[-1], 7.0)
        self.assertTrue(Js[0] >= Js[-1])

    def test_cumulative_cost_different_terminal_state(self):
        """
        Regression Test to ensure the cost is using the last state for terminal cost!
        """
        xs = np.zeros([self.N + 1, self.n])
        us = np.zeros([self.N, self.n])
        # Set the terminal state to ones
        xs[self.N] = np.ones([self.n])
        # Evaluate costs
        Js = self.cost.cumulative_cost(xs, us)
        # Last one is terminal cost
        self.assertEqual(Js[-1], 5.0)
        # Since other costs are 0
        self.assertTrue(Js[0] == Js[-1])

    def test_stagewise_grads(self):
        x = np.random.sample(self.n)
        u = np.random.sample(self.m)
        L, jac, hess = self.cost.stagewise_cost(0, x, u, True)
        np_testing.assert_almost_equal(jac[0], self.Q * x)
        np_testing.assert_almost_equal(jac[1], self.R * u)
        np_testing.assert_almost_equal(np.diag(hess[0]), self.Q)
        np_testing.assert_almost_equal(np.diag(hess[1]), self.R)
        np_testing.assert_almost_equal(hess[2], 0)

    def test_terminal_grads(self):
        x = np.random.sample(self.n)
        Lf, Qfx, Qfxx = self.cost.terminal_cost(x, True)
        np_testing.assert_almost_equal(Qfx, self.Qf * x)
        np_testing.assert_almost_equal(np.diag(Qfxx), self.Qf)


if __name__ == "__main__":
    unittest.main()
