#!/usr/bin/env python

from optimal_control_framework.costs import LQRObstacleCost, SphericalObstacle
from scipy import optimize
import unittest
import numpy as np
import numpy.testing as np_testing

class TestLQRObstacleCost(unittest.TestCase):
  def setUp(self):
    self.N = 5  # Length of traj
    self.n = 2  # lenght of state
    self.m = 1  # Length of control
    self.Q = np.ones(self.n)
    self.Qf = 5*np.ones(self.n)
    self.R = 2*np.ones(self.m)
    obs1 = SphericalObstacle(np.array([1, 1]), 0.1)
    obs2 = SphericalObstacle(np.array([2, 2]), 0.2)
    self.cost = LQRObstacleCost(self.N, self.Q, self.R, self.Qf,
                                obstacles = [obs1, obs2], ko=2)
    self.obstacles = [obs1, obs2]

  def test_stagewise_cost(self):
    x = np.array([0.95, 0.95])
    cost = 0.5*(2*(0.95**2) + 2)
    obs1_cost = (self.obstacles[0].distance(x)[0]**2)
    self.assertEqual(self.cost.stagewise_cost(0, x, np.ones(self.m)),
                     cost + obs1_cost)

  def test_terminal_cost(self):
    obs1_cost = 0.5*self.cost.ko*(0.01)
    self.assertEqual(self.cost.terminal_cost(np.ones(self.n)), 5.0 + obs1_cost)

  def test_stagewise_grads(self):
    x = np.random.sample(self.n)
    u = np.random.sample(self.m)
    L, jac, hess = self.cost.stagewise_cost(0, x, u, True)
    fx = lambda x1: (self.cost.stagewise_cost(0, x1, u, True)[0])
    fu = lambda u1: (self.cost.stagewise_cost(0, x, u1, True)[0])
    fx_approx = optimize.approx_fprime(x, fx, 1e-6)
    fu_approx = optimize.approx_fprime(u, fu, 1e-6)
    np_testing.assert_almost_equal(jac[0], fx_approx, decimal=4)
    np_testing.assert_almost_equal(jac[1], fu_approx, decimal=4)

  def test_terminal_grads(self):
    x = np.random.sample(self.n)
    Lf, Qfx,Qfxx = self.cost.terminal_cost(x, True)
    fx = lambda x: self.cost.terminal_cost(x, True)[0]
    fx_approx = optimize.approx_fprime(x, fx, 1e-6)
    np_testing.assert_almost_equal(Qfx, fx_approx, decimal=4)

if __name__=="__main__":
  unittest.main()
