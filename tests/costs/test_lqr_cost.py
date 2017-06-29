#!/usr/bin/env python

from optimal_control_framework.costs import LQRCost
import unittest
import numpy as np

class TestLQRCost(unittest.TestCase):
  def setUp(self):
    self.N = 5  # Length of traj
    self.n = 2  # lenght of state
    self.m = 1  # Length of control
    self.Q = np.ones(self.n)
    self.Qf = 5*np.ones(self.n)
    self.R = 2*np.ones(self.m)
    self.cost = LQRCost(self.N, self.Q, self.R, self.Qf)

  def test_stagewise_cost(self):
    self.assertEqual(self.cost.stagewise_cost(0, np.ones(self.n), np.ones(self.m)),4)

  def test_terminal_cost(self):
    self.assertEqual(self.cost.terminal_cost(np.ones(self.n)),10.0)

  def test_cumulative_cost(self):
    Js = self.cost.cumulative_cost(np.ones([self.N+1, self.n]), np.ones([self.N, self.m]))
    self.assertEqual(len(Js), self.N)
    self.assertEqual(Js[-1], 14.0)
    self.assertTrue(Js[0] >= Js[-1])
