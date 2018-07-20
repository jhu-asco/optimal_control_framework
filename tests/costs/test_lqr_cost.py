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

  def test_cumulative_cost_different_terminal_state(self):
    """
    Regression Test to ensure the cost is using the last state for terminal cost!
    """
    xs = np.zeros([self.N+1, self.n])
    us = np.zeros([self.N, self.n])
    # Set the terminal state to ones
    xs[self.N] = np.ones([self.n])
    # Evaluate costs
    Js = self.cost.cumulative_cost(xs, us)
    # Last one is terminal cost
    self.assertEqual(Js[-1], 10.0)
    # Since other costs are 0
    self.assertTrue(Js[0] == Js[-1])
