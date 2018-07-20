#!/usr/bin/env python

from optimal_control_framework.controllers import UnicycleController
from optimal_control_framework.dynamics import UnicycleDynamics
import unittest
import numpy as np
import numpy.testing as np_testing
import matplotlib.pyplot as plt

class TestUnicycleController(unittest.TestCase):
  def setUp(self):
    self.unicycle_dynamics = UnicycleDynamics()
    self.n = self.unicycle_dynamics.n
    self.m = self.unicycle_dynamics.m
    self.controller = UnicycleController(self.unicycle_dynamics)
    self.plot = True

  def testControl(self):
    #x0 = -1*np.ones(self.n)
    x0 = 2*np.ones(self.n)
    x0[2] = 1
    max_iters = 1000
    iters = 0
    converged = False
    x = x0
    dt = 0.1
    t = 0
    xs = []
    ts = []
    while iters < max_iters:
        iters = iters+1
        u = self.controller.control(t, x)
        xdot = self.unicycle_dynamics.xdot(t, x, u, np.zeros(self.n))
        x = x + xdot*dt
        t = t + dt
        xs.append(x)
        ts.append(t)
        error = np.linalg.norm(x[:2])
        if error < 1e-2:
            converged = True
            break
    xs = np.vstack(xs)
    ts = np.array(ts)
    if self.plot:
        plt.figure(1)
        plt.subplot(2,1,1)
        plt.plot(xs[:,0], xs[:,1], 'r')
        plt.plot(0, 0, 'b*')
        plt.subplot(2,1,2)
        theta = np.remainder(xs[:,2], 2*np.pi)
        inds_theta_greater = theta > np.pi
        inds_theta_smaller = theta < -np.pi
        theta[inds_theta_greater] = theta[inds_theta_greater] - 2*np.pi
        theta[inds_theta_smaller] = theta[inds_theta_smaller] + 2*np.pi
        plt.plot(ts, theta, 'r')
        theta_d = np.arctan2(-xs[:,1], -xs[:,0])
        plt.plot(ts, theta_d, 'b')
        plt.show(block=True)
    self.assertTrue(converged)
