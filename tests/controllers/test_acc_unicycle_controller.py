#!/usr/bin/env python

from optimal_control_framework.controllers import AccUnicycleController
from optimal_control_framework.dynamics import AccUnicycleDynamics
import unittest
import numpy as np
import numpy.testing as np_testing
import matplotlib.pyplot as plt

class TestAccUnicycleController(unittest.TestCase):
  def setUp(self):
    self.unicycle_dynamics = AccUnicycleDynamics()
    self.n = self.unicycle_dynamics.n
    self.m = self.unicycle_dynamics.m
    self.controller = AccUnicycleController(self.unicycle_dynamics)
    self.plot = True
  
  def testGetVelocity(self):
    x = np.array([1, 1, 1, 1])
    # xdot, ydot
    pdot = self.controller.getVelocity(x)
    self.assertAlmostEqual(pdot[0], x[3]*np.cos(x[2]))
    self.assertAlmostEqual(pdot[1], x[3]*np.sin(x[2]))

  def testControlValue(self):
    # x, y, theta, v
    x = np.array([1, 1, -1, 3])
    self.controller.setGoal(np.array([1, 0, 0, -2]))
    # Check double derivative
    dt = 0.0001
    u = self.controller.control(0, x)
    x1 = x + self.unicycle_dynamics.xdot(0, x, u, np.zeros(self.n))*dt
    u = self.controller.control(0, x1)
    x2 = x1 + dt*self.unicycle_dynamics.xdot(0, x1, u, np.zeros(self.n))
    xddot_1 = (x2 + x - 2*x1)/(dt*dt)
    # Check xddot , yddot from discrete differentiation
    p = x1[:2]
    pdot = self.controller.getVelocity(x1)
    pd = self.controller._xd[:2]
    pd_dot = self.controller.getVelocity(self.controller._xd)
    pddot_expected = -1*self.controller._gains[0]*(p-pd) -1*self.controller._gains[1]*(pdot - pd_dot)
    np_testing.assert_allclose(pddot_expected, xddot_1[:2], atol=1e-3)


  def testControl(self):
    self.controller.setGains(np.array([5, 5]))
    #x0 = -1*np.ones(self.n)
    x0 = -1*np.ones(self.n)
    x0[2] = 1
    x0[3] = 1
    xd = np.array([0, 0, 0, 0.1])
    u_ff = np.array([0, 0])
    self.controller.setGoal(xd)
    max_iters = 1000
    iters = 0
    converged = False
    x = x0
    dt = 0.01
    t = 0
    xd_s = []
    xs = []
    ts = []
    while iters < max_iters:
        iters = iters+1
        u = u_ff + self.controller.control(t, x)
        xdot = self.unicycle_dynamics.xdot(t, x, u, np.zeros(self.n))
        ud = u_ff + self.controller.control(t, xd)
        xd_dot =self.unicycle_dynamics.xdot(t, xd, ud, np.zeros(self.n))
        x = x + xdot*dt
        xd = xd + xd_dot*dt
        t = t + dt
        self.controller.setGoal(xd)
        xd_s.append(xd)
        xs.append(x)
        ts.append(t)
        error = np.linalg.norm(x-xd)
        if error < 1e-2:
            converged = True
            break
    xd_s = np.vstack(xd_s)
    xs = np.vstack(xs)
    ts = np.array(ts)
    if self.plot:
        plt.figure(1)
        plt.subplot(2,1,1)
        plt.plot(xs[:,0], xs[:,1], 'r')
        plt.plot(xd_s[:,0], xd_s[:,1], 'b*-')
        plt.subplot(2,2,3)
        theta = np.remainder(xs[:,2], 2*np.pi)
        inds_theta_greater = theta > np.pi
        inds_theta_smaller = theta < -np.pi
        theta[inds_theta_greater] = theta[inds_theta_greater] - 2*np.pi
        theta[inds_theta_smaller] = theta[inds_theta_smaller] + 2*np.pi
        plt.plot(ts, theta, 'r')
        theta_d = self.controller._xd[2]*np.ones_like(ts)
        plt.plot(ts, theta_d, 'b')
        plt.subplot(2,2,4)
        vd = np.ones_like(ts)*self.controller._xd[3]
        plt.plot(ts, xs[:,3], 'r')
        plt.plot(ts, vd, 'b')
        plt.show(block=True)
    self.assertTrue(converged)
