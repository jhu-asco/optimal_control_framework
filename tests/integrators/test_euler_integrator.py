#!/usr/bin/env python

from dynamics.linear_dynamics import LinearDynamics
from controllers.feedforward_controller import FeedforwardController
from integrators.euler_integrator import EulerIntegrator
import numpy as np
import numpy.testing as np_testing


def testIntegration():
  n = 2
  m = 1
  tf = 1
  dt = 0.1
  ts = np.arange(0, tf+dt, dt)
  N = len(ts)-1
  A = np.array([[0, 1],[0, 0]])
  B = np.array([[0], [1]])
  x0 = np.array([0,1])
  us = np.zeros([N, m])
  dynamics = LinearDynamics([A,B])
  controller = FeedforwardController(us)
  # No noise
  ws_sample_fun = lambda *args : np.zeros(n)
  integrator = EulerIntegrator(dynamics)
  xs, us_out = integrator.integrate(x0, ts, controller, ws_sample_fun)
  np_testing.assert_allclose(us_out, us,atol=1e-8)
  np_testing.assert_allclose(xs[-1], np.array([1, 1]), atol=1e-8)
