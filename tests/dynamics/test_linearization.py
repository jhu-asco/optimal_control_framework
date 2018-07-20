#!/usr/bin/env python

from optimal_control_framework.dynamics import (LinearDynamics,
                                                LinearizeDynamics)
from optimal_control_framework.controllers import FeedforwardController
import numpy as np
import numpy.testing as np_testing

def testLinearization():
  n = 5
  m = 2
  A = np.ones([n, n])
  B = np.ones([n, m])
  x0 = np.array([0,0,0,0,1])
  ts = np.arange(0, 1.1, 0.1)
  N = len(ts) - 1
  us = np.zeros([N, m])
  dynamics = LinearDynamics([A,B])
  controller = FeedforwardController(us)
  linearize_dynamics = LinearizeDynamics(dynamics)
  wbar_sample_fun = lambda *args : np.zeros(n)
  As, Bs, Cs = linearize_dynamics.linearize(ts, controller, x0, wbar_sample_fun)
  np_testing.assert_allclose(As[0], np.eye(n) + 0.1*A, atol=1e-8)
  np_testing.assert_allclose(Bs[0], 0.1*B, atol=1e-8)
  np_testing.assert_allclose(Cs[0], 0.1*np.eye(n), atol=1e-8)
