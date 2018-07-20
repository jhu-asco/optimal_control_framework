#!/usr/bin/env python

from optimal_control_framework.sampling import SampleTrajectories
from optimal_control_framework.dynamics import LinearDynamics
from optimal_control_framework.controllers import FeedforwardController
from optimal_control_framework.integrators import EulerIntegrator
from optimal_control_framework.costs import LQRCost
import numpy as np
import numpy.testing as np_testing

def testSampling():
  n = 2
  m = 1
  M = 10
  tf = 1
  dt = 0.1
  ts = np.arange(0, tf+dt, dt)
  N = len(ts)-1
  A = np.array([[0, 1],[0, 0]])
  B = np.array([[0], [1]])
  Q = np.ones(n)
  Qf = 2*np.ones(n)
  R = np.ones(m)
  us = np.zeros([N, m])
  dynamics = LinearDynamics([A,B])
  controller = FeedforwardController(us)
  cost = LQRCost(N, Q, R, Qf)
  # No noise
  ws_sampling_fun = lambda *args : np.zeros(n)
  x0_sampling_fun = lambda *args : np.zeros(n)
  integrator = EulerIntegrator(dynamics)
  # Create sampler
  sample_trajectories = SampleTrajectories(dynamics, integrator,
                                           cost, ws_sampling_fun,
                                           x0_sampling_fun)
  # Sample M trajectories
  xss, uss, Jss = sample_trajectories.sample(M, ts, controller)
  assert(xss.shape == (M, N+1, n))
  assert(uss.shape == (M, N, m))
  assert(Jss.shape == (M, N))
  np_testing.assert_allclose(xss[0], xss[-1], atol=1e-8)
  np_testing.assert_allclose(Jss[0], Jss[-1], atol=1e-8)

