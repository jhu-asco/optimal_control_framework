#!/usr/bin/env python

from optimal_control_framework.integrators import AbstractIntegrator
from optimal_control_framework.costs import AbstractCost
from optimal_control_framework.dynamics import AbstractDynamicSystem
import numpy as np

"""
The Goal here is to sample a bunch of system trajectories and their
costs given controller, integrator, dynamics, sampling function for noise,
and cost classes. The result of the sampling is the state matrix,
control matrix, cumulative Cost matrix.
"""

class SampleTrajectories:
  def __init__(self, dynamics, integrator, cost, ws_sampling_fun, x0_sampling_fun):
    """
    Store static classes used to sample
    """
    assert(isinstance(dynamics, AbstractDynamicSystem))
    assert(isinstance(integrator, AbstractIntegrator))
    assert(isinstance(cost, AbstractCost))
    self.dynamics = dynamics
    self.integrator = integrator
    self.cost = cost
    self.ws_sampling_fun = ws_sampling_fun
    self.x0_sampling_fun = x0_sampling_fun

  def sample(self, M, ts, controller):
    # Define storage structures:
    N = ts.size-1
    xss = np.empty([M, N+1, self.dynamics.n])
    uss = np.empty([M, N, self.dynamics.m])
    Jss = np.empty([M, N])
    for i in range(M):
      # Sample initial condition
      x0 = self.x0_sampling_fun()
      # Integrate
      xss[i], uss[i] = self.integrator.integrate(x0, ts, controller, self.ws_sampling_fun)
      Jss[i] = self.cost.cumulative_cost(xss[i], uss[i])
    return [xss, uss, Jss]
