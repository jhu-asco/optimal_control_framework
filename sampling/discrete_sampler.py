#!/usr/bin/env python

from optimal_control_framework.discrete_integrators import AbstractIntegrator
from optimal_control_framework.costs import AbstractCost
from optimal_control_framework.dynamics import AbstractDynamicSystem
import numpy as np


"""
The Goal here is to sample a bunch of system trajectories and their
costs given controller, integrator, dynamics, sampling function for noise,
and cost classes. The result of the sampling is the state matrix,
control matrix, cumulative Cost matrix.
"""

class DiscreteSampleTrajectories:
  def default_ws_sampling_fun(self):
    return 0

  def default_x0_sampling_fun(self):
    return np.zeros(self.dynamics.n)

  def isColliding(self, obs_list, xs):
      for x in xs:
          for obs in obs_list:
              distance,_ = obs.distance(x)
              if distance < 0:
                  return True
      return False

  def __init__(self, dynamics, discrete_integrator, cost=None, ws_sampling_fun=None, x0_sampling_fun=None):
    """
    Store static classes used to sample
    """
    assert(isinstance(dynamics, AbstractDynamicSystem))
    assert(isinstance(discrete_integrator, AbstractIntegrator))
    if cost is not None:
        assert(isinstance(cost, AbstractCost))
    self.dynamics = dynamics
    self.integrator = discrete_integrator
    self.cost = cost
    if ws_sampling_fun is None:
        ws_sampling_fun = self.default_ws_sampling_fun
    if x0_sampling_fun is None:
        x0_sampling_fun = self.default_x0_sampling_fun
    self.ws_sampling_fun = ws_sampling_fun
    self.x0_sampling_fun = x0_sampling_fun

  def sample(self, M, ts, controller, x0=None):
    # Define storage structures:
    N = ts.size-1
    xss = np.empty([M, N+1, self.dynamics.n])
    uss = np.empty([M, N, self.dynamics.m])
    Jss = np.empty([M, N])
    for i in range(M):
      # Sample initial condition
      if x0 is None:
        xss[i, 0, :] = self.x0_sampling_fun()
      else:
        xss[i, 0, :] = x0
      # Integrate
      for j in range(N):
        uss[i, j, :] = controller.control(j, xss[i][j])
        dt = ts[j+1] - ts[j]
        w = self.ws_sampling_fun()
        xss[i, j+1, :] = self.integrator.step(i, dt, xss[i][j], uss[i][j], w)
      if self.cost is not None:
        Jss[i] = self.cost.cumulative_cost(xss[i], uss[i])
    return [xss, uss, Jss]
