#!/usr/bin/env python

from optimal_control_framework.integrators import AbstractIntegrator
from optimal_control_framework.costs import AbstractCost
from optimal_control_framework.dynamics import AbstractDynamicSystem
import numpy as np
import matplotlib.pyplot as plt

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
    self.plot_init = False

  @classmethod
  def plot_vector(self, ts, vector_inp, figure_id, name):
    """
    Vector input should be [Nxn]
    """
    ncol = 2
    n = vector_inp.shape[1]
    nrow = int(n/ncol)
    if nrow*ncol != n:
      nrow += 1
    # Plot vector
    plt.figure(figure_id)
    for i in range(0, n):
      plt.subplot(nrow, ncol, i+1)
      plt.plot(ts, vector_inp[:, i], 'b')
      plt.ylabel(name+'_'+str(i))
      ax = plt.gca()
      ax.set_xticklabels([])

  def plot_sample(self, ts, xs, us, Js):
      self.plot_vector(ts, xs, 1,'xs')
      self.plot_vector(ts[:-1], us, 2,'us')
      plt.figure(3)
      plt.plot(ts[:-1], Js,'b')

  def sample(self, M, ts, controller, plot_samples=False):
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
      if plot_samples is True:
          self.plot_sample(ts, xss[i], uss[i], Jss[i])
    return [xss, uss, Jss]
