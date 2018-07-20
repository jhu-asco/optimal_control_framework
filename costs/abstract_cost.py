#!/usr/bin/env python

from abc import ABCMeta, abstractmethod
import numpy as np

class AbstractCost(object):
  __metaclass__ = ABCMeta

  def __init__(self, N):
    self.N = N

  @abstractmethod
  def stagewise_cost(self, i, x, u):
    """
    Stage wise cost L_i(x_i, u_i)
    returns a scalar cost
    """
    pass

  @abstractmethod
  def terminal_cost(self, xf):
    """
    Terminal cost L_n(x_n) where
    n refers to the length of the
    trajectory
    """
    pass
  def cumulative_cost(self, xs, us):
    """
    Creates cost to go matrix using array of states
    and controls
    """
    N = len(us)
    Js = np.empty([N,])
    cumulative_cost = self.terminal_cost(xs[N])

    for i in range(N-1,-1,-1):
      cumulative_cost += self.stagewise_cost(i, xs[i],us[i])
      Js[i] = cumulative_cost
    return Js
