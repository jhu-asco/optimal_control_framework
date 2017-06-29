#!/usr/bin/env python

from abstract_integrator import AbstractIntegrator
from controllers.abstract_controller import AbstractController
import numpy as np

class EulerIntegrator(AbstractIntegrator):

  def integrate(self, x0, ts, controller, ws_sample_fun):
    N = ts.size
    xs = np.empty([N, self.dynamics.n])
    us = np.empty([N-1, self.dynamics.m])
    xs[0] = x0;
    for i, t in enumerate(ts[:-1]):
      dt = ts[i+1] - t
      if controller.discrete:
        us[i] = controller.control(i, xs[i])
      else:
        us[i] = controller.control(ts[i], xs[i])
      ws = ws_sample_fun(i, ts[i])
      xs[i+1] = xs[i] + dt*self.dynamics.xdot(t, xs[i], us[i], ws)
    return [xs, us]
