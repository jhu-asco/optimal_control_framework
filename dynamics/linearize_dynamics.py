#!/usr/bin/env python

from optimal_control_framework.dynamics import AbstractDynamicSystem
import numpy as np

class LinearizeDynamics:
  def __init__(self, dynamics):
    """
    Constructor that stores dynamics and integrator
    Parameters:
    dynamics   -- Nonlinear dynamics to linearize
    """
    assert(isinstance(dynamics, AbstractDynamicSystem))
    self.dynamics = dynamics

  def linearize(self, ts, controller, x0, wbar_fun):
    """
    Linearize dynamics around the controls
    \delta x_{i+1} = A_i \delta x_i + B_i \delta u_i C_i \delta w_i
    Parameters:
    ts         -- Time steps
    us         -- Controls or controllers
    x0         -- Initial state
    wbar_fun   -- Mean disturbance parameter function
    Returns:
    [As, Bs, Cs] where each is a matrix of appropriate dimensions
    As - [Nxnxn]; Bs - [Nxnxm]; Cs - [Nxnxn]
    """
    As = []
    Bs = []
    Cs = []
    N = len(ts) - 1
    xtemp = x0
    # Linearize
    for i, t in enumerate(ts[:-1]):
      dt = ts[i+1] - t
      if controller.discrete:
        control = controller.deterministic_control(i, xtemp)
      else:
        control = controller.deterministic_control(ts[i], xtemp)
      jac = self.dynamics.jacobian(t, xtemp, control, wbar_fun(i))
      xtemp = xtemp + dt*self.dynamics.xdot(t, xtemp, control, wbar_fun(i))
      As.append(np.eye(self.dynamics.n) + jac[0]*dt)
      Bs.append(jac[1]*dt)
      Cs.append(jac[2]*dt)
    return [np.stack(As), np.stack(Bs), np.stack(Cs)]
