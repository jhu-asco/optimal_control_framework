from abstract_dynamics import AbstractDynamicSystem
import casadi as cs
import numpy as np

class LinearDynamics(AbstractDynamicSystem):

  def __init__(self, theta):
    """
    Constructor. Defines the symbolic function
    to compute xdot = Ax + Bu. Assumes constant
    A, B throughout
    Parameters:
    theta - List of matrices [A, B]
    """
    self.A = theta[0]
    self.B = theta[1]
    self.n = self.A.shape[0]
    self.m = self.B.shape[1]
    x = cs.SX.sym('x',self.n)
    u = cs.SX.sym('u',self.m)
    w = cs.SX.sym('w', self.n)
    xdot = cs.mtimes(self.A, x) + cs.mtimes(self.B, u) + w
    self.xdot_fun = cs.Function('xdot',[x,u,w],[xdot])

  def jacobian(self, t, x, u, w):
    return [self.A, self.B, np.eye(self.n)]

  def xdot(self, t, x, u, w):
    return np.squeeze(np.array(self.xdot_fun(x,u,w)))
