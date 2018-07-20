from optimal_control_framework.costs import AbstractCost
import numpy as np

class LQRCost(AbstractCost):
  def __init__(self, N, Q, R, Qf, xd=None):
    """
    Assuming Q,R,Qf are vectors of appropriate length
    xd can be a single goal state or a matrix [N+1, n].
    If nothing provided then treated as zero
    vector
    """
    super(LQRCost, self).__init__(N)
    self.Q = Q
    self.R = R
    self.Qf = Qf
    n = self.Q.shape[0]
    if xd is None:
        self.xd = np.zeros((N+1, n))
    elif xd.shape[0] == N+1:
        self.xd = xd
    elif xd.shape[0] == n:
        self.xd = np.tile(xd, (N+1, 1))

  def getQxRu(self, xdiff, u):
      if self.Q.ndim == 1:
          Qx = self.Q*xdiff
      else:
          Qx = np.dot(self.Q, xdiff)
      if self.R.ndim == 1:
          Ru = self.R*u
      else:
          Ru = np.dot(self.R, u)
      return Qx, Ru

  def stagewise_cost(self, i, x, u):
    xdiff =  x - self.xd[i]
    Qx, Ru = self.getQxRu(xdiff, u)
    return np.dot(xdiff,Qx) + np.dot(u,Ru)

  def terminal_cost(self, xf):
    xdiff =  xf - self.xd[-1]
    if self.Qf.ndim == 1:
        Qfx = self.Qf*xdiff
    else:
        Qfx = np.dot(self.Qf, xdiff)
    return np.dot(xdiff, Qfx)

  def stagewise_jacobian(self, i, x, u):
    xdiff =  x - self.xd[i]
    return self.getQxRu(xdiff, u)

  def terminal_jacobian(self, xf):
    xdiff =  xf - self.xd[-1]
    if self.Qf.ndim == 1:
        Qfx = self.Qf*xdiff
    else:
        Qfx = np.dot(self.Qf, xdiff)
    return Qfx

  def stagewise_hessian(self, i, x, u):
    if self.Q.ndim == 1:
        Q = np.diag(self.Q)
    else:
        Q = self.Q
    if self.R.ndim == 1:
        R = np.diag(self.R)
    else:
        R = self.R
    return Q, R, 0

  def terminal_hessian(self, x):
     if self.Qf.ndim == 1:
         Qf = np.diag(self.Qf)
     else:
         Qf = self.Qf
     return Qf
