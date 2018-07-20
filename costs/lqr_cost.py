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

  def stagewise_cost(self, i, x, u):
    xdiff =  x - self.xd[i]
    return np.dot(xdiff,self.Q*xdiff) + np.dot(u,self.R*u)

  def terminal_cost(self, xf):
    xdiff =  xf - self.xd[-1]
    return np.dot(xdiff,self.Qf*xdiff)

  def stagewise_jacobian(self, i, x, u):
    xdiff =  x - self.xd[i]
    return self.Q*xdiff, self.R*u

  def terminal_jacobian(self, x):
    xdiff =  xf - self.xd[-1]
    return self.Qf*xdiff

  def stagewise_hessian(self, i, x, u):
    return np.diag(self.Q), np.diag(self.R), 0

  def terminal_hessian(self, x):
    return np.diag(self.Qf)
