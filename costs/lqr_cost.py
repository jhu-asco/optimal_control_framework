from abstract_cost import AbstractCost
import numpy as np

class LQRCost(AbstractCost):
  def __init__(self, N, Q, R, Qf):
    """
    Assuming Q,R,Qf are vectors of appropriate length
    """
    super(LQRCost, self).__init__(N)
    self.Q = Q
    self.R = R
    self.Qf = Qf

  def stagewise_cost(self, i, x, u):
    return np.dot(x,self.Q*x) + np.dot(u,self.R*u)

  def terminal_cost(self, xf):
    return np.dot(xf,self.Qf*xf)
    
