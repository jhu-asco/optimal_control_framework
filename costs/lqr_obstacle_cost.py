from optimal_control_framework.costs import LQRCost
import numpy as np

class LQRObstacleCost(LQRCost):
    def __init__(self, N, Q, R, Qf, xd=None, obstacles=[], ko = 1):
        super(LQRObstacleCost, self).__init__(N, Q, R, Qf, xd)
        self.obstacles = obstacles
        self.ko = ko

    def obstacle_cost(self, x, obstacle, compute_grads=False):
        obst_dist, obst_jac = obstacle.distance(x, compute_grads)
        obst_cost = 0.5*(self.ko*(obst_dist**2))
        obst_cost_x = obst_cost_xx = 0
        if compute_grads:
            obst_cost_x = self.ko*obst_dist*obst_jac
            # Simplified approximate
            obst_cost_xx = self.ko*np.outer(obst_jac, obst_jac)
        return obst_cost, obst_cost_x, obst_cost_xx

    def stagewise_cost(self, i, x, u, compute_grads=False):
        out = super(LQRObstacleCost, self).stagewise_cost(i, x, u,
                                                          compute_grads)
        obst_cost = 0
        obst_cost_x = 0
        obst_cost_xx = 0
        for obstacle in self.obstacles:
            obst_out = self.obstacle_cost(x, obstacle, compute_grads)
            obst_cost = obst_cost + obst_out[0]
            if compute_grads:
                obst_cost_x = obst_cost_x + obst_out[1]
                obst_cost_xx = obst_cost_xx + obst_out[2]
        #if obst_cost > 0:
        #    print("Hitting obstacle: ", obst_cost)
        if not compute_grads:
            return out + obst_cost
        else:
            final_out = []
            final_out.append(out[0] + obst_cost)
            Lx, Lu = out[1]
            final_out.append((Lx+obst_cost_x, Lu))
            Q, R, Lxu = out[2]
            final_out.append((Q+obst_cost_xx, R, Lxu))
            return final_out

    def terminal_cost(self, xf, compute_grads=False):
        out = super(LQRObstacleCost, self).terminal_cost(xf, compute_grads)
        obst_cost = 0
        obst_cost_x = 0
        obst_cost_xx = 0
        for obstacle in self.obstacles:
            obst_out = self.obstacle_cost(xf, obstacle, compute_grads)
            if compute_grads:
                obst_cost = obst_cost + obst_out[0]
                obst_cost_x = obst_cost_x + obst_out[1]
                obst_cost_xx = obst_cost_xx + obst_out[2]
            else:
                obst_cost = obst_cost + obst_out[0]
        if not compute_grads:
            return out + obst_cost
        else:
            final_out = []
            final_out.append(out[0] + obst_cost)
            final_out.append(out[1] + obst_cost_x)
            final_out.append(out[2] + obst_cost_xx)
            return final_out
