#!/usr/bin/env python3
import numpy as np
import scipy.linalg as scipy_linalg
from optimal_control_framework.costs import SphericalObstacle
from optimal_control_framework.discrete_integrators import EulerIntegrator


class Ddp(object):
    def __init__(self, dynamics, cost, us0, x0, dt, max_step,
                 integrator=None):
        self.dynamics = dynamics
        if integrator is None:
          self.integrator = EulerIntegrator(self.dynamics)
        else:
          self.integrator = integrator
        self.cost = cost
        self.us = us0  # N x m
        self.min_hessian_value = 1.0 / max_step
        N = self.cost.N
        n = self.integrator.dynamics.n
        m = self.integrator.dynamics.m
        self.w = np.zeros(n)  # Mean noise
        self.Ks = np.zeros((N, m, n + 1))
        self.xs = np.empty((N + 1, n))
        self.xs[0] = x0
        self.dt = dt
        self.us_up = np.empty_like(us0)
        self.xs_up = np.empty_like(self.xs)
        self.xs_up[0] = x0
        self.V = np.Inf  # Total value function
        self.alpha = 1
        self.N = N
        self.status = True
        # Initialization
        self.update_dynamics(self.us, self.xs)

    def regularize(self, Q, min_hessian_value):
        w, v = np.linalg.eigh(Q)
        min_w = np.min(w)
        if min_w < min_hessian_value:
            delta_reg = min_hessian_value - min_w
            return Q + delta_reg * np.eye(Q.shape[0])
        else:
            return Q

    def update_dynamics(self, us, xs):
        self.V = 0
        for i, u in enumerate(us):
            x = xs[i]
            self.V = self.V + self.cost.stagewise_cost(i, x, u)
            xs[i + 1] = self.integrator.step(i, self.dt, x, u, self.w)
            # TODO Change instead of dynamics take in an integrator that
            # integrates continuous dynamics using a fancy integrator maybe
        self.V = self.V + self.cost.terminal_cost(xs[-1])

    def backward_pass(self, xs, us, Ks, min_hessian_value):
        V, Vx, Vxx = self.cost.terminal_cost(xs[-1], True)
        for k in range(self.N - 1, -1, -1):
            L, jac, hess = self.cost.stagewise_cost(k, xs[k], us[k], True)
            Lx, Lu = jac
            Lxx, Luu, Lxu = hess
            fx, fu, _ = self.integrator.jacobian(k, self.dt, xs[k], us[k],
                                                 self.w)
            Qx = Lx + np.dot(fx.T, Vx)
            Qu = Lu + np.dot(fu.T, Vx)
            Qxx = Lxx + np.dot(fx.T, np.dot(Vxx, fx))
            Quu = Luu + np.dot(fu.T, np.dot(Vxx, fu))
            Qux = Lxu + np.dot(fu.T, np.dot(Vxx, fx))
            Quu_reg = self.regularize(Quu, min_hessian_value)
            Qux_u = np.hstack((Qux, Qu[:, np.newaxis]))
            K = -scipy_linalg.solve(Quu_reg, Qux_u, sym_pos=True)
            Vx = Qx + np.dot(K[:, :-1].T, Qu)
            Vxx = Qxx + np.dot(K[:, :-1].T, Qux)
            Ks[k] = K

    def forward_pass_step(self, Ks, xs, us, alpha):
        Vnew = 0
        for k in range(self.N):
            x = self.xs_up[k]
            delta_x = x - self.xs[k]
            K_k = Ks[k]
            u = self.us[k] + alpha * K_k[:, -1] + np.dot(K_k[:, :-1], delta_x)
            Vnew = Vnew + self.cost.stagewise_cost(k, x, u)
            self.xs_up[k+1] = self.integrator.step(k, self.dt, x, u, self.w)
            self.us_up[k] = u
        Vnew = Vnew + self.cost.terminal_cost(self.xs_up[-1])
        return Vnew

    def forward_pass(self, Ks):
        Vnew = np.Inf
        alpha = self.alpha
        while (Vnew >= self.V) and alpha > 1e-16:
            Vnew = self.forward_pass_step(Ks, self.xs, self.us, alpha)
            if Vnew < self.V:
                alpha = 2 * alpha
            else:
                alpha = 0.5 * alpha
        if alpha < 1e-16:
            print("Failed to find a reasonable step size! Probably converged!")
            Vnew = self.V
            self.status = False
        else:
            self.xs, self.xs_up = self.xs_up, self.xs  # Swap xs
            self.us, self.us_up = self.us_up, self.us  # Swap us
        return Vnew

    def iterate(self):
        self.backward_pass(self.xs, self.us, self.Ks,
                           self.min_hessian_value)
        self.V = self.forward_pass(self.Ks)
        # print("Cost: ", self.V)
