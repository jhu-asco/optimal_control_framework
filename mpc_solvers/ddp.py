#!/usr/bin/env python3
import numpy as np
import scipy.linalg as scipy_linalg

class Ddp:
    def __init__(self, dynamics, cost, us0, x0, dt, max_step):
        self.dynamics = dynamics
        self.cost = cost
        self.us = us0  # N x m
        self.min_hessian_value = 1.0/max_step
        N = self.cost.N
        n = self.dynamics.n
        m = self.dynamics.m
        self.w = np.zeros(n) # Mean noise
        self.Ks = np.zeros((N, m, n+1))
        self.xs = np.empty((N+1, n))
        self.xs[0] = x0
        self.dt = dt
        self.us_up = np.empty_like(us0)
        self.xs_up = np.empty_like(self.xs)
        self.xs_up[0] = x0
        self.V = np.Inf # Total value function
        self.alpha = 1
        self.N = N
        # Initialization
        self.update_dynamics(self.us, self.xs)

    def regularize(self, Q, min_hessian_value):
        w, v = np.linalg.eigh(Q)
        min_w = np.min(w)
        if min_w < min_hessian_value:
            delta_reg = min_hessian_value - min_w
            return Q + delta_reg*np.eye(Q.shape[0])
        else:
            return Q

    def update_dynamics(self, us, xs):
        self.V = 0
        for i, u in enumerate(us):
            x = xs[i]
            self.V = self.V + self.cost.stagewise_cost(i, x, u)
            xdot = self.dynamics.xdot(i, x, u, self.w)
            xs[i+1] = x + self.dt*xdot  # For now euler integration
            # TODO Change instead of dynamics take in an integrator that
            # integrates continuous dynamics using a fancy integrator maybe
        self.V = self.V + self.cost.terminal_cost(xs[-1])

    def backward_pass(self, xs, us, Ks, min_hessian_value):
        Vx = self.cost.terminal_jacobian(xs[-1])
        Vxx = self.cost.terminal_hessian(xs[-1])
        for k in range(self.N-1, -1, -1):
            Lx, Lu = self.cost.stagewise_jacobian(k, xs[k], us[k])
            Lxx, Luu, Lxu = self.cost.stagewise_hessian(k, xs[k], us[k])
            fx, fu, _ = self.dynamics.jacobian(k, xs[k], us[k], self.w)
            # Find better jacobians based on notes!!
            fx_bar = np.eye(self.dynamics.n) + fx*self.dt
            fu_bar = fu*self.dt
            Qx = Lx + np.dot(fx_bar.T, Vx)
            Qu = Lu + np.dot(fu_bar.T, Vx)
            Qxx = Lxx + np.dot(fx_bar.T, np.dot(Vxx, fx_bar))
            Quu = Luu + np.dot(fu_bar.T, np.dot(Vxx, fu_bar))
            Qux = Lxu + np.dot(fu_bar.T, np.dot(Vxx, fx_bar))
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
            u = self.us[k] + alpha*K_k[:, -1] + np.dot(K_k[:, :-1], delta_x)
            xdot = self.dynamics.xdot(k, x, u, self.w)
            Vnew = Vnew + self.cost.stagewise_cost(k, x, u)
            self.xs_up[k+1] = x + self.dt*xdot  # For now euler integration
            self.us_up[k] = u
        Vnew = Vnew + self.cost.terminal_cost(self.xs_up[-1])
        return Vnew

    def forward_pass(self, Ks):
        Vnew = np.Inf
        alpha = self.alpha
        while (Vnew >= self.V) and alpha > 1e-16:
            Vnew = self.forward_pass_step(Ks, self.xs, self.us, alpha)
            if Vnew < self.V:
                alpha = 2*alpha
            else:
                alpha = 0.5*alpha
        if alpha < 1e-16:
            print("Failed to find a reasonable step size! Probably converged!")
            Vnew = self.V
        else:
            self.xs, self.xs_up = self.xs_up, self.xs  # Swap xs
            self.us, self.us_up = self.us_up, self.us  # Swap us
        return Vnew

    def iterate(self):
        self.backward_pass(self.xs, self.us, self.Ks,
                           self.min_hessian_value)
        self.V = self.forward_pass(self.Ks)
        # print("Cost: ", self.V)
