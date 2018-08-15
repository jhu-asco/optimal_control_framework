#!/usr/bin/env python3
import numpy as np
import scipy.linalg as scipy_linalg
from optimal_control_framework.discrete_integrators import EulerIntegrator

class Armiho:
    def __init__(self):
        self.dV = [0, 0]
        self.sigma = 0.1
        self.beta = 0.25
        self.alpha_min = 1e-10
        self.cVmin = 1e-10
        self.alpha_tol = 1e-2
        self.s1 = 0.1
        self.s2 = 0.5
        self.b1 = 0.25
        self.b2 = 2.0
        self.alpha_max = 1.0
        self.step_converged = False

    def termination(self, V, Vnew, alpha):
        delta_V = Vnew - V
        return (abs(delta_V) < self.cVmin or alpha < self.alpha_min or
                self.step_converged)
    
    def updateStep(self, V, Vnew, alpha):
        delta_V = Vnew - V
        alpha_new = alpha
        if delta_V > 0:
            alpha_new = alpha*self.b1
        else:
            r = delta_V/(alpha*self.dV[0] + alpha*alpha*self.dV[1])
            if r < self.s1:
                alpha_new = self.b1*alpha
            else:
                alpha_new = alpha
                self.step_converged = True
        return alpha_new


class Ddp(object):
    def __init__(self, dynamics, cost, us0, x0, dt, max_step,
                 integrator=None, Ks=None):
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
        if Ks is not None:
            self.Ks = Ks
        else:
            self.Ks = np.zeros((N, m, n+1))
        self.xs = np.empty((N + 1, n))
        self.xs[0] = x0
        self.dt = dt
        self.us_up = np.empty_like(us0)
        self.xs_up = np.empty_like(self.xs)
        self.xs_up[0] = x0
        self.V = np.Inf  # Total value function
        self.N = N
        self.status = True
        self.step_search = Armiho()
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

    def control(self, k, x):
        delta_x = x - self.xs[k]
        K_k = self.Ks[k]
        u = self.us[k] + np.dot(K_k[:, :-1], delta_x)
        return u

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
        dV = [0, 0]
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
            dV[0] = dV[0] + np.dot(K[:, -1], Qu)
            dV[1] = dV[1] + 0.5*np.dot(K[:, -1], np.dot(Quu_reg, K[:, -1]))
        self.step_search.dV = dV

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
        alpha = self.step_search.alpha_max
        self.step_search.step_converged = False
        while not self.step_search.termination(self.V, Vnew, alpha):
            Vnew = self.forward_pass_step(Ks, self.xs, self.us, alpha)
            alpha = self.step_search.updateStep(self.V, Vnew, alpha)
        if alpha < self.step_search.alpha_min:
            print("Failed to find a reasonable step size! Probably converged!")
            Vnew = self.V
            self.status = False
        else:
            self.xs, self.xs_up = self.xs_up, self.xs  # Swap xs
            self.us, self.us_up = self.us_up, self.us  # Swap us
            self.status = True
        return Vnew

    def iterate(self):
        self.backward_pass(self.xs, self.us, self.Ks,
                           self.min_hessian_value)
        self.V = self.forward_pass(self.Ks)
        # print("Cost: ", self.V)
