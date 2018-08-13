# -*- coding: utf-8 -*-
"""
Created on Mon Aug 13 15:52:02 2018

@author: gowtham
"""

import numpy as np
import pickle
from optimal_control_framework.dynamics import CasadiUnicycleDynamics
from optimal_control_framework.discrete_integrators import SemiImplicitCarIntegrator
from optimal_control_framework.sampling import DiscreteSampleTrajectories

class RandomController(object):
    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma

    def control(self, i, x):
        return np.random.multivariate_normal(self.mu, self.sigma)

np.random.seed(1000)
# Controller params
mu_u = np.zeros(2)
sigma_u = np.diag([5, 5])
# X0 params
xy_bnd = 5
theta_bnd = 1
scale_noise = [0.5, 0.5, 0.5]
# Create stuff
x0_sampling_fun = lambda : np.hstack([np.random.uniform(-xy_bnd, xy_bnd, 2),np.random.uniform(-theta_bnd, theta_bnd)])
ws_sampling_fun = lambda : np.random.multivariate_normal(np.zeros(3), np.eye(3))
dynamics = CasadiUnicycleDynamics(use_nonlinear_noise_model=True, scale=scale_noise)
integrator = SemiImplicitCarIntegrator(dynamics)
sampler = DiscreteSampleTrajectories(dynamics, integrator,x0_sampling_fun=x0_sampling_fun, ws_sampling_fun=ws_sampling_fun)
controller = RandomController(mu_u, sigma_u)
# Sample
M = 1000
N = 100
dt = 0.1
ts = np.arange(N+1)*dt
xss, uss, Jss = sampler.sample(M, ts, controller)
pickle.dump({'xss': xss, 'uss': uss}, open('samples.pickle', 'wb'))
