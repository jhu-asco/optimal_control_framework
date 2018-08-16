#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
import pickle
from optimal_control_framework.dynamics import CasadiUnicycleDynamics
from optimal_control_framework.mpc_solvers import Ddp
from optimal_control_framework.costs import LQRObstacleCost, SphericalObstacle
from optimal_control_framework.discrete_integrators import SemiImplicitCarIntegrator
from optimal_control_framework.sampling import DiscreteSampleTrajectories
from matplotlib.patches import Circle as CirclePatch


np.set_printoptions(precision=3, suppress=True)
dynamics = CasadiUnicycleDynamics()
integrator = SemiImplicitCarIntegrator(dynamics)
# Trajectory info
dt = 0.1
N = 20
Q = dt*np.zeros(dynamics.n)
R = 5*dt*np.eye(dynamics.m)
Qf = 30*np.eye(dynamics.n)
Qf[-1, -1] = 0
ts = np.arange(N + 1) * dt
# Obstacles
ko = 1000  # Obstacle gain
SphericalObstacle.buf = 0.5
obs1 = SphericalObstacle(np.array([4, 4]), 1.4)
#obs2 = SphericalObstacle(np.array([6, 5]), 1.4)
obs_list = [obs1]
# Desired terminal condition
xd = np.array([8.0, 8.0, 0.0])
ud = np.array([5.0, 0.0])
cost = LQRObstacleCost(N, Q, R, Qf, xd, ko=ko, obstacles=obs_list, ud=ud)
max_step = 5.0  # Allowed step for control

x0 = np.array([0, 0, np.pi/6])
us0 = np.zeros([N, dynamics.m])
ddp = Ddp(dynamics, cost, us0, x0, dt, max_step,
          integrator=integrator)
V = ddp.V
max_iters = 15
max_ko = 1800
ko_gain = 2.0/(0.8*max_iters)
for i in range(max_iters):
    ddp.iterate()
    V = ddp.V
    print("V: ", V)
    print("ko: ", cost.ko)
    cost.ko = np.tanh(i*ko_gain)*max_ko
    ddp.update_dynamics(ddp.us, ddp.xs)
# Sample example trajectories
scale = 0.01*np.array([1, 1, 0.0])
ws_sampling_fun = lambda : (np.random.sample(dynamics.n)-0.5)*0.0
x0_sampling_fun = lambda : ((np.random.sample(dynamics.n)-0.5)*scale + x0)
sampler = DiscreteSampleTrajectories(dynamics, integrator, cost, ws_sampling_fun, x0_sampling_fun)
M = 100
xss, uss, Jss = sampler.sample(M, ts, ddp)
pickle.dump({'xss':xss, 'uss':uss, 'Jss': Jss, 'obs_list': obs_list}, open('unicycle_samples.pickle', 'wb'))
f = plt.figure(1)
plt.clf()
ax = f.add_subplot(111)
ax.set_aspect('equal')
plt.plot(ddp.xs[:, 0], ddp.xs[:, 1], 'b*-')
for j in range(M):
    plt.plot(xss[j][:,0], xss[j][:, 1], 'g*-')
plt.plot(xd[0], xd[1], 'r*')
plt.xlabel('x (m)')
plt.ylabel('y (m)')
for obs in obs_list:
    circ_patch = CirclePatch(obs.center, obs.radius, fill=False,
                             ec='r')
    ax.add_patch(circ_patch)
    ax.plot(obs.center[0], obs.center[1], 'r*')
plt.figure(2)
plt.subplot(2, 1, 1)
plt.plot(ts[:-1], ddp.us[:, 0])
plt.ylabel('Velocity (m/s)')
plt.subplot(2, 1, 2)
plt.plot(ts[:-1], ddp.us[:, 1])
plt.ylabel('Angular rate (rad/s)')
plt.figure(3)
plt.plot(ts, ddp.xs[:, 2])
plt.xlabel('Time (seconds)')
plt.ylabel('Angle (radians)')
plt.show()
