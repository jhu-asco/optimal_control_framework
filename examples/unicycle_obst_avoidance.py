#!/usr/bin/env python

from optimal_control_framework.dynamics import CasadiUnicycleDynamics
from optimal_control_framework.mpc_solvers import Ddp
from optimal_control_framework.costs import LQRObstacleCost, SphericalObstacle
from optimal_control_framework.discrete_integrators import SemiImplicitCarIntegrator
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle as CirclePatch


np.set_printoptions(precision=3, suppress=True)
dynamics = CasadiUnicycleDynamics()
integrator = SemiImplicitCarIntegrator(dynamics)
# Trajectory info
dt = 0.1
N = 30
Q = dt * np.zeros(dynamics.n)
R = 0.5 * dt * np.eye(dynamics.m)
Qf = 10 * np.eye(dynamics.n)
ts = np.arange(N + 1) * dt
# Obstacles
ko = 1000  # Obstacle gain
obs1 = SphericalObstacle(np.array([0.3, 0.1]), 0.15)
obs2 = SphericalObstacle(np.array([0.7, -0.12]), 0.05)
obs_list = [obs1, obs2]
# Desired terminal condition
xd = np.array([1.0, 0.0, 0.0])
ud = np.array([0.5, 0.0])
cost = LQRObstacleCost(N, Q, R, Qf, xd, ko=ko, obstacles=obs_list, ud=ud)
max_step = 5.0  # Allowed step for control

x0 = np.array([0, 0, 0])
us0 = np.zeros([N, dynamics.m])
ddp = Ddp(dynamics, cost, us0, x0, dt, max_step, use_prev_x=False,
          integrator=integrator)
V = ddp.V
for i in range(50):
    ddp.iterate()
    V = ddp.V
    print("V: ", V)
    print("xn: ", ddp.xs[-1])
    if not ddp.status:
        break
f = plt.figure(1)
plt.clf()
ax = f.add_subplot(111)
ax.set_aspect('equal')
plt.plot(ddp.xs[:, 0], ddp.xs[:, 1], 'b*-')
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
