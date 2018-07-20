#!/usr/bin/env python

from optimal_control_framework.dynamics import SimpleCarDynamics
from optimal_control_framework.mpc_solvers import Ddp
from optimal_control_framework.costs import LQRCost
import matplotlib.pyplot as plt
import numpy as np


np.set_printoptions(precision=3, suppress=True)
dynamics = SimpleCarDynamics()
# Trajectory info
dt = 0.05
N = 20
Q = dt*np.zeros(dynamics.n)
R = 0.01*dt*np.eye(dynamics.m)
Qf = 100*np.eye(dynamics.n)
Qf[-1,-1] = 1  # Don't care about end steering angle
ts = np.arange(N+1)*dt
#xd = np.array([1, 0.5, np.pi/4, 0, 0])
xd = np.array([0.5, 0.5, 0, 0, 0])
#xd = np.array([1, 0, 0, 0, 0])
cost = LQRCost(N, Q, R, Qf, xd)
max_step = 10.0  # Allowed step for control

x0 = np.array([0, 0, 0, 0, 0])
us0 = np.zeros([N, dynamics.m])
ddp =Ddp(dynamics, cost, us0, x0, dt, max_step)
V = ddp.V
for i in range(50):
    ddp.iterate()
    V = ddp.V
    print("V: ", V)
    print("xn: ", ddp.xs[-1])
plt.figure(1)
plt.plot(ddp.xs[:, 0], ddp.xs[:, 1])
plt.plot(xd[0], xd[1], 'r*')
plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.figure(2)
plt.subplot(2,1,1)
plt.plot(ts[:-1], ddp.us[:, 0])
plt.ylabel('Accelertion (m/ss)')
plt.subplot(2,1,2)
plt.plot(ts[:-1], ddp.us[:, 1])
plt.ylabel('Steering rate (rad/s)')
plt.figure(3)
plt.subplot(2,1,1)
plt.plot(ts, ddp.xs[:, 3])
plt.ylabel('Velocity (m/s)')
plt.subplot(2,1,2)
plt.plot(ts, ddp.xs[:, 4])
plt.ylabel('Steering angle (rad)')
plt.show()
