#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import pickle
from optimal_control_framework.dynamics import CasadiUnicycleDynamics
from optimal_control_framework.mpc_solvers import Ddp
from optimal_control_framework.costs import LQRObstacleCost, SphericalObstacle
from optimal_control_framework.discrete_integrators import EulerIntegrator
from optimal_control_framework.sampling import DiscreteSampleTrajectories
from matplotlib.patches import Circle as CirclePatch

def createObstacleList(obs_params):
    obs_list = []
    for i, param in enumerate(obs_params):
        radius = max(param[2], 0.1)
        obs_list.append(SphericalObstacle(param[:2], radius))
    return obs_list

def sampleObsParams(M, obs_mu, obs_cov):
    obs_params = np.empty((M, obs_mu.shape[0], obs_mu.shape[1]))
    for i , mu in enumerate(obs_mu):
        obs_params[:, i, :] = np.random.multivariate_normal(mu, np.diag(obs_cov[i]), M)
    return obs_params
    
np.random.seed(1000)
sns.set_style('whitegrid')
sns.set(font_scale=1.0)
np.set_printoptions(precision=3, suppress=True)

dynamics = CasadiUnicycleDynamics()
integrator = EulerIntegrator(dynamics)
# Trajectory info
dt = 0.1
N = 20
Q = dt*np.zeros(dynamics.n)
R = 5*dt*np.eye(dynamics.m)
Qf = 30*np.eye(dynamics.n)
Qf[-1, -1] = 0
ts = np.arange(N + 1) * dt
# Obstacles
obs_mu = np.array([[4,4.5,1.4]])
obs_cov = np.square(np.tile(np.array([0.2, 0.2, 0.5]), (1, 1)))
#obs_mu = np.array([[3,2,1.4],
#                   [6,5,1.4]])
#obs_cov = np.square(np.tile(np.array([0.2, 0.2, 0.5]), (2, 1)))
obs_list = createObstacleList(obs_mu)
# Desired terminal condition
mud = np.array([8.0, 8.0, 0])
Sigmad = 0.001*np.array([0.2, 0.2, 0.1])
xd = mud
ud = np.array([5.0, 0.0])
max_step = 5.0  # Allowed step for control
mu0 = np.array([0, 0, np.pi/6])
Sigma0 = np.array([0.2, 0.2, 0.05])
x0 = mu0
Sigma_w = 5*np.array([0.01, 0.01, 0.01])
max_iters = 30
max_ko = 5000
ko_gain = 2.0/(0.8*max_iters)
ko_start = 500

def singleTrial(M=100, plot=False, buf=0.0, return_ddp=False):
    SphericalObstacle.buf = buf
    us0 = np.tile(ud, (N, 1))
    cost = LQRObstacleCost(N, Q, R, Qf, xd, ko=ko_start, obstacles=obs_list, ud=ud)
    ddp = Ddp(dynamics, cost, us0, x0, dt, max_step,
              integrator=integrator)
    V = ddp.V
    print("V0: ", V)
    for i in range(max_iters):
        cost.ko = np.tanh(i*ko_gain)*(max_ko-ko_start) + ko_start
        ddp.update_dynamics(ddp.us, ddp.xs)
        ddp.iterate()
        V = ddp.V
    print("Vfinal: ", V)
    # Sample example trajectories
    # Stdeviation!!:
    Sigma0_sqr = np.diag(np.square(Sigma0))
    Sigmaw_sqr = np.diag(np.square(Sigma_w))
    ws_sampling_fun = lambda : np.random.multivariate_normal(np.zeros(dynamics.n),
                                                             Sigmaw_sqr)
    x0_sampling_fun = lambda : np.random.multivariate_normal(x0, Sigma0_sqr)
    sampler = DiscreteSampleTrajectories(dynamics, integrator, cost,
                                         ws_sampling_fun, x0_sampling_fun)
    cost.ko = 0  #Ignore obstacle avoidance when computing costs
    SphericalObstacle.buf = 0 #Ignore buffer when checking actual collisions
    xss, uss, Jss = sampler.sample(M, ts, ddp)
    collision_array = np.full(M, False)
    for i, sample_traj in enumerate(xss):
        collision_array[i] = sampler.isColliding(obs_list, sample_traj)
    Ncollisions = np.sum(collision_array)
    print("Ncollisions: ", Ncollisions)
    print("Jmean: ", np.mean(Jss[:, 0]))
    print("Jstd: ", np.std(Jss[:, 0]))
    buf_str = '{0:.3f}'.format(buf)
    buf_str = buf_str.replace('.','_')
    if plot:
        f = plt.figure(1)
        plt.clf()
        ax = f.add_subplot(111)
        ax.set_aspect('equal')
        plt.plot(ddp.xs[:, 0], ddp.xs[:, 1], 'b*-')
        for j in range(M):
            if collision_array[j]:
                color='m*-'
            else:
                color='g*-'
            plt.plot(xss[j][:,0], xss[j][:, 1], color)
        plt.plot(xd[0], xd[1], 'r*')
        plt.xlabel('x (m)')
        plt.ylabel('y (m)')
        for obs in obs_list:
            circ_patch = CirclePatch(obs.center, obs.radius, fill=False,
                                     ec='r')
            ax.add_patch(circ_patch)
            ax.plot(obs.center[0], obs.center[1], 'r*')
        plt.tight_layout()
        plt.savefig('trajectories_'+buf_str+'.eps', bbox_inches='tight')
        plt.figure(2)
        plt.clf()
        plt.subplot(2, 1, 1)
        plt.plot(ts[:-1], ddp.us[:, 0])
        plt.ylabel('Velocity (m/s)')
        plt.subplot(2, 1, 2)
        plt.plot(ts[:-1], ddp.us[:, 1])
        plt.ylabel('Angular rate (rad/s)')
        plt.tight_layout()
        plt.savefig('controls_'+buf_str+'.eps', bbox_inches='tight')
        plt.figure(3)
        plt.clf()
        plt.plot(ts, ddp.xs[:, 2])
        plt.xlabel('Time (seconds)')
        plt.ylabel('Angle (radians)')
        plt.tight_layout()
        plt.savefig('angle_'+buf_str+'.eps', bbox_inches='tight')
    if return_ddp:
        return Ncollisions, Jss, ddp

    return Ncollisions, Jss

if __name__ == "__main__":
    Ncollisions_buf = []
    buf_array = np.linspace(0, 0.2, 5)
    delta_Js_frame = pd.DataFrame()
    M = 100 # Number of samples per trial
    Mtrials = 20 #Number of trials other than nominal one
    obs_params = sampleObsParams(Mtrials, obs_mu, obs_cov)
    xds = np.random.multivariate_normal(mud, np.diag(np.square(Sigmad)), Mtrials)
    for buf in buf_array:
        xd = mud
        obs_list = createObstacleList(obs_mu)
        if buf == buf_array[-1]:
            Ncollisions, Jss, ddp = singleTrial(M, True, buf, return_ddp=True)
        else:
            Ncollisions, Jss = singleTrial(M, True, buf)
        Jss_list = [Jss[:,0]]
        Mtotal = M
        # Sample x0
        for trial_number in range(Mtrials):
            print("Trial number: ", trial_number)
            obs_list = createObstacleList(obs_params[trial_number]) 
            xd = xds[trial_number]
            Ncol_trial, Js_trial = singleTrial(M, False, buf)
            Jss_list.append(Js_trial[:,0])
            Ncollisions = Ncollisions+Ncol_trial
            Mtotal = Mtotal + M
        Ncollisions_buf.append(float(Ncollisions)/Mtotal*1000)
        if buf == 0:
            J0 = np.hstack(Jss_list)
        else:
            delta_Js_frame[buf] = np.hstack(Jss_list) - J0
    
    pickle.dump({'delta_Js_frame': delta_Js_frame, 'Ncollisions_buf': Ncollisions_buf,
                 'buf_array': buf_array,
                 'J0': J0, 'obs_params': obs_params,
                 'xds': xds, 'obs_mu': obs_mu, 'obs_cov': obs_cov,
                 'mud': mud, 'sigmad': Sigmad,
                 'mu0': mu0, 'sigma0': Sigma0, 'sigmaw': Sigma_w,
                 'M': M, 'Mtrials': Mtrials,
                 'ddp_K': ddp.Ks, 'ddp_us': ddp.us},
                 open('costs_collisions.pickle', 'wb'))
    plt.figure(4)
    ax = sns.barplot(data=delta_Js_frame)
    ax.set_xlabel('Buffer Length(m)')
    ax.set_ylabel('Change Trajectory Cost')
    plt.tight_layout()
    plt.savefig('cost_vs_buf_length.eps', bbox_inches='tight')
    plt.figure(5)
    plt.semilogy(buf_array, Ncollisions_buf, 'b*-')
    plt.xlabel('Buffer Length(m)')
    plt.ylabel('Ncollisions per 1000 samples')
    plt.tight_layout()
    plt.savefig('ncollisions_vs_buf_length.eps', bbox_inches='tight')
    plt.show()
