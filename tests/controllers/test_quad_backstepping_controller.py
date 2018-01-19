#!/usr/bin/env python

from optimal_control_framework.controllers import QuadBacksteppingController
from optimal_control_framework.dynamics import QuadrotorDynamicsExt
from transforms3d.euler import euler2mat
import unittest
import numpy as np
import numpy.testing as np_testing
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from transforms3d.euler import mat2euler, euler2mat

def getTrajectoryState(t, dynamics, controller):
    pd = np.array([np.cos(t), np.sin(t), 0.01*t])
    vd = np.array([-np.sin(t), np.cos(t), 0.01])
    ad = np.array([-np.cos(t), -np.sin(t), 0])
    jerk_d = np.array([np.sin(t), -np.cos(t), 0])
    snap_d = np.array([np.cos(t), np.sin(t), 0])
    gd = dynamics.mass*(ad - dynamics.g)
    ud = np.linalg.norm(gd)
    bz = gd/ud
    by = controller.normalize_cross(bz, np.array([1, 0, 0]))
    Rd = np.vstack((controller.normalize_cross(by,bz), by, bz)).transpose()
    rpyd = mat2euler(Rd, 'rzyx')[::-1]  # Get roll first
    R_e = euler2mat(rpyd[2], rpyd[1], rpyd[0], 'rzyx')
    r_jerk_d = dynamics.mass*np.dot(Rd.T, jerk_d)
    dud = np.dot(controller.e3, r_jerk_d)
    # Works for omegaz = 0 i.e no yawing
    residual = r_jerk_d - dud*controller.e3
    omegad = np.cross(controller.e3, residual)/ud
    return np.hstack((pd, vd, rpyd, omegad, ud, dud)), snap_d


class TestBacksteppingController(unittest.TestCase):
    def setUp(self):
        self.dynamics = QuadrotorDynamicsExt(mass=1.0)
        #self.dynamics = QuadrotorDynamicsExt()
        self.controller = QuadBacksteppingController(self.dynamics)
        self.plot = True

    def testGoal(self):
        pd = np.array([1,2,3])
        vd = np.array([1,2,3])
        rpyd = np.array([0,0,np.pi/2])
        omegad = np.array([0,0,0])
        ud = 1
        dud = 0
        xd = np.hstack((pd, vd, rpyd, omegad, ud, dud))
        self.controller.setGoal(xd)
        np_testing.assert_allclose(self.controller.pd, pd)
        np_testing.assert_allclose(self.controller.vd, vd)
        np_testing.assert_allclose(self.controller.ad, np.array([0,0,0]))
        np_testing.assert_allclose(self.controller.jerk_d, np.zeros(3))

    def testDecomposeState(self):
        pd = np.array([1,2,3])
        vd = np.array([1,2,3])
        rpyd = np.array([np.pi/4,-np.pi/6,np.pi/2])
        Rd = euler2mat(rpyd[2], rpyd[1], rpyd[0], 'rzyx')
        omegad = np.array([0.1,0.2,0.3])
        ud = 1
        dud = 0
        xd = np.hstack((pd, vd, rpyd, omegad, ud, dud))
        out = self.controller.decomposeState(xd)
        np_testing.assert_allclose(out[0], pd)
        np_testing.assert_allclose(out[1], vd)
        np_testing.assert_allclose(out[2], Rd)
        np_testing.assert_allclose(out[3], omegad)
        np_testing.assert_allclose(out[4], ud)
        np_testing.assert_allclose(out[5], dud)

    def testSetGains(self):
        kp = np.array([1,2,3])
        kv = np.array([4,5,6])
        k1 = 1
        k2 = 2
        self.controller.setGains(np.hstack((kp, kv, k1, k2)))
        K = np.hstack((np.diag(kp),np.diag(kv)))
        np_testing.assert_allclose(self.controller.K, K)
        self.assertEqual(K.shape, (3, 6))
        Aprime = self.controller.A - np.dot(self.controller.B, K)
        Qexpected = np.dot(Aprime.T, self.controller.P) + np.dot(self.controller.P, Aprime)
        np_testing.assert_allclose(Qexpected, -1*np.eye(6), atol=1e-6)

    def testHat(self):
        omega = np.array([1,2,3])
        v = np.array([3,2.5,1.8])
        omega_v = np.cross(omega, v)
        omega_hat_v = np.dot(self.controller.hat(omega), v)
        np_testing.assert_allclose(omega_v, omega_hat_v)

    def testNormalizeCross(self):
        x = np.array([1,1,0])
        y = np.array([2,-1,0])
        z = self.controller.normalize_cross(x,y)
        np_testing.assert_allclose(z, np.array([0,0,-1]))
        np.seterr(all='raise')
        self.assertRaises(FloatingPointError, self.controller.normalize_cross, x, x)

    def testEvaluateControl(self):
        xd = np.zeros(14)
        xd[12] = 1.0 # Thrust
        self.controller.setGoal(xd)
        control = self.controller.control(0, xd)
        np_testing.assert_allclose(control, np.zeros(4))

    def testZControl(self):
        xd = np.zeros(14)
        xd[12] = 1.0 # Thrust
        self.controller.setGoal(xd)
        x = np.copy(xd)
        x[2] = 0.5
        control = self.controller.control(0, x)
        np_testing.assert_allclose(control[1:], np.zeros(3))
        self.assertLess(control[0], 0)

    def testTorque(self):
        xd = np.zeros(14)
        xd[12] = 1.0 # Thrust
        xd[0] = 1.0
        self.controller.setGoal(xd)
        x = np.copy(xd)
        x[0] = 0
        # Torque_y should be positive to move in x direction
        control = self.controller.control(0, x)
        self.assertAlmostEqual(control[0],0.0)
        self.assertAlmostEqual(control[1], 0.0)
        self.assertGreater(control[2], 0.0)
        self.assertAlmostEqual(control[3], 0.0)


    def testPointControl(self):
        # Set external force for quad dynamics:
        self.dynamics.g = np.array([0,0,-9.81])
        #Define goal
        xd = np.zeros(14)
        xd[:3] = np.array([0.5,0.5,1])
        xd[3:6] = np.array([0,0,0])
        xd[12] = 9.81
        self.controller.setGoal(xd)
        # Set gains
        kp = np.array([0.2,0.2,0.2])
        kv = np.array([0.8,0.8,0.8])
        k1 = 5
        k2 = 5
        self.controller.setGains(np.hstack((kp, kv, k1, k2)))
        # Run controller
        N = 1000
        dt = 0.01
        x = np.zeros(14)
        x[12] = 9.81
        xs = [x]
        LF = []
        for i in range(N):
            u = self.controller.control(i, x)
            xdot  = self.dynamics.xdot(i*dt, x, u, np.zeros(14))
            x = x + xdot*dt
            xs.append(x)
            LF.append(self.controller.LF)
        xs = np.vstack(xs)
        np_testing.assert_allclose(x[:3], xd[:3], atol=0.05)
        np_testing.assert_allclose(x[3:6], xd[3:6], atol=0.05) 
        np_testing.assert_allclose(x[6:9], xd[6:9], atol=0.01)
        if self.plot:
            fig = plt.figure(1)
            ax = fig.add_subplot(111, projection='3d')
            ax.plot(xs[:,0],xs[:,1],zs=xs[:,2])
            ax.plot([0],[0],[0], 'm*')
            ax.axis('equal')
            plt.figure(2)
            plt.plot(LF)
            plt.show(block=True)



    def testTrajectoryControl(self):
        # Convert a trajectory into states and feedforward
        # Set the goals and try track a trajectory
        # Run controller
        self.dynamics.g = np.array([0,0,-9.81])
        N = 1000
        dt = 0.01
        xd, snap_d = getTrajectoryState(0, self.dynamics, self.controller)
        self.controller.setGoal(xd, snap_d)
        x = xd.copy()
        x[:3] = np.array([0,0,0])
        xds = [xd]
        xs = [x]
        LF = []
        # Set gains
        kp = np.array([0.2,0.2,0.2])
        kv = np.array([0.8,0.8,0.8])
        k1 = 5
        k2 = 5
        self.controller.setGains(np.hstack((kp, kv, k1, k2)))
        for i in range(N):
            u = self.controller.control(i, x)
            xdot  = self.dynamics.xdot(i*dt, x, u, np.zeros(14))
            x = x + xdot*dt
            xs.append(x)
            LF.append(self.controller.LF)
            # Update goal
            xd, snap_d = getTrajectoryState((i+1)*dt, self.dynamics, self.controller)
            xds.append(xd)
            self.controller.setGoal(xd, snap_d)
        if self.plot:
            xs = np.vstack(xs)
            xds = np.vstack(xds)
            fig = plt.figure(1)
            ax = fig.add_subplot(111, projection='3d')
            ax.plot(xs[:,0],xs[:,1],zs=xs[:,2])
            ax.plot(xds[:,0],xds[:,1], xds[:,2], 'r-')
            ax.plot([0],[0],[0], 'm*')
            ax.axis('equal')
            plt.figure(2)
            plt.plot(LF)
            plt.ylabel('Lyapunov Function')
            plt.show(block=True)
