from optimal_control_framework.controllers import AbstractController
import numpy as np
import sys
import control
from transforms3d.euler import euler2mat

class QuadBacksteppingController(AbstractController):
    # Goal (p, pdot, pddot, pdddot)
    pd = np.zeros(3)
    vd = np.zeros(3)
    ad = np.zeros(3)
    jerk_d = np.zeros(3)
    snap_d = np.zeros(3)
    # Gains
    #Backstepping gains
    k1 = 1
    k2 = 1
    K = np.zeros((3,6))
    P = np.eye(6)
    Q = np.eye(6)
    # Tolerance for checking thrust close to zero
    tol = 1e-2

    def __init__(self, dynamics):
      super(QuadBacksteppingController, self).__init__(dynamics)
      self.m = 2  # Feedforward trajectory pddot
      self.max_thetadot = np.pi/2
      self.A = np.zeros((6,6))
      self.A[:3, 3:] = np.eye(3)
      self.B = np.vstack((np.zeros((3,3)), (1.0/self.dynamics.mass)*np.eye(3)))
      self.e3 = np.array([0,0,1]) #Thrust direction
      self.setGains(np.ones(8))
      self.LF = 0  # Debug lyapunov function

    def setGoal(self, xd, snap_d=np.zeros(3)):
        """
        Set goal trajectory derivatives
        """
        pd, vd, Rd, omegad, ud, dud = self.decomposeState(xd)
        Red = np.dot(Rd, self.e3)
        self.pd = pd
        self.vd = vd
        self.ad = (1/self.dynamics.mass)*Red*ud + self.dynamics.g
        self.jerk_d = (1/self.dynamics.mass)*(Red*dud + np.dot(Rd, np.cross(omegad, self.e3))*ud)
        self.snap_d = snap_d

    def decomposeState(self, x):
        """
        Get position, velocity, orientation, angular velocity,
        thrust and derivative thrust given the state vector.
        """
        p = x[:3]
        v = x[3:6]
        R = euler2mat(x[8], x[7], x[6], 'rzyx')
        omega = x[9:12]
        u = x[12]
        udot = x[13]
        return (p,v,R,omega,u,udot)

    def setGains(self, gains):
        """
        set gain array np.array([kp, kv, k1, k2])
        """
        # Kp, Kv
        self.K = np.hstack((np.diag(gains[:3]), np.diag(gains[3:6])))
        Aprime = self.A - np.dot(self.B, self.K)
        self.P = control.lyap(Aprime.T, self.Q)
        # Backstepping gains
        self.k1 = gains[6]
        self.k2 = gains[7]

    def hat(self, x):
        return np.array([[0, -x[2], x[1]],
                         [x[2], 0, -x[0]],
                         [-x[1], x[0], 0]])

    def normalize_cross(self, x, y):
        v = np.cross(x,y)
        return v / np.linalg.norm(v)

    def deterministic_control(self, i, x):
        """
        Compute the double derivative of thrust and
        torque needed to track a reference trajectory
        """
        m = self.dynamics.mass
        J = self.dynamics.J
        ag = self.dynamics.g
        p, v, R, omega, u, du = self.decomposeState(x)
        f_ext = m*ag

        # Thrust direction
        Re3 = np.dot(R, self.e3)

        # Input thrust vector
        g = Re3*u

        x = np.hstack((p, v))
        dx = np.dot(self.A,x) + np.dot(self.B,(f_ext + g))

        xd = np.hstack((self.pd, self.vd))
        dxd = np.hstack((self.vd, self.ad))
        d2xd = np.hstack((self.ad, self.jerk_d))


        z0 = x - xd
        dz0 = dx - dxd

        gd = m*self.ad - np.dot(self.K,z0) - f_ext
        z1 = g - gd

        dgd = m*self.jerk_d - np.dot(self.K,dz0)

        BTP = np.dot(self.B.T, self.P)

        ad = dgd - np.dot(BTP,z0) - self.k1*z1


        #sd.u = norm(gd)
        u_d = np.linalg.norm(gd)
        if u_d < self.tol:
            print 'Desired thrust close to zero!'
            u_d = u_d + self.tol

        du_d = np.dot(ad, gd)/u_d

        wh = self.hat(omega)

        whe3 = np.dot(wh, self.e3)

        dg = np.dot(R, whe3*u) + Re3*du

        dz1 = dg - dgd

        d2x = np.dot(self.A, dx) + np.dot(self.B, dg)
        d2z0 = d2x - d2xd

        d2gd = m*self.snap_d - np.dot(self.K, d2z0)

        dad = d2gd - np.dot(BTP, dz0) - self.k1*dz1

        z2 = dg - ad

        bd = dad - z1 - self.k2*z2

        #bz = gd/u_d
        #by = self.normalize_cross(bz, np.array([1, 0, 0]))
        #Rd = np.vstack((self.normalize_cross(by,bz), by, bz)).transpose()

        c = np.dot(R.T, bd) - np.dot(wh, whe3)*u + 2*whe3*du

        ddu = np.dot(self.e3, c)

        T = np.dot(J, np.cross(self.e3, c/u)) - np.cross(np.dot(J, omega), omega)

        self.LF = (np.dot(z0, np.dot(self.P,z0)) + np.dot(z1, z1) + np.dot(z2, z2))/2.0
        return np.hstack((ddu, T))
