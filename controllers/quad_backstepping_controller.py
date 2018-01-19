from optimal_control_framework.controllers import AbstractController
import numpy as np
import sys
import control.lyap
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
    tol = 1e-4

    def __init__(self, dynamics):
      super(AccUnicycleController, self).__init__(dynamics)
      self.m = 2  # Feedforward trajectory pddot
      self.max_thetadot = np.pi/2
      self.A = np.zeros((6,6))
      self.A[:3, 3:] = np.eye(3)
      self.B = np.vstack((np.zeros((3,3)), (1.0/self.dynamics.mass)*np.eye(3)))
      self.setGains(np.ones(8))
      self.Lf = 0  # Debug lyapunov function

    def setGoal(self, xd, pddot=np.zeros(2)):
        """
        Set goal x, y positions
        """
        p = xd[:2]
        pdot = self.getVelocity(xd)
        self._xd = np.hstack([p, pdot, pddot])

    def getState(self, x):
        p = x[:3]
        v = x[3:6]
        R = euler2mat(x[6], x[7], x[8], 'rzyx')
        omega = x[9:12]
        u = x[12]
        udot = x[13]
        return (p,v,R,omega,u,udot)

    def setGains(self, gains):
        """
        set gain array np.array([kp, ktheta])
        """
        self.K = np.hstack((np.diag(self.gains[:3]), np.diag(self.[3:6])))
        Aprime = self.A - np.dot(self.B, self.K)
        self.P = control.lyap(Aprime.T, self.Q)
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
        ag = self.dynamics.g
        e3 = np.array([0,0,1])
        p, v, R, omega, u, udot = self.getState(x)
        f = m*ag;
        g = np.dot(R,e3)*u;
        J = self.dynamics.J

        x = np.hstack((p, v));
        dx = np.dot(A,x) + np.dot(B,(f + g))

        xd = np.hstack((self.pd, self.vd))
        dxd = np.hstack((self.vd, self.ad))
        d2xd = np.hstack((self.ad, self.jerk_d))

        z0 = x - xd;
        dz0 = dx - dxd;

        gd = m*self.ad - np.dot(self.K,z0) - f;
        z1 = g - gd;

        dgd = m*self.jerk_d - np.dot(self.K,dz0);

        ad = dgd - np.dot(self.B.T,np.dot(self.P,z0)) - self.k1*z1;

        #sd.u = norm(gd);
        u_d = np.linalg.norm(gd)
        if u_d < self.tol
          print 'Desired thrust close to zero!'

        du_d = np.dot(ad, gd)/u_d

        wh = self.hat(omega)

        dg = np.dot(R,(np.dot(wh, e3)*u + e*du))

        dz1 = dg - dgd

        d2x = np.dot(self.A, dx) + np.dot(self.B, dg)

        d2gd = m*self.snap_d - np.dot(K, d2x - d2xd);

        dad = d2gd - np.dot(B.T, np.dot(self.P, dz0)) - self.k1*dz1

        z2 = dg - ad

        bd = dad - z1 - self.k2*z2

        bz = gd/u_d;
        by = self.normalize_cross(bz, np.array([1, 0, 0]))
        Rd = np.vstack((self.normalize_cross(by,bz), by, bz)).transpose()

        c = np.dot(R.T, bd) - np.dot(wh, (np.dot(wh, e3)*u + 2*e3*du));

        ddu = np.dot(e3, c)
        T = np.dot(J, np.cross(e3, c/u)) - np.cross(np.dot(J, omega), omega);

        self.LF = (np.dot(z0, np.dot(self.P,z0)) + np.dot(z1, z1) + np.dot(z2, z2))/2.0;
        return np.hstack((ddu, T))
