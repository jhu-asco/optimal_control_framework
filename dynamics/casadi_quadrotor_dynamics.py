#!/usr/bin/env python2
from optimal_control_framework.dynamics import AbstractCasadiSystem
import casadi as cs
import numpy as np

class CasadiQuadrotorDynamics(AbstractCasadiSystem):
    """
    Zero order dynamics with n = m = 1
    """

    def __init__(self, mass=1.0, J=np.eye(3), g=[0,0,-1]):
        """
        X = [x, y, z, vx, vy, vz, r,p,y, omegax, omegay, omegaz]
        Constructor that initializes the order
        """
        self.mass = cs.DM(mass)
        self.J = cs.DM(J)
        self.mg = cs.DM(g)*self.mass
        Jinv = np.linalg.solve(J, np.eye(3))
        self.Jinv = cs.DM(Jinv)
        super(CasadiQuadrotorDynamics, self).__init__(12, 4)

    def omegaToRpyDot(self, rpy, omega):
        secant_pitch= (1.0/cs.cos(rpy[1]))
        c_roll = cs.cos(rpy[0])
        s_roll = cs.sin(rpy[0])
        tan_pitch = cs.tan(rpy[1])
        c0 = cs.DM([1, 0, 0])
        c1 = cs.vertcat(s_roll*tan_pitch, c_roll, s_roll*secant_pitch)
        c2 = cs.vertcat(c_roll*tan_pitch, -s_roll, c_roll*secant_pitch)
        Momega_to_rpydot = cs.horzcat(c0, c1, c2)
        rpy_dot = cs.mtimes(Momega_to_rpydot, omega)
        return rpy_dot

    def euler2mat(self, rpy):
        c_rpy = cs.cos(rpy)
        s_rpy = cs.sin(rpy)
        c0 = cs.vertcat(c_rpy[2]*c_rpy[1], s_rpy[2]*c_rpy[1], -s_rpy[1])
        c1 = cs.vertcat(c_rpy[2]*s_rpy[1]*s_rpy[0] - s_rpy[2]*c_rpy[0],
                        s_rpy[2]*s_rpy[1]*s_rpy[0] + c_rpy[2]*c_rpy[0],
                        c_rpy[1]*s_rpy[0])
        c2 = cs.vertcat(c_rpy[2]*s_rpy[1]*c_rpy[0] + s_rpy[2]*s_rpy[0],
                        s_rpy[2]*s_rpy[1]*c_rpy[0] - c_rpy[2]*s_rpy[0],
                        c_rpy[1]*c_rpy[0])
        M = cs.horzcat(c0, c1, c2)
        return M

    def casadi_ode(self, t, x, u, w):
        """
        Basic dynamics of xdot for unicycle
        """
        tau = u[1:]
        thrust= self.mass*u[0]
        e3 = cs.DM([0, 0, 1])
        v = x[3:6]
        rpy = x[6:9]
        R = self.euler2mat(rpy)
        omega = x[9:12]
        J_omega_dot = cs.cross(cs.mtimes(self.J, omega), omega) + tau
        thrust_vec = cs.mtimes(R, e3)
        acceleration = (1.0/self.mass)*(thrust_vec*thrust + self.mg)
        omega_dot = cs.mtimes(self.Jinv, J_omega_dot)
        rpy_dot = self.omegaToRpyDot(rpy, omega)
        return cs.vertcat(v, acceleration, rpy_dot, omega_dot) + w
