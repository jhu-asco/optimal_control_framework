#!/usr/bin/env python

from abc import ABCMeta, abstractmethod
import numpy as np


class AbstractObstacle(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def mapState(self, x):
        pass

    @abstractmethod
    def mapStateJacobian(self, x):
        pass

    @abstractmethod
    def distance(self, x, compute_grads=False):
        pass


class SphericalObstacle(AbstractObstacle):
    tol = 1e-12
    x_prev = None
    use_xprev = False

    def __init__(self, center, radius, use_xprev=False):
        self.center = center
        self.radius = radius
        self.n = self.center.size
        self.projectionMatrix = None
        SphericalObstacle.use_xprev = use_xprev

    def getProjectionMatrix(self, x_size):
        projection_matrix = np.zeros((self.n, x_size))
        projection_matrix[:self.n, :self.n] = np.eye(self.n)
        return projection_matrix

    def mapState(self, x):
        if x.size >= self.n:
            return x[:self.n]
        else:
            raise RuntimeError("x size bigger than center")

    def mapStateJacobian(self, x):
        if self.projectionMatrix is None:
            self.projectionMatrix = self.getProjectionMatrix(x.size)
        return self.projectionMatrix

    def distance_substep(self, error, compute_grads=False):
        distance = min(np.linalg.norm(error) - self.radius, 0)
        jac = None
        if compute_grads:
            if distance >= -self.tol:
                jac = None
            else:
                jac = (1.0 / (distance + self.radius)) * error
        return distance, jac

    def findLambda(self, e_0, e_1):
        v = e_0 - e_1
        tol = 1e-3
        v_norm_square = np.dot(v, v)
        v_norm_square = max(v_norm_square, tol)
        out = np.dot(v, e_0) / v_norm_square
        num_grad = - e_0 + 2 * out * v
        out_grad = num_grad / v_norm_square
        return out, out_grad

    def findError(self, x):
        z1 = self.mapState(x)
        e1 = z1 - self.center
        if self.use_xprev and self.x_prev is not None:
            z0 = self.mapState(self.x_prev)
            e0 = z0 - self.center
            l, l_e = self.findLambda(e0, e1)
            ebar = e1 * l + e0 * (1 - l)
            ebar_e1_T = np.outer(l_e, (e1 - e0)) + l * np.eye(e1.size)
        else:
            ebar = e1
            ebar_e1_T = 1
        z_x = self.mapStateJacobian(x)
        ebar_x_T = np.dot(z_x.T, ebar_e1_T)
        return ebar, ebar_x_T

    @classmethod
    def updatePreviousX(self, x):
        if self.use_xprev:
            # Update previous state
            self.x_prev = x

    def distance(self, x, compute_grads=False):
        ebar, ebar_x_T = self.findError(x)
        distance, jac = self.distance_substep(ebar, compute_grads)
        if compute_grads:
            if distance < -self.tol:
                jac = np.dot(ebar_x_T, jac)
            else:
                jac = np.zeros_like(x)
        return distance, jac
