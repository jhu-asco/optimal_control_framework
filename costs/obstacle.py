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
    def __init__(self, center, radius):
        self.center = center
        self.radius = radius
        self.n = self.center.size

    def mapState(self, x):
        if x.size >= self.n:
            return x[:self.n]
        else:
            raise RuntimeError("x size bigger than center")

    def mapStateJacobian(self, x):
        jacobian = np.zeros((self.n, x.size))
        jacobian[:self.n, :self.n] = np.eye(self.n)
        return jacobian

    def distance_substep(self, error, compute_grads=False):
        distance = min(np.linalg.norm(error)-self.radius, 0)
        jac = None
        if compute_grads:
            if distance >= -self.tol:
                jac = None
            else:
                jac = (1.0/(distance + self.radius))*error
        return distance, jac


    def distance(self, x, compute_grads=False):
        z = self.mapState(x)
        error = z - self.center
        distance, jac = self.distance_substep(error, compute_grads)
        if compute_grads:
            if distance < -self.tol:
                z_x = self.mapStateJacobian(x)
                jac = np.dot(z_x.T, jac)
            else:
                jac = np.zeros_like(x)
        return distance, jac
