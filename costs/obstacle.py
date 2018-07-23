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

    def distance(self, x, compute_grads=False):
        z = self.mapState(x)
        error = z - self.center
        distance = min(np.linalg.norm(error)-self.radius, 0)
        jac = None
        if compute_grads:
            if distance >= -1e-12:
                jac = np.zeros_like(x)
            else:
                z_x = self.mapStateJacobian(x)
                jac = (1.0/(distance + self.radius))*(np.dot(z_x.T, error))
        return distance, jac
