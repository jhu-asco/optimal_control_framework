#!/usr/bin/env python

from optimal_control_framework.costs import SphericalObstacle
from scipy import optimize
import unittest
import numpy as np
import numpy.testing as np_testing


class TestObstacle(unittest.TestCase):
    def setUp(self):
        self.n = 3  # lenght of state
        self.center = np.array([1., 2.])
        self.radius = 2.0
        self.obstacle = SphericalObstacle(self.center, self.radius)

    def test_mapState(self):
        np_testing.assert_almost_equal(
            self.obstacle.mapState(np.array([1, 3, 4])),
            np.array([1, 3]))

    def test_mapStateJacobian(self):
        np_testing.assert_almost_equal(
            self.obstacle.mapStateJacobian(np.array([1, 3, 4])),
            np.array([[1, 0, 0], [0, 1, 0]]))

    def test_findError(self):
        np_testing.assert_almost_equal(
            self.obstacle.findError(np.array([1, 0, 8]))[0],
            np.array([0, -2]))
        np_testing.assert_almost_equal(
            self.obstacle.findError(np.array([-1, 2, 8]))[0],
            np.array([-2, 0]))

    def test_distance(self):
        self.assertEqual(self.obstacle.distance(np.array([10, 0, 0]))[0], 0.0)
        self.assertEqual(self.obstacle.distance(np.array([3, 2, 3]))[0], 0.0)
        self.assertLess(self.obstacle.distance(np.array([1, 1, 3]))[0], 0.0)

    def test_distance_jacobian(self):
        def distance_fun(x): return self.obstacle.distance(x)[0]
        x = np.random.sample(3)
        grad = optimize.approx_fprime(x, distance_fun, 1e-6)
        dist, jac = self.obstacle.distance(x, True)
        np_testing.assert_almost_equal(jac, grad, decimal=4)


if __name__ == "__main__":
    unittest.main()
