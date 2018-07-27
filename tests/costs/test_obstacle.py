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

    def test_findError_no_previous_state(self):
        np_testing.assert_almost_equal(
            self.obstacle.findError(np.array([1, 0, 8]))[0],
            np.array([0, -2]))
        np_testing.assert_almost_equal(
            self.obstacle.findError(np.array([-1, 2, 8]))[0],
            np.array([-2, 0]))

    def test_findError_with_previous_state(self):
        SphericalObstacle.use_xprev = True
        x1 = np.array([1, 0, 8])
        x2 = np.array([-1, 2, 8])
        x3 = np.array([-1, 0, 8])
        SphericalObstacle.updatePreviousX(x1)
        e1, _ = self.obstacle.findError(x2)
        delta_x = (x2 - x1)
        self.assertAlmostEqual(np.dot(e1, delta_x[:2]), 0.0, places=4)
        SphericalObstacle.updatePreviousX(x2)
        e2, _ = self.obstacle.findError(x3)
        np_testing.assert_almost_equal(e2, x2[:2] - self.obstacle.center)

    def test_findLambda(self):
        e0 = np.array([0, 0]) - self.obstacle.center
        e1 = np.array([1, 0]) - self.obstacle.center
        l, l_e1 = self.obstacle.findLambda(e0, e1)
        self.assertEqual(l, 1)
        e0 = np.array([1, 0]) - self.obstacle.center
        e1 = np.array([-1, 2]) - self.obstacle.center
        l, l_e1 = self.obstacle.findLambda(e0, e1)
        self.assertLess(l, 1.0)
        self.assertGreater(l, 0.0)

    def test_findLambda_jacobian(self):
        e0 = np.array([1, 0]) - self.obstacle.center
        e1 = np.array([-1, 2]) - self.obstacle.center

        def l_fun(x): return self.obstacle.findLambda(e0, x)[0]
        l_e1_fd = optimize.approx_fprime(e1, l_fun, 1e-6)
        l, l_e1 = self.obstacle.findLambda(e0, e1)
        np_testing.assert_almost_equal(l_e1, l_e1_fd, decimal=4)

    def test_findError_jacobian_wprevx_none(self):
        SphericalObstacle.use_xprev = True

        def error_fun(x): return np.linalg.norm(
            self.obstacle.findError(x)[0])**2
        x = np.random.sample(3)
        grad = optimize.approx_fprime(x, error_fun, 1e-6)
        error, jac = self.obstacle.findError(x)
        grad_analytic = 2 * np.dot(jac, error)
        np_testing.assert_almost_equal(grad_analytic, grad, decimal=4)

    def test_findError_jacobian_wprevx(self):
        SphericalObstacle.use_xprev = True
        SphericalObstacle.updatePreviousX(np.random.sample(3))

        def error_fun(x): return np.linalg.norm(
            self.obstacle.findError(x)[0])**2
        x = np.random.sample(3)
        grad = optimize.approx_fprime(x, error_fun, 1e-6)
        error, jac = self.obstacle.findError(x)
        grad_analytic = 2 * np.dot(jac, error)
        np_testing.assert_almost_equal(grad_analytic, grad, decimal=4)

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

    def test_distance_jacobian_wprevx(self):
        SphericalObstacle.use_xprev = True
        SphericalObstacle.updatePreviousX(np.random.sample(3))

        def distance_fun(x): return self.obstacle.distance(x)[0]
        x = np.random.sample(3)
        grad = optimize.approx_fprime(x, distance_fun, 1e-6)
        dist, jac = self.obstacle.distance(x, True)
        np_testing.assert_almost_equal(jac, grad, decimal=4)


if __name__ == "__main__":
    unittest.main()
