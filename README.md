# Optimal Control Framework
This project is a simple framework for specifying dynamics and jacobians of dynamics and plug the dynamics into optimal control problems. The main components in the package are:

- dynamics    : Dynamic systems provide xdot and also jacobians of xdot wrt x and u and w (Supports stochastic systems)
- controllers : Controllers provide control at a given time x and u (supports stochastic controllers)
- integrators : Integrators take in dynamics, x0, and find the state at specified times ts (Supports controllers)
- costs       : Cost function provides stage-wise and terminal costs along a trajectory
- sampling    : Samples a specified number of trajectories using a specified integrator, controller, and provides the costs, states and controls for the sampled trajectories

# Installing Module
To install module, either copy the folder to a folder in python path or extend the the python path using bashrc to include the parent folder containing this module.
You also need to install
1. Casadi using `pip install casadi`
2. Transforms3d using `pip install transforms3d`
3. Slycot and controls using `pip install slycot control` (Install gfortran on Ubuntu)

# Testing
To test the repo, use `nose2 -v` in the main folder of the project. The python path should be extended before running tests
