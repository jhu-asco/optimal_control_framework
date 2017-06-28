# Optimal Control Framework
This project is a simple framework for specifying dynamics and jacobians of dynamics and plug the dynamics into optimal control problems. The main components in the package are:

- dynamics    : Dynamic systems provide xdot and also jacobians of xdot wrt x and u and w (Supports stochastic systems)
- controllers : Controllers provide control at a given time x and u (supports stochastic controllers)
- integrators : Integrators take in dynamics, x0, and find the state at specified times ts (Supports controllers)
- costs       : Cost function provides stage-wise and terminal costs along a trajectory
- sampling    : Samples a specified number of trajectories using a specified integrator, controller, and provides the costs, states and controls for the sampled trajectories
