# pyNACX

!!! A jax compatible periodic spline is needed and can only be found in this repo https://github.com/semzit/interpax
  clone it locally and then run pyQSX with the interpax virtual environment
  
Refactor of pyQSC to use JAX for automatic differentiation and GPU acceleration.

The Python implementation of the Quasisymmetric Stellarator Construction method
done by pyQSC implements the equations derived by Garren and Boozer (1991) for MHD equilibrium near the magnetic axis. The documentation for the original pyQSC can be found [here](https://landreman.github.io/pyQSC/).


# To Do List

- [ ] Refactor the code to use JAX for automatic differentiation (while refactoring run the tests to make sure everything is still working, ```python3 -m unittest```)
- [ ] Refactor Tests to use JAX
- [ ] Refactor to use tensors to create several stellarators at once, can be used for machine learning or optimization purposes
- [ ] Add possibility to use plotly for plotting, specially it is useful for 3D plots of the stellarator
