#!/usr/bin/env python3

import unittest
import numpy as np
import jax.numpy as jnp
from qsc.newton import new_newton, econ_newton, new_new_newton
import qsc

class NewtonTests(unittest.TestCase):

    def test_2d(self):
        """
        Try a little 2D nonlinear example.
        """
        def func(x):
            return jnp.array([x[1] - jnp.exp(x[0]), x[0] + x[1]])
        
        def jac(x):
            return np.array([[-np.exp(x[0]), 1],
                             [1, 1]])
        x0 = np.zeros(2)
        
        #soln = new_newton(func, x0, jac)
        soln = new_new_newton(func, x0, jac, niter=4, nlinesearch=20)
        np.testing.assert_allclose(soln, [-0.5671432904097838, 0.5671432904097838]) 
           
        
if __name__ == "__main__":
    unittest.main()
