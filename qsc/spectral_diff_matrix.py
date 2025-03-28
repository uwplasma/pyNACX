#!/usr/bin/env python3

"""
This module contains a subroutine for making spectral differentiation matrices.
"""
import jax
import jax.numpy as jnp
from jax.scipy.linalg import toeplitz
from jax import jit

def spectral_diff_matrix(n, xmin=0, xmax=2*jnp.pi):
    """
    Return the spectral differentiation matrix for n grid points
    on the periodic domain [xmin, xmax). This routine is based on the
    MATLAB code in the DMSuite package by S.C. Reddy and J.A.C. Weideman, available at
    http://www.mathworks.com/matlabcentral/fileexchange/29
    or here:
    http://dip.sun.ac.za/~weideman/research/differ.html  
    """

    h = 2 * jnp.pi / n
    kk = jnp.arange(1, n)
    n1 = int(jnp.floor((n - 1) / 2))
    n2 = int(jnp.ceil((n - 1) / 2))

    if n % 2 == 0:
        topc = 1 / jnp.tan(jnp.arange(1, n2 + 1) * h / 2)
        temp = jnp.concatenate((topc, -jnp.flip(topc[:n1])))
    else:
        topc = 1 / jnp.sin(jnp.arange(1, n2 + 1) * h / 2)
        temp = jnp.concatenate((topc, jnp.flip(topc[:n1])))

    col1 = jnp.concatenate((jnp.array([0]), 0.5 * ((-1) ** kk) * temp))
    row1 = -col1
    D = 2 * jnp.pi / (xmax - xmin) * toeplitz(col1, r=row1)
    return D

def jax_spectral_diff_matrix(n, xmin=0, xmax=2*jnp.pi): 
    """
    Return the spectral differentiation matrix for n grid points 
    on the periodic domain [xmin, xmax). This routine is based on the
    MATLAB code in the DMSuite package by S.C. Reddy and J.A.C. Weideman, available at
    http://www.mathworks.com/matlabcentral/fileexchange/29
    or here:
    http://dip.sun.ac.za/~weideman/research/differ.html  
    """
    h = 2 * jnp.pi / n
    kk = jnp.arange(1, n)
    n1 = int(jnp.floor((n - 1) / 2))
    n2 = int(jnp.ceil((n - 1) / 2))

    def if_case(): 
        topc = 1 / jnp.tan(jnp.arange(1, n2 + 1) * h / 2)
        temp = jnp.concatenate((topc, -jnp.flip(topc[:n1])))
        return topc, temp
    
    def else_case():
        topc = 1 / jnp.sin(jnp.arange(1, n2 + 1) * h / 2)
        temp = jnp.concatenate((topc, jnp.flip(topc[:n1])))
        return topc, temp

    topc, temp = jax.lax.cond((n % 2 == 0),
            lambda _: if_case(),
            lambda _: else_case(),
            None
        )

    col1 = jnp.concatenate((jnp.array([0]), 0.5 * ((-1) ** kk) * temp))
    row1 = -col1
    D = 2 * jnp.pi / (xmax - xmin) * toeplitz(col1, r=row1)
    return D
    
