#!/usr/bin/env python3

"""
Various utility functions
"""

import logging
import numpy as np
import scipy.optimize
from qsc.fourier_interpolation import fourier_interpolation
from interpax import CubicSpline


import jax
import jax.scipy.optimize as jso
import jax.numpy as jnp
from jaxopt import ScipyMinimize

from qsc.types import Results


#logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

mu0 = 4 * jnp.pi * 1e-7

# Define periodic spline interpolant conversion used in several scripts and plotting

# Define periodic spline interpolant conversion used in several scripts and plotting
def convert_to_spline(array, phi, nfp):
    sp = CubicSpline(jnp.append(phi,2*jnp.pi/nfp), jnp.append(array,array[0]), bc_type='periodic', check=False) #need to get open source to work here
    return sp

class Struct():
    """
    This class is just an empty mutable object to which we can attach
    attributes.
    """
    pass

def jax_fourier_minimum(y):
    """
    Given uniformly spaced data y on a periodic domain, find the
    minimum of the spectral interpolant.
    """
    y = jnp.array(y)  # Ensure correct argument

    # Handle the case of a constant
    def handle_constant_case(y):
        return jnp.repeat(y[0], 3)

    def func(x):
            interp = fourier_interpolation(y, jnp.array([x]))
            return interp[0]
        
    def handle_non_constant_case(y):
        n = len(y)
        dx = 2 * jnp.pi / n
        
        index = jnp.argmin(y)
        
        
    
        f0 = func(index * dx)
        
        #look for bracket 
        def body(j, carry): 
            fm = func(carry[0])
            fp = func(carry[2])
            
            #cond = f0 < fm and f0 < fp

            carry = jax.lax.cond(jnp.logical_and(f0 < fm, f0 < fp),
                                 lambda _: jnp.array([index - j, index, index + j]) * dx,
                                 lambda _: carry,
                                 None)        
            
            return carry
        
        carry = jnp.array([index, index, index]) * dx 
        
        bracket = jax.lax.fori_loop(1, 4, body, carry)
        
        return bracket

    
    max_min_diff = jnp.max(y) - jnp.min(y)
    mean_y_abs = jnp.abs(jnp.mean(y))
    
    is_constant = (max_min_diff / jnp.maximum(1e-14, mean_y_abs) < 1e-14)
    
    result = jax.lax.cond(is_constant, handle_constant_case, handle_non_constant_case, y)
    
    
    return jso.minimize(func, result, method = 'BFGS')

def fourier_minimum(y):
    """
    Given uniformly spaced data y on a periodic domain, find the
    minimum of the spectral interpolant.
    """
    # Handle the case of a constant:
    y = jnp.array(y) # ensure correct argument
    print(y)
    if (jnp.max(y) - jnp.min(y)) / jnp.max(jnp.array([1e-14, jnp.abs(jnp.mean(y))])) < 1e-14:
        return y[0]
    
    n = len(y)
    dx = 2 * jnp.pi / n
    # Compute a rough guess for the minimum, given by the minimum of
    # the discrete data:
    index = jnp.argmin(y)
    def func(x):
        interp = fourier_interpolation(y, jnp.array([x]))
        logger.debug('fourier_minimum.func called at x={}, y={}'.format(x, interp[0]))
        return interp[0]

    # Try to find a bracketing interval, using successively wider
    # intervals.
    f0 = func(index * dx)
    found_bracket = False
    for j in range(1, 4):
        bracket = jnp.array([index - j, index, index + j]) * dx
        fm = func(bracket[0])
        fp = func(bracket[2])
        
        if f0 < fm and f0 < fp:
            found_bracket = True
            break
    if not found_bracket:
        # We could throw an exception, though scipy will do that anyway
        pass

    logger.info('bracket={}, f(bracket)={}'.format(bracket, [func(bracket[0]), func(bracket[1]), func(bracket[2])]))
    #solution = scipy.optimize.minimize_scalar(func, bracket=bracket, options={"disp": True})
    solution = scipy.optimize.minimize_scalar(func, bracket=bracket)
    return solution.fun

def to_Fourier(R_2D, Z_2D, nfp, mpol, ntor, lasym):
    """
    This function takes two 2D arrays (R_2D and Z_2D), which contain
    the values of the radius R and vertical coordinate Z in cylindrical
    coordinates of a given surface and Fourier transform it, outputing
    the resulting cos(theta) and sin(theta) Fourier coefficients

    The first dimension of R_2D and Z_2D should correspond to the
    theta grid, while the second dimension should correspond to the
    phi grid.

    Args:
        R_2D: 2D array of the radial coordinate R(theta, phi) of a given surface
        Z_2D: 2D array of the vertical coordinate Z(theta, phi) of a given surface
        nfp: number of field periods of the surface
        mpol: resolution in poloidal Fourier space
        ntor: resolution in toroidal Fourier space
        lasym: False if stellarator-symmetric, True if not
    """
    shape = jnp.array(R_2D).shape
    ntheta = shape[0]
    nphi_conversion = shape[1]
    theta = jnp.linspace(0, 2 * jnp.pi, ntheta, endpoint=False)
    phi_conversion = jnp.linspace(0, 2 * jnp.pi / nfp, nphi_conversion, endpoint=False)
    RBC = jnp.zeros((int(2 * ntor + 1), int(mpol + 1)))
    RBS = jnp.zeros((int(2 * ntor + 1) , int(mpol + 1)))
    ZBC = jnp.zeros((int(2 * ntor + 1), int(mpol + 1)))
    ZBS = jnp.zeros((int(2 * ntor + 1), int(mpol + 1)))
    factor = 2 / (ntheta * nphi_conversion)
    phi2d, theta2d = jnp.meshgrid(phi_conversion, theta)
    for m in range(mpol+1):
        nmin = -ntor
        if m==0: nmin = 1
        for n in range(nmin, ntor+1):
            angle = m * theta2d - n * nfp * phi2d
            sinangle = jnp.sin(angle)
            cosangle = jnp.cos(angle)
            factor2 = factor
            # The next 2 lines ensure inverse Fourier transform(Fourier transform) = identity
            if jnp.mod(ntheta,2) == 0 and m  == (ntheta/2): factor2 = factor2 / 2
            if jnp.mod(nphi_conversion,2) == 0 and abs(n) == (nphi_conversion/2): factor2 = factor2 / 2
            RBC = RBC.at[n + ntor, m].set(jnp.sum(R_2D * cosangle * factor2))
            RBS = RBS.at[n + ntor, m].set(jnp.sum(R_2D * sinangle * factor2))
            ZBC = ZBC.at[n + ntor, m].set(jnp.sum(Z_2D * cosangle * factor2))
            ZBS = ZBS.at[n + ntor, m].set(jnp.sum(Z_2D * sinangle * factor2))
    RBC = RBC.at[ntor,0].set(jnp.sum(R_2D) / (ntheta * nphi_conversion))
    ZBC = ZBC.at[ntor,0].set(jnp.sum(Z_2D) / (ntheta * nphi_conversion))

    if not lasym:
        RBS = 0
        ZBC = 0

    return RBC, RBS, ZBC, ZBS

def B_mag(results: Results, r, theta, phi, Boozer_toroidal = False):
    '''
    Function to calculate the modulus of the magnetic field B for a given
    near-axis radius r, a Boozer poloidal angle theta (not vartheta) and
    a cylindrical toroidal angle phi if Boozer_toroidal = True or the
    Boozer angle varphi if Boozer_toroidal = True

    Args:
      r: the near-axis radius
      theta: the Boozer poloidal angle
      phi: the cylindrical or Boozer toroidal angle
      Boozer_toroidal: False if phi is the cylindrical toroidal angle, True for the Boozer one
    '''
    if Boozer_toroidal == False:
        thetaN = theta - (results.pre_calculation_results.solve_sigma_equation_results.iota - results.pre_calculation_results.solve_sigma_equation_results.iotaN) * (phi + results.pre_calculation_results.init_axis_results.nu_spline(phi))
    else:
        thetaN = theta - (results.pre_calculation_results.solve_sigma_equation_results.iota - results.pre_calculation_results.solve_sigma_equation_results.iotaN) * phi

    B = results.inputs.B0*(1 + r * results.inputs.etabar * jnp.cos(thetaN))

    # Add O(r^2) terms if necessary:
    if results.inputs.order != 'r1':
        if Boozer_toroidal == False:
            B20_spline = convert_to_spline(results.complete_calculation_results.complete_r2_results.r2_results.B20, results.pre_calculation_results.init_axis_results.phi, results.inputs.nfp)
        else:
            B20_spline = spline(jnp.append(results.pre_calculation_results.init_axis_results.varphi, 2 * jnp.pi / results.inputs.nfp),
                                     jnp.append(results.complete_calculation_results.complete_r2_results.r2_results.B20, results.complete_calculation_results.complete_r2_results.r2_results.B20[0]),
                                     bc_type='periodic')

        B += (r**2) * (B20_spline(phi) + results.inputs.B2c * jnp.cos(2 * thetaN) + results.inputs.B2s * jnp.sin(2 * thetaN))

    return B
