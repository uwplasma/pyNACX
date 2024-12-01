"""
This module contains a function for Newton's method refactored to JAX.
"""

import jax.numpy as jnp
import numpy as np
from jax.scipy.linalg import toeplitz
from jax import jit

#logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def newton(f, x0, jac, niter=20, tol=1e-13, nlinesearch=10):
    """
    Solve a system of nonlinear equations using Newton's method with a
    line search.

    f = function providing the residual vector.
    x0 = initial guess
    jac = function providing the Jacobian.
    niter = max number of Newton iterations.
    tol = stop when the residual norm is less than this.
    """
    x = jnp.copy(x0)
    x_best = jnp.copy(x0)
    residual = f(x0)
    initial_residual_norm = calc_residual_norm(residual)
    residual_norm = initial_residual_norm
    logger.info('Beginning Newton method. residual {}'.format(residual_norm))

    newton_tolerance_achieved = False
    for jnewton in range(niter):
        last_residual_norm = residual_norm
        if residual_norm < tol:
            newton_tolerance_achieved = True
            break

        j = jac(x)
        x0 = jnp.copy(x)
        logger.info('Newton iteration {}'.format(jnewton))
        step_direction = compute_newton_step_direction(j, residual)

        step_scale = 1.0
        for jlinesearch in range(nlinesearch):
            x = x0 + step_scale * step_direction
            residual = f(x)
            residual_norm = jnp.sqrt(jnp.sum(residual * residual))
            logger.info('  Line search step {} residual {}'.format(jlinesearch, residual_norm))
            if residual_norm < last_residual_norm:
                x_best = jnp.copy(x)
                break

            step_scale /= 2
            
        if residual_norm >= last_residual_norm:
            logger.info('Line search failed to reduce residual')
            break

    if last_residual_norm > tol * 1e4:
        logger.warning('Newton solve did not get close to desired tolerance. '
                       f'Final residual: {last_residual_norm}')

    return x_best


def new_newton(f, x0, jac, niter=20, tol=1e-13, nlinesearch=10): 
    """
    Solve a system of nonlinear equations using Newton's method with a
    line search.

    Broken down to improve refactorability. 

    f = function providing the residual vector.
    x0 = initial guess
    jac = function providing the Jacobian.
    niter = max number of Newton iterations.
    tol = stop when the residual norm is less than this.
    """
    x = jnp.copy(x0)
    residual = f(x0)
    residual_norm = calc_residual_norm(residual)
    logger.info('Beginning Newton method. residual {}'.format(residual_norm))

    for iter in range(niter): 
        if check_convergence(residual_norm,tol): 
            logger.info(f'Converged in {iter} iteration. Residual norm: {residual_norm}')
            return x 
        
        jacobian = jac(x)
        step_direction = compute_newton_step_direction(jacobian, residual)

        x, residual, residual_norm = perform_line_search(f, x, step_direction, residual_norm, nlinesearch)

    logger.warning('Newton solve did not get close to desired tolerance. 'f'Final residual: {residual_norm}')
    
    return x
    

def calc_residual_norm(residual): 
    """
    compute the norm of residual vector. 
    """
    return jnp.sqrt(jnp.sum(residual**2))

def compute_newton_step_direction(jacobian, residual): 
    """
    compute the step direction
    """
    return -jnp.linalg.solve(jacobian, residual)

def perform_line_search(f, x0, step_direction, last_residual_norm, nlinesearch=10): 
    """
    perform a line search for the best step size
    """
    step_scale = 1.0
    x_1 = jnp.copy(x0)
    for jlinesearch in range(nlinesearch): 
        x = x0 + step_scale * step_direction
        residual = f(x)
        residual_norm = calc_residual_norm(residual)
        logger.info('  Line search step {} residual {}'.format(jlinesearch, residual_norm))

        if residual_norm < last_residual_norm: 
            return x, residual, residual_norm

        step_scale /= 2

    logger.info('Line search failed to reduce residual')
    return x_1, residual, last_residual_norm

def check_convergence(residual_norm, tol): 
    """
    return true if residual norm is less than tolerance
    """
    return residual_norm < tol 

