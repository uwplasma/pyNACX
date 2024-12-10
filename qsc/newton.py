"""
This module contains a function for Newton's method refactored to JAX.
"""
import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.linalg import toeplitz
from jax import jit
import logging

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
            residual_norm = calc_residual_norm(residual)
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

def new_new_newton(f, x0, jac, niter=20, tol=1e-13, nlinesearch=10): 
    x = jnp.copy(x0)
    x_best = jnp.copy(x0)
    residual = f(x0)
    initial_residual_norm = calc_residual_norm(residual)
    residual_norm = initial_residual_norm
    logger.info('Beginning Newton method. residual {}'.format(residual_norm))

    for jnewton in range(niter):
        last_residual_norm = residual_norm
        j = jac(x)
        x0 = jnp.copy(x)
        logger.info('Newton iteration {}'.format(jnewton))
        step_direction = compute_newton_step_direction(j, residual)

        y, residual_norm, residual = new_new_line_search(f, x0, step_direction, last_residual_norm,  nlinesearch=nlinesearch)
        
        con = residual_norm < last_residual_norm
        
        x = jax.lax.cond(con, lambda _ : y, lambda _ : x, None)
        
        x_best = jnp.copy(x)
        #print(x_best)
        #print(step_direction)
        
        con2 = residual_norm < tol 
        
        nlinesearch = jax.lax.cond(con2, lambda _ : 1, lambda _ : nlinesearch, None)
        
        
    return x_best
    

def new_new_line_search(f, x0, step_direction, last_residual_norm,  nlinesearch=10): 
    step_scale = 1.0
    for jlinesearch in range(nlinesearch):
        x = x0 + step_scale * step_direction
        residual = f(x)
        residual_norm = calc_residual_norm(residual)
        con = residual_norm < last_residual_norm
        logger.info('  Line search step {} residual {}'.format(jlinesearch, residual_norm))
        step_scale = jax.lax.cond(con, lambda _ : step_scale,  lambda _ : step_scale/2, None)
        print(x)
        #print(step_scale)
    return x, residual_norm, residual
        
        

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

    def newton_body(i, state): # i is required by jax
    
        x ,last_residual = state
        residual = f(x)
        residual_norm = calc_residual_norm(residual)

        converged = check_convergence(residual_norm, tol)
        
        x = jax.lax.cond(converged, lambda _: x, lambda _: x, None) #when converged nothing is updated

        jacobian = jac(x)
        step_direction = compute_newton_step_direction(jacobian, residual)

        x, residual_norm = perform_line_search(f, x, step_direction, last_residual, nlinesearch)
        return (x, residual_norm) 
    
        
    #initial values
    initial_res = f(x0)
    last_residual = calc_residual_norm(initial_res)
    state = (x0, last_residual)

    _, results = jax.lax.scan(newton_body, state, jnp.arange(niter))

    return results[-1][0] # return the top item in the stack and only the x value

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
    def line_search_body(i, state):
        step_scale = state[0]
        x = state[1]

        x_test = x0 + step_scale * step_direction
        residual = f(x_test)

        trial_residual_norm = calc_residual_norm(residual)

        improved = check_improvement(trial_residual_norm, residual_norm) # boolean used in jax cond
        step_scale = jax.lax.cond(improved, lambda _: step_scale, lambda _: step_scale / 2, None)
        x = jax.lax.cond(improved, lambda _: x_test, lambda _: x, None)
        residual_norm = jax.lax.cond(improved, lambda _: trial_residual_norm, lambda _: residual_norm, None)

        return (step_scale, x, residual_norm)

    initial_state = (1.0, x0, last_residual_norm)

    #jax iteration 
    final_state = jax.lax.fori_loop(0, nlinesearch, line_search_body, initial_state)
    
    _, x, residual_norm = final_state

    return x, residual_norm

def check_improvement(trial, last): 
    return trial < last

def check_convergence(residual_norm, tol): 
    """
    return true if residual norm is less than tolerance
    """
    return residual_norm < tol 

def econ_newton(f, x0, niter=20, tol=1e-13, nlinesearch=10):
    """
    newton solver taken from https://jax.quantecon.org/newtons_method.html
    """
    x = x0
    f_jac = jax.jacobian(f)
    @jax.jit
    def q(x): 
        return x - jnp.linalg.solve(f_jac(x), f(x))
    
    error = tol + 1
    n = 0 
    while error > tol : 
        n += 1
        if (n > niter): 
            raise Exception("max iter reached")
        
        y = q(x)
        error = jnp.linalg.norm(x-y)
        x = y 
    return x          
    
    