"""
This module contains methods for performing mathematical operations in calculate_r1.py
"""
import jax.numpy as jnp
from .init_axis_helpers import *
from .util import fourier_minimum

"""
helpers for solve_sigma_equation
"""

def calc_iotaN(iota, helicity, nfp): 
  return iota + helicity * nfp 


"""
helpers for r1_diagnostics
"""

def derive_calc_X1s(nphi): 
  """
  calculate X1s as a function of inputed parameters
  """
  return jnp.zeros(nphi)

def derive_calc_X1c(etabar, nphi, nfp, rc, rs, zc, zs): 
  """
  calulate the X1c as a function of inputed parameters
  """
  curvature = calc_curvature(nphi, nfp, rc, rs, zc, zs)
  return etabar / curvature

def derive_calc_Y1s(sG, spsi, nphi, nfp, rc, rs, zc, zs, etabar):
  """
  calulate the Y1s as a function of inputed parameters
  """
  curvature = calc_curvature(nphi, nfp, rc, rs, zc, zs)
  return sG * spsi * curvature / etabar 

def calc_Y1s(sG, spsi, curvature, etabar):
  return sG * spsi * curvature / etabar 

def derive_calc_Y1c(sG, spsi, nphi, nfp, rc, rs, zc, zs, sigma0, etabar):
  from .calculate_r1 import solve_sigma_equation
  """
  calulate the Y1c as a function of inputed parameters
  """
  helicity = derive_helicity(nphi, nfp, rc, rs, zc, zs)
  sigma = solve_sigma_equation(nphi, sigma0, helicity, nfp)[0] #not yet derivable because I dont have the newtons method working with is within this 
  # will also need to create a derivable version of solve sigma equation that first derives helicity and nfp
  curvature = calc_curvature(nphi, nfp, rc, rs, zc, zs)
  return  sG * spsi * curvature * sigma / etabar 

def calc_Y1c(sG, spsi, curvature, sigma, etabar): 
  return sG * spsi * curvature * sigma / etabar 

def calc_angle(helicity, nfp, varphi): 
  return -helicity * nfp * varphi

def calc_sinangle(angle): 
  return jnp.sin(angle)

def calc_cosangle(angle): 
  return jnp.cos(angle)

def calc_X1s_untwisted(X1s, cosangle, X1c, sinangle): 
  return X1s * cosangle + X1c * sinangle

def calc_X1c_untwisted(X1s, sinangle, X1c, cosangle): 
  return X1s * (-sinangle) + X1c * cosangle 

def calc_Y1s_untwisted(Y1s, cosangle, Y1c, sinangle): 
  return Y1s * cosangle + Y1c * sinangle

def calc_Y1c_untwisted(Y1s, sinangle, Y1c, cosangle): 
  return Y1s * (-sinangle) + Y1c * cosangle

def calc_p(X1s, X1c, Y1s, Y1c): 
  return X1s * X1s + X1c * X1c + Y1s * Y1s + Y1c * Y1c

def calc_q(X1s, Y1c, X1c, Y1s): 
  return X1s * Y1c - X1c * Y1s

def calc_elongation(p, q): 
  return (p + jnp.sqrt(p * p - 4 * q * q)) / (2 * jnp.abs(q))

def derive_elongation(sG, spsi, nphi, nfp, rc, rs, zc, zs, sigma0, etabar): 
  """
  calculate elongation as a function of inputed parameters
  """
  X1s = derive_calc_X1s(nphi)
  X1c = derive_calc_X1c(etabar, nphi, nfp, rc, rs, zc, zs)
  Y1s = derive_calc_Y1s(sG, spsi, nphi, nfp, rc, rs, zc, zs, etabar)
  Y1c =  derive_calc_Y1c(sG, spsi, nphi, nfp, rc, rs, zc, zs, sigma0, etabar)
  p = calc_p(X1s, X1c, Y1s, Y1c)
  q = calc_q(X1s, X1c, Y1s, Y1c)
  return calc_elongation(p,q)

def calc_mean_elongation(elongation, d_l_d_phi): 
  return jnp.sum(elongation * d_l_d_phi) / jnp.sum(d_l_d_phi)

def derive_mean_elongation(sG, spsi, sigma0, etabar, nphi, nfp, rc, rs, zc, zs):
  """
  calulate the mean_elongation as a function of inputed parameters
  """
  X1s = derive_calc_X1s(nphi)
  X1c = derive_calc_X1c(etabar, nphi, nfp, rc, rs, zc, zs)
  Y1s = derive_calc_X1s(sG, spsi, nphi, nfp, rc, rs, zc, zs, etabar)
  Y1c = derive_calc_Y1c(sG, spsi, nphi, nfp, rc, rs, zc, zs, sigma0, etabar)
  p = calc_p(X1s, X1c, Y1s, Y1c)
  q = calc_q(X1s, X1c, Y1s, Y1c)
  elongation = calc_elongation(p,q) 
  d_l_d_phi = calc_d_l_d_phi(nphi, nfp, rc, rs, zc, zs)
  return calc_mean_elongation(elongation, d_l_d_phi)

def derive_max_elongation(sG, spsi, nphi, nfp, rc, rs, zc, zs, sigma0, etabar): 
  elongation = derive_elongation(sG, spsi, nphi, nfp, rc, rs, zc, zs, sigma0, etabar)
  return -fourier_minimum(-elongation) # not sure if derivable

def derive_d_X1c_d_varphi(etabar, nphi, nfp, rc, rs, zc, zs): 
  d_d_varphi = calc_d_d_varphi(rc, zs, rs, zc, nfp,  nphi)
  X1c =  derive_calc_X1c(etabar, nphi, nfp, rc, rs, zc, zs)
  jnp.matmul(d_d_varphi, X1c)
  
def derive_d_X1s_d_varphi(rc, zs, rs, zc, nfp,  nphi): 
  d_d_varphi = calc_d_d_varphi(rc, zs, rs, zc, nfp,  nphi)
  X1s = derive_calc_X1s(nphi)
  return jnp.matmul(d_d_varphi, X1s)
  pass

def derive_d_Y1s_d_varphi(sG, spsi, nphi, nfp, rc, rs, zc, zs, etabar): 
  d_d_varphi = calc_d_d_varphi(rc, zs, rs, zc, nfp,  nphi)
  Y1s = derive_calc_Y1s(sG, spsi, nphi, nfp, rc, rs, zc, zs, etabar)
  return jnp.matmul(d_d_varphi, Y1s)

def derive_d_Y1c_d_varphi(sG, spsi, nphi, nfp, rc, rs, zc, zs, sigma0, etabar): 
  d_d_varphi = calc_d_d_varphi(rc, zs, rs, zc, nfp,  nphi)
  Y1c = derive_calc_Y1c(sG, spsi, nphi, nfp, rc, rs, zc, zs, sigma0, etabar)
  return jnp.matmul(d_d_varphi, Y1c)
  