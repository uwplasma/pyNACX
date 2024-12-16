import jax.numpy as jnp
from init_axis_helpers import *
from init_axis import calculate_helicity

def helper_residual(rc, rs, zs, zc, etabar, nphi, nfp, spsi, I2, B0, sigma0, sG, x): 
  """
  this helper residual runs without anything from self 
  """
  sigma = jnp.copy(x)
  sigma[0] = sigma0
  iota = x[0]
    
  nfourier = jnp.max([len(rc), len(zs), len(rs), len(zc)])

  phi = jnp.linspace(0, 2 * jnp.pi / nfp, nphi, endpoint=False)
  
  n = jnp.arange(0, nfourier) * nfp
  angles = jnp.outer(n, phi)
  sinangles = jnp.sin(angles)
  cosangles = jnp.cos(angles)

  
  R0 = jnp.dot(rc, cosangles) + jnp.dot(rs, sinangles)
  
  R0p = jnp.dot(rc, -n[:, jnp.newaxis] * sinangles) + jnp.dot(rs, n[:, jnp.newaxis] * cosangles)
  Z0p = jnp.dot(zc, -n[:, jnp.newaxis] * sinangles) + jnp.dot(zs, n[:, jnp.newaxis] * cosangles)
  R0pp = jnp.dot(rc, -n[:, jnp.newaxis]**2 * cosangles) + jnp.dot(rs, -n[:, jnp.newaxis]**2 * sinangles)
  Z0pp = jnp.dot(zc, -n[:, jnp.newaxis]**2 * cosangles) + jnp.dot(zs, -n[:, jnp.newaxis]**2 * sinangles)
  
  d_l_d_phi = jnp.sqrt(R0 * R0 + R0p * R0p + Z0p * Z0p)
  d2_l_d_phi2 = (R0 * R0p + R0p * R0pp + Z0p * Z0pp) / d_l_d_phi
  
  d_r_d_phi_cylindrical = jnp.array([R0p, R0, Z0p]).transpose()
  d2_r_d_phi2_cylindrical = jnp.array([R0pp - R0, 2 * R0p, Z0pp]).transpose()
  
  d_tangent_d_l_cylindrical = ((-d_r_d_phi_cylindrical * d2_l_d_phi2[:, jnp.newaxis] / d_l_d_phi[:, jnp.newaxis]) \
                                + d2_r_d_phi2_cylindrical) / (d_l_d_phi[:, jnp.newaxis] * d_l_d_phi[:, jnp.newaxis])

  
  curvature = calc_curvature(nphi, nfp, rc, rs, zc, zs)
  
  normal_cylindrical = d_tangent_d_l_cylindrical / curvature[:, jnp.newaxis]
  
  etabar_squared_over_curvature_squared = etabar ** 2 / curvature ** 2

  G0 = calc_G0(sG,  nphi,  B0, nfp, rc, rs, zc, zs)

  
  d_d_varphi = calc_d_d_varphi(rc, zs, rs, zc, nfp, nphi) # i wonder if original rc and rs are used 
  helicity = calculate_helicity(phi, normal_cylindrical, spsi, sG)
  torsion = calc_torsion(nphi, nfp, rc, rs, zc, zs, sG, etabar, spsi, sigma0)
  r = jnp.matmul(d_d_varphi, sigma) \
        + (iota + helicity * nfp) * \
        (etabar_squared_over_curvature_squared * etabar_squared_over_curvature_squared + 1 + sigma * sigma) \
        - 2 * etabar_squared_over_curvature_squared * (-spsi * torsion + I2 / B0) * G0 / B0
  return r