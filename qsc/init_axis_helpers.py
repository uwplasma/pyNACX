import numpy as jnp 



def calc_curvature(nphi, nfp, rc, rs, zc, zs): 
  """
  this function returns curvature as a fucntion of inputed parameters within qsc.py 
  """
  # Generate phi
  phi = jnp.linspace(0, 2 * jnp.pi / nfp, nphi, endpoint=False)
  
  nfourier = jnp.max([len(rc), len(zs), len(rs), len(zc)])

  # Compute n and the angles
  n = jnp.arange(0, nfourier) * nfp
  angles = jnp.outer(n, phi)
  sinangles = jnp.sin(angles)
  cosangles = jnp.cos(angles)

  # Compute R0, Z0, R0p, Z0p, R0pp, Z0pp, R0ppp, Z0ppp
  R0 = jnp.dot(rc, cosangles) + jnp.dot(rs, sinangles)
  R0p = jnp.dot(rc, -n[:, jnp.newaxis] * sinangles) + jnp.dot(rs, n[:, jnp.newaxis] * cosangles)
  Z0p = jnp.dot(zc, -n[:, jnp.newaxis] * sinangles) + jnp.dot(zs, n[:, jnp.newaxis] * cosangles)
  R0pp = jnp.dot(rc, -n[:, jnp.newaxis]**2 * cosangles) + jnp.dot(rs, -n[:, jnp.newaxis]**2 * sinangles)
  Z0pp = jnp.dot(zc, -n[:, jnp.newaxis]**2 * cosangles) + jnp.dot(zs, -n[:, jnp.newaxis]**2 * sinangles)
  
  
  d_l_d_phi = jnp.sqrt(R0 * R0 + R0p * R0p + Z0p * Z0p)
  d2_l_d_phi2 = (R0 * R0p + R0p * R0pp + Z0p * Z0pp) / d_l_d_phi
 # For these next arrays, the first dimension is phi, and the 2nd dimension is (R, phi, Z).
  d_r_d_phi_cylindrical = jnp.array([R0p, R0, Z0p]).transpose()
  d2_r_d_phi2_cylindrical = jnp.array([R0pp - R0, 2 * R0p, Z0pp]).transpose()
  
  d_tangent_d_l_cylindrical = ((-d_r_d_phi_cylindrical * d2_l_d_phi2[:, jnp.newaxis] / d_l_d_phi[:, jnp.newaxis]) \
                                + d2_r_d_phi2_cylindrical) / (d_l_d_phi[:, jnp.newaxis] * d_l_d_phi[:, jnp.newaxis])
  
  return jnp.sqrt(d_tangent_d_l_cylindrical[:,0] * d_tangent_d_l_cylindrical[:,0] + \
                        d_tangent_d_l_cylindrical[:,1] * d_tangent_d_l_cylindrical[:,1] + \
                        d_tangent_d_l_cylindrical[:,2] * d_tangent_d_l_cylindrical[:,2])

def calc_d_l_d_phi(nphi, nfp, rc, rs, zc, zs): 
  """
  this function returns d_l_d_phi as a fucntion of inputed parameters within qsc.py 
  """
  phi = jnp.linspace(0, 2 * jnp.pi / nfp, nphi, endpoint=False)

  nfourier = jnp.max([len(rc), len(zs), len(rs), len(zc)])

  # Compute n and the angles
  n = jnp.arange(0, nfourier) * nfp
  angles = jnp.outer(n, phi)
  sinangles = jnp.sin(angles)
  cosangles = jnp.cos(angles)

  R0 = jnp.dot(rc, cosangles) + jnp.dot(rs, sinangles)
  R0p = jnp.dot(rc, -n[:, jnp.newaxis] * sinangles) + jnp.dot(rs, n[:, jnp.newaxis] * cosangles)
  Z0p = jnp.dot(zc, -n[:, jnp.newaxis] * sinangles) + jnp.dot(zs, n[:, jnp.newaxis] * cosangles)

  return jnp.sqrt(R0 * R0 + R0p * R0p + Z0p * Z0p)