import numpy as jnp 

from .spectral_diff_matrix import *

def calc_torsion(_residual, _jacobian, nphi, nfp, rc, rs, zc, zs, sG, etabar, spsi, sigma0, B0):
  """
  calculate torsion as a function of inputed parameters
  """
  from .derive_r2 import recalc_rc, recalc_rs
  from .calculate_r1_helpers import derive_calc_Y1c, derive_calc_Y1s, derive_calc_X1c
  
  nfourier = jnp.max(jnp.array([len(rc), len(zs), len(rs), len(zc)]))

  
  #Y1c = derive_calc_Y1c(_residual, _jacobian, sG, spsi, nphi, nfp, rc, rs, zc, zs, sigma0, etabar)
  #Y1s = derive_calc_Y1s(sG, spsi, nphi, nfp, rc, rs, zc, zs, etabar)
  #X1c = derive_calc_X1c(etabar, nphi, nfp, rc, rs, zc, zs)
  
  #rs = recalc_rs(_residual, _jacobian, sG, spsi, nphi, nfp, rc, rs, zc, zs, sigma0, etabar)
  #rc = recalc_rc( Y1c, Y1s, X1c, rc, zs, rs, zc, nfp, nphi, sG, etabar, spsi, sigma0, B0)
  
  phi = jnp.linspace(0, 2 * jnp.pi / nfp, nphi, endpoint=False)
  n = jnp.arange(0, nfourier) * nfp
  angles = jnp.outer(n, phi)
  sinangles = jnp.sin(angles)
  cosangles = jnp.cos(angles)

  R0 = jnp.dot(rc, cosangles) + jnp.dot(rs, sinangles)
  Z0p = jnp.dot(zc, -n[:, jnp.newaxis] * sinangles) + jnp.dot(zs, n[:, jnp.newaxis] * cosangles)
  R0p = jnp.dot(rc, -n[:, jnp.newaxis] * sinangles) + jnp.dot(rs, n[:, jnp.newaxis] * cosangles)
  Z0p = jnp.dot(zc, -n[:, jnp.newaxis] * sinangles) + jnp.dot(zs, n[:, jnp.newaxis] * cosangles)
  R0pp = jnp.dot(rc, -n[:, jnp.newaxis]**2 * cosangles) + jnp.dot(rs, -n[:, jnp.newaxis]**2 * sinangles)
  Z0pp = jnp.dot(zc, -n[:, jnp.newaxis]**2 * cosangles) + jnp.dot(zs, -n[:, jnp.newaxis]**2 * sinangles)
  R0ppp = jnp.dot(rc, n[:, jnp.newaxis]**3 * sinangles) + jnp.dot(rs, -n[:, jnp.newaxis]**3 * cosangles)
  Z0ppp = jnp.dot(zc, n[:, jnp.newaxis]**3 * sinangles) + jnp.dot(zs, -n[:, jnp.newaxis]**3 * cosangles)

  d_r_d_phi_cylindrical = d_r_d_phi_cylindrical = jnp.array([R0p, R0, Z0p]).transpose()
  d2_r_d_phi2_cylindrical = jnp.array([R0pp - R0, 2 * R0p, Z0pp]).transpose()
  d3_r_d_phi3_cylindrical = jnp.array([R0ppp - 3 * R0p, 3 * R0pp - R0, Z0ppp]).transpose()
  
  torsion_numerator = (
        d_r_d_phi_cylindrical[:,0] * (d2_r_d_phi2_cylindrical[:,1] * d3_r_d_phi3_cylindrical[:,2] - d2_r_d_phi2_cylindrical[:,2] * d3_r_d_phi3_cylindrical[:,1])
        + d_r_d_phi_cylindrical[:,1] * (d2_r_d_phi2_cylindrical[:,2] * d3_r_d_phi3_cylindrical[:,0] - d2_r_d_phi2_cylindrical[:,0] * d3_r_d_phi3_cylindrical[:,2])
        + d_r_d_phi_cylindrical[:,2] * (d2_r_d_phi2_cylindrical[:,0] * d3_r_d_phi3_cylindrical[:,1] - d2_r_d_phi2_cylindrical[:,1] * d3_r_d_phi3_cylindrical[:,0])
  )

  torsion_denominator = (
        (d_r_d_phi_cylindrical[:,1] * d2_r_d_phi2_cylindrical[:,2] - d_r_d_phi_cylindrical[:,2] * d2_r_d_phi2_cylindrical[:,1]) ** 2
        + (d_r_d_phi_cylindrical[:,2] * d2_r_d_phi2_cylindrical[:,0] - d_r_d_phi_cylindrical[:,0] * d2_r_d_phi2_cylindrical[:,2]) ** 2
        + (d_r_d_phi_cylindrical[:,0] * d2_r_d_phi2_cylindrical[:,1] - d_r_d_phi_cylindrical[:,1] * d2_r_d_phi2_cylindrical[:,0]) ** 2
  ) 
  return torsion_numerator / torsion_denominator

def calc_curvature(nphi, nfp, rc, rs, zc, zs): 
  """
  this function returns curvature as a fucntion of inputed parameters within qsc.py 
  """
  # Generate phi
  phi = jnp.linspace(0, 2 * jnp.pi / nfp, nphi, endpoint=False)
  
  nfourier = jnp.max(jnp.array([len(rc), len(zs), len(rs), len(zc)]))

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
  print("hello")
  phi = jnp.linspace(0, 2 * jnp.pi / nfp, nphi, endpoint=False)

  nfourier = jnp.max(jnp.array([len(rc), len(zs), len(rs), len(zc)]))

  print(f"nfourier {nfourier}")
  print(f"nfp {nfp}")
  """
  # semo : adding padding 
  rc = jnp.pad(rc, (0, nfourier - len(rc)), constant_values=0)
  rs = jnp.pad(rs, (0, nfourier - len(rs)), constant_values=0)
  zc = jnp.pad(zc, (0, nfourier - len(zc)), constant_values=0)
  zs = jnp.pad(zs, (0, nfourier - len(zs)), constant_values=0)
  """
  
  # Compute n and the angles
  
  n = jnp.arange(0, nfourier) * nfp
  angles = jnp.outer(n, phi)
  
  sinangles = jnp.sin(angles)
  cosangles = jnp.cos(angles)      
  
  print(f"rc shape {rc.shape}")
  print(f"cosangles {cosangles.shape}")
  print(f"rs shape {rs.shape}")
  R0 = jnp.dot(rc, cosangles) + jnp.dot(rs, sinangles)
  R0p = jnp.dot(rc, -n[:, jnp.newaxis] * sinangles) + jnp.dot(rs, n[:, jnp.newaxis] * cosangles)
  Z0p = jnp.dot(zc, -n[:, jnp.newaxis] * sinangles) + jnp.dot(zs, n[:, jnp.newaxis] * cosangles)

  return jnp.sqrt(R0 * R0 + R0p * R0p + Z0p * Z0p)

def calc_G0(sG,  nphi,  B0, nfp, rc, rs, zc, zs):
  """
  Calculates G0 as a function of inputs. 
  Used in derive_B20 - derive_r2.py
  """
  d_l_d_phi = calc_d_l_d_phi(nphi, nfp, rc, rs, zc, zs)
  B0_over_abs_G0 = nphi / jnp.sum(d_l_d_phi) 
  abs_G0_over_B0 = 1 / B0_over_abs_G0
  return sG * abs_G0_over_B0 * B0

def calc_d_d_varphi(rc, zs, rs, zc, nfp,  nphi): 
  """
  Calculates d_d_varphi as a function of inputs. 
  """
  d_d_phi = spectral_diff_matrix(nphi, xmax=2 * jnp.pi / nfp)
  d_l_d_phi = calc_d_l_d_phi(nphi, nfp, rc, rs, zc, zs)
  B0_over_abs_G0 = nphi / jnp.sum(d_l_d_phi) 
  d_varphi_d_phi = B0_over_abs_G0 * d_l_d_phi
  return  d_d_phi / d_varphi_d_phi[:, jnp.newaxis]

def derive_helicity(rc, nfp, zs, rs, zc, nphi, sG, spsi): 
  """
  calculate helicity as a function of inputed values
      
  Determine the integer N associated with the type of quasisymmetry
  by counting the number
  of times the normal vector rotates
  poloidally as you follow the axis around toroidally.
  """
  nfourier = jnp.max(jnp.array([len(rc), len(zs), len(rs), len(zc)]))

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
  normal_cylindrical =   normal_cylindrical = d_tangent_d_l_cylindrical / curvature[:, jnp.newaxis]

  
  quadrant = jnp.zeros(nphi + 1)
  
  for j in range(nphi):
      if normal_cylindrical[j,0] >= 0:
          if normal_cylindrical[j,2] >= 0:
              quadrant = quadrant.at[j].set(1)
          else:
              quadrant = quadrant.at[j].set(4)
      else:
          if normal_cylindrical[j,2] >= 0:
              quadrant = quadrant.at[j].set(2)
          else:
              quadrant = quadrant.at[j].set(3)
  quadrant = quadrant.at[nphi].set(quadrant[0])

  counter = 0
  for j in range(nphi):
      if quadrant[j] == 4 and quadrant[j+1] == 1:
          counter += 1
      elif quadrant[j] == 1 and quadrant[j+1] == 4:
          counter -= 1
      else:
          counter += quadrant[j+1] - quadrant[j]

    # It is necessary to flip the sign of axis_helicity in order
    # to maintain "iota_N = iota + axis_helicity" under the parity
    # transformations.
  counter *= spsi * sG
  return  counter / 4
  
def derive_varphi(nphi, nfp, rc, rs, zc, zs): 
  nfourier = jnp.max(jnp.array([len(rc), len(zs), len(rs), len(zc)]))
  n = jnp.arange(0, nfourier) * nfp
  phi = jnp.linspace(0, 2 * jnp.pi / nfp, nphi, endpoint=False)
  angles = jnp.outer(n, phi)
  sinangles = jnp.sin(angles)
  cosangles = jnp.cos(angles)
  R0 = jnp.dot(rc, cosangles) + jnp.dot(rs, sinangles)
  R0p = jnp.dot(rc, -n[:, jnp.newaxis] * sinangles) + jnp.dot(rs, n[:, jnp.newaxis] * cosangles)
  Z0p = jnp.dot(zc, -n[:, jnp.newaxis] * sinangles) + jnp.dot(zs, n[:, jnp.newaxis] * cosangles)
  d_l_d_phi = jnp.sqrt(R0 * R0 + R0p * R0p + Z0p * Z0p)
  d_phi = phi[1] - phi[0]
  axis_length = jnp.sum(d_l_d_phi) * d_phi * nfp
  varphi = jnp.zeros(nphi)
  for j in range(1, nphi):
      # To get toroidal angle on the full mesh, we need d_l_d_phi on the half mesh.
      varphi = varphi.at[j].set(varphi[j-1] + (d_l_d_phi[j-1] + d_l_d_phi[j]))
  varphi = varphi * (0.5 * d_phi * 2 * jnp.pi / axis_length)
  
  return varphi