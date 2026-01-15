"""
this is needed to make pyNACX non-reliant on self
"""


from qsc.calculate_r1 import solve_sigma_equation, r1_diagnostics
from qsc.calculate_r2 import calc_r2
from qsc.calculate_r3 import calc_r3
from qsc.init_axis import init_axis
from qsc.types import *

import jax
import jax.numpy as jnp

                
def Qsc_method(rc, zs, rs=[], zc=[], nfp=1, etabar=1., sigma0=0., B0=1., I2=0., sG=1, spsi=1, nphi=61, B2s=0., B2c=0., p2=0., order="r1") -> Results: 
  
  inputs = Inputs(
    rc = rc, 
    zs = zs, 
    rs = rs, 
    zc = zc, 
    nfp = nfp,
    etabar = etabar,
    sigma0 = sigma0,
    B0 = B0,
    I2 = I2,
    sG = sG,
    spsi = spsi,
    nphi = nphi,
    B2s = B2s,
    B2c = B2c,
    p2 = p2,
    order = order
  )

  find_max = jnp.array([len(rc), len(zs), len(rs), len(zc)])
  nfourier = jnp.max(find_max)

  rc_temp = rc
  zs_temp = zs
  rs_temp = rs
  zc_temp = zc

  rc = jnp.zeros(nfourier)
  zs = jnp.zeros(nfourier)
  rs = jnp.zeros(nfourier)
  zc = jnp.zeros(nfourier)
  
  rc = rc.at[:len(rc_temp)].set(rc_temp)
  zs = zs.at[:len(zs_temp)].set(zs_temp)
  rs = rs.at[:len(rs_temp)].set(rs_temp)
  zc = zc.at[:len(zc_temp)].set(zc_temp)
  
  if jnp.mod(nphi, 2) == 0:
            nphi += 1
  
  
  pre_calculations_results = pre_calculations(rc, zs, rs, zc, nfp, etabar, sigma0, B0, I2, sG, spsi, nphi, B2s, order, nfourier)
  
  d_l_d_varphi = pre_calculations_results.init_axis_results.abs_G0_over_B0 
  
  
  calculation_results = calculate(nfp, etabar, pre_calculations_results.init_axis_results.curvature, 
                                  pre_calculations_results.solve_sigma_equation_results.sigma,
                                  pre_calculations_results.init_axis_results.helicity,
                                  pre_calculations_results.init_axis_results.varphi,
                                  pre_calculations_results.init_axis_results.X1s,
                                  pre_calculations_results.init_axis_results.X1c,
                                  pre_calculations_results.init_axis_results.d_l_d_phi,
                                  pre_calculations_results.init_axis_results.d_d_varphi,
                                  sG, spsi, B0,
                                  pre_calculations_results.init_axis_results.G0,
                                  pre_calculations_results.solve_sigma_equation_results.iotaN,
                                  pre_calculations_results.init_axis_results.torsion,
                                  pre_calculations_results.init_axis_results.abs_G0_over_B0,
                                  B2s, B2c, p2, I2, nphi, order,
                                  pre_calculations_results.solve_sigma_equation_results.iota,
                                  d_l_d_varphi,
                                  pre_calculations_results.init_axis_results.tangent_cylindrical,
                                  pre_calculations_results.init_axis_results.normal_cylindrical,
                                  pre_calculations_results.init_axis_results.binormal_cylindrical,
                                  pre_calculations_results.init_axis_results.d_phi,
                                  pre_calculations_results.init_axis_results.axis_length
                                )
  
  return Results(inputs, pre_calculations_results, calculation_results)


def pre_calculations(
  rc,
  zs,
  rs,
  zc,
  nfp,
  etabar,
  sigma0,
  B0,
  I2,
  sG,
  spsi,
  nphi,
  B2s,
  order,
  nfourier) -> Pre_Calculation_Results:
  """
    calculates init_axis_results, solve_sigma_equation_results
  """
  
  init_axis_results = init_axis(nphi, nfp, rc, rs, zc, zs, nfourier, sG, B0, etabar, spsi, sigma0, order, B2s)
  print("\nInit axis completed...")
              
  solve_sigma_equation_results = solve_sigma_equation(nphi, sigma0, init_axis_results.helicity, nfp, init_axis_results.d_d_varphi, init_axis_results.etabar_squared_over_curvature_squared, spsi, init_axis_results.torsion, I2, B0, init_axis_results.G0)
  print("\nSigma equation solved...")
  
  return Pre_Calculation_Results(
    init_axis_results, 
    solve_sigma_equation_results
  )
        
        
def calculate(nfp, etabar, curvature, sigma, helicity, varphi, X1s, X1c, d_l_d_phi, d_d_varphi, sG, spsi, B0, G0, iotaN, torsion, abs_G0_over_B0, B2s, B2c, p2, I2, nphi, order, iota, d_l_d_varphi, tangent_cylindrical, normal_cylindrical, binormal_cylindrical, d_phi, axis_length) -> Complete_Calculation_Results:      
  """
    A jax compatible driver of main calculations.
  """
  print("\nCalculating R1...")
  r1_results = r1_diagnostics(nfp, etabar, sG, spsi, curvature, sigma, helicity, varphi, X1s, X1c, d_l_d_phi, d_d_varphi, B0, d_l_d_varphi, tangent_cylindrical, normal_cylindrical, binormal_cylindrical, iotaN, torsion)

  Y1s = r1_results.r1_results.Y1s
  Y1c = r1_results.r1_results.Y1c
  d_X1c_d_varphi = r1_results.r1_results.d_X1c_d_varphi
  d_Y1s_d_varphi = r1_results.r1_results.d_Y1s_d_varphi
  d_Y1c_d_varphi = r1_results.r1_results.d_Y1c_d_varphi
                
  dummy_r2 = jax.eval_shape(calc_r2, X1c, Y1c, Y1s, B0 / jnp.abs(G0), d_d_varphi, iotaN, torsion, abs_G0_over_B0, B2s, B0, curvature, etabar, B2c, spsi, sG, p2, sigma, I2/B0, nphi, d_l_d_phi, helicity, nfp, G0, iota, I2, varphi, d_X1c_d_varphi, d_Y1c_d_varphi, d_Y1s_d_varphi, d_phi, axis_length)
  
  zero_r2 = jax.tree_util.tree_map(jnp.zeros_like, dummy_r2)

  r2_results = jax.lax.cond(order != 'r1',
                      lambda _:  calc_r2(X1c, Y1c, Y1s, B0 / jnp.abs(G0), d_d_varphi, iotaN, torsion, abs_G0_over_B0, B2s, B0, curvature, etabar, B2c, spsi, sG, p2, sigma, I2/B0, nphi, d_l_d_phi, helicity, nfp, G0, iota, I2, varphi, d_X1c_d_varphi, d_Y1c_d_varphi, d_Y1s_d_varphi, d_phi, axis_length),
                      lambda _: zero_r2,
                      operand=None)
        
 
  print("\nCalculating R2...")

  X20 = r2_results.  r2_results[2][20]
  X2c = r2_results[2][22]
  X2s = r2_results[2][21]
  B20 = r2_results[2][30]
  Y20 = r2_results[2][23]
  Y2c = r2_results[2][25]
  Y2s = r2_results[2][24]
  Z20 = r2_results[2][26]
  Z2c = r2_results[2][28]
  Z2s = r2_results[2][27]
  d_Z20_d_varphi = r2_results[2][10]
  G2 = r2_results[2][1]
  N_helicity  = r2_results[2][0]
        
  print("\nCalculating R3...")

  dummy_r3 = jax.eval_shape(calc_r3, B0, G0, X20, Y1c, X2c, X2s, etabar * B0, 
                          X1c, X1s, Y1s, I2, iotaN, B20, Y20, Y2c, Y2s, Z20, 
                          abs_G0_over_B0, Z2c, Z2s, torsion, d_X1c_d_varphi, 
                          d_Y1c_d_varphi, d_d_varphi, spsi, p2, curvature, 
                          d_Z20_d_varphi, sG, G2, N_helicity, helicity, nfp, varphi)
  
  zero_r3 = jax.tree_util.tree_map(jnp.zeros_like, dummy_r3)


  order_is_r3 = jnp.array(order == 'r3')
  r3_result = jax.lax.cond(order_is_r3,
                      lambda _: calc_r3(B0, G0, X20, Y1c, X2c, X2s, etabar*B0, X1c, X1s, Y1s, I2, iotaN, B20, Y20, Y2c, Y2s, Z20, abs_G0_over_B0, Z2c, Z2s, torsion, d_X1c_d_varphi, d_Y1c_d_varphi, d_d_varphi, spsi, p2, curvature, d_Z20_d_varphi, sG, G2, N_helicity, helicity, nfp, varphi),
                      lambda _: zero_r3,
                    operand=None)
        
  return Complete_Calculation_Results(
      r1_results,
      r2_results,
      r3_result
    ) 