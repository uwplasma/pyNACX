from .calculate_r3 import *
from .calculate_r1_helpers import derive_calc_X1c, derive_calc_X1s, derive_calc_Y1c, derive_calc_Y1s
from .init_axis_helpers import * 
from .derive_r2 import * 

def derive_flux_constraint_coefficient(rc, zs, rs, zc, nfp, etabar, sigma0, I2, B0, sG, spsi, nphi, B2s, p2, B2c): 
  G0 = calc_G0(sG,  nphi,  B0, nfp, rc, rs, zc, zs)
  solution = calc_solution(rc, zs, rs, zc, nfp, etabar, sigma0, I2, B0, sG, spsi, nphi, B2s, p2, B2c)
  X20 = calc_X20(solution, nphi)  
  Y1c = derive_calc_Y1c(sG, spsi, nphi, nfp, rc, rs, zc, zs, sigma0, etabar)
  X2c = derive_X2c(rc, zs, rs, zc, nfp, etabar, sigma0, B0, sG, spsi, nphi, B2c)
  X2s = derive_X2s(rc, zs, rs, zc, nfp, etabar, sigma0, B0, sG, spsi, nphi, B2s, B2c)
  B1c =
  X1c = derive_calc_X1c(etabar, nphi, nfp, rc, rs, zc, zs)
  Y1s = derive_calc_Y1s(sG, spsi, nphi, nfp, rc, rs, zc, zs, etabar)
  B20 = derive_B20(rc, zs, rs, zc, nfp, etabar, sigma0, B0, I2, sG, spsi, nphi, B2s, B2c, p2)
  Y20 = 
  Y2c = 
  Y2s = 
  Z20 = derive_Z20(sG, spsi, nphi, nfp, rc, rs, zc, zs, sigma0, etabar, B0)
  iotaN = solve_sigma_equation(self, nphi, sigma0, helicity, nfp)
  Z2c = derive_Z2c(sG, spsi, nphi, nfp, rc, rs, zc, zs, sigma0, etabar, B0)
  abs_G0_over_B0 = calc_abs_G0_over_B0(sG, nphi, B0, nfp, rc, rs, zc, zs)
  Z2s = derive_Z2s(sG, spsi, nphi, nfp, rc, rs, zc, zs, sigma0, etabar, B0)
  torsion = calc_torsion(nphi, nfp, rc, rs, zc, zs, sG, etabar, spsi, sigma0)
  d_X1c_d_varphi = 
  d_Y1c_d_varphi = 
  flux =  (-4*B0**2*G0*X20**2*Y1c**2 + 8*B0**2*G0*X20*X2c*Y1c**2 - 4*B0**2*G0*X2c**2*Y1c**2 - \
        4*B0**2*G0*X2s**2*Y1c**2 + 8*B0*G0*B1c*X1c*X2s*Y1c*Y1s + 16*B0**2*G0*X20*X2s*Y1c*Y1s + \
        2*B0**2*I2*iotaN*X1c**2*Y1s**2 - G0*B1c**2*X1c**2*Y1s**2 - 4*B0*G0*B20*X1c**2*Y1s**2 - \
        8*B0*G0*B1c*X1c*X20*Y1s**2 - 4*B0**2*G0*X20**2*Y1s**2 - 8*B0*G0*B1c*X1c*X2c*Y1s**2 - \
        8*B0**2*G0*X20*X2c*Y1s**2 - 4*B0**2*G0*X2c**2*Y1s**2 - 4*B0**2*G0*X2s**2*Y1s**2 + \
        8*B0**2*G0*X1c*X20*Y1c*Y20 - 8*B0**2*G0*X1c*X2c*Y1c*Y20 - 8*B0**2*G0*X1c*X2s*Y1s*Y20 - \
        4*B0**2*G0*X1c**2*Y20**2 - 8*B0**2*G0*X1c*X20*Y1c*Y2c + 8*B0**2*G0*X1c*X2c*Y1c*Y2c + \
        24*B0**2*G0*X1c*X2s*Y1s*Y2c + 8*B0**2*G0*X1c**2*Y20*Y2c - 4*B0**2*G0*X1c**2*Y2c**2 + \
        8*B0**2*G0*X1c*X2s*Y1c*Y2s - 8*B0*G0*B1c*X1c**2*Y1s*Y2s - 8*B0**2*G0*X1c*X20*Y1s*Y2s - \
        24*B0**2*G0*X1c*X2c*Y1s*Y2s - 4*B0**2*G0*X1c**2*Y2s**2 - 4*B0**2*G0*X1c**2*Z20**2 - \
        4*B0**2*G0*Y1c**2*Z20**2 - 4*B0**2*G0*Y1s**2*Z20**2 - 4*B0**2*abs_G0_over_B0*I2*Y1c*Y1s*Z2c + \
        8*B0**2*G0*X1c**2*Z20*Z2c + 8*B0**2*G0*Y1c**2*Z20*Z2c - 8*B0**2*G0*Y1s**2*Z20*Z2c - \
        4*B0**2*G0*X1c**2*Z2c**2 - 4*B0**2*G0*Y1c**2*Z2c**2 - 4*B0**2*G0*Y1s**2*Z2c**2 + \
        2*B0**2*abs_G0_over_B0*I2*X1c**2*Z2s + 2*B0**2*abs_G0_over_B0*I2*Y1c**2*Z2s - 2*B0**2*abs_G0_over_B0*I2*Y1s**2*Z2s + \
        16*B0**2*G0*Y1c*Y1s*Z20*Z2s - 4*B0**2*G0*X1c**2*Z2s**2 - 4*B0**2*G0*Y1c**2*Z2s**2 - \
        4*B0**2*G0*Y1s**2*Z2s**2 + B0**2*abs_G0_over_B0*I2*X1c**3*Y1s*torsion + B0**2*abs_G0_over_B0*I2*X1c*Y1c**2*Y1s*torsion + \
        B0**2*abs_G0_over_B0*I2*X1c*Y1s**3*torsion - B0**2*I2*X1c*Y1c*Y1s*d_X1c_d_varphi + \
        B0**2*I2*X1c**2*Y1s*d_Y1c_d_varphi)/(16*B0**2*G0*X1c**2*Y1s**2)
  return flux 


def derive_X3c1(): 
  # requires X1c & flux_constraint_coefficient
  X1c = derive_calc_X1c()
  flux_constraint_coefficient = derive_flux_constraint_coefficient()
  return X1c * flux_constraint_coefficient

def derive_Y3c1(): 
  Y1c = derive_calc_Y1c(sG, spsi, nphi, nfp, rc, rs, zc, zs, sigma0, etabar)
  flux_constraint_coefficient = derive_flux_constraint_coefficient()
  return Y1c * flux_constraint_coefficient

def derive_Y3s1(): 
  Y1s = derive_calc_Y1s(sG, spsi, nphi, nfp, rc, rs, zc, zs, etabar)
  flux_constraint_coefficient = derive_flux_constraint_coefficient()
  return Y1s * flux_constraint_coefficient

def derive_X3s1(): 
  X1s = derive_calc_X1s(nphi)
  flux_constraint_coefficient = derive_flux_constraint_coefficient()
  return X1s * flux_constraint_coefficient
def derive_d_X3c1_d_varphi():
  d_d_varphi = calc_d_d_varphi(rc, zs, rs, zc, nfp, nphi)
  X3c1 = derive_X3c1()
  return d_d_varphi @ X3c1

def derive_d_Y3c1_d_varphi(): 
  d_d_varphi = calc_d_d_varphi(rc, zs, rs, zc, nfp, nphi)
  Y3c1 = derive_Y3c1()
  return d_d_varphi @ Y3c1

def derive_d_Y3s1_d_varphi(): 
  d_d_varphi = calc_d_d_varphi(rc, zs, rs, zc, nfp, nphi)
  Y3s1 = derive_Y3s1()
  return d_d_varphi @ Y3s1

def derive_Q(): 
  Q = -sign_psi * B0 * abs_G0_over_B0 / (2*G0*G0) * (self.iotaN * I2 + mu0 * self.p2 * G0 / (B0 * B0)) + 2 * (X2c * Y2s - X2s * Y2c) \
            + sign_psi * B0 / (2*G0) * (abs_G0_over_B0 * X20 * curvature - d_Z20_d_varphi) \
            + I2 / (4 * G0) * (-abs_G0_over_B0 * torsion * (X1c*X1c + Y1s*Y1s + Y1c*Y1c) + Y1c * d_X1c_d_varphi - X1c * d_Y1c_d_varphi)
  return Q
def derive_predicted_flux_constraint_coefficient(): 
  sign_G = sG
  sign_psi = spsi
  Q = derive_Q()
  return - Q / (2 * sign_G * sign_psi)

def derive_B0_order_a_squared_to_cancel(): 
  sign_G = sG
  
  B0_order_a_squared_to_cancel = -sign_G * B0 * B0 * (G2 + I2 * N_helicity) * abs_G0_over_B0 / (2*G0*G0) \
        -sign_G * sign_psi * B0 * 2 * (X2c * Y2s - X2s * Y2c) \
        -sign_G * B0 * B0 / (2*G0) * (abs_G0_over_B0 * X20 * curvature - d_Z20_d_varphi) \
        -sign_G * sign_psi * B0 * I2 / (4*G0) * (-abs_G0_over_B0 * torsion * (X1c*X1c + Y1c*Y1c + Y1s*Y1s) + Y1c * d_X1c_d_varphi - X1c * d_Y1c_d_varphi)
  return B0_order_a_squared_to_cancel

