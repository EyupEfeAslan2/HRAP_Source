# Purpose: Model regression of motor grains
# Authors: Thomas A. Scott

from hrap.core import store_x, make_part

import numpy as np

import jax.numpy as jnp
from jax.lax import cond

from functools import partial

def d_grain_constOF(s, x, xmap, fshape):
    mdot_inj = x[xmap['tnk_mdot_inj']] # TODO: using vent?
    A = x[xmap['grn_A']]
    d = x[xmap['grn_d']]
    L   = s['grn_L']
    rho = s['grn_rho']
    OF = s['grn_OF']
    
    # Current arc length of exposed grain on the cross section
    arc = fshape(d, s, x, xmap)
    
    # Current volume
    V = L * A
    
    # Grain consumption rate (positive)
    mdot = mdot_inj / OF
    mdot = cond(A <= 0.0, lambda val: 0.0, lambda val: val, mdot)
    
    # Rate of volume consumption (positive)
    Vdot = mdot / rho
    
    # Rate of cross section area loss
    Adot = Vdot / L
    
    # Cross sectional area linearization (i.e. volume of thin shell, Adot = arc * ddot)
    ddot = Adot / arc
    
    # Store result
    x = store_x(x, xmap, grn_Adot=-Adot, grn_ddot=ddot, grn_V=V, grn_mdot=mdot, grn_Vdot=Vdot, cmbr_OF=OF)

    return x

def d_grain_shiftOF(s, x, xmap, fshape):
    mdot_inj = x[xmap['tnk_mdot_inj']]
    A = x[xmap['grn_A']]
    d = x[xmap['grn_d']]
    L   = s['grn_L']
    rho = s['grn_rho']
    Reg = s['grn_Reg']
    
    # Correctly fetch DTI variables using the 'grn_' prefix that make_part adds
    K = s.get('grn_K_dti', 0.0)         
    D_inj = s.get('grn_D_inj_dti', 0.0) 
    
    arc = fshape(d, s, x, xmap)
    V = L * A
    
    # --- PHYSICS FIX: Annulus Flow Area Calculation ---
    D_m = s['grn_shape_ID'] + (2.0 * d)
    # Area = (pi/4) * (Port_Diameter^2 - Injector_Tube_Diameter^2)
    # Using jnp.maximum to prevent division by zero during JAX tracing
    A_flow = jnp.maximum((jnp.pi / 4.0) * (D_m**2 - D_inj**2), 1e-6)
    
    # Axial mass flow is reduced by the radial injection fraction (K)
    mdot_axial = mdot_inj * (1.0 - K)
    G = mdot_axial / A_flow
    # --------------------------------------------------
    
    # 1. Base HRAP Regression
    ddot_ax = 0.001 * Reg[0] * (G**Reg[1]) * (L**Reg[2])
    
    # 2. DTI Multiplier
    def apply_dti(ddot_base):
        term1 = K / (1.0 - K)
        term2 = 0.25 / (L / D_m)
        term3 = jnp.maximum(1.0 - ((D_inj**2) / (D_m**2)), 0.001)
        multiplier = 1.1216 * ((1.0 + 100.0 * ((term1 * term2 * term3)**0.8))**0.482)
        return ddot_base * multiplier
        
    # K comes from `s` which is traced inside fori_loop — must use jax.lax.cond
    # Positional signature: cond(pred, true_fun, false_fun, *operands)
    ddot = cond(K > 0.0, lambda _: apply_dti(ddot_ax), lambda _: ddot_ax, None)
    
    Adot = ddot * arc
    Vdot = Adot * L
    
    mdot = Vdot * rho
    mdot = cond(A <= 0.0, lambda val: 0.0, lambda val: val, mdot)
    
    OF = mdot_inj / mdot
    
    x = store_x(x, xmap, grn_Adot=-Adot, grn_ddot=ddot, grn_V=V, grn_mdot=mdot, grn_Vdot=Vdot, cmbr_OF=OF)
    return x

def u_grain(s, x, xmap):
    x = store_x(x, xmap,
        grn_A = jnp.maximum(x[xmap['grn_A']], 0.0),
        grn_d = jnp.minimum(x[xmap['grn_d']], s['grn_OD']/2) # TODO: shouldnt be necessary
    )
    
    return x

def make_circle_shape(**kwargs):
    def fcircle(d, s, x, xmap):
        return np.pi * (s['grn_shape_ID'] + 2*d)
    
    def preprs(s, x, xmap):
        OD, ID = s['grn_OD'], s['grn_shape_ID']
        A = np.pi/4 * (OD**2 - ID**2)
        x = x.at[xmap['grn_A']].set(A)
        
        return x
    
    return make_part(
        # Default static and initial dynamic variables
        s = {
            'ID': 0.1,
        },
        x = {
        },
        
        # Required and integrated variables
        req_s = ['ID'],
        req_x = [],
        dx    = { },

        typename = 'shape',

        fshape = fcircle,
        fpreprs = preprs,

        # The user-specified static and initial dynamic variables
        **kwargs,
    )

def make_constOF_grain(shape, **kwargs):
    return make_part(
        # Default static and initial dynamic variables
        s = {
            'OD': 0.1,
            'L': 0.1,
            'OF': 1.0,
            'rho': 1000.0,
            **shape['s'],
        },
        x = {
            'A': 0.1,
            'd': 0.0,   # Distance regressed, i.e. increasing during burn
            
            # Calculated variables
            'V': 0.0,
            'Vdot': 0.0,
            'P': 101e3,
            'mdot': 0.0,
            
            **shape['x'],
        },
        
        # Required and integrated variables
        req_s = ['OD', 'L', 'OF'],
        req_x = ['A'],
        dx    = {'A': 'Adot', 'd': 'ddot'},

        # Designation and associated functions
        typename = 'grn',
        fderiv  = partial(d_grain_constOF, fshape=shape['shape_fshape']),
        fupdate = u_grain,
        fpreprs = shape['fpreprs'],

        # The user-specified static and initial dynamic variables
        **kwargs,
    )

def make_shiftOF_grain(shape, **kwargs):
    return make_part(
        # Default static and initial dynamic variables
        s = {
            'OD': 0.1,
            'L': 0.1,
            'Reg': jnp.zeros(3), # Regression coefficient (mm/s), regression exponent, length exponent
            'rho': 1000.0,
            'K_dti': 0.0,
            'D_inj_dti': 0.0,
            **shape['s'],
        },
        x = {
            'A': 0.1,
            'd': 0.0,   # Distance regressed, i.e. increasing during burn
            
            # Calculated variables
            'V': 0.0,
            'Vdot': 0.0,
            'P': 101e3,
            'mdot': 0.0,
            
            **shape['x'],
        },
        
        # Required and integrated variables
        req_s = ['OD', 'L'],  # No 'OF' — it's computed dynamically in shiftOF mode
        req_x = ['A'],
        dx    = {'A': 'Adot', 'd': 'ddot'},

        # Designation and associated functions
        typename = 'grn',
        fderiv  = partial(d_grain_shiftOF, fshape=shape['shape_fshape']),
        fupdate = u_grain,
        fpreprs = shape['fpreprs'],

        # The user-specified static and initial dynamic variables
        **kwargs,
    )