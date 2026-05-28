# Purpose: Demonstrate API usage and validate for a typical hybrid, case is similar to GUI defaults
# HRAP_Source/HRAP - Python/examples/nitrous_plastisol.py

import scipy
import numpy as np
from pathlib import Path
from importlib.resources import files as imp_files

import matplotlib.pyplot as plt

import hrap.core as core
import hrap.chem as chem
import hrap.fluid as fluid
from hrap.tank    import *
from hrap.grain   import *
from hrap.chamber import *
from hrap.nozzle  import *
from hrap.units   import _in, _ft, _lbf, _atm

jax.config.update("jax_enable_x64", True)
hrap_root = Path(imp_files('hrap'))
file_prefix = 'nitrous_plastisol'

print('Building combustion chemistry table...')
plastisol = chem.make_basic_reactant(
    formula = 'Plastisol-362',
    composition = { 'C': 7.200, 'H': 10.82, 'O': 1.14, 'Cl': 0.669 },
    M = 140.86, # kg/kmol
    T0 = 298.15, # K
    h0 = -2.6535755e7, # J/kmol
)
comb = chem.ChemSolver([hrap_root/'thermo.dat', plastisol])
chem_Pc = np.linspace(1*_atm, 20*_atm, 20)

chem_OF = np.linspace(0.5, 15.0, 50)
chem_k, chem_M, chem_T = [np.zeros((chem_Pc.size, chem_OF.size)) for i in range(3)]
ox, fu_1, fu_2 = 'N2O(L),298.15K', 'Plastisol-362', 'AL(cr)'
mfrac_al = 0.0 
internal_state = None
for j, OF in enumerate(chem_OF):
    for i, Pc in enumerate(chem_Pc):
        o = OF / (1 + OF) # o/f = OF, o+f=1 => o=OF/(1 + OF)
        flame, internal_state = comb.solve(Pc, {ox: o, fu_1: (1-mfrac_al)*(1-o), fu_2: mfrac_al*(1-o)}, max_iters=150, internal_state=internal_state)
        chem_k[i,j], chem_M[i,j], chem_T[i,j] = flame.gamma, flame.M, flame.T

print('Baking NOS saturated property curves...')
get_sat_nos_props = fluid.bake_sat_coolprop('NitrousOxide', np.linspace(183.0, 309.0, 20))

# ══════════════════════════════════════════════════════════════════════════════
#  REGULATED TANK MODEL — N2 Supercharged, Constant Pressure Feed System
# ══════════════════════════════════════════════════════════════════════════════
# Physical system: External N2 regulator maintains constant head pressure on
# the liquid N2O column.  No blowdown, no phase-change thermodynamics.
#
# Interface contract (all variables use the 'tnk_' prefix via make_part):
#   Static (s):  P_reg, inj_CdA, rho_liq
#   Dynamic (x): m_ox (integrated), T, P, mdot_ox_total, mdot_ox, mdot_vnt,
#                m_ox_liq, m_ox_vap, Pdot_sum, Pdot_N
#   dx:          m_ox ← mdot_ox (integrated by the engine's fori_loop)
# ──────────────────────────────────────────────────────────────────────────────
from hrap.core import store_x, make_part

def d_regulated_tank(s, x, xmap):
    """Derivative function for N2-regulated constant-pressure tank.
    
    Physics:
      - Tank pressure = P_reg (constant) while liquid remains
      - Injector flow = Cd·A · sqrt(2·ρ_liq · (P_tank - P_chamber))
      - When m_ox_liq ≤ 0, all flow stops
      - Main valve closes at t = valve_close_t (tracked via t_elapsed state)
    """
    m_ox      = x[xmap['tnk_m_ox']]
    t_elapsed = x[xmap['tnk_t_elapsed']]   # Simulation time [s]
    Pc        = x[xmap['cmbr_P']]
    P_reg     = s['tnk_P_reg']              # Regulated pressure [Pa]
    inj_CdA   = s['tnk_inj_CdA']           # Cd × A_inj [m²]
    rho_liq   = s['tnk_rho_liq']            # Liquid N2O density [kg/m³]
    t_close   = s['tnk_valve_close_t']      # Main valve close time [s]

    # ── Main Valve State ────────────────────────────────────────────
    # JAX-compatible step function: valve_open = 1.0 if t < t_close, else 0.0
    valve_open = jnp.where(t_elapsed < t_close, 1.0, 0.0)

    # Tank pressure: constant while liquid remains, else zero gauge
    P_tank = jnp.where(m_ox > 0.0, P_reg, s['Pa'])

    # Pressure drop across injector
    dP = jnp.maximum(P_tank - Pc, 0.0)

    # Incompressible Bernoulli injector flow — gated by valve state
    mdot_ox_total = jnp.where(
        m_ox > 0.0,
        inj_CdA * jnp.sqrt(2.0 * rho_liq * dP),
        0.0
    )
    mdot_ox_total = mdot_ox_total * valve_open   # ← Main valve shut-off

    # Total oxidizer consumption rate (negative = mass leaving tank)
    mdot_ox = -mdot_ox_total

    # All liquid (no phase separation in regulated system)
    m_ox_liq = jnp.maximum(m_ox, 0.0)
    m_ox_vap = 0.0

    # No venting in regulated system
    mdot_vnt = 0.0

    # No temperature/pressure dynamics
    Tdot = 0.0
    Pdot = 0.0

    # Time integration: t_rate = 1.0 → dt_elapsed/dt = 1.0 (clock)
    t_rate = 1.0

    x = store_x(x, xmap,
        tnk_P=P_tank,
        tnk_Tdot=Tdot,
        tnk_Pdot=Pdot,
        tnk_mdot_ox=mdot_ox,
        tnk_mdot_ox_total=mdot_ox_total,
        tnk_mdot_vnt=mdot_vnt,
        tnk_m_ox_liq=m_ox_liq,
        tnk_m_ox_vap=m_ox_vap,
        tnk_rho_ox_liq=rho_liq,
        tnk_rho_ox_vap=0.0,
        tnk_t_rate=t_rate,
    )
    return x

def u_regulated_tank(s, x, xmap):
    """Update function — clamp oxidizer mass to non-negative."""
    x = store_x(x, xmap,
        tnk_m_ox = jnp.maximum(x[xmap['tnk_m_ox']], 0.0),
        # Accumulate Pdot stats (required by state vector shape, always zero here)
        tnk_Pdot_sum = x[xmap['tnk_Pdot_sum']] + x[xmap['tnk_Pdot']],
        tnk_Pdot_N   = x[xmap['tnk_Pdot_N']] + 1,
    )
    return x

def make_regulated_tank(**kwargs):
    """Create a constant-pressure N2-regulated tank part.
    
    Required kwargs:
        P_reg        : Regulated pressure [Pa]  (e.g. 65e5)
        rho_liq      : Liquid oxidizer density [kg/m³]
        inj_CdA      : Injector Cd × Area [m²]
        m_ox         : Initial liquid oxidizer mass [kg]
        valve_close_t: Main valve close time [s] (default: 8.0)
    """
    return make_part(
        s = {
            'V': 0.0,              # Tank volume (informational only)
            'P_reg': 65e5,         # Regulated pressure [Pa]
            'rho_liq': 745.0,      # Liquid N2O density [kg/m³]
            'vnt_S': 0,            # No vent in regulated mode
            'vnt_CdA': 0.0,
            'inj_CdA': 0.0,
            'inj_N': 1,
            'valve_close_t': 8.0,  # Main valve close time [s]
        },
        x = {
            'T':   293.0,          # Constant (no thermal dynamics)
            'm_ox': 1.0,           # Initial oxidizer mass [kg] — overridden by user
            't_elapsed': 0.0,      # Simulation clock [s] (integrated at rate 1.0)
            
            # Calculated variables (same shape as sat_tank for compatibility)
            'Pdot': 0.0,
            'mdot_ox_total': 0.0,
            'mdot_vnt': 0.0,
            'P':   101e3,
            'm_ox_liq': 0.0,
            'm_ox_vap': 0.0,
            'rho_ox_liq': 0.0,
            'rho_ox_vap': 0.0,
            'Pdot_sum': 0.0,
            'Pdot_N': 0,
        },
        
        req_s = ['P_reg', 'inj_CdA'],
        req_x = ['m_ox'],
        
        # m_ox integrated by mdot_ox, T by Tdot (always 0),
        # t_elapsed integrated by t_rate (always 1.0 → real-time clock)
        dx = { 'm_ox': 'mdot_ox', 'T': 'Tdot', 't_elapsed': 't_rate' },
        
        typename = 'tnk',
        fderiv   = d_regulated_tank,
        fupdate  = u_regulated_tank,
        
        **kwargs,
    )

print('Regulated tank model loaded.')

# 1. GİRDİLER VE TASARIM NOKTASI (Inputs & Design Point)
design_point = "Optimum Expansion at 0.9033 bar (Ambient)"
target_thrust = 100.0  # N
burn_time = 8.0  # s
Pc_target = 10.0 # bar
Cf_target = 1.2413        # O/F 5.5 için Cf
cstar_target = 1490.72    # O/F 5.5 için c*
OF_ratio = 5.5            # Makine ekibinin hedefi
fuel_density = 900.0 # kg/m³
reg_a = 0.0001435 # m/s (Bunu kendi modelimiz çarpacak)
reg_n = 0.5275
ambient_pressure_pa = 90300.0
N2_supercharge_bar = 65.0  # N2 süperşarj basıncı

# 2. GEOMETRİ VE HESAPLANAN ÇIKTILAR
# --- Nozzle Sizing (F = Cf * At * Pc) ---
Pc_pa = Pc_target * 1e5
throat_area_m2 = target_thrust / (Cf_target * Pc_pa)       # At = F / (Cf * Pc)
throat_diameter_mm = np.sqrt(4.0 * throat_area_m2 / np.pi) * 1000.0  # ~32.04 mm
# Expansion ratio computed for optimum expansion (Pe=Pa)
# Using iterative approach from isentropic relations with gamma~1.2
_k_design = 1.2  # approximate gamma for N2O/Plastisol at design point
_Pe_Pc = ambient_pressure_pa / Pc_pa
_Me2 = (2.0/(_k_design-1.0)) * ((_Pe_Pc)**(-((_k_design-1.0)/_k_design)) - 1.0)
_Me = np.sqrt(_Me2)
expansion_ratio = (1.0/_Me) * ((2.0/(_k_design+1.0)) * (1.0 + (_k_design-1.0)/2.0 * _Me2))**((_k_design+1.0)/(2.0*(_k_design-1.0)))
exit_diameter_mm = throat_diameter_mm * np.sqrt(expansion_ratio)

# --- Grain Geometry ---
grain_length_mm = 16.87   # Gereken Yakıt Boyu Lf
inner_diameter_mm = 30.00  # Başlangıç İç Çapı Di
initial_port_radius_mm = inner_diameter_mm / 2.0  # 15.0 mm
outer_diameter_mm = 100.0  # Yakıt Dış Çapı Do

# --- Mass Flows ---
oxidizer_mass_flow = 0.04573   # 45.73 g/s
fuel_mass_flow = 0.00831       # 8.31 g/s
total_mass_flow = 0.05404      # 54.04 g/s
port_area_m2 = np.pi * (initial_port_radius_mm / 1000.0)**2
oxidizer_mass_flux = oxidizer_mass_flow / port_area_m2
regression_rate_m_s = reg_a * oxidizer_mass_flux**reg_n  # r = a * G^n [m/s]
web_thickness_mm = regression_rate_m_s * burn_time * 1000.0  # Regression Rate × Burn Time [mm]
chamber_inner_diameter_mm = 110.0  # Do (100mm) + 2×Nylon kılıf (5mm) → min 110mm

# --- DTI Enjektör Parametreleri ---
Cd_injector = 0.55
axial_hole_dia_mm = 0.2   # Aksiyel delik çapı
radial_hole_dia_mm = 0.2  # Radyal delik çapı
n_axial_holes = 21         # Aksiyel delik sayısı
n_radial_holes = 7         # Radyal delik sayısı
K_dti_target = 0.25        # Hedef radyal yükleme oranı
DTI_tube_OD_mm = 6.0       # DTI metal tüp dış çapı
nylon_casing_mm = 5.0      # Nylon 6-6 kılıf kalınlığı

print("\n" + "="*65)
print("=== MOTOR BOYUTLARI RAPORU ===")
print(f"Design Point: {design_point}")
print("-" * 65)
print("1. SABİT PERFORMANS GEREKSİNİMLERİ:")
print(f"  Hedef İtki F:         {target_thrust} N")
print(f"  Yanma Odası Pc:       {Pc_target} bar")
print(f"  N2 Süperşarj:         {N2_supercharge_bar} bar")
print(f"  Yanma Süresi tb:      {burn_time} s")
print(f"  Hedef O/F:            {OF_ratio}")
print(f"  c* / Cf:              {cstar_target} m/s / {Cf_target}")
print(f"  Fuel Density:         {fuel_density} kg/m³")
print(f"  Regression Law:       a = {reg_a}, n = {reg_n}")
print(f"  Ambient Pressure:     {ambient_pressure_pa} Pa")
print("-" * 65)
print("2. MOTOR GEOMETRİSİ & NOZZLE:")
print(f"  Throat Dia (calc):    {throat_diameter_mm:.2f} mm")
print(f"  Exit Dia (calc):      {exit_diameter_mm:.2f} mm")
print(f"  Expansion Ratio:      {expansion_ratio:.3f}")
print(f"  Grain Di / Do:        {inner_diameter_mm} mm / {outer_diameter_mm} mm")
print(f"  Grain Length Lf:      {grain_length_mm} mm")
print(f"  Web Thickness:        {web_thickness_mm:.2f} mm")
print(f"  Do/Di Oranı:          {outer_diameter_mm/inner_diameter_mm:.2f}")
print(f"  Lf/Do Oranı:          {grain_length_mm/outer_diameter_mm:.2f}")
print("-" * 65)
print("3. KÜTLESEL DEBİLER:")
print(f"  Toplam Debi:          {total_mass_flow*1000:.2f} g/s")
print(f"  Oksitleyici:          {oxidizer_mass_flow*1000:.2f} g/s")
print(f"  Yakıt:                {fuel_mass_flow*1000:.2f} g/s")
print(f"  Ox Mass Flux:         {oxidizer_mass_flux:.2f} kg/m²s")
print("-" * 65)
print("4. ENJEKTÖR (0.2 mm EDM):")
print(f"  Cd:                   {Cd_injector}")
print(f"  Aksiyel Delikler:     {n_axial_holes} × ø{axial_hole_dia_mm} mm")
print(f"  Radyal Delikler:      {n_radial_holes} × ø{radial_hole_dia_mm} mm")
print(f"  Toplam Delik:         {n_axial_holes + n_radial_holes} adet")
print(f"  Hedef K:              {K_dti_target*100:.0f}%")
print(f"  DTI Tüp OD:          {DTI_tube_OD_mm} mm")
print("-" * 65)
print("5. NYLON 6-6 KONTROL:")
ablation_thickness = 0.6 * burn_time  # ~0.6 mm/s ablation rate estimate
print(f"  Tahmini ablasyon ({burn_time}s): {ablation_thickness:.1f} mm")
remaining = nylon_casing_mm - ablation_thickness
status = "✓ Yeterli" if remaining > 0 else "✗ YETERSİZ"
print(f"  Kalan güvenlik payı:  {remaining:.1f} mm  {status}")
print("-" * 65)
print("6. CONSISTENCY CHECKS:")
print(f"  Grain OD ({outer_diameter_mm}mm) <= Chamber ID ({chamber_inner_diameter_mm}mm): {'✓ VALIDATED' if outer_diameter_mm <= chamber_inner_diameter_mm else '✗ FAIL'}")
print("="*65 + "\n")

print('Initializing engine...')

# ── Tank & Injector Sizing ────────────────────────────────────────────────────
m_ox_required = 0.400     # Exactly 400 g liquid N2O loaded
rho_n2o_liquid = 745.0    # kg/m³ (subcooled liquid at ~20°C under 65 bar)
P_regulated = N2_supercharge_bar * 1e5  # 65 bar → 6.5e6 Pa

# Injector CdA from design-point Bernoulli: mdot = CdA × √(2ρΔP)
# ΔP = P_tank - Pc = 65 - 10 = 55 bar
dP_design = (N2_supercharge_bar - Pc_target) * 1e5  # 5.5e6 Pa
inj_CdA = oxidizer_mass_flow / np.sqrt(2.0 * rho_n2o_liquid * dP_design)
# NOTE: inj_CdA here absorbs the Cd factor — it IS Cd×A.
# Verify: mdot = inj_CdA × √(2·745·5.5e6) ≈ 0.04573 kg/s ✓
print(f"  Design inj_CdA = {inj_CdA:.6e} m²  →  mdot_check = {inj_CdA * np.sqrt(2*rho_n2o_liquid*dP_design)*1000:.2f} g/s")

tank_volume = (m_ox_required / rho_n2o_liquid) * 1.5  # ullage factor

# 1. TANK — N2 Regulated Constant-Pressure Model
tnk = make_regulated_tank(
    P_reg   = P_regulated,     # 65 bar constant
    rho_liq = rho_n2o_liquid,  # 745 kg/m³
    V       = tank_volume,
    inj_CdA = inj_CdA,
    m_ox    = m_ox_required,   # 0.400 kg
    T       = 293.0,           # Room temp (not used dynamically)
)

# 2. GRAIN (YAKIT)
shape = make_circle_shape(
    ID = inner_diameter_mm / 1000.0,   # 30 mm → 0.030 m
)
grn = make_shiftOF_grain(
    shape,
    Reg = np.array([reg_a * 1000.0, reg_n, 0.0]), 
    OD = outer_diameter_mm / 1000.0,   # 100 mm → 0.100 m
    L = grain_length_mm / 1000.0,      # 16.87 mm → 0.01687 m
    rho = fuel_density,  
    K_dti = K_dti_target,              # 0.25 (hedef radyal yükleme)
    D_inj_dti = DTI_tube_OD_mm / 1000.0  # 6 mm → 0.006 m
)

# 3. CHAMBER (YANMA ODASI)
prepost_ID = chamber_inner_diameter_mm / 1000.0  
prepost_V  = (3.5+1.7)*_in * np.pi/4*prepost_ID**2  
rings_V    = 3 * (1/8*_in) * np.pi*(2.5/2 * _in)**2  
fuel_V     = (grain_length_mm/1000.0) * np.pi*((outer_diameter_mm/1000.0)/2)**2  
cmbr = make_chamber(
    V0 = prepost_V + rings_V + fuel_V, 
    cstar_eff = 0.92,  
)

# 4. NOZZLE (LÜLE)
noz = make_cd_nozzle(
    thrt = throat_diameter_mm / 1000.0,  
    ER = expansion_ratio,   # Optimum expansion hesabından
    eff = 0.9796,          
    C_d = 0.995,
)

from jax.scipy.interpolate import RegularGridInterpolator
chem_interp_k = RegularGridInterpolator((chem_OF, chem_Pc), chem_k.T, fill_value=1.4)
chem_interp_M = RegularGridInterpolator((chem_OF, chem_Pc), chem_M.T, fill_value=29.0)
chem_interp_T = RegularGridInterpolator((chem_OF, chem_Pc), chem_T.T, fill_value=293.0)

s, x, method = core.make_engine(
    tnk, grn, cmbr, noz,
    chem_interp_k=chem_interp_k, chem_interp_M=chem_interp_M, chem_interp_T=chem_interp_T,
    Pa=ambient_pressure_pa,
)

fire_engine = core.make_integrator(
    core.step_fe,
    method,
)

T = 12.0
print('Running...')
import time
t1 = time.time()
t, _x, xstack = fire_engine(s, x, dt=1E-3, T=T)
jax.block_until_ready(xstack)
t2  = time.time()
t, x, xstack = fire_engine(s, x, dt=1E-3, T=T)
jax.block_until_ready(xstack)
t3 = time.time()
print('done, first run was {a:.2f}s, second run was {b:.2f}s'.format(a=t2-t1, b=t3-t2))

N_t = xstack.shape[0]
tnk, grn, cmbr, noz = core.unpack_engine(s, xstack, method)

results_path = Path('./results')
results_path.mkdir(parents=True, exist_ok=True)

OD, L = outer_diameter_mm/1000.0, grain_length_mm/1000.0
core.export_rse(
    results_path/(file_prefix+'.rse'),
    t, noz['thrust'].ravel(), noz['mdot_total'].ravel(), t*0, t*0,
    OD=OD, L=L, D_throat=s['noz_thrt'], D_exit=np.sqrt(s['noz_ER'])*s['noz_thrt'],
    motor_type='hybrid', mfg='HRAP',
)
core.export_eng(
    results_path/(file_prefix+'.eng'),
    t, noz['thrust'], t*0,
    OD=OD, L=L,
    mfg='HRAP',
)

# ──────────────────────────────────────────────────────────────────────────────
# Visualization — Interactive 3×3 dashboard with per‑graph HD export buttons
# ──────────────────────────────────────────────────────────────────────────────
import button

time_arr = np.linspace(0.0, T, N_t)

# JAX Dizilerini Tek Seferde Numpy'a Çeviriyoruz (Açılış Lag'ını Yok Etmek İçin)
def to_numpy_dict(d):
    return {k: np.array(v) for k, v in d.items()}

tnk = to_numpy_dict(tnk)
grn = to_numpy_dict(grn)
cmbr = to_numpy_dict(cmbr)
noz = to_numpy_dict(noz)

# Web Thickness = Regression Rate × Burn Time
web_thickness_mm = regression_rate_m_s * burn_time * 1000.0  # [mm]

valve_close_t = 8.0  # Main valve close time [s] — for plot annotations

def _plot_thrust(ax):
    ax.plot(time_arr, noz['thrust'], label='sim', linewidth=2)
    ax.axhline(y=target_thrust, color='red', linestyle='--', label=f'target={target_thrust}N')
    ax.axvline(x=valve_close_t, color='orange', linestyle=':', linewidth=1.5, label='valve close t=8.0s')
    ax.set_xlim(left=-0.1)
    ax.set_ylim(bottom=0)
    ax.legend()

def _plot_mdot(ax): # mdot Graphic
    mask = time_arr < valve_close_t
    ax.plot(time_arr[mask], tnk['mdot_ox_total'][mask], label='mdot_ox_total', linewidth=1.5)
    ax.plot(time_arr[mask], grn['mdot_fuel'][mask], label='mdot_fuel', linewidth=1.5)
    ax.plot(time_arr[mask], noz['mdot_total'][mask], label='mdot_total', linewidth=1.5)
    ax.axhline(y=oxidizer_mass_flow, color='red', linestyle='--', label=f'target ox={oxidizer_mass_flow} kg/s')
    ax.axvline(x=valve_close_t, color='orange', linestyle=':', linewidth=1.5, label='valve close t=8.0s')
    ax.set_xlim(left=-0.1)
    ax.set_ylim(bottom=0)
    ax.set_ylabel('Mass Flow (kg/s)', fontsize=9)
    ax.legend(loc='upper right', fontsize=8)

def _plot_mdot_split(ax): # Oksitleyici Debi Dağılımı (Axial vs Radial)
    mask = time_arr < valve_close_t
    
    # Verileri hesapla
    mdot_tot = tnk['mdot_ox_total'][mask]
    mdot_rad = mdot_tot * K_dti_target          # %25'lik kısım
    mdot_ax = mdot_tot * (1.0 - K_dti_target)   # %75'lik kısım
    
    # Çizgiler
    ax.plot(time_arr[mask], mdot_tot, label='Total Ox Flow', color='black', linewidth=2)
    ax.plot(time_arr[mask], mdot_ax, label=f'Axial Flow (%{(1.0 - K_dti_target)*100:.0f})', color='blue', linewidth=1.5)
    ax.plot(time_arr[mask], mdot_rad, label=f'Radial Flow (%{K_dti_target*100:.0f})', color='red', linewidth=1.5)
    
    # Formatlama
    ax.axvline(x=valve_close_t, color='orange', linestyle=':', linewidth=1.5, label='valve close t=8.0s')
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.set_ylabel('Mass Flow (kg/s)', fontsize=9)
    ax.legend(loc='upper right', fontsize=8) 

def _plot_pressure(ax): # P Graphic
    mask = time_arr < valve_close_t
    ax.plot(time_arr[mask], cmbr['P'][mask] / 1e5, label='chamber', linewidth=2)
    ax.plot(time_arr[mask], tnk['P'][mask] / 1e5, label='tank', linewidth=1.5)
    ax.axhline(y=Pc_target, color='red', linestyle='--', alpha=0.6, label=f'target={Pc_target} bar')
    ax.axvline(x=valve_close_t, color='orange', linestyle=':', linewidth=1.5, label='valve close t=8.0s')
    ax.set_xlim(left=-0.1)
    ax.legend(loc='upper right')

def _plot_T_tank(ax): # T Graphic
    ax.plot(time_arr, tnk['T'], label='tank T', linewidth=2)
    ax.axvline(x=valve_close_t, color='orange', linestyle=':', linewidth=1.5, label='valve close t=8.0s')
    ax.set_xlim(left=-0.1)
    ax.set_ylabel('Temperature (K)', fontsize=9)
    ax.annotate('Regulated — constant T', xy=(0.5, 0.9),
               xycoords='axes fraction', ha='center', fontsize=8, color='gray')
    ax.legend()

def _plot_mass(ax): # Mass Graphic
    ax.plot(time_arr, tnk['m_ox_liq'], label='ox liq (=m_ox)', linewidth=1.5)
    ax.plot(time_arr, cmbr['m_g'], label='cmbr stored', linewidth=1.5)
    ax.plot(time_arr, grn['V']*grn['rho'], label='grain', linewidth=1.5)
    ax.axhline(y=m_ox_required, color='red', linestyle='--', alpha=0.5, label=f'initial={m_ox_required} kg')
    ax.axvline(x=valve_close_t, color='orange', linestyle=':', linewidth=1.5, label='valve close t=8.0s')
    ax.set_xlim(left=-0.1)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=8)

def _plot_regression(ax): # Regression Rate & Fuel Port Area Graphic
    mask = time_arr < valve_close_t

    # Left axis: Regression rate
    ax.plot(time_arr[mask], grn['ddot'][mask] * 1000.0, label='Regression Rate (mm/s)', color='blue', linewidth=2)
    ax.axvline(x=valve_close_t, color='orange', linestyle=':', linewidth=1.5, label='valve close t=8.0s')
    ax.set_xlim(left=-0.1)
    ax.set_ylim(bottom=0)
    ax.set_ylabel('Rate (mm/s)', fontsize=9)
    ax.legend(loc='upper left', fontsize=8)

    # Right axis: Fuel Port Area  (expanding circle: r = r_initial + d_burned)
    current_radius_mm = initial_port_radius_mm + (grn['d'] * 1000.0)   # [mm]
    fuel_port_area = np.pi * (current_radius_mm ** 2)                  # [mm²]
    ax2 = ax.twinx()
    ax2.plot(time_arr[mask], fuel_port_area[mask], label='Fuel Port Area (mm²)', color='red', linewidth=1.5)
    ax2.set_ylim(bottom=0)
    ax2.set_ylabel('Fuel Port Area (mm²)', fontsize=9)
    ax2.legend(loc='upper right', fontsize=8)

def _plot_mach(ax): #Mach Graphic
    ax.plot(time_arr, noz['Me'], label='Mach exit', linewidth=2)
    ax.axvline(x=valve_close_t, color='orange', linestyle=':', linewidth=1.5, label='valve close t=8.0s')
    ax.set_xlim(left=-0.1)
    ax.legend()

def _plot_cstar_T(ax): # C_star Graphic
    ax.plot(time_arr, cmbr['cstar'], label='cstar', linewidth=2)
    ax2_twin = ax.twinx()
    ax2_twin.plot(time_arr, cmbr['T'], label='cmbr T', color='purple', linewidth=1.5)
    ax.axvline(x=valve_close_t, color='orange', linestyle=':', linewidth=1.5, label='valve close t=8.0s')
    ax.set_xlim(left=-0.1)
    ax.legend(loc='upper left')
    ax2_twin.legend(loc='upper right')
    ax2_twin.set_ylabel('Temperature (K)', fontsize=10)

def _plot_OF(ax): # OF Graphic
    ax.plot(time_arr, cmbr['OF'], label='OF', linewidth=2)
    ax.axhline(y=OF_ratio, color='red', linestyle='--', alpha=0.6, label=f'target={OF_ratio}')
    ax.axvline(x=valve_close_t, color='orange', linestyle=':', linewidth=1.5, label='valve close t=8.0s')
    ax.set_xlim(left=-0.1)
    ax.legend()

def _plot_flux(ax): # Dynamic Radial and Axial Flux
    mask = time_arr < valve_close_t
    # Port area expands as the grain regresses
    current_radius_m = (initial_port_radius_mm / 1000.0) + grn['d']   # [m]
    A_port = np.pi * (current_radius_m ** 2)                          # [m²]
    # Axial (oxidizer) mass flux: Ga = mdot_ox / A_port
    Ga = tnk['mdot_ox_total'] / np.maximum(A_port, 1e-12)            # [kg/m²s]
    # Radial flux fraction (DTI lateral loading)
    Gr = 0.25 * Ga
    ax.plot(time_arr[mask], Ga[mask], label='Axial Flux $G_a$', color='blue', linewidth=2)
    ax.plot(time_arr[mask], Gr[mask], label='Radial Flux $G_r$ (0.25·$G_a$)', color='red', linewidth=1.5)
    ax.axvline(x=valve_close_t, color='orange', linestyle=':', linewidth=1.5, label='valve close t=8.0s')
    ax.set_xlim(left=-0.1)
    ax.set_ylabel('Flux (kg/m²s)', fontsize=9)
    ax.legend(loc='upper right', fontsize=8)

def _plot_pressure_pa(ax): # Main Dashboard Pressure Graphic (in Pa)
    mask = time_arr < valve_close_t
    ax.plot(time_arr[mask], cmbr['P'][mask], label='chamber', linewidth=1.5)
    ax.plot(time_arr[mask], tnk['P'][mask], label='tank', linewidth=1.5)
    ax.axvline(x=valve_close_t, color='orange', linestyle=':', linewidth=1.5, label='valve close t=8.0s')
    ax.set_xlim(left=-0.1)
    ax.legend(loc='upper right', fontsize=7)

_hd_registry = [
    (_plot_thrust,     'İtki',                             'Zaman (s)', 'İtki (N)',          'HD_Thrust'),
    (_plot_mdot,       'Kütle Akış Hızları',                    'Zaman (s)', 'Kütle Akış Hızı (kg/s)',    'HD_Mdot'),
    (_plot_mdot_split, 'Oksitleyici Debi Dağılımı (Axial vs Radial)', 'Zaman (s)', 'Kütle Akış Hızı (kg/s)',    'HD_Mdot_Split'),
    (_plot_pressure,   'Basınç',                           'Zaman (s)', 'Basınç (Bar)',      'HD_Pressure'),
    (_plot_T_tank,     'Tank Sıcaklığı',                   'Zaman (s)', 'Sıcaklık (K)',     'HD_Tank_Temp'),
    (_plot_mass,       'Kütle Dağılımı',                  'Zaman (s)', 'Kütle (kg)',           'HD_Mass'),
    (_plot_regression, 'Regression Rate & Fuel Port Area',  'Zaman (s)', '',                    'HD_Regression'),
    (_plot_mach,       'Nozzle Exit Mach',                   'Time (s)', 'Mach',                'HD_Mach'),
    (_plot_cstar_T,    'c* and Chamber Temperature',         'Time (s)', 'c* (m/s)',            'HD_Cstar_T'),
    (_plot_OF,         'OF',                                 'Time (s)', 'OF',                  'HD_OF'),
    (_plot_flux,       'Flux (Radial & Axial)',              'Time (s)', 'Flux (kg/m²s)',       'HD_Flux'),
]

dashboard_items = [
    (_plot_thrust, 'Thrust'),
    (_plot_mdot, 'mdot (kg/s)'),
    (_plot_mdot_split, 'mdot Split (Axial/Radial)'),
    (_plot_pressure_pa, 'P (Pa)'),
    (_plot_T_tank, 'T tank'),
    (_plot_mass, 'm'),
    (_plot_regression, 'Regression Rate & Fuel Port Area'),
    (_plot_mach, 'Mach'),
    (_plot_cstar_T, 'c* / T'),
    (_plot_OF, 'OF'),
    (_plot_flux, 'Radial & Axial Flux'),
]

def _open_hd(idx):
    plot_fn, title, xlabel, ylabel, fname = _hd_registry[idx]
    
    hd_fig, hd_ax = plt.subplots(figsize=(10, 5))
    hd_fig.canvas.manager.set_window_title(f"HD Analysis: {title}")
    
    plot_fn(hd_ax)
    hd_ax.set_xlabel(xlabel, fontsize=12, fontweight='bold')
    
    if ylabel:
        hd_ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
        
    hd_ax.set_title(title, fontsize=14, fontweight='bold')
    hd_ax.grid(True, linestyle=':', alpha=0.7)
    hd_fig.tight_layout()
    
    save_path = results_path / f'{fname}.png'
    hd_fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    hd_fig.canvas.draw_idle()
    hd_fig.show()

# ── Static 3×4 Dashboard Layout ──────────────────────────────────────────────
fig, axs = plt.subplots(nrows=3, ncols=4, figsize=(20, 13)) # Matris büyütüldü
fig.canvas.manager.set_window_title('HRAP Simulation — Nitrous / Plastisol')
axs_flat = np.array(axs).ravel()

# Sadece dolu olanları (ilk 11 grafik) çiz, sonuncuyu (12. boş kutuyu) gizle
for i in range(len(axs_flat)):
    ax = axs_flat[i]
    if i < len(dashboard_items):
        plot_fn, title = dashboard_items[i]
        plot_fn(ax)
        ax.set_title(title, fontsize=9)
    else:
        ax.set_visible(False) # 12. boş paneli gizle

# Butonları bağla (Sadece görünür olan ilk 11 panele)
_hd_buttons = button.attach_hd_buttons(fig, axs_flat[:len(dashboard_items)], _open_hd)

fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.suptitle('HRAP Simulation Results   (Click + for High-Res plots)',
             fontsize=11, fontweight='bold', y=0.98)

fig.savefig(results_path/(file_prefix+'_plots.pdf'), format='pdf', bbox_inches='tight', pad_inches=0.1)

plt.show()