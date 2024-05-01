# %%
import numpy as np
import scipy.constants as ct
from bunch_profiles import gaussian_bunch_from_twiss
from fbpic.main import Simulation
from fbpic.lpa_utils.bunch import add_elec_bunch_from_arrays
from fbpic.lpa_utils.boosted_frame import BoostConverter
from fbpic.openpmd_diag import BackTransformedFieldDiagnostic, \
    BackTransformedParticleDiagnostic

# %%
# Define plasma density
np0 = 1e21
kp = np.sqrt(np0 * ct.e**2 / (ct.epsilon_0 * ct.m_e * ct.c**2))

plasma_start = 5 / kp
plasma_ramp_length = 6 / kp
plasma_plateau_length = 595 / kp

def density_profile(z, r):
    # Allocate relative density
    n = np.zeros_like(z)
    z_0 = plasma_start
    z_1 = z_0 + plasma_ramp_length
    z_2 = z_1 + plasma_plateau_length
    z_3 = z_2 + plasma_ramp_length
    # Ramp up
    n = np.where((z > z_0) & (z < z_1), (z - z_0) / (z_1 - z_0), n)
    # Plateau
    n = np.where((z > z_1) & (z < z_2), 1, n)
    # Ramp down
    n = np.where((z > z_2) & (z < z_3), 1 - (z - z_2) / (z_3 - z_2), n)
    # Return absolute density
    return n * np0

# %%
# Simulation parameters
L_box = 50 / kp                # Box length
zmax = 0.0
zmin = zmax - L_box            # Left  edge of the simulation box (meters)
rmax_plasma = 10 / kp          # radius of the plasma column
rmax = 2.0 * rmax_plasma       # radius of the simulation box
dz_adv = 0.02 / kp             # Advised longitudinal resolution
Nz_adv = int(L_box / dz_adv)
Nz   = Nz_adv                  # Number of gridpoints along z
dr_adv = 0.02 / kp
Nr_adv = int(rmax / dr_adv)
Nr = Nr_adv                    # Number of gridpoints along r
# Resolution
dz = (zmax - zmin) / Nz
dr = rmax / Nr
dt = (zmax - zmin) / Nz / ct.c # Timestep
Nm = 1                         # Number of modes used
# Windows velocity
v_window = ct.c

# %%
# Define PITZ beam
E = 21.5e6 * ct.eV
gamma0 = E / (ct.m_e * ct.c**2)
s_g = (60e3 * ct.eV / E) * gamma0
Q = 100e-12
length = (40.46 - 9.54) / kp
s_edge = 1.85 / kp
current0 = 5.0
en_x = 0.372e-6
en_y = 0.372e-6
s_x = 42e-6
s_y = 42e-6
b_x = s_x**2 / (en_x / gamma0)
b_y = s_y**2 / (en_y / gamma0)
a_x = 0.0
a_y = 0.0
Np = 10000000
beam_data = gaussian_bunch_from_twiss(q_tot=Q, n_part=Np, gamma0=gamma0, s_g=s_g,
                a_x=a_x, a_y=a_y, b_x=b_x, b_y=b_y, en_x=en_x, en_y=en_y,
                z_c=(zmin + zmax) / 2,
                lon_profile='flattop', s_z=length, s_edge=s_edge)
x, y, z, ux, uy, uz, q = beam_data


# Boosted frame setup
# -------------------
# Boost factor
gamma_boost = 2.8
boost = BoostConverter(gamma_boost)
# corrected dt (if applies)
dt = min(dr / (2 * gamma_boost) / ct.c, dt)
# Velocity of the Galilean frame (for suppression of the NCI)
v_comoving = -np.sqrt(gamma_boost**2 - 1.) / gamma_boost * ct.c

# Propagation distance
dt_os = 25 * 0.024
time_os = 705 * dt_os
L_lab_interact = time_os / kp
T_lab_interact = (L_lab_interact + (zmax - zmin)) / v_window
N_lab_diag = 11  # Number of diagnostics
lab_diag_period = L_lab_interact / (N_lab_diag - 1) / ct.c
write_period = 500

# number of timesteps for the moving window to slide across the plasma
N_iter_lab = int(T_lab_interact / dt)
# Interaction time (seconds)
T_interact = boost.interaction_time( L_lab_interact, (zmax - zmin), v_window)
# In boosted frame:
L_box_boost, dz_boost, dt_boost = boost.copropag_length([L_box, dz, dt],
                                    beta_object=v_window / ct.c)
L_boost_interact, = boost.static_length([L_lab_interact])
N_iter_boost = int(T_interact / dt_boost)

print()
print('Box and channel lengths:')
print('Interaction length           = %6.3f mm -> %i iterations' % (L_lab_interact / 1e-3, N_iter_lab))
print('Box length                   = %6.3f mm' % (L_box / 1e-3))
print('Interaction length (boosted) = %6.3f mm -> %i iterations' % (L_boost_interact / 1e-3, N_iter_boost))
print('Box length (boosted)         = %6.3f mm' % (L_box_boost / 1e-3))
print()
print('Sanity checks:')
print('see -> https://fbpic.github.io/advanced/boosted_frame.html')
print('- Cerenkov condition:')
print('c dt_boost            = %.2e < dr_boost = %.2e -> %r -> ratio = %.2f' % (ct.c * dt_boost, dr, ct.c * dt_boost < dr, dr / (ct.c * dt_boost)))
print('- Performance conditions (speed up):')
print('g_boost^2             = %.2f < Lp / Lb  = %.2f -> %r -> ratio = %.2f' %
    (gamma_boost**2, L_lab_interact / L_box, gamma_boost**2 < L_lab_interact / L_box, L_lab_interact / L_box / gamma_boost**2))
print('Lp_boost / Lb_boost   = %.2f' % (L_boost_interact / L_box_boost))
gamma_beam = gamma0
print('gamma_boost           = %.2f < gamma_beam / 2 = %.2f -> %r -> ratio = %.2f' %
    (gamma_boost, gamma_beam / 2., gamma_boost < gamma_beam / 2, gamma_beam / 2. / gamma_boost))
print()
print('Propagation distance  = %.3f cm' % (L_lab_interact / 1e-2))
print('Diagnostics period    = %.3f cm' % (lab_diag_period * ct.c / 1e-2))
print('Number of diagnostics = %i' % N_lab_diag)
print()

# %%
# Initialize the simulation object
sim = Simulation(Nz, zmax, Nr, rmax, Nm, dt, zmin=zmin,
                 v_comoving=v_comoving,
                 gamma_boost=boost.gamma0,
                 particle_shape='cubic',
                 n_order=-1, use_cuda=True,
                 boundaries={'z': 'open', 'r': 'open'})

# The particles
p_zmin = zmax    # Position of the beginning of the plasma (meters)
p_zmax = L_lab_interact
p_rmax = rmax_plasma
p_nz = 2     # Number of particles per cell along z
p_nr = 2     # Number of particles per cell along r
p_nt = 4     # Number of particles per cell along theta
plasma_elec = sim.add_new_species(q=-ct.e, m=ct.m_e, n=1,
    dens_func=density_profile, boost_positions_in_dens_func=True,
    p_nz=p_nz, p_nr=p_nr, p_nt=p_nt, p_zmin=p_zmin, p_rmax=p_rmax)

plasma_ions = sim.add_new_species(q=ct.e, m=ct.m_p, n=1,
    dens_func=density_profile, boost_positions_in_dens_func=True,
    p_zmin=p_zmin, p_zmax=p_zmax, p_rmax=p_rmax,
    p_nz=p_nz, p_nr=p_nr, p_nt=p_nt )

# Add PITZ bunch
w = q / ct.e
beam = add_elec_bunch_from_arrays(sim, x, y, z, ux, uy, uz, w,
                                  boost=boost)

# Convert parameter to boosted frame
v_window, = boost.velocity([v_window])
# Configure the moving window
sim.set_moving_window(v=v_window)

# Add a diagnostics
sim.diags = [BackTransformedFieldDiagnostic( zmin, zmax, ct.c,
             lab_diag_period, N_lab_diag, boost.gamma0,
             period=write_period, fldobject=sim.fld, comm=sim.comm,
             fieldtypes=["E", "rho"]),
             BackTransformedParticleDiagnostic( zmin, zmax, ct.c,
             lab_diag_period, N_lab_diag, boost.gamma0,
             period=write_period, fldobject=sim.fld, comm=sim.comm,
             select={'uz': [10., None]}, species={"beam": beam})]

# %%
# Number of iterations to perform
N_step = int(T_interact / sim.dt)
# round total number of iterations to accomodate the data writing period
N_step = N_step + write_period - N_step % write_period + 1
print('Number of iterations     = %i' % (N_step))

# Run the simulation
sim.step(N_step)
