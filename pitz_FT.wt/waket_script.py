# %%
import numpy as np
import scipy.constants as ct
from wake_t.beamline_elements import PlasmaStage
from wake_t.particles.particle_bunch import ParticleBunch
from bunch_profiles import gaussian_bunch_from_twiss
import matplotlib.pyplot as plt
from pip4sims.plotting import plot_lps, plot_phasespace

# %%
np0 = 1e21
kp = np.sqrt(np0 * ct.e**2 / (ct.epsilon_0 * ct.m_e * ct.c**2))

plasma_start = 5 / kp
plasma_ramp_length = 6 / kp
plasma_plateau_length = 595 / kp

def density_profile(z):
    # Allocate relative density
    n = np.ones_like(z) * 1e-10
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
rmax_plasma = 10 / kp           # radius of the plasma column
rmax = 2.0 * rmax_plasma       # radius of the simulation box
dz_adv = 0.02 / kp             # Advised longitudinal resolution
Nz_adv = int(L_box / dz_adv)
Nz   = Nz_adv                  # Number of gridpoints along z
dr_adv = 0.02 / kp
Nr_adv = int(rmax / dr_adv)
Nr = Nr_adv                    # Number of gridpoints along r
ppc = 4                        # Particles per cell
dz_fields = 0.5 * L_box        # Field update step

# Propagation distance
dt_os = 25 * 0.024
time_os = 705 * dt_os
print(f'{time_os = }')
L_interact = time_os / kp
N_diag = 10  # Number of diagnostics

# %%
# Create plasma stage.
plasma = PlasmaStage(
    length=L_interact, density=density_profile,
    wakefield_model='quasistatic_2d', n_out=N_diag,
    laser_evolution=True,
    r_max=rmax, r_max_plasma=rmax_plasma, xi_min=zmin, xi_max=zmax,
    n_r=Nr, n_xi=Nz, dz_fields=dz_fields, ppc=ppc,
)

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

# %%
w = q / ct.e
bunch = ParticleBunch(w, x, y, z, ux, uy, uz, name='beam')

# %%
bunch_list = plasma.track([bunch], opmd_diag=True)


