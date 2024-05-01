import numpy as np
from scipy.constants import c


def gaussian_bunch_from_twiss(q_tot, n_part, gamma0, s_g, s_z,
        a_x, a_y, b_x, b_y, en_x, en_y, lon_profile='gauss',
        s_edge=0., z_c=0., x_c=0., y_c=0., chirp=None, seed=None):

    n_part = int(n_part)

    if seed is not None:
        np.random.seed(seed)

    # LPS
    if lon_profile == 'gauss':
        z = np.random.normal(z_c, s_z, n_part)
    elif lon_profile == 'flattop':
        length = s_z
        norma = length + np.sqrt(2 * np.pi) * s_edge
        n_plat = int(n_part * length / norma)
        n_gaus = int(n_part * np.sqrt(2 * np.pi) * s_edge / norma)
        # Create flattop and gaussian profiles
        z_plat = np.random.uniform(0., length, n_plat)
        z_gaus = s_edge * np.random.standard_normal(n_gaus)
        # Concatenate both profiles
        z = np.concatenate((z_gaus[np.where(z_gaus <= 0)],
                            z_plat,
                            z_gaus[np.where(z_gaus > 0)] + length))
        z = z - length / 2. + z_c  # shift position
        n_part = n_plat + n_gaus

    gamma = np.random.normal(gamma0, s_g, n_part)
    if chirp is not None:
        zmean = np.mean(z)
        gamma = gamma + chirp * gamma0 * (z - zmean)
    pz = np.sqrt(gamma ** 2 - 1)

    # TPS        
    em_x = en_x / gamma0
    em_y = en_y / gamma0
    g_x = (1 + a_x**2) / b_x
    g_y = (1 + a_y**2) / b_y
    s_x = np.sqrt(em_x * b_x)
    s_y = np.sqrt(em_y * b_y)
    s_xp = np.sqrt(em_x * g_x)
    s_yp = np.sqrt(em_y * g_y)
    p_x = -a_x * em_x / (s_x * s_xp)
    p_y = -a_y * em_y / (s_y * s_yp)
    # Create normalized gaussian distributions
    u_x = np.random.standard_normal(n_part)
    v_x = np.random.standard_normal(n_part)
    u_y = np.random.standard_normal(n_part)
    v_y = np.random.standard_normal(n_part)
    # Calculate transverse particle distributions
    x = x_c + s_x * u_x
    xp = s_xp * (p_x * u_x + np.sqrt(1 - np.square(p_x)) * v_x)
    y = y_c + s_y * u_y
    yp = s_yp * (p_y * u_y + np.sqrt(1 - np.square(p_y)) * v_y)
    # Change from slope to momentum
    px = xp * pz
    py = yp * pz
    # Charge
    q = np.ones(n_part) * q_tot / n_part

    return [x, y, z, px, py, pz, q]
