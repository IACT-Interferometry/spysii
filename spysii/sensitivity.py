import astropy.units as u
import numpy as np


@u.quantity_input(zd='angle', observation_time='time')
def signal_to_noise(observatory, telescopes_pair, b_mag, zd, squared_visibility,
                    observation_time, moon_to_star_ratio, is_checkboard=False, mirror_areas=False,
                    identical_telescopes=False):
    # TODO: Add mirror configs.
    site_constants = observatory['constants']

    # If identical_telescopes is true, it does not matter which telescope we choose, so we just use the first entry:
    if identical_telescopes:
        tx_index = 0
        ty_index = 0
    else:
        # Identify which indexes are the two telescopes of the correlation:
        tx_index = np.where(np.array(observatory['telescope_names']) == telescopes_pair[0])[0][0]
        ty_index = np.where(np.array(observatory['telescope_names']) == telescopes_pair[1])[0][0]

    # Mirror area [m^2]
    area = np.sqrt(site_constants['area'][tx_index]*site_constants['area'][ty_index])  #
    if is_checkboard:
        area = np.sqrt(site_constants['area'][tx_index]/2. * site_constants['area'][ty_index]/2.)
    if mirror_areas is not False:
        area = np.sqrt(mirror_areas[0]*mirror_areas[1])
    # quantum efficiency of the PMTs
    alpha_qe = np.sqrt(site_constants['alpha_QE'][tx_index]*site_constants['alpha_QE'][ty_index])

    # quantum efficiency of the remaining optics
    g = np.sqrt(site_constants['g'][tx_index]*site_constants['g'][ty_index])

    # (Hz) effective cross-correlation electrical bandwidth
    b_v = np.sqrt(float(site_constants['b_v'][tx_index])*float(site_constants['b_v'][ty_index]))
    # normalized spectral distribution of the light after the filter
    sigma = np.average([site_constants['sigma'][tx_index],
                        site_constants['sigma'][ty_index]])
    # F^{-1} excess noise factor of the PMT
    f_1 = 1.0/np.average([site_constants['F'][tx_index],
                          site_constants['F'][ty_index]])
    # excess noise introduced by background light
    bck_light_noise = np.average([site_constants['bck_light_noise'][tx_index],
                                  site_constants['bck_light_noise'][ty_index]])

    electronic_noise = np.average([site_constants['elec_noise'][tx_index],
                                   site_constants['elec_noise'][ty_index]])
    # zero baseline correlation (1 for single stars, <1 for multiple systems)
    zbc = np.average([site_constants['cN'][tx_index],
                      site_constants['cN'][ty_index]])
    # polarization factor
    beta0 = np.average([site_constants['beta0'][tx_index],
                        site_constants['beta0'][ty_index]])
    # Term accounting for the star to moon ratio (background)
    if hasattr(moon_to_star_ratio, '__len__'):
        a = np.array(moon_to_star_ratio)
        # Calculate the geometric mean of a values:
        a_geom = a.prod() ** (1.0 / len(a))
        moon_noise = 1/(1+a_geom)
    else:
        moon_noise = 1/(1+moon_to_star_ratio)

    b_mag_corr = b_mag + (0.23 / np.cos(zd.to('radian').value))
    n = 0.0000869 / (2.5 ** b_mag_corr)  # star's differential photon flux

    # S/N over 1 s
    s_n = area * alpha_qe * g * n * squared_visibility * np.sqrt(b_v) * f_1 * \
        np.sqrt(1. / 2.0) * zbc * beta0 * bck_light_noise * electronic_noise * sigma * moon_noise

    # If the observatory has a "multiple_channels" entry, we assume that the S/N improves as sqrt(N_channels):
    if 'multiple_channels' in observatory.keys():
        s_n = s_n * np.sqrt(observatory['multiple_channels'])

    # from seconds to minutes
    # t_sigma = (((target_sigmas ** 2) * 2 / b_v) * (area * alpha_qe * g * n * squared_visibility * f_1 * zbc *
    #  beta0 * bck_light_noise * electronic_noise * sigma) ** (-2)) * u.min
    return s_n * np.sqrt(observation_time.to('s').value)
