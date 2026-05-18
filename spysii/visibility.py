import numpy as np
from scipy import special
import astropy.units as u

loaded_interpolator = None

def prepare_uv_coords(coord):
    if isinstance(coord, float):
        return np.array([coord])
    elif len(np.shape(coord)) == 1:
        return coord[:, np.newaxis]
    elif len(np.shape(coord)) == 2:
        return coord[:, :, np.newaxis]

def visibility_uniform_disk(baseline, theta, wavelength):
    bessel_first = special.j1
    return (2*bessel_first(np.pi * theta * (baseline/wavelength))) / \
           (np.pi * theta * (baseline/wavelength))


def abs_visibility_uniform_disk(baseline, theta, norm, wavelength):
    bessel_first = special.j1
    result = np.ones(np.shape(baseline))
    result[baseline > 0.] = np.abs((2*bessel_first(np.pi * theta * (baseline[baseline > 0.]/wavelength))) /
                                   (np.pi * theta * (baseline[baseline > 0.]/wavelength)))
    return norm * result


def squared_visibility_uniform_disk(baseline, theta, norm, wavelength):
    return abs_visibility_uniform_disk(baseline, theta, norm, wavelength)**2


def visibility_fast_rotator(u_m, v_m, semi_major_axis, semi_major_axis_orientation, compression_factor, wavelength):
    u_coord = u_m / wavelength
    v_coord = v_m / wavelength
    u_theta = v_coord*np.cos(semi_major_axis_orientation) + u_coord*np.sin(semi_major_axis_orientation)
    v_theta = -v_coord*np.sin(semi_major_axis_orientation) + u_coord*np.cos(semi_major_axis_orientation)
    r = np.sqrt((u_theta**2) + ((v_theta**2)*(compression_factor**2)))
    # bessel_zero = special.j0
    bessel_first = special.j1
    uniform_disc_distorted = (2*bessel_first(2*np.pi*semi_major_axis*r)) / (2*np.pi*semi_major_axis*r)
    # return bessel_zero(2*np.pi*semi_major_axis*r)
    return uniform_disc_distorted


def squared_visibility_fast_rotator(u_m, v_m, semi_major_axis, semi_major_axis_orientation,
                                    compression_factor, wavelength):
    return visibility_fast_rotator(u_m, v_m, semi_major_axis, semi_major_axis_orientation,
                                   compression_factor, wavelength)**2

def protected_squared_visibility_uniform_disk(x, theta, norm, wavelength=420e-9):
    mas_to_radian = 4.84814e-9
    # In MAGIC we currently use the Semrock 425-26nm, which is weirdly centered at 420 nm...
    # http://wiki.magic.pic.es/index.php/Intensity_Interferometry#Transmission_curve_of_a_narrower_bandpass_filter
    if norm < 0.:
        norm = 0.
    return abs_visibility_uniform_disk(np.array(x, dtype=float), theta*mas_to_radian, np.sqrt(norm), wavelength)**2


def protected_squared_visibility_fast_rotator(u_m, v_m,
                                              semi_major_axis, semi_major_axis_orientation,
                                              compression_factor, norm, wavelength=420e-9):
    mas_to_radian = 4.84814e-9
    # In MAGIC we currently use the Semrock 425-26nm, which is weirdly centered at 420 nm...
    # http://wiki.magic.pic.es/index.php/Intensity_Interferometry#Transmission_curve_of_a_narrower_bandpass_filter
    if norm < 0.:
        norm = 0.
    result = np.ones(np.shape(u_m))
    mask = np.abs(u_m**2 + v_m**2) > 0.
    result[mask] = squared_visibility_fast_rotator(u_m[mask], 
                                                   v_m[mask], 
                                                   semi_major_axis*mas_to_radian, 
                                                   semi_major_axis_orientation,
                                                   compression_factor, 
                                                   wavelength)
    return result * norm

# Limb darkened profile:
def squared_visibility_limb_darkened(baseline, theta, norm, wavelength, u_term):
    # Make sure baselines with values below or equal to 0 have values extremely close to 0 (so that we don't get
    # NaNs when dividing by 0):
    bs = np.array(baseline)
    bs[bs <= 0.] = 1e-20
    x = np.array((np.pi*theta*bs)/wavelength)
    first_term = np.power(((1.-u_term)/2.) + (u_term/3.), -2.)
    second_term = np.power(((1-u_term)*(special.j1(x)/x)) + u_term*np.sqrt(np.pi/2.)*(special.jv(3/2.,
                                                                                                 x)/(x**(3/2.))),
                           2.)
    return norm*first_term*second_term

def protected_squared_visibility_limb_darkened(x, theta, norm, u_term, wavelength=420e-9):
    mas_to_radian = 4.84814e-9
    if norm < 0.:
        norm = 0.
    if theta < 0.:
        theta = 0.0000001
    return squared_visibility_limb_darkened(np.array(x, dtype=float), theta*mas_to_radian, norm, wavelength,
                                            u_term)
