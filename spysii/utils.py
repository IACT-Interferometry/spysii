import numpy as np
import logging

from astropy.utils.data import conf
from astropy import coordinates as c
from astropy.time import Time

from spysii.errors import WrongTimeFormat

def format_time(time_stamp):
    """
    Format the given time stamp.

    Parameters
    ----------
    time_stamp : str or `~astropy.time.Time`
        The time stamp to format.

    Returns
    -------
    `~astropy.time.Time`
        Formatted time object.
    """
    if type(time_stamp) == str:
        return Time(time_stamp, format='isot', scale='utc')
    elif isinstance(time_stamp, Time):
        return time_stamp
    elif time_stamp is None:
        return None
    else:
        raise WrongTimeFormat()

def get_sky_coordinate(source_name):
    # Set time limit for vizier queries to 20 s:
    with conf.set_temp('remote_timeout', 20):
        try:
            target_coord = c.SkyCoord.from_name(source_name)
            print("Found star with c.SkyCoord.from_name")
            print(target_coord)
        except Exception:
            from spysii.errors import MissingTargetCoordinateError
            raise MissingTargetCoordinateError(source_name)
    return target_coord


def power_law(x, a, b):
    return a*np.power(x, b)


def load_pickle_file(input_file):
    """
    Load the content of a pickle file

    Parameters
    ----------
    input_file: str
        Filename of the pickle file. It will give an error if it cannot be loaded by pickle.

    Returns
    -------
    dict()
        Dictionary containing all data stored by 'reduce_binaries'.
    """
    import pickle
    file = open(input_file, 'rb')
    try:
        loaded_data = pickle.load(file)
    except MemoryError:
        logging.error("MemoryError loading the pickle file: {}".format(input_file))
        raise MemoryError
    file.close()
    return loaded_data

def free_gaussian(x, a, x0, sigma, dense=False):
    return a*np.exp(-(x-x0)**2/(2*sigma**2))

def get_azimuth_angle(u_m, v_m):
    """
    Function used to compute the azimuth angle  of the projected baseline
    in the uv plane. The angle is defined as the angle from north to east in the uv plane.
    Parameters
    ----------
    u_m: Value of the u axis in the UV-plane (we assume it is positive)
    v_m: Value of the v axis in the UV-plane

    Returns
    -------
    azimuth_angle: Value of the position angle of the projected baseline (deg)
    """
    if (u_m < 0).any():
        azimuth_angle = np.array([np.arctan2(u_m[i], v_m[i]) * 180 / np.pi if u_m[i] >= 0 else
                                    np.arctan2(u_m[i], v_m[i]) * 180 / np.pi + 180
                                    for i in range(len(u_m))])
    else:
        azimuth_angle = np.arctan2(u_m, v_m) * 180 / np.pi
    return azimuth_angle

def validate_observatory(observatory):
    """
    Validate the observatory configuration.

    Parameters
    ----------
    observatory : dict
        Dictionary containing the observatory configuration.

    Returns
    -------
    dict
        Validated observatory configuration.
    """
    # First check the number of telescope names:
    number_of_telescopes = len(observatory['telescope_names'])
    # Make sure there are as many X/Y coordinates as telescope names:
    if len(observatory['telescope_locations']) != number_of_telescopes:
        raise ValueError("The number of telescope coordinates is different than the number of telescope names.")
    # Check the rest of parameters. If some entries have only one parameter, we assume it is the same for all
    # telescopes:
    ones = np.ones_like(observatory['telescope_names'], dtype=np.float64)
    if len(observatory['telescope_cabling_delays']) == 1:
        observatory['telescope_cabling_delays'] = ones * observatory['telescope_cabling_delays']
    # Do the same with all constants:
    for constant_name in ["area", "alpha_QE", "g", "b_v", "sigma", "F",
                          "bck_light_noise", "elec_noise", "cN", "beta0"]:
        if len(observatory['constants'][constant_name]) == 1:
            observatory['constants'][constant_name] = ones * float(observatory['constants'][constant_name][0])
    return observatory

