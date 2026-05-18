from astropy.time import Time
import astropy.units as u
import numpy as np


def get_light_travel_delay(skycoord, obs_time, location, tx_xyz_position,
                           ty_xyz_position, verbose=False, get_distances=False,
                           lst1_present=False):
    """
    General function to calculate light-travel delays and baselines

    Unnecessarily slow, but works.

    Parameters
    ----------
    skycoord: astropy.coordinates.SkyCoord
        SkyCoord of the target star

    obs_time: astropy.time.Time
        Observation times considered in the light travel delay correction.

    location: astropy.coordinates.EarthLocation
        EarthLocation of the observation site

    tx_xyz_position: array[3]
        Relative X, Y, Z coordinates of telescope 1, in meters.

    ty_xyz_position: array[3]
        Relative X, Y, Z coordinates of telescope 2, in meters.

    verbose: bool
        If true, print some info.

    get_distances: bool
        If true, also return the projected inter-telescope distance (1-D baseline)

    Returns
    -------
    astropy.units.Quantity
        Time of the light-travel delay (on the order of 200 ns).
    """
    from scipy.spatial.transform import Rotation
    from astropy.coordinates import HADec

    if not isinstance(obs_time, Time):
        obs_time = Time(obs_time, format='isot', scale='utc')
    # Refraction index of the air:
    n = 1.000293
    # Light speed in medium:
    c_speed = 299792458e-9/n

    # We need to calculate the hour angle for the UV calculation:
    hadec_frame = HADec(obstime=obs_time, location=location)
    target_hadec_frame = skycoord.transform_to(hadec_frame)
    # H = times.sidereal_time('apparent') - target_coord.ra
    ha = target_hadec_frame.ha
    dec = target_hadec_frame.dec

    if verbose:
        print("dec: {}   HA: {}   LAT: {}".format(dec.deg, ha.hour, location.lat.deg))
        print("HAdec: {:2.3f}   &  star dec: {:2.3f}".format(target_hadec_frame.dec.deg, skycoord.dec.deg))
    r1 = Rotation.from_euler('x', dec.rad)
    r2 = Rotation.from_euler('y', ha.rad)
    r3 = Rotation.from_euler('x', -location.lat.rad)
    rot = r1 * r2 * r3
    # We compute baseline in u,v,w coordinates (in m):
    baseline = rot.apply(np.array(ty_xyz_position) - np.array(tx_xyz_position))
    # Optical path delay from w coordinate:
    optical_path_delay = baseline[2] / c_speed * u.ns
    # if lst1_present:
    #     # In case LST1 is present, the difference in focal distance lengths also needs to be taken into account:
    #     optical_path_delay += 2 * ((23 - 17) / 0.299792) * u.ns
    # 1D baseline from the uv coordinates:
    projected_intertelescope_distance = np.sqrt(np.power(baseline[0], 2.) +
                                                np.power(baseline[1], 2.)) * u.m

    if get_distances:
        return optical_path_delay, projected_intertelescope_distance, baseline
    else:
        return optical_path_delay, baseline


def get_array_baselines(skycoord, obs_time, location, tx_xyz_positions, ty_xyz_positions, verbose=False):
    from scipy.spatial.transform import Rotation
    from astropy.coordinates import HADec

    if not isinstance(obs_time, Time):
        obs_time = Time(obs_time, format='isot', scale='utc')

    # We need to calculate the hour angle for the UV calculation:
    hadec_frame = HADec(obstime=obs_time, location=location)
    target_hadec_frame = skycoord.transform_to(hadec_frame)
    # H = times.sidereal_time('apparent') - target_coord.ra
    ha = target_hadec_frame.ha
    dec = target_hadec_frame.dec

    if verbose:
        print("dec: {}   HA: {}   LAT: {}".format(dec.deg, ha.hour, location.lat.deg))
        print("HAdec: {:2.3f}   &  star dec: {:2.3f}".format(target_hadec_frame.dec.deg, skycoord.dec.deg))
    r1 = Rotation.from_euler('x', dec.rad)
    r2 = Rotation.from_euler('y', ha.rad)
    r3 = Rotation.from_euler('x', -location.lat.rad)
    rot = r1 * r2 * r3
    # We compute baseline in u,v,w coordinates (in m):
    return rot.apply(np.array(ty_xyz_positions) - np.array(tx_xyz_positions))
