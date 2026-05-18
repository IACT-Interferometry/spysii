from astropy import coordinates as c
import astropy.units as u
import numpy as np
from astropy.coordinates import Angle
from itertools import combinations

from pygments.lexer import inherit
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import matplotlib
import scipy
import os
import logging
import yaml

from spysii.catalog import catalog
from spysii import PACKAGE_DATA
from spysii.utils import get_sky_coordinate, validate_observatory, get_azimuth_angle
from spysii.sensitivity import signal_to_noise
from spysii.light_travel import get_light_travel_delay, get_array_baselines
from spysii.visibility import protected_squared_visibility_uniform_disk, protected_squared_visibility_fast_rotator
from astropy.time import Time

valid_observatory_names = ["MAGIC", "MAGIC-LST", "MAGIC-4LST", "VERITAS",
                           "MAGIC-4LST-microstars", "CTA-N", "CTA-N_alpha",
                           "MAGIC-CTA-N", "CTA-N-LSTs", "CTA-S", "CTA-S_alpha", "CTA-S_alpha_2LSTs",
                           "CTA-N_alpha_phase_II", "CTA-S_alpha_2LSTs_phase_II",
                           "CTA-S_alpha_phase_II", "MAGIC-butterfleye", "MAGIC2-facets-butterfleye",
                           "MAGIC-4LST-butterfleye", "MAGIC-CTA-N-butterfleye",
                           "4LST", "4LST-butterfleye"
                           ]

class SimulateSiiRuns(object):
    """
    A class to simulate SII runs.

    """

    @u.quantity_input(stellar_diameter='angle')
    def __init__(self, source_name, stellar_diameter=None, zbc_norm=1., observatory_name="MAGIC-LST",
                 wavelength=420e-9, semi_major_axis_orientation=None, compression_factor=None, sky_image=None):
        """
        A class to read MAGIC SII runs.
        Input files should be generated with "reduce_binaries", making sure it was executed
        with the '--source_name' parameter (so all metadata associated to the star is already there)

        Parameters
        ----------
        source_name: str
            Name of the source to be simulated

        stellar_diameter: astropy Quantity
            Stellar diameter of the source to be simulated. If not provided, it will be obtained from the JMDC catalog.
            The value should be in mas.

        zbc_norm: float
            ZBC normalization factor. This is the value of the ZBC at 420 nm, and it is used to normalize the
            squared visibility values.

        observatory_name: str
            Name of the observatory where the SII run is to be simulated. Valid names are:
                ["MAGIC", "MAGIC-LST", "MAGIC-4LST", "VERITAS",
                 "MAGIC-4LST-microstars", "CTA-N", "CTA-N_alpha",
                 "MAGIC-CTA-N", "CTA-N-LSTs", "CTA-S", "CTA-S_alpha", "CTA-N_alpha_phase_II",
                 "CTA-S_alpha_phase_II", "MAGIC-butterfleye", "MAGIC2-facets-butterfleye",
                 "MAGIC-4LST-butterfleye", "MAGIC-CTA-N-butterfleye",
                 "4LST", "4LST-butterfleye"
                 ]

        wavelength: astropy Quantity
            Wavelength at which the SII run is to be simulated. Default is 420 nm.

        semi_major_axis_orientation: astropy Quantity
            Orientation of the semi-major axis of the fast rotator star model. If provided, the
            compression_factor parameter should also be provided.

        compression_factor: float
            Compression factor of the fast rotator star model. If provided, the
            semi_major_axis_orientation parameter should also be provided.

        sky_image: spysii.sky_image.SkyImage
            Sky image object containing the brightness distribution of the target star.
            It should have the visibility already computed.
        """
        # TODO: This should be a dictionary containing all the quality cuts we apply to the data
        # TODO: For the moment, ignore... We need to explore if they will be necessary at all.

        self.observatory_name = observatory_name
        self.source_name = source_name
        self.wavelength = wavelength
        self.semi_major_axis_orientation = semi_major_axis_orientation
        self.compression_factor = compression_factor
        if sky_image is not None:
            # Make sure the sky image is the right class:
            from spysii.sky_image import SkyImage
            if not isinstance(sky_image, SkyImage):
                logging.error("The input sky_image is not an instance of spysii.sky_image.SkyImage")
                raise TypeError
            else:
                # Make sure V² has been computed:
                if sky_image.fft_magnitude is None:
                    logging.error("The input sky_image has not computed the visibility yet.")
                    raise ValueError
        self.sky_image = sky_image
        # Some telescope layouts have a telescope multiplicity value larger than 1 (e.g. MAGIC butterfleye, as
        # there are two identical telescopes taking the same short-baseline measurements.
        self.tel_multiplicity = 1
        # Container of the figure:
        self.f = None
        # Check that the observatory name is valid:
        if observatory_name not in valid_observatory_names:
            logging.error("The input observatory name ({}) is not within the list of valid names.".format(
                observatory_name))
            raise ValueError()
        # Load observatory configuration file
        config_filename = os.path.join(PACKAGE_DATA, 'telescope_layouts/{}.yml'.format(observatory_name))
        with open(config_filename) as file:
            conf_file = yaml.load(file, Loader=yaml.FullLoader)
        self.observatory = validate_observatory(conf_file['observatory'])
        # If baselines do not change while slewing (e.g. submirror observations):
        self.is_fixed_baselines = self.observatory['fixed_baselines']
        # Update multiplicity if needed:
        if 'tel_multiplicity' in self.observatory.keys():
            self.tel_multiplicity = self.observatory['tel_multiplicity']
        # Baselines will only be generated once:
        self.fixed_baselines = None
        self.multiplicity = None
        # UV-dependent baselines:
        self.uv_fixed_baselines = None
        self.uv_multiplicity = None
        # Using the observatory coordinates, we fix its location:
        lat = self.observatory['location']['lat']
        lon = self.observatory['location']['lon']
        self.site = c.EarthLocation(lat=Angle(lat), lon=Angle(lon))

        self.target_skycoord = None
        self.target_b_mag = None
        # Container for the uniform disc fit:
        self.uniform_disc_fit = None
        self.fast_rotator_fit = None
        self.stellar_diameter = stellar_diameter
        # In order to provide simulations consistent with adquired data, we need to specify the ZBC value.
        self.zbc_norm = zbc_norm
        self.identify_source(source_name)
        # If the stellar diameter was not provided, check value in the catalog:
        if self.stellar_diameter is None:
            logging.info("Using stellar diameter value of the JMDC catalog.")
            self.stellar_diameter = catalog.get_jmdc_diameter_from_coordinate(self.target_skycoord) * u.mas
            logging.info("Diameter: {} mas".format(self.stellar_diameter.to('mas').value))

        # Identify telescope pair combinations from this observatory:
        self.telescope_pair_combinations = list(combinations(self.observatory['telescope_names'], 2))
        # We extract telescope x/y/z positions for all telescope pairs:
        self.tx_xyz_positions = []
        self.ty_xyz_positions = []

        for telescope_pair in self.telescope_pair_combinations:
            tx_index = np.where(np.array(self.observatory['telescope_names']) == telescope_pair[0])[0][0]
            ty_index = np.where(np.array(self.observatory['telescope_names']) == telescope_pair[1])[0][0]
            self.tx_xyz_positions.append(self.observatory['telescope_locations'][tx_index])
            self.ty_xyz_positions.append(self.observatory['telescope_locations'][ty_index])

        self.simulated_runs = []

        # Container with a dictionary, storing stuff for efficiency reasons
        # TODO: Store some stuff from previous simulation.

    def identify_source(self, target_name):
        self.target_skycoord = get_sky_coordinate(target_name)
        if not self.target_skycoord:
            logging.warning(self.target_name)
            from spysii.errors import MissingTargetCoordinateError
            raise MissingTargetCoordinateError(target_name)
        self.target_b_mag = catalog.get_bsc_bmag_from_coordinate(self.target_skycoord)

    @u.quantity_input(duration='time')
    def simulate_run(self, datetime, duration, moon_to_star_ratio=0.1, ignore_zd=False,
                     duration_scaling=1.0, ignore_larger_baselines=False):
        # TODO: If datetime array, telescope array and target are the same, do not recompute baselines (slow!)
        if not isinstance(datetime, Time):
            logging.error("Input datetime is not an instance of astropy.time")
            raise TypeError
        # The frame times are calculated from the datetime, and the duration:
        if ignore_zd:
            frame_times = np.arange(datetime, datetime + 1 * u.min, 1 * u.min)
        else:
            frame_times = np.arange(datetime, datetime + duration, 1 * u.min)
        # Times array at the right location:
        loc_times = Time([t.isot for t in frame_times], format='isot', scale='utc',
                         location=c.EarthLocation(lat=self.observatory['location']['lat'],
                                                  lon=self.observatory['location']['lon']))
        star_alt_az = self.target_skycoord.transform_to(c.AltAz(obstime=loc_times, location=self.site))
        zds = (90. * u.deg) - star_alt_az.alt
        for i, alt in enumerate(star_alt_az.alt):
            if alt < 5. * u.deg:
                logging.error("The altitude of the source at the time {} is just too low.".format(frame_times[i]))
                # raise ValueError("The source altitude is not valid.")

        result = dict()
        # Create empty lists for the following entries:
        keys = ['telescope_pair', 'baseline', 'baseline_min', 'baseline_max', 'sim_g2', 'sim_g2_error',
                'U', 'V', 'W']
        for key in keys:
            result[key] = []

        # If the array assumes fixed baselines (e.g. submirrors within a slewing telescope), we can calculate
        # the baselines, and bundle all identical ones:
        if self.is_fixed_baselines:
            if self.fixed_baselines is None:
                # UV dependent try:
                tx_x_positions = []
                tx_y_positions = []
                ty_x_positions = []
                ty_y_positions = []
                # For each telescope pair:
                for telescope_pair in self.telescope_pair_combinations:
                    # Calculate baseline for this telescope pair:
                    tx_index = np.where(np.array(self.observatory['telescope_names']) == telescope_pair[0])[0][0]
                    ty_index = np.where(np.array(self.observatory['telescope_names']) == telescope_pair[1])[0][0]
                    tx_x_positions.append(self.observatory['telescope_locations'][tx_index][0])
                    tx_y_positions.append(self.observatory['telescope_locations'][tx_index][1])
                    ty_x_positions.append(self.observatory['telescope_locations'][ty_index][0])
                    ty_y_positions.append(self.observatory['telescope_locations'][ty_index][1])

                # TODO: Remove the 1e-2 once we correct the units from the facet positions
                tx_x_positions = np.array(tx_x_positions) * 1e-2
                tx_y_positions = np.array(tx_y_positions) * 1e-2
                ty_x_positions = np.array(ty_x_positions) * 1e-2
                ty_y_positions = np.array(ty_y_positions) * 1e-2

                baselines = np.sqrt((tx_x_positions - ty_x_positions) ** 2 + (tx_y_positions - ty_y_positions) ** 2)
                uvw_baselines = np.array([np.abs(tx_x_positions - ty_x_positions),
                                          np.abs(tx_y_positions - ty_y_positions),
                                          np.zeros_like(tx_x_positions)]).T

                # Let's identify the unique baselines and their multiplicity:
                unique_baselines, counts = np.unique(baselines, return_counts=True)
                unique_uv_baselines, uv_counts = np.unique(uvw_baselines[:, 0:2], axis=0, return_counts=True)

                self.fixed_baselines = unique_baselines
                self.multiplicity = counts
                self.uv_fixed_baselines = unique_uv_baselines
                self.uv_multiplicity = uv_counts
            baselines = self.fixed_baselines
            # Now store one result per unique baseline:
            for i, baseline in enumerate(self.fixed_baselines):
                if ignore_larger_baselines:
                    if baseline > ignore_larger_baselines:
                        continue
                result['telescope_pair'].append("Average B = {:3.0f}".format(baseline))
                result['baseline'].append(baseline)
                result['baseline_min'].append(baseline - (np.sqrt(self.observatory['constants']['area'][0]) / 2.))
                result['baseline_max'].append(baseline + (np.sqrt(self.observatory['constants']['area'][0]) / 2.))
                # In this simplified 1D approach, it makes no sense to include UVW coordinates:
                result['U'].append(np.nan)
                result['V'].append(np.nan)
                result['W'].append(np.nan)
                baselines = np.array(baselines)
                squared_visibilities = protected_squared_visibility_uniform_disk(baseline,
                                                                                 self.stellar_diameter.to('mas').value,
                                                                                 self.zbc_norm,
                                                                                 wavelength=self.wavelength)
                telescope_pair = None
                s_to_n = signal_to_noise(self.observatory, telescope_pair, self.target_b_mag, np.average(zds),
                                         np.average(squared_visibilities), duration * duration_scaling,
                                         moon_to_star_ratio, identical_telescopes=True)
                # As we now have many measurements of this same baseline, we scale S/N.
                s_to_n *= np.sqrt(self.multiplicity[i])
                # Also take into account telescope multiplicity:
                s_to_n *= np.sqrt(self.tel_multiplicity)
                # We "simulate" the measurement, by adding noise to the expected squared visibility:
                g2_error = np.average(squared_visibilities) / s_to_n
                sim_g2 = np.random.normal(loc=np.average(squared_visibilities), scale=g2_error)

                result['sim_g2'].append(sim_g2)
                result['sim_g2_error'].append(g2_error)
        else:
            # For each telescope pair:
            for telescope_pair in self.telescope_pair_combinations:
                # Calculate baseline for this telescope pair:
                tx_index = np.where(np.array(self.observatory['telescope_names']) == telescope_pair[0])[0][0]
                ty_index = np.where(np.array(self.observatory['telescope_names']) == telescope_pair[1])[0][0]
                tx_xyz_position = self.observatory['telescope_locations'][tx_index]
                ty_xyz_position = self.observatory['telescope_locations'][ty_index]
                # from IPython import embed
                # embed()
                baselines = []
                uvw_baselines = []
                for time in frame_times:
                    t, dis, uvw_baseline = get_light_travel_delay(self.target_skycoord, time, self.site,
                                                                  tx_xyz_position, ty_xyz_position, get_distances=True)
                    baselines.append(dis.to('m').value)
                    uvw_baselines.append(uvw_baseline)
                baselines = np.array(baselines)
                uvw_baselines = np.array(uvw_baselines)

                if ignore_larger_baselines:
                    if np.all(baselines > ignore_larger_baselines):
                        continue
                    else:
                        # Let's select only those values of baseline below the baseline limit:
                        mask = baselines < ignore_larger_baselines
                        baselines = baselines[mask]
                        uvw_baselines = uvw_baselines[mask]

                # TODO: We can calculate this once for many baseline values, and later on interpolate. This would speed
                # things up
                if self.semi_major_axis_orientation is not None and self.compression_factor is not None:
                    semi_major_axis = self.stellar_diameter.to('mas').value / 2.
                    squared_visibilities = protected_squared_visibility_fast_rotator(uvw_baselines[:, 0],
                                                                                 uvw_baselines[:, 1],
                                                                                 semi_major_axis,
                                                                                 self.semi_major_axis_orientation,
                                                                                 self.compression_factor,
                                                                                 self.zbc_norm,
                                                                                 wavelength=self.wavelength)
                elif self.sky_image is not None:
                    # Interpolate V² using simulated UV coverage and sky image:
                    interpolator = self.sky_image.sq_visibility_interpolator
                    squared_visibilities = interpolator((uvw_baselines[:, 0], uvw_baselines[:, 1]))
                else:
                    squared_visibilities = protected_squared_visibility_uniform_disk(baselines,
                                                                                     self.stellar_diameter.to('mas').value,
                                                                                     self.zbc_norm,
                                                                                     wavelength=self.wavelength)
                # Calculate signal to noise of this simulated run:
                s_to_n = signal_to_noise(self.observatory, telescope_pair, self.target_b_mag, np.average(zds),
                                         np.average(squared_visibilities), duration * duration_scaling,
                                         moon_to_star_ratio)
                # With this signal to noise, we now know the uncertainty of the g2 measurement:
                g2_error = np.average(squared_visibilities) / s_to_n

                # The simulated g2 measurement will be a simulated observation of the expected g2 with a
                # sigma of g2_error:
                sim_g2 = np.random.normal(loc=np.average(squared_visibilities), scale=g2_error)
                result['telescope_pair'].append(telescope_pair)
                result['baseline'].append(np.average(baselines))
                result['baseline_min'].append(np.min(baselines))
                result['baseline_max'].append(np.max(baselines))

                result['U'].append(np.average(uvw_baselines[:, 0]))
                result['V'].append(np.average(uvw_baselines[:, 1]))
                result['W'].append(np.average(uvw_baselines[:, 2]))
                result['sim_g2'].append(sim_g2)
                result['sim_g2_error'].append(g2_error)
        return result

    @u.quantity_input(duration='time')
    def simulate_run_fast(self, datetime, duration, moon_to_star_ratio=0.1, ignore_zd=False,
                          duration_scaling=1.0, ignore_larger_baselines=False):

        # In this fast simulation, we just assume the duration of the run is "instantaneous" regarding S/N. So we
        # calculate S/N, baselines, etc. just once.

        if not isinstance(datetime, Time):
            logging.error("Input datetime is not an instance of astropy.time")
            raise TypeError

        # Times array at the right location:
        loc_time = Time(datetime, format='isot', scale='utc',
                         location=c.EarthLocation(lat=self.observatory['location']['lat'],
                                                  lon=self.observatory['location']['lon']))
        star_alt_az = self.target_skycoord.transform_to(c.AltAz(obstime=loc_time, location=self.site))
        zds = (90. * u.deg) - star_alt_az.alt

        if star_alt_az.alt < 5. * u.deg:
            logging.error("The altitude of the source at the time {} is just too low.".format(loc_time))
            # raise ValueError("The source altitude is not valid.")

        result = dict()
        # Create empty lists for the following entries:
        keys = ['squared_visibilities', 'telescope_pair', 'baseline', 'baseline_min', 'baseline_max', 'sim_g2',
                'sim_g2_error', 'U', 'V', 'W']
        for key in keys:
            result[key] = []

        # If the array assumes fixed baselines (e.g. submirrors within a slewing telescope), we can calculate
        # the baselines, and bundle all identical ones:
        if self.is_fixed_baselines:
            if self.fixed_baselines is None:
                # UV dependent try:
                tx_x_positions = []
                tx_y_positions = []
                ty_x_positions = []
                ty_y_positions = []
                # For each telescope pair:
                for telescope_pair in self.telescope_pair_combinations:
                    # Calculate baseline for this telescope pair:
                    tx_index = np.where(np.array(self.observatory['telescope_names']) == telescope_pair[0])[0][0]
                    ty_index = np.where(np.array(self.observatory['telescope_names']) == telescope_pair[1])[0][0]
                    tx_x_positions.append(self.observatory['telescope_locations'][tx_index][0])
                    tx_y_positions.append(self.observatory['telescope_locations'][tx_index][1])
                    ty_x_positions.append(self.observatory['telescope_locations'][ty_index][0])
                    ty_y_positions.append(self.observatory['telescope_locations'][ty_index][1])

                # TODO: Remove the 1e-2 once we correct the units from the facet positions
                tx_x_positions = np.array(tx_x_positions) * 1e-2
                tx_y_positions = np.array(tx_y_positions) * 1e-2
                ty_x_positions = np.array(ty_x_positions) * 1e-2
                ty_y_positions = np.array(ty_y_positions) * 1e-2

                baselines = np.sqrt((tx_x_positions - ty_x_positions) ** 2 + (tx_y_positions - ty_y_positions) ** 2)
                uvw_baselines = np.array([np.abs(tx_x_positions - ty_x_positions),
                                          np.abs(tx_y_positions - ty_y_positions),
                                          np.zeros_like(tx_x_positions)]).T

                # baselines = []
                # uvw_baselines = []
                # for telescope_pair in self.telescope_pair_combinations:
                #     tx_index = np.where(np.array(self.observatory['telescope_names']) == telescope_pair[0])[0][0]
                #     ty_index = np.where(np.array(self.observatory['telescope_names']) == telescope_pair[1])[0][0]
                #     tx_xyz_position = self.observatory['telescope_locations'][tx_index]
                #     ty_xyz_position = self.observatory['telescope_locations'][ty_index]
                #
                #     # Let's try a UVW approach:
                #     # t, dis, uvw_baseline = get_light_travel_delay(self.target_skycoord, frame_times[0], self.site,
                #     #                                               tx_xyz_position, ty_xyz_position,
                #     #                                               get_distances=True)
                #     # baselines.append(dis.to('m').value)
                #     # uvw_baselines.append(uvw_baseline)
                #     # baselines = np.array(baselines)*1e-2
                #     # uvw_baselines = np.array(uvw_baselines)*1e-2
                #
                #     b = np.sqrt((tx_xyz_position[0]-ty_xyz_position[0])**2 + (tx_xyz_position[1]-ty_xyz_position[1])**2)
                #     # TODO: Right now sub-mirrors are added in cm. This should be fixed in the future.
                #     baselines.append(b*1e-2)

                # Let's identify the unique baselines and their multiplicity:
                unique_baselines, counts = np.unique(baselines, return_counts=True)
                unique_uv_baselines, uv_counts = np.unique(uvw_baselines[:, 0:2], axis=0, return_counts=True)

                self.fixed_baselines = unique_baselines
                self.multiplicity = counts
                self.uv_fixed_baselines = unique_uv_baselines
                self.uv_multiplicity = uv_counts
            baselines = self.fixed_baselines
            # Now store one result per unique baseline:
            for i, baseline in enumerate(self.fixed_baselines):
                if ignore_larger_baselines:
                    if baseline > ignore_larger_baselines:
                        continue
                result['telescope_pair'].append("Average B = {:3.0f}".format(baseline))
                result['baseline'].append(baseline)
                result['baseline_min'].append(baseline - (np.sqrt(self.observatory['constants']['area'][0]) / 2.))
                result['baseline_max'].append(baseline + (np.sqrt(self.observatory['constants']['area'][0]) / 2.))
                # In this simplified 1D approach, it makes no sense to include UVW coordinates:
                result['U'].append(np.nan)
                result['V'].append(np.nan)
                result['W'].append(np.nan)
                baselines = np.array(baselines)
                squared_visibilities = protected_squared_visibility_uniform_disk(baseline,
                                                                                 self.stellar_diameter.to('mas').value,
                                                                                 self.zbc_norm,
                                                                                 wavelength=self.wavelength)
                telescope_pair = None
                s_to_n = signal_to_noise(self.observatory, telescope_pair, self.target_b_mag, np.average(zds),
                                         np.average(squared_visibilities), duration * duration_scaling,
                                         moon_to_star_ratio, identical_telescopes=True)
                # As we now have many measurements of this same baseline, we scale S/N.
                s_to_n *= np.sqrt(self.multiplicity[i])
                # Also take into account telescope multiplicity:
                s_to_n *= np.sqrt(self.tel_multiplicity)
                # We "simulate" the measurement, by adding noise to the expected squared visibility:
                g2_error = np.average(squared_visibilities) / s_to_n
                sim_g2 = np.random.normal(loc=np.average(squared_visibilities), scale=g2_error)

                result['sim_g2'].append(sim_g2)
                result['sim_g2_error'].append(g2_error)
        else:

            uvw_baseline = get_array_baselines(self.target_skycoord, loc_time, self.site,
                                               self.tx_xyz_positions, self.ty_xyz_positions)
            projected_intertelescope_distance = np.sqrt(np.power(uvw_baseline[:, 0], 2.) +
                                                        np.power(uvw_baseline[:, 1], 2.)) * u.m

            baselines = np.array(projected_intertelescope_distance.to('m').value)
            uvw_baselines = np.array(uvw_baseline)

            if ignore_larger_baselines:
                # Let's select only those values of baseline below the baseline limit:
                mask = baselines < ignore_larger_baselines
                baselines = baselines[mask]
                uvw_baselines = uvw_baselines[mask]
                telescope_pair_combinations = np.array(self.telescope_pair_combinations)[mask]
            else:
                telescope_pair_combinations = self.telescope_pair_combinations

            for i_pair, telescope_pair in enumerate(telescope_pair_combinations):
                # TODO: We can calculate this once for many baseline values, and later on interpolate. This would speed
                # things up
                if self.semi_major_axis_orientation is not None and self.compression_factor is not None:
                    semi_major_axis = self.stellar_diameter.to('mas').value / 2.
                    squared_visibilities = protected_squared_visibility_fast_rotator(uvw_baselines[i_pair, 0],
                                                                                     uvw_baselines[i_pair, 1],
                                                                                     semi_major_axis,
                                                                                     self.semi_major_axis_orientation,
                                                                                     self.compression_factor,
                                                                                     self.zbc_norm,
                                                                                     wavelength=self.wavelength)
                elif self.sky_image is not None:
                    # Interpolate in UV, using a regular grid:
                    interpolator = self.sky_image.sq_visibility_interpolator
                    squared_visibilities = interpolator((uvw_baselines[i_pair, 0], uvw_baselines[i_pair, 1]))





                    
                else:
                    squared_visibilities = protected_squared_visibility_uniform_disk(baselines[i_pair],
                                                                                     self.stellar_diameter.to('mas').value,
                                                                                     self.zbc_norm,
                                                                                     wavelength=self.wavelength)
                # Calculate signal to noise of this simulated run:
                s_to_n = signal_to_noise(self.observatory, telescope_pair, self.target_b_mag, zds,
                                         squared_visibilities, duration * duration_scaling,
                                         moon_to_star_ratio)
                # With this signal to noise, we now know the uncertainty of the g2 measurement:
                # print("np.average(squared_visibilities) = {}".format(np.average(squared_visibilities)))
                # print("s_to_n = {}".format(s_to_n))
                # print("g2_error = {}".format(np.average(squared_visibilities)/s_to_n))
                # print("zds = {}".format(np.average(zds)))

                # print("Telescope pair: {}".format(telescope_pair))
                # print("U: {}".format(np.average(uvw_baselines[:, 0])))
                # print("V: {}".format(np.average(uvw_baselines[:, 1])))
                g2_error = np.average(squared_visibilities) / s_to_n

                # The simulated g2 measurement will be a simulated observation of the expected g2 with a
                # sigma of g2_error:
                sim_g2 = np.random.normal(loc=np.average(squared_visibilities), scale=g2_error)
                # Let's also store the squared visibility:
                result['squared_visibilities'].append(np.average(squared_visibilities))

                result['telescope_pair'].append(telescope_pair)
                result['baseline'].append(np.average(baselines[i_pair]))
                result['baseline_min'].append(np.min(baselines[i_pair]))
                result['baseline_max'].append(np.max(baselines[i_pair]))

                result['U'].append(np.average(uvw_baselines[i_pair, 0]))
                result['V'].append(np.average(uvw_baselines[i_pair, 1]))
                result['W'].append(np.average(uvw_baselines[i_pair, 2]))
                result['sim_g2'].append(sim_g2)
                result['sim_g2_error'].append(g2_error)
        return result

    @u.quantity_input(duration='time')
    def simulate_zbc_run(self, datetime, duration, moon_to_star_ratio=0.15, ignore_zd=False):
        if not isinstance(datetime, Time):
            logging.error("Input datetime is not an instance of astropy.time")
            raise TypeError
        # The frame times are calculated from the datetime, and the duration:
        if ignore_zd:
            frame_times = np.arange(datetime, datetime+1*u.min, 1*u.min)
        else:
            frame_times = np.arange(datetime, datetime+duration, 1*u.min)
        star_alt_az = self.target_skycoord.transform_to(c.AltAz(obstime=frame_times, location=self.site))
        zds = (90. * u.deg) - star_alt_az.alt
        for i, alt in enumerate(star_alt_az.alt):
            if alt < 5. * u.deg:
                logging.error("The altitude of the source at the time {} is just too low.".format(frame_times[i]))
                # raise ValueError("The source altitude is not valid.")

        result = dict()
        # Create empty lists for the following entries:
        keys = ['telescope_pair', 'baseline', 'baseline_min', 'baseline_max', 'sim_g2', 'sim_g2_error',
                'U', 'V', 'W']
        for key in keys:
            result[key] = []
        # For each telescope in the array:
        for telescope in self.observatory['telescope_names']:
            # TODO: We can calculate this once for many baseline values, and later on interpolate. This would speed
            # TODO: things up
            squared_visibilities = protected_squared_visibility_uniform_disk(np.array([0.01]),
                                                                             self.stellar_diameter.to('mas').value,
                                                                             self.zbc_norm,
                                                                             wavelength=self.wavelength)
            # Calculate signal to noise of this simulated run:
            s_to_n = signal_to_noise(self.observatory, [telescope, telescope], self.target_b_mag, np.average(zds),
                                     np.average(squared_visibilities), duration, moon_to_star_ratio, is_checkboard=True)
            # With this signal to noise, we now know the uncertainty of the g2 measurement:
            g2_error = np.average(squared_visibilities)/s_to_n

            # The simulated g2 measurement will be a simulated observation of the expected g2 with a sigma of g2_error:
            sim_g2 = np.random.normal(loc=np.average(squared_visibilities), scale=g2_error)
            result['telescope_pair'].append([telescope, telescope])
            result['baseline'].append(0.01)
            result['baseline_min'].append(0.01)
            result['baseline_max'].append(0.01)
            result['U'].append(0.01)
            result['V'].append(0.01)
            result['W'].append(0.01)
            result['sim_g2'].append(sim_g2)
            result['sim_g2_error'].append(g2_error)
        return result

    @u.quantity_input(run_duration='time')
    def simulate_observations(self, start_times, run_duration, moon_to_star_ratio=0.1, duration_scaling=1.0,
                              ignore_zd=False, ignore_larger_baselines=False, fast_simulation=False):
        self.validate_times(start_times)
        # Now that entries are valid, simulate all runs:
        for start_time in start_times:
            # TODO: If we had a way to estimate the sky brightness due to the moon, this would be the place
            #  to incorporate it...
            if fast_simulation:
                result = self.simulate_run_fast(start_time, run_duration, moon_to_star_ratio=moon_to_star_ratio,
                                                ignore_zd=ignore_zd, duration_scaling=duration_scaling,
                                                ignore_larger_baselines=ignore_larger_baselines)
                self.simulated_runs.append(result)
            else:
                result = self.simulate_run(start_time, run_duration, moon_to_star_ratio=moon_to_star_ratio,
                                           ignore_zd=ignore_zd, duration_scaling=duration_scaling,
                                           ignore_larger_baselines=ignore_larger_baselines)
                self.simulated_runs.append(result)

    @u.quantity_input(run_duration='time')
    def simulate_zbc_observations(self, start_times, run_duration, moon_to_star_ratio=0.15, ignore_zd=False):
        self.validate_times(start_times)
        # Now that entries are valid, simulate all runs:
        for start_time in start_times:
            # TODO: If we had a way to estimate the sky brightness due to the moon, this would be the place
            #  to incorporate it...
            result = self.simulate_zbc_run(start_time, run_duration, moon_to_star_ratio=moon_to_star_ratio,
                                           ignore_zd=ignore_zd)
            self.simulated_runs.append(result)

    def add_zbc_measurement(self, norm, norm_uncertainty, telescopes):
        result = dict()
        # Create empty lists for the following entries:
        keys = ['squared_visibilities', 'telescope_pair', 'baseline', 'baseline_min', 'baseline_max', 'sim_g2', 'sim_g2_error',
                'U', 'V', 'W']
        for key in keys:
            result[key] = []

        result['squared_visibilities'].append(1.0)
        result['telescope_pair'].append(telescopes)
        result['baseline'].append(0.01)
        result['baseline_min'].append(0.01)
        result['baseline_max'].append(0.01)
        result['U'].append(0.01)
        result['V'].append(0.01)
        result['W'].append(0.01)
        result['sim_g2'].append(norm)
        result['sim_g2_error'].append(norm_uncertainty)
        self.simulated_runs.append(result)

    @u.quantity_input(run_duration='time')
    def add_zbc_measurements(self, start_times, run_duration, telescopes):
        self.validate_times(start_times)
        self.validate_telescopes(telescopes)
        # Now that entries are valid, simulate all runs:
        for start_time in start_times:
            # TODO: If we had a way to estimate the sky brightness due to the moon, this would be the place
            #  to incorporate it...
            for telescope in telescopes:
                result = self.simulate_zbc_run(start_time, run_duration, telescope)
                self.simulated_runs.append(result)

    @staticmethod
    def validate_times(times):
        # First, make sure that all entries within start_times are instances of the astropy Time class:
        for time in times:
            if not isinstance(time, Time):
                logging.error("The members within the start_times input are not instances of the astropy Time class.")
                raise ValueError("Wrong times provided.")

    def validate_telescopes(self, telescopes):
        # Make sure all telescopes are within the telescope names within the observatory:
        for telescope in telescopes:
            if telescope not in self.observatory['telescope_names']:
                logging.error("The following telescope name is not within the observatory telescope name list.")
                logging.error("- Telescope name: {}".format(telescope))
                logging.error("- Observatory telescope names: {}".format(self.observatory['telescope_names']))
                raise ValueError("Wrong telescope name provided.")

    def plot_v2(self, set_log_yscale=True, color=None, axis=None, plot_legend=True, plot_label=False,
                plot_true_visibility=True, plot_fit_visibility=True, plot_title=False):

        import re

        if axis is not None:
            ax1 = axis
        else:
            self.f = plt.figure(figsize=(15, 9))
            ax1 = self.f.add_subplot(111)
        # Extract the values that have been simulated:
        telescope_pairs = []
        baselines = []
        amplitudes = []
        amplitude_uncertainties = []
        us = []
        vs = []
        colors = []
        for sim_run in self.simulated_runs:
            # Loop through the number of correlations within each simulated run:
            for i, baseline in enumerate(sim_run['baseline']):
                telescopes = sim_run['telescope_pair'][i]
                # Remove the number from each telescope name:
                telescope_pairs.append("{}-{}".format(re.sub(r'[0-9]+', '', telescopes[0]),
                                                      re.sub(r'[0-9]+', '', telescopes[1])))
                baselines.append(sim_run['baseline'][i])
                amplitudes.append(sim_run['sim_g2'][i])
                amplitude_uncertainties.append(sim_run['sim_g2_error'][i])
                us.append(sim_run['U'][i])
                vs.append(sim_run['V'][i])
        telescope_pairs = np.array(telescope_pairs)
        baselines = np.array(baselines)
        amplitudes = np.array(amplitudes)
        amplitude_uncertainties = np.array(amplitude_uncertainties)
        star_azimuth_angle = get_azimuth_angle(np.array(us), np.array(vs)) * u.deg

        if self.compression_factor is not None and self.semi_major_axis_orientation is not None:
            cm = plt.cm.get_cmap('RdYlBu')
            ax1.errorbar(baselines, amplitudes, yerr=amplitude_uncertainties, fmt='o')
            sc = plt.scatter(baselines, amplitudes,
                             c=star_azimuth_angle.to('deg').value,
                             vmin=np.min(star_azimuth_angle.to('deg').value),
                             vmax=np.max(star_azimuth_angle.to('deg').value),
                             s=50, cmap=cm, zorder=32)
            cbar = plt.colorbar(sc)
            cbar.set_label('Star azimuth angle', rotation=270, labelpad=20)
        else:
            for i, telescope_pair in enumerate(np.unique(telescope_pairs)):
                if color is not None:
                    marker_color = color
                else:
                    # Using different colors per specific telescope pair:
                    marker_color = "C{}".format(i)
                # Plot only the points from a given telescope pair:
                mask = telescope_pairs == telescope_pair

                measurement_label = plot_label if plot_label else telescope_pair
                ax1.errorbar(baselines[mask], amplitudes[mask], yerr=amplitude_uncertainties[mask], fmt='o',
                             color=marker_color, label=measurement_label)

        all_baselines = np.arange(0, 500, 0.1)
        true_visibilities = protected_squared_visibility_uniform_disk(all_baselines,
                                                                      self.stellar_diameter.to('mas').value,
                                                                      self.zbc_norm,
                                                                      wavelength=self.wavelength)
        if plot_true_visibility:
            true_color = color if color is not None else 'b'
            ax1.plot(all_baselines, true_visibilities, color=true_color, linestyle='--',
                     label=r"$\theta_{Ref}$" + " = {:1.3f} mas".format(self.stellar_diameter.to('mas').value))
            if self.compression_factor is not None and self.semi_major_axis_orientation is not None:
                small_diameter = self.stellar_diameter.to('mas').value * self.compression_factor
                true_visibilities = protected_squared_visibility_uniform_disk(np.arange(0, 500, 0.1),
                                                                              small_diameter,
                                                                              self.zbc_norm,
                                                                              wavelength=self.wavelength)
                ax1.plot(all_baselines, true_visibilities, color=true_color, linestyle='--',
                         label=r"$\theta_{Ref}$" + " = {:1.3f} mas".format(small_diameter))

        if self.uniform_disc_fit is not None:
            fit_results = self.uniform_disc_fit
            rng = np.random.RandomState()
            # rng = np.random.RandomState(seed=30)
            parameter_samples = rng.multivariate_normal(fit_results['means'],
                                                        fit_results['cov_matrix'], 500)
            vs = [protected_squared_visibility_uniform_disk(all_baselines,
                                                            pars[0],
                                                            pars[1],
                                                            wavelength=self.wavelength) for pars in parameter_samples]
            realization = np.array(vs)
            q = 100 * scipy.stats.norm.cdf(-1)  # 1 is the 1 sigma
            y_low = np.percentile(realization, q, axis=0)
            q = 100 * scipy.stats.norm.cdf(1)  # 1 is the 1 sigma
            y_high = np.percentile(realization, q, axis=0)
            ax1.plot(all_baselines, y_low, color="C7", alpha=0.2)
            ax1.plot(all_baselines, y_high, color="C7", alpha=0.2)
            ax1.fill_between(all_baselines, y_low, y_high, color="C7", alpha=0.1)
            visibility_fit = protected_squared_visibility_uniform_disk(all_baselines, fit_results['theta'],
                                                                       fit_results['norm'], wavelength=self.wavelength)
            fit_color = color if color is not None else 'C7'
            if plot_fit_visibility:
                ax1.plot(all_baselines, visibility_fit, color=fit_color,
                         label='Best fit:\n ' + r'$\theta$ = {:1.3f} $\pm$ {:1.3f}'.format(fit_results['theta'],
                                                                                           fit_results['theta_e']) +
                               '\n norm = {:1.3f} $\pm$ {:1.3f}'.format(fit_results['norm'],
                                                                        fit_results['norm_e']))

        ax1.yaxis.set_label_text('g($\tau$)')
        ax1.xaxis.set_label_text('Baseline [m]')
        if set_log_yscale:
            ax1.set_yscale('log')
        if np.max(baselines) < 20:
            ax1.set_xlim([-1, 20])
        elif np.max(baselines) < 200:
            ax1.set_xlim([-2, 200])
        elif np.max(baselines) < 500:
            ax1.set_xlim([-2, 500])
        elif np.max(baselines) < 750:
            ax1.set_xlim([-2, 750])
        elif np.max(baselines) < 1000:
            ax1.set_xlim([-2, 1000])
        ax1.set_ylim([1.e-3, self.zbc_norm*1.3])
        figure_title = plot_title if plot_title else "{} observations on {}".format(self.observatory_name,
                                                                                    self.source_name)
        ax1.set_title(figure_title)
        ax1.axhline(y=0.0, color='grey', linestyle='-')
        if plot_legend:
            # ax1.legend(loc=2, bbox_to_anchor=(1.02, 0.98), prop={'size': 12})
            ax1.legend()
        if axis is None:
            plt.show()

    def plot_uv(self, fixed_uv=False):

        fuv = plt.figure(figsize=(12, 9))
        ax1 = fuv.add_subplot(111)
        # If fast rotator parameters were provided, we can plot the UV coverage for this case:
        if self.compression_factor is not None and self.semi_major_axis_orientation is not None:
            max_uv_distance = 300

            u_array, v_array = np.mgrid[-max_uv_distance:max_uv_distance, -max_uv_distance:max_uv_distance]
            v2 = protected_squared_visibility_fast_rotator(u_array, v_array, self.stellar_diameter.to('mas').value/2.,
                                                           self.semi_major_axis_orientation, self.compression_factor,
                                                           self.zbc_norm, wavelength=self.wavelength)

            ax1.imshow(v2, vmin=0, vmax=self.zbc_norm, extent=[-max_uv_distance,max_uv_distance,
                                                               -max_uv_distance,max_uv_distance])

        if fixed_uv:
            cm = plt.cm.get_cmap('RdYlBu')
            sc = plt.scatter(self.uv_fixed_baselines[:, 0], self.uv_fixed_baselines[:, 1], c=self.uv_multiplicity,
                             vmin=0, vmax=np.max(self.uv_multiplicity), s=25, cmap=cm)
            plt.scatter(self.uv_fixed_baselines[:, 0], -self.uv_fixed_baselines[:, 1], c=self.uv_multiplicity,
                        vmin=0, vmax=np.max(self.uv_multiplicity), s=25, cmap=cm)
            plt.scatter(-self.uv_fixed_baselines[:, 0], self.uv_fixed_baselines[:, 1], c=self.uv_multiplicity,
                        vmin=0, vmax=np.max(self.uv_multiplicity), s=25, cmap=cm)
            plt.scatter(-self.uv_fixed_baselines[:, 0], -self.uv_fixed_baselines[:, 1], c=self.uv_multiplicity,
                        vmin=0, vmax=np.max(self.uv_multiplicity), s=25, cmap=cm)
            plt.colorbar(sc)
            ax1.xaxis.set_label_text('U [m]')
            ax1.yaxis.set_label_text('V [m]')
            ax1.set_title("UV coverage and multiplicity by {}.".format(self.observatory_name))
            # Find max value of U or V:
            max_uv_distance = np.max([self.uv_fixed_baselines[:, 0], self.uv_fixed_baselines[:, 1]])
        else:
            for k, run in enumerate(self.simulated_runs):
                for i, pair in enumerate(run['telescope_pair']):
                    if k == 0:
                        ax1.scatter(run['U'][i], run['V'][i], c="C{}".format(i),
                                    s=150,
                                    #             s=area, c=colors, alpha=0.5,
                                    label="{}-{}".format(pair[0], pair[1]))
                    else:
                        ax1.scatter(run['U'][i], run['V'][i], c="C{}".format(i),
                                    s=150,)
                    ax1.scatter(-np.array(run['U'][i]), -np.array(run['V'][i]), c="C{}".format(i),
                                s=150)
            ax1.set_title("UV coverage by {} observing {}".format(self.observatory_name, self.source_name))
            # Find max value of U or V:
            max_uv_distance = np.max([np.max([np.max(np.abs(run['U'])),
                                              np.max(np.abs(run['V']))]) for run in self.simulated_runs])
        # A bit larger for "breathing":
        max_uv_distance *= 1.1
        ax1.set_xlim([-max_uv_distance, max_uv_distance])
        ax1.set_ylim([-max_uv_distance, max_uv_distance])

        ax1.xaxis.set_label_text('U [m]')
        ax1.yaxis.set_label_text('V [m]')
        # ax1.legend(loc=2, bbox_to_anchor=(1.02, 0.98), prop={'size': 12})
        ax1.legend(loc=2, bbox_to_anchor=(1.02, 0.98))
        plt.show()

    def fit_uniform_disc(self):
        # Fit uniform disc profile to simulated data.
        # Extract the values that have been simulated:
        telescope_pair = []
        baselines = []
        amplitudes = []
        amplitude_uncertainties = []
        for sim_run in self.simulated_runs:
            # Loop through the number of correlations within each simulated run:
            for i, baseline in enumerate(sim_run['baseline']):
                telescopes = sim_run['telescope_pair'][i]
                telescope_pair.append("{}-{}".format(telescopes[0], telescopes[1]))
                baselines.append(sim_run['baseline'][i])
                amplitudes.append(sim_run['sim_g2'][i])
                amplitude_uncertainties.append(sim_run['sim_g2_error'][i])
        baselines = np.array(baselines)
        amplitudes = np.array(amplitudes)
        amplitude_uncertainties = np.array(amplitude_uncertainties)

        # We define a new function in order to fix the wavelength:
        def sq_vis(b, theta, norm):
            return protected_squared_visibility_uniform_disk(b, theta, norm, wavelength=self.wavelength)

        # Fit parameter bounds:
        param_bounds = ((0.01, 0.), (50., 20.))
        means, cov_matrix = curve_fit(sq_vis,
                                      baselines,
                                      amplitudes,
                                      # np.array([av_g[0] for av_g in self.av_g[i]]),
                                      p0=[0.5, 2.],
                                      sigma=amplitude_uncertainties,
                                      bounds=param_bounds)
        method_fit_result = dict()
        method_fit_result['means'] = means
        method_fit_result['cov_matrix'] = cov_matrix
        method_fit_result['theta'] = means[0]
        method_fit_result['theta_e'] = cov_matrix[0, 0] ** 0.5
        method_fit_result['norm'] = means[1]
        method_fit_result['norm_e'] = cov_matrix[1, 1] ** 0.5
        self.uniform_disc_fit = method_fit_result

    def fit_fast_rotator(self):
        # Fit uniform disc profile to simulated data.
        # Extract the values that have been simulated:
        telescope_pair = []
        u_m = []
        v_m = []
        amplitudes = []
        amplitude_uncertainties = []
        for sim_run in self.simulated_runs:
            # Loop through the number of correlations within each simulated run:
            for i, baseline in enumerate(sim_run['baseline']):
                telescopes = sim_run['telescope_pair'][i]
                telescope_pair.append("{}-{}".format(telescopes[0], telescopes[1]))
                u_m.append(sim_run['U'][i])
                v_m.append(sim_run['V'][i])
                amplitudes.append(sim_run['sim_g2'][i])
                amplitude_uncertainties.append(sim_run['sim_g2_error'][i])
        u_m = np.array(u_m)
        v_m = np.array(v_m)
        amplitudes = np.array(amplitudes)
        amplitude_uncertainties = np.array(amplitude_uncertainties)

        def fast_rotator_residual(params, u_m, v_m, y_data, y_errors):
            theta = params['theta']
            norm = params['norm']
            semi_major_axis_orientation = params['semi_major_axis_orientation']
            compression_factor = params['compression_factor']

            # In this method, "baselines" contains a long array of baselines for each point in Y.
            # So the model values we want to compare with our measurements is the average V² value
            # resulting from evaluating the model with all these baselines
            model = protected_squared_visibility_fast_rotator(u_m,
                                                              v_m,
                                                              theta/2.,
                                                              semi_major_axis_orientation,
                                                              compression_factor,
                                                              norm,
                                                              wavelength=self.wavelength)

            model = np.array(model)
            try:
                return (y_data - model) / y_errors
            except ValueError:
                from IPython import embed
                embed()

        from lmfit import Parameters, Minimizer, conf_interval, conf_interval2d, printfuncs
        from lmfit.minimizer import MinimizerException

        params = Parameters()
        params.add('theta', value=0.5, min=0.01, max=5)
        params.add('norm', value=0.0, min=0, max=5)
        params.add('semi_major_axis_orientation', value=0.5*np.pi, min=0, max=2*np.pi)
        params.add('compression_factor', value=0.0, min=0, max=1)
        #
        mini = Minimizer(fast_rotator_residual, params,
                         fcn_args=(u_m, v_m, amplitudes, amplitude_uncertainties))

        out = mini.minimize(method='least_squares')

        method_fit_result = dict()
        try:
            method_fit_result['cov_matrix'] = out.covar
        except:
            logging.error("Could not retrieve covariance matrix. Setting it to None")
            method_fit_result['cov_matrix'] = None
        method_fit_result['means'] = [out.params[p].value for p in ['theta', 'norm', 'semi_major_axis_orientation',
                                                                    'compression_factor']]
        method_fit_result['theta'] = out.params['theta'].value
        method_fit_result['theta_e'] = out.params['theta'].stderr
        method_fit_result['norm'] = out.params['norm'].value
        method_fit_result['norm_e'] = out.params['norm'].stderr
        method_fit_result['semi_major_axis_orientation'] = out.params['semi_major_axis_orientation'].value
        method_fit_result['semi_major_axis_orientation_e'] = out.params['semi_major_axis_orientation'].stderr
        method_fit_result['compression_factor'] = out.params['compression_factor'].value
        method_fit_result['compression_factor_e'] = out.params['compression_factor'].stderr
        self.fast_rotator_fit = method_fit_result

    def reset(self):
        self.simulated_runs = []
        self.uniform_disc_fit = None

    def randomize_simulation(self):
        for i, sim_run in enumerate(self.simulated_runs):
            randomized_values = np.random.normal(loc=sim_run['squared_visibilities'], scale=sim_run['sim_g2_error'])
            self.simulated_runs[i]['sim_g2'] = randomized_values.copy()

    def print(self):
        """
        Print the content of the object, mainly fit results.
        """
        if self.uniform_disc_fit is not None:
            print("Uniform disc fit:")
            print("theta = {:1.3f} +/- {:1.3f}".format(self.uniform_disc_fit['theta'],
                                                        self.uniform_disc_fit['theta_e']))
            print("norm = {:1.3f} +/- {:1.3f}".format(self.uniform_disc_fit['norm'],
                                                        self.uniform_disc_fit['norm_e']))
        if self.fast_rotator_fit is not None:
            print("Fast rotator fit:")
            print(r"$\theta$_Diam = {:1.3f} +/- {:1.3f}".format(self.fast_rotator_fit['theta'],
                                                                self.fast_rotator_fit['theta_e']))
            print("norm = {:1.3f} +/- {:1.3f}".format(self.fast_rotator_fit['norm'],
                                                      self.fast_rotator_fit['norm_e']))
            angles = [self.fast_rotator_fit['semi_major_axis_orientation'] * u.rad,
                      self.fast_rotator_fit['semi_major_axis_orientation_e'] * u.rad]

            print(r"$\Theta$ = {:1.3f} +/- {:1.3f}".format(angles[0].to('deg'),
                                                           angles[1].to('deg')))
            print("compression_factor = {:1.3f} +/- {:1.3f}".format(self.fast_rotator_fit['compression_factor'],
                                                                    self.fast_rotator_fit['compression_factor_e']))


class CombineSimulatedRuns(object):
    """
    A class to simulate SII runs.

    """

    def __init__(self, observations):
        """
        Constructor of the CombineSimulatedRuns class.

        Parameters
        ----------
        observations: list
            List of SimulateSiiRuns instances, to be used for the combined analysis.

        """
        self.observations = observations
        # Container for the uniform disc fit:
        self.uniform_disc_fit = None

        # Container of the figure:
        self.f = None

        # Assume all measurements come from the same target, same wavelength:
        # self.observatory_name = observatory_name
        # self.source_name = source_name
        # self.wavelength = wavelength
        self.wavelength = self.observations[0].wavelength
        # We extract a list of attributes from each run, and store everything into a dictionary
        data = dict()
        for attribute in ["source_name", "target_b_mag", "zbc_norm", "stellar_diameter"]:
            data[attribute] = [getattr(r, attribute) for r in self.observations]
            if not isinstance(data[attribute][0], u.Quantity):
                data[attribute] = np.array(data[attribute])
            # Check that all runs were simulated using the same parameters:
            if not np.all(data[attribute] == data[attribute][0]):
                raise ValueError("Not all runs have the same {}: {}".format(attribute,
                                                                            np.unique(data[attribute])))
        # Now that we know the list of observations is valid, extract relevant parameters:
        self.stellar_diameter = data['stellar_diameter'][0]
        self.zbc_norm = data['zbc_norm'][0]
        self.target_b_mag = data['target_b_mag'][0]
        self.source_name = data['source_name'][0]

    def fit_uniform_disc(self):
        # Fit uniform disc profile to simulated data.
        # Extract the values that have been simulated:
        observatory = []
        telescope_pair = []
        baselines = []
        amplitudes = []
        amplitude_uncertainties = []
        for observation in self.observations:
            for sim_run in observation.simulated_runs:
                # Loop through the number of correlations within each simulated run:
                for i, baseline in enumerate(sim_run['baseline']):
                    observatory.append(observation.observatory_name)
                    telescopes = sim_run['telescope_pair'][i]
                    telescope_pair.append("{}-{}".format(telescopes[0], telescopes[1]))
                    baselines.append(sim_run['baseline'][i])
                    amplitudes.append(sim_run['sim_g2'][i])
                    amplitude_uncertainties.append(sim_run['sim_g2_error'][i])
        baselines = np.array(baselines)
        amplitudes = np.array(amplitudes)
        amplitude_uncertainties = np.array(amplitude_uncertainties)

        def sq_vis(b, theta, norm):
            return magic_squared_visibility_uniform_disk(b, theta, norm, wavelength=self.wavelength)

        # Fit parameter bounds:
        param_bounds = ((0.01, 0.), (100., 20.))
        means, cov_matrix = curve_fit(sq_vis,
                                      baselines,
                                      amplitudes,
                                      # np.array([av_g[0] for av_g in self.av_g[i]]),
                                      p0=[0.1, 2.],
                                      sigma=amplitude_uncertainties,
                                      bounds=param_bounds)
        method_fit_result = dict()
        method_fit_result['means'] = means
        method_fit_result['cov_matrix'] = cov_matrix
        method_fit_result['theta'] = means[0]
        method_fit_result['theta_e'] = cov_matrix[0, 0] ** 0.5
        method_fit_result['norm'] = means[1]
        method_fit_result['norm_e'] = cov_matrix[1, 1] ** 0.5
        self.uniform_disc_fit = method_fit_result

    def plot_v2(self, set_log_yscale=True, color=None, axis=None, plot_legend=True, plot_label=False,
                plot_true_visibility=True, plot_fit_visibility=True, plot_title=False):
        font = {'weight': 'normal',
                'size': 18}
        import re
        matplotlib.rc('font', **font)

        if axis is not None:
            ax1 = axis
        else:
            self.f = plt.figure(figsize=(15, 9))
            ax1 = self.f.add_subplot(111)
        # Extract the values that have been simulated:
        telescope_pairs = []
        baselines = []
        amplitudes = []
        amplitude_uncertainties = []
        colors = []
        for observation in self.observations:
            for sim_run in observation.simulated_runs:
                # Loop through the number of correlations within each simulated run:
                for i, baseline in enumerate(sim_run['baseline']):
                    telescopes = sim_run['telescope_pair'][i]
                    # Remove the number from each telescope name:
                    telescope_pairs.append("{}-{}".format(re.sub(r'[0-9]+', '', telescopes[0]),
                                                          re.sub(r'[0-9]+', '', telescopes[1])))
                    baselines.append(sim_run['baseline'][i])
                    amplitudes.append(sim_run['sim_g2'][i])
                    amplitude_uncertainties.append(sim_run['sim_g2_error'][i])
        telescope_pairs = np.array(telescope_pairs)
        baselines = np.array(baselines)
        amplitudes = np.array(amplitudes)
        amplitude_uncertainties = np.array(amplitude_uncertainties)
        for i, telescope_pair in enumerate(np.unique(telescope_pairs)):
            if color is not None:
                marker_color = color
            else:
                # Using different colors per specific telescope pair:
                marker_color = "C{}".format(i)
            # Plot only the points from a given telescope pair:
            mask = telescope_pairs == telescope_pair

            measurement_label = plot_label if plot_label else telescope_pair
            ax1.errorbar(baselines[mask], amplitudes[mask], yerr=amplitude_uncertainties[mask], fmt='o',
                         color=marker_color, label=measurement_label)

        all_baselines = np.arange(0, 500, 0.1)
        true_visibilities = protected_squared_visibility_uniform_disk(all_baselines,
                                                                  self.stellar_diameter.to('mas').value,
                                                                  self.zbc_norm,
                                                                  wavelength=self.wavelength)
        if plot_true_visibility:
            true_color = color if color is not None else 'b'
            ax1.plot(all_baselines, true_visibilities, color=true_color, linestyle='--',
                     label=r"$\theta_{Ref}$" + " = {:1.3f} mas".format(self.stellar_diameter.to('mas').value))

        if self.uniform_disc_fit is not None:
            fit_results = self.uniform_disc_fit
            rng = np.random.RandomState()
            # rng = np.random.RandomState(seed=30)
            parameter_samples = rng.multivariate_normal(fit_results['means'],
                                                        fit_results['cov_matrix'], 500)
            vs = [protected_squared_visibility_uniform_disk(all_baselines,
                                                        pars[0],
                                                        pars[1],
                                                        wavelength=self.wavelength) for pars in parameter_samples]
            realization = np.array(vs)
            q = 100 * scipy.stats.norm.cdf(-1)  # 1 is the 1 sigma
            y_low = np.percentile(realization, q, axis=0)
            q = 100 * scipy.stats.norm.cdf(1)  # 1 is the 1 sigma
            y_high = np.percentile(realization, q, axis=0)
            ax1.plot(all_baselines, y_low, color="C7", alpha=0.2)
            ax1.plot(all_baselines, y_high, color="C7", alpha=0.2)
            ax1.fill_between(all_baselines, y_low, y_high, color="C7", alpha=0.1)
            visibility_fit = protected_squared_visibility_uniform_disk(all_baselines, fit_results['theta'],
                                                                   fit_results['norm'], wavelength=self.wavelength)
            fit_color = color if color is not None else 'C7'
            if plot_fit_visibility:
                ax1.plot(all_baselines, visibility_fit, color=fit_color,
                         label='Best fit:\n ' + r'$\theta$ = {:1.3f} $\pm$ {:1.3f}'.format(fit_results['theta'],
                                                                                           fit_results['theta_e']) +
                               '\n norm = {:1.3f} $\pm$ {:1.3f}'.format(fit_results['norm'],
                                                                        fit_results['norm_e']))

        ax1.yaxis.set_label_text('g($\tau$)')
        ax1.xaxis.set_label_text('Baseline [m]')
        if set_log_yscale:
            ax1.set_yscale('log')
        if np.max(baselines) < 20:
            ax1.set_xlim([-1, 20])
        elif np.max(baselines) < 100:
            ax1.set_xlim([-2, 100])
        elif np.max(baselines) < 200:
            ax1.set_xlim([-2, 200])
        elif np.max(baselines) < 500:
            ax1.set_xlim([-2, 500])
        elif np.max(baselines) < 750:
            ax1.set_xlim([-2, 750])
        elif np.max(baselines) < 1000:
            ax1.set_xlim([-2, 1000])
        ax1.set_ylim([1.e-3, self.zbc_norm*1.3])

        ax1.axhline(y=0.0, color='grey', linestyle='-')
        if plot_legend:
            ax1.legend(loc=2, bbox_to_anchor=(1.02, 0.98), prop={'size': 12})
        if axis is None:
            plt.show()

    def reset(self):
        self.observations = []
        self.uniform_disc_fit = None
