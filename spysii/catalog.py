import numpy as np
import astropy.units as u
from astropy.io import ascii
from astropy.coordinates import SkyCoord
import os
import logging

from spysii import PACKAGE_DATA
from spysii.utils import format_time


class Catalog(object):
    """
    Load the Yale Bright Star Catalog, and allow queries to grab the Bmag

    """

    def __init__(self):
        bsc_file = os.path.join(PACKAGE_DATA, 'catalogs/bsc_brighter_than_5_bmag.dat')
        jmdc_file = os.path.join(PACKAGE_DATA, 'catalogs/jmdc_catalog.ecsv')
        jsdc_file = os.path.join(PACKAGE_DATA, 'catalogs/jsdc_catalog_above_6mag.ecsv')
        self.table_bsc = ascii.read(bsc_file)
        self.table_jmdc = ascii.read(jmdc_file)
        self.table_jsdc = ascii.read(jsdc_file)
        self.coords_bsc = SkyCoord(ra=self.table_bsc['RAJ2000'], dec=self.table_bsc['DEJ2000'], frame='icrs',
                                   unit=(u.hourangle, u.deg))
        self.coords_jmdc = SkyCoord(ra=self.table_jmdc['_RA'], dec=self.table_jmdc['_DE'],
                                    frame='icrs', unit=(u.deg, u.deg))
        self.coords_jsdc = SkyCoord(ra=self.table_jsdc['RAJ2000'], dec=self.table_jsdc['DEJ2000'], frame='icrs',
                                    unit=(u.hourangle, u.deg))

    def find_closest_bsc_star(self, skycoord):
        """
        Calculate separation of skycoord with respect to all stars in the catalog, and pick the closest:
        """
        separations = skycoord.separation(self.coords_bsc)
        argmin = np.argmin(separations)
        return self.table_bsc[argmin]

    def get_jsdc_diameter_from_coordinate(self, skycoord):
        """
        Calculate separation of skycoord with respect to all stars in the catalog, and pick the closest:
        """
        separations = skycoord.separation(self.coords_jsdc)
        argmin = np.argmin(separations)
        logging.info("Closest star found within the JSDC is:\n -Star id: {}\n -B mag: {}\n -Spectral type: {}\n".format(
            self.table_jsdc['Name'][argmin], self.table_jsdc['Bmag'][argmin], self.table_jsdc['SpType'][argmin]
        ))
        print("Closest star found within the JSDC is:\n -Star id: {}\n -B mag: {}\n -Spectral type: {}\n".format(
            self.table_jsdc['Name'][argmin], self.table_jsdc['Bmag'][argmin], self.table_jsdc['SpType'][argmin]))
        diameter = self.table_jsdc['UDDB'][argmin]
        print("- Expected stellar diameter: {}".format(diameter))
        return diameter

    def get_jmdc_diameter_from_coordinate(self, skycoord):
        """
        Calculate separation of skycoord with respect to all stars in the catalog, and pick the closest:
        """
        separations = skycoord.separation(self.coords_jmdc)
        argmin = np.argmin(separations)
        # logging.info("Closest star found within the JMDC is:\n -Star ids: {} or {}".format(
        #     self.table_jmdc['ID1'][argmin], self.table_jmdc['ID2'][argmin]
        # ))
        print("Closest star found within the JMDC is:\n -Star ids: {} or {}".format(
             self.table_jmdc['ID1'][argmin], self.table_jmdc['ID2'][argmin]))
        diameter = self.table_jmdc['UDdiam'][argmin]
        if diameter < 0.:
            diameter = self.get_jsdc_diameter_from_coordinate(skycoord)
        print("- Stellar diameter: {}".format(diameter))
        return diameter

    def get_bsc_bmag_from_coordinate(self, skycoord):
        """
        Calculate separation of skycoord with respect to all stars in the catalog, and pick the closest:
        """
        separations = skycoord.separation(self.coords_bsc)
        argmin = np.argmin(separations)
        return self.table_bsc['Bmag'][argmin]

catalog = Catalog()
