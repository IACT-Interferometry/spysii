import numpy as np
from astropy.io import ascii
import os
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable, axes_size
from astropy import units as u
from scipy.optimize import curve_fit
from scipy.ndimage import rotate
import logging

from spysii import PACKAGE_DATA
from spysii.utils import power_law, format_time


class SkyImage(object):
    """
    Class contaning all the information needed to define a mono-chromatic sky image
    """

    def __init__(self, side_in_pixels, pixel_to_mas):
        # Size of the image:
        self.side_in_pixels = side_in_pixels
        # Proyected angular size of a pixel side, in mas:
        self.pixel_to_mas = pixel_to_mas
        self.image = np.zeros((side_in_pixels, side_in_pixels))
        # Calculate coordinates in X and Y in the sky reference frame:
        x = np.arange(-self.side_in_pixels/2, self.side_in_pixels/2, 1)
        sky_x, sky_y = np.meshgrid(x * pixel_to_mas, x * pixel_to_mas)
        self.sky_x = sky_x
        self.sky_y = sky_y
        # Parameters from visibility computation:
        self.wavelength = None
        self.fft_magnitude = None
        self.norm_visibility = None
        self.u_meters = None
        self.v_meters = None
        self.sq_visibility_interpolator = None

    def plot_image(self):
        """
        Plot the sky image.

        :return:
        """
        plt.figure(figsize=(14, 6))
        plt.subplot(1, 2, 1)
        plt.imshow(
            self.image,
            extent=(-(np.shape(self.image)[0] / 2) * self.pixel_to_mas,
                    (np.shape(self.image)[0] / 2) * self.pixel_to_mas,
                    -(np.shape(self.image)[1] / 2) * self.pixel_to_mas,
                    (np.shape(self.image)[1] / 2) * self.pixel_to_mas),
            #     origin='lower',
            #    cmap=grey_cm,
            #    norm=norm,
            #    aspect='equal'
        )
        # plt.colorbar(label='Intensity')
        # plt.xlim(-1.5, 1.5)
        # plt.ylim(-1.5, 1.5)
        plt.title('Intensity Map')
        plt.xlabel('X [mas]')
        plt.ylabel('Y [mas]')
        plt.gca().invert_xaxis()

        plt.subplot(1, 2, 2)
        plt.imshow(
            self.image,
            extent=(-(np.shape(self.image)[0] / 2) * self.pixel_to_mas,
                    (np.shape(self.image)[0] / 2) * self.pixel_to_mas,
                    -(np.shape(self.image)[1] / 2) * self.pixel_to_mas,
                    (np.shape(self.image)[1] / 2) * self.pixel_to_mas),
            #     origin='lower',
            #    cmap=grey_cm,
            #    norm=norm,
            #    aspect='equal'
        )
        # plt.colorbar(label='Intensity')
        plt.xlim(-1.5, 1.5)
        plt.ylim(-1.5, 1.5)
        plt.title('Zoomed-in 2D Intensity Map')
        plt.xlabel('X [mas]')
        plt.ylabel('Y [mas]')
        plt.gca().invert_xaxis()

       
    def compare_with(self, sky_image, squared_visibility=True):
        """
        Compare one image model with another SkyImage.

        :return
        A figure with four subplots arranged in a 2×2 grid:

        1) The model image in mas coordinates (self).
        2) The comparison model image in mas coordinates (sky_image).
        3) The difference in squared visibility (or V) computed via FFT between the model and comparison image.
        4) The difference in V² (or V) using amplitude interpolated values from both images.
        
        """
        
        # Ensure that comparison is only performed between SkyImage instances and that they are different
        if not isinstance(self, SkyImage):
            raise TypeError("The image must be an instance of SkyImage")
        if not isinstance(sky_image, SkyImage):
            raise TypeError("The image to be compared with must be an instance of SkyImage")
        if self is sky_image:
            raise ValueError("Cannot compare the same SkyImage instance with itself")

        # Check if visibility was calculated. If not, try to calculate it.
        if self.fft_magnitude is None:
            logging.warning("Visibility from fast rotator not calculated. Calculating it now")
            self.calculate_visibility()
        if sky_image.fft_magnitude is None:
            logging.warning("Visibility from input image not calculated. Calculating it now")
            sky_image.calculate_visibility()

        #In order to compare both images we should request that they have the same dimensions
        # and proyected angular size of a pixel side
        if self.side_in_pixels != sky_image.side_in_pixels:
            raise ValueError("Both images should have the same dimensions (side_in_pixels)")
        if self.pixel_to_mas != sky_image.pixel_to_mas:
            raise ValueError("Both images should have the same proyected angular size of a pixel side (pixel_to_mas)") 

        
        plt.figure(figsize=(14, 12), dpi = 150)
        # Model image
        plt.subplot(2,2,1)
        plt.imshow(
            self.image,
            extent=(-(np.shape(self.image)[0] / 2) * self.pixel_to_mas,
                    (np.shape(self.image)[0] / 2) * self.pixel_to_mas,
                    -(np.shape(self.image)[1] / 2) * self.pixel_to_mas,
                    (np.shape(self.image)[1] / 2) * self.pixel_to_mas))
        
        
        plt.xlim([-(np.shape(self.image)[0] / 2) * self.pixel_to_mas, 
                  (np.shape(self.image)[0] /2) * self.pixel_to_mas])
        plt.ylim([-(np.shape(self.image)[1] / 2) * self.pixel_to_mas, 
                  (np.shape(self.image)[1] /2) * self.pixel_to_mas])
        
        if self.model_type == "uniform_disc":
            label = 'UD'
        elif self.model_type == 'fast_rotator':
            label = 'FR'
        else:
            label = 'LD'


        plt.title('2D Intensity Map for the ' f'{label}' ' model')
        plt.xlabel('X [mas]')
        plt.ylabel('Y [mas]')
        plt.gca().invert_xaxis()

        #Comparison model image
        plt.subplot(2,2,2)
        plt.imshow(
            sky_image.image,
            extent=(-(np.shape(sky_image.image)[0] / 2) * sky_image.pixel_to_mas,
                    (np.shape(sky_image.image)[0] / 2) * sky_image.pixel_to_mas,
                    -(np.shape(sky_image.image)[1] / 2) * sky_image.pixel_to_mas,
                    (np.shape(sky_image.image)[1] / 2) * sky_image.pixel_to_mas))

        
        plt.xlim([-(np.shape(sky_image.image)[0] / 2) * sky_image.pixel_to_mas, 
                  (np.shape(sky_image.image)[0] /2) * sky_image.pixel_to_mas])
        plt.ylim([-(np.shape(sky_image.image)[1] / 2) * sky_image.pixel_to_mas, 
                  (np.shape(sky_image.image)[1] /2) * sky_image.pixel_to_mas])

        if sky_image.model_type == "uniform_disc":
            label = 'UD'
        elif sky_image.model_type == 'fast_rotator':
            label = 'FR'
        else:
            label = 'LD'


        plt.title('2D Intensity Map for the ' f'{label}' ' model')
        plt.xlabel('X [mas]')
        plt.ylabel('Y [mas]')
        plt.gca().invert_xaxis()


        plt.subplot(2,2,3)
        # Difference in squared visibility (or V) via FFT
        if squared_visibility:
            diff = self.norm_visibility**2 - sky_image.norm_visibility**2
            amp_label = 'V²' 
            mask = diff > 0.001
        else:
            diff = self.norm_visibility - sky_image.norm_visibility
            amp_label = 'V'
            mask = diff > 0.009
            
        ax = plt.gca()
        fig = ax.get_figure()
        pc = plt.imshow(diff, 
           extent=(self.u_meters.min(), self.u_meters.max(),
                   self.v_meters.min(), self.v_meters.max()),
           origin='lower', cmap='plasma')
        
        
        u_valid = self.u_meters[mask]
        v_valid = self.v_meters[mask]
        lim = np.maximum(u_valid.max(), v_valid.max())
        plt.xlim(-lim, lim)
        plt.ylim(-lim, lim)
        plt.xlabel('u (Meters)')
        plt.ylabel('v (Meters)')
        plt.title(f'{amp_label}' ' difference')
        
        
        CS = plt.contour(self.u_meters, self.v_meters, 
                        diff, 
                        [-0.1, -0.05, 0.02])
        ax.clabel(CS)

        
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size=axes_size.Fixed(.18), pad=axes_size.Fixed(.05))
        cbar = fig.colorbar(pc, cax=cax)
        cbar.set_label('Difference in 'f'{amp_label}')


        plt.subplot(2,2,4)
        # Difference in V² (or V) using amplitude interpolated values
        x = np.linspace(-lim, lim, self.side_in_pixels*2)
        y = np.linspace(-lim, lim, self.side_in_pixels*2)
        X, Y = np.meshgrid(x,y)
        
        points = np.column_stack([X.ravel(), Y.ravel()])

        label_abs = None
        if squared_visibility:
          diff_interp = (self.sq_visibility_interpolator(points).reshape(X.shape) -
                         sky_image.sq_visibility_interpolator(points).reshape(X.shape))
        else:
          diff_interp = np.sqrt(
              abs( self.sq_visibility_interpolator(points).reshape(X.shape) -
                  sky_image.sq_visibility_interpolator(points).reshape(X.shape)))
          label_abs = 'Absolute'

        ax = plt.gca()
        fig = ax.get_figure()
        pc = plt.imshow(diff_interp, 
           extent=(-lim, lim,
                   -lim, lim),
           origin='lower', cmap='plasma')

        
        plt.xlabel('u (Meters)')
        plt.ylabel('v (Meters)')
        plt.title(f'{amp_label}' ' difference using amplitude interpolated values')
        CS = plt.contour(X, Y, 
                        diff_interp, 
                        [-0.1, -0.05, 0.02])
        ax.clabel(CS)

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size=axes_size.Fixed(.18), pad=axes_size.Fixed(.05))
        cbar = fig.colorbar(pc, cax=cax)
    
        if label_abs:
            cbar.set_label(f'{label_abs} difference in {amp_label}')
        else:
            cbar.set_label(f'Difference in {amp_label}')

        plt.subplots_adjust(wspace=0.35, hspace=0.2)

        
    def plot_visibility(self, sq_visibility=True, sq_visibility_interpolated=False):
        """
        Plot the sky image in the fourier space.

        return:
        """

        # Check if visibility was calculated. If not, try to calculate it.
        if self.fft_magnitude is None:
            logging.warning("Visibility not calculated. Calculating it now.")
            self.calculate_visibility()

        # Plot the visibility
        plt.figure(figsize=(14, 6))
        plt.subplot(1, 2, 1)
        visibilities = self.norm_visibility**2 if sq_visibility else self.norm_visibility
        pc = plt.imshow(visibilities,
                   extent=(self.u_meters.min(), self.u_meters.max(),
                           self.v_meters.min(), self.v_meters.max()),
                   origin='lower', cmap='plasma')

        ax = plt.gca()
        fig = ax.get_figure()

        mask = visibilities > 0.009

        u_valid = self.u_meters[mask]
        v_valid = self.v_meters[mask]
        lim = np.maximum(u_valid.max(), v_valid.max())
        plt.xlim(-lim, lim)
        plt.ylim(-lim, lim)
		
        plt.xlabel('u (Meters)')
        plt.ylabel('v (Meters)')
        plt.title('FFT in Meters')

        CS = ax.contour(self.u_meters,
                        self.v_meters,
                        self.norm_visibility, 5)

        ax.clabel(CS, fontsize=10)

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size=axes_size.Fixed(.18), pad=axes_size.Fixed(.05))
        cbar = fig.colorbar(pc, cax=cax)
        cbar.set_label('Amplitude')


        plt.tight_layout()
        

        plt.subplot(1, 2, 2)
        pc = plt.imshow(visibilities,
                   extent=(self.u_meters.min(), self.u_meters.max(),
                           self.v_meters.min(), self.v_meters.max()),
                   origin='lower', cmap='plasma')

        plt.xlim(-200, 200)
        plt.ylim(-200, 200)
        plt.xlabel('u (Meters)')
        plt.ylabel('v (Meters)')
        plt.title('FFT in Meters')

        ax = plt.gca()
        fig =ax.get_figure()
        CS = ax.contour(self.u_meters,
                        self.v_meters,
                        self.norm_visibility, 5)
        ax.clabel(CS, fontsize=10)

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size=axes_size.Fixed(.18), pad=axes_size.Fixed(.05))
        cbar = fig.colorbar(pc, cax=cax)
        cbar.set_label('Amplitude')

        plt.tight_layout()

        if sq_visibility_interpolated:
            # Plot the interpolated squared visibility
            x = np.linspace(-lim, lim, self.side_in_pixels*2)
            y = np.linspace(-lim, lim, self.side_in_pixels*2)
            X, Y = np.meshgrid(x,y)
            points = np.column_stack([X.ravel(), Y.ravel()])

            plt.figure(figsize=(6, 6))
            vis_interp = self.sq_visibility_interpolator(points).reshape(X.shape)
            pc = plt.imshow(vis_interp,
                       extent=(-lim, lim,
                               -lim, lim),
                       origin='lower', cmap='plasma')

            plt.xlim(-lim, lim)
            plt.ylim(-lim, lim)
            plt.xlabel('u (Meters)')
            plt.ylabel('v (Meters)')
            plt.title('Interpolated V²')

            ax = plt.gca()
            fig =ax.get_figure()
            CS = ax.contour(X,
                            Y,
                            vis_interp, 5)
            ax.clabel(CS, fontsize=10)

            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size=axes_size.Fixed(.18), pad=axes_size.Fixed(.05))
            cbar = fig.colorbar(pc, cax=cax)
            cbar.set_label('V²')

            ax.invert_xaxis()
            plt.tight_layout()


    def plot_v2_vs_baseline(self, angle=None):
        """
        Plot the squared visibility as a function of the baseline.

        :param angle: angle of the uv straight line to be drawn. If None, the average uv vs baseline will be calculated.

        :return:
        """

        raise NotImplementedError("This method is not implemented yet. Please implement it if needed.")

        # Check if visibility was calculated. If not, try to calculate it.
        if self.fft_magnitude is None:
            logging.warning("Visibility not calculated. Calculating it now.")
            self.calculate_visibility()

        from scipy.interpolate import RegularGridInterpolator
        u_m = np.linspace(self.u_meters.min(), self.u_meters.max(), np.shape(self.u_meters)[0])
        v_m = np.linspace(self.v_meters.min(), self.v_meters.max(), np.shape(self.v_meters)[0])
        interp_disc = RegularGridInterpolator((u_m, v_m), self.norm_visibility)

        # Define the baseline range we want to plot:
        baseline_range = np.arange(0, 500, 0.1)

        # Plot the visibility
        plt.figure(figsize=(14, 6))
        plt.subplot(1, 2, 1)
        plt.imshow(self.norm_visibility,
                   extent=(self.u_meters.min(), self.u_meters.max(),
                           self.v_meters.min(), self.v_meters.max()),
                   origin='lower', cmap='plasma')
        plt.colorbar(label='Amplitude')

        ax = plt.gca()
        CS = ax.contour(self.u_meters,
                        self.v_meters,
                        self.norm_visibility, 5)
        ax.clabel(CS, fontsize=10)

        plt.xlim(-500, 500)
        plt.ylim(-500, 500)
        plt.xlabel('u (Meters)')
        plt.ylabel('v (Meters)')
        plt.title('FFT in Meters')


        plt.subplot(1, 2, 2)
        plt.imshow(self.norm_visibility,
                   extent=(self.u_meters.min(), self.u_meters.max(),
                           self.v_meters.min(), self.v_meters.max()),
                   origin='lower', cmap='plasma')
        plt.colorbar(label='Amplitude')

        ax = plt.gca()
        CS = ax.contour(self.u_meters,
                        self.v_meters,
                        self.norm_visibility, 5)
        ax.clabel(CS, fontsize=10)

        plt.xlim(-200, 200)
        plt.ylim(-200, 200)
        plt.xlabel('u (Meters)')
        plt.ylabel('v (Meters)')
        plt.title('FFT in Meters')

    def compute_uv_average(self, num_bins=100):
        """
        Compute the average value of the FFT magnitude as a function of the radial baseline (1D).

        Parameters:
            num_bins (int): The number of bins in the baseline array.

        Returns:
            bin_centers (1D array): The centers of the radial bins.
            radial_avg (1D array): The average FFT magnitude for each radial bin.
        """
        # Compute the radial baseline (r) in meters
        radial_baseline = np.sqrt(self.u_meters ** 2 + self.v_meters ** 2)

        # Flatten the arrays for easier processing
        r_flat = radial_baseline.flatten()
        fft_flat = self.fft_magnitude.flatten()

        # Define the bins for the radial baseline
        r_min, r_max = r_flat.min(), r_flat.max()
        bins = np.linspace(r_min, r_max, num_bins + 1)
        bin_centers = 0.5 * (bins[:-1] + bins[1:])  # Midpoints of bins

        # Initialize array to hold the average values
        radial_avg = np.zeros(len(bin_centers))

        # Compute average FFT magnitude for each bin
        for i in range(len(bins) - 1):
            indices = np.where((r_flat >= bins[i]) & (r_flat < bins[i + 1]))
            if len(indices[0]) > 0:
                radial_avg[i] = np.mean(fft_flat[indices])
            else:
                radial_avg[i] = np.nan  # Handle empty bins

        return bin_centers, radial_avg

    def add_uniform_disc(self, radius, xy_offset=(0,0), intensity=1):
        """
        Add a uniform disc to the sky image.

        :param radius: angular radius of the disc in mas
        :param xy_offset: location of the center of the disc in mas, in the form (x, y)
        :param intensity: intensity of the disc, in arbitrary units (important when adding several sources)
        :return:
        """
        current_image = self.image.copy()
        # We invert x/y coordinates to match the image, that is stored "upside down":
        xy_offset = np.array(xy_offset) * -1
        mask = np.where(np.sqrt((self.sky_x - xy_offset[0]) ** 2 + (self.sky_y - xy_offset[1]) ** 2) < radius)
        current_image[mask] = intensity
        self.image = current_image
        self.model_type = "uniform_disc"

    def add_limb_darkened_disc(self, radius, u_l, xy_offset=(0,0), intensity=1):
        """
        Add a uniform disc to the sky image.

        :param radius: angular radius of the disc in mas
        :param u_l: linear limb darkening coefficient
        :param xy_offset: location of the center of the disc in mas, in the form (x, y)
        :param intensity: intensity of the disc, in arbitrary units (important when adding several sources)
        :return:
        """
        current_image = self.image.copy()
        # We invert x/y coordinates to match the image, that is stored "upside down":
        xy_offset = np.array(xy_offset) * -1
        mask = np.where(np.sqrt((self.sky_x - xy_offset[0]) ** 2 + (self.sky_y - xy_offset[1]) ** 2) < radius)
        r = np.sqrt((self.sky_x[mask] - xy_offset[0]) ** 2 + (self.sky_y[mask] - xy_offset[1]) ** 2)
        current_image[mask] = intensity*(1 - u_l*(1 - np.sqrt(1 - ((r/radius)**2))))
        current_image[current_image<0] = 0
        self.image = current_image
        self.model_type = "limb_darkened_disc"


    def calculate_visibility(self, wavelength=420 * 1e-9):
        # Pixel scale in radians
        pixel_radian_scale = (self.pixel_to_mas * u.mas).to('rad').value

        # Compute 2D FFT
        fft_image = np.fft.fftshift(np.fft.fft2(self.image))
        fft_magnitude = np.abs(fft_image)

        # FFT frequency grid
        nx, ny = self.image.shape
        kx = np.fft.fftshift(np.fft.fftfreq(nx, d=1))  # Normalized pixel frequencies
        ky = np.fft.fftshift(np.fft.fftfreq(ny, d=1))
        kx_grid, ky_grid = np.meshgrid(kx, ky)

        # Baseline in meters. The factor two comes from the Nyquist sampling of the grid
        u_meters = (kx_grid * wavelength) / pixel_radian_scale
        v_meters = (ky_grid * wavelength) / pixel_radian_scale

        self.wavelength = wavelength
        self.fft_magnitude = fft_magnitude
        self.norm_visibility = fft_magnitude/np.max(fft_magnitude)
        self.u_meters = u_meters
        self.v_meters = v_meters

        # Calculate regular grid interpolator:
        from scipy.interpolate import RegularGridInterpolator
        self.sq_visibility_interpolator = RegularGridInterpolator((u_meters[0, :],
                                                                   v_meters[:, 0]),
                                                                  (self.norm_visibility**2).T)

        # Baseline in Mega Lambda (Mλ)
        # u_megalambda = u_meters / wavelength / 1e6
        # v_megalambda = v_meters / wavelength / 1e6


    def extend_with_zeros(self, input_shape):
        """
        Extends the intensity grid with zeros to match the desired input shape,
        keeping the original grid centered in the new shape.

        Parameters:
            input_shape (tuple): The desired output shape (rows, cols).

        Returns:
            ndarray: The extended grid with zeros added to match input_shape.
        """
        # Get current shape of the intensity grid
        current_shape = self.image.shape
        rows_diff = input_shape[0] - current_shape[0]
        cols_diff = input_shape[1] - current_shape[1]

        # Check if the input_shape is valid
        if rows_diff < 0 or cols_diff < 0:
            raise ValueError("Input shape must be larger than or equal to the intensity grid shape.")

        # Calculate padding on all sides to center the intensity grid
        pad_top = rows_diff // 2
        pad_bottom = rows_diff - pad_top
        pad_left = cols_diff // 2
        pad_right = cols_diff - pad_left

        # Pad the array with zeros
        extended_grid = np.pad(
            self.image,
            pad_width=((pad_top, pad_bottom), (pad_left, pad_right)),
            mode='constant',
            constant_values=0
        )
        self.image = extended_grid
        self.side_in_pixels = np.shape(self.image)[0]
