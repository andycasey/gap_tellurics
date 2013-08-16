# coding: utf-8

""" Basic functionality to treat spectra as mutable objects. """

from __future__ import division, print_function

__author__ = "Andy Casey <acasey@mso.anu.edu.au>"

# Standard library
import json
import logging
import os

from shutil import copyfile

# Third-party
import numpy as np
import pyfits

from scipy import interpolate, ndimage, polyfit, poly1d
from scipy.optimize import leastsq


__all__ = ['Spectrum', 'Spectrum1D']


class Spectrum(object):
    """A class to represent multi-dimensional spectral objects. """

    def __init__(self, disp, flux, uncertainty=None, headers={}):
        """Initializes a `Spectrum` object with the given (multi-dimensional)
        dispersion and flux arrays.

        Inputs
        ------
        disp : `np.ndarray`
            Dispersion of the spectra.

        flux : `np.ndarray`
            Flux values for each dispersion point.

        uncertainty : `np.ndarray`
            Uncertainty values for each dispersion point.

        headers : `dict`
            Headers.
        """

        self.disp = disp
        self.flux = flux
        self.uncertainty = uncertainty
        self.headers = headers
        self.num_orders = self.flux.shape[1] if len(self.flux.shape) > 1 else len(self.flux)
        return None


    def remove_invalid_orders(self, band=None):
        """Discards any invalid orders where no finite flux values greater than
        zero exist.

        Inputs
        ------
        band : `int`
            Dimensional index to search for invalid orders on.
        """

        if band > self.flux.shape[0]:
            raise ValueError

        remove_indices = []
        for i in xrange(self.flux.shape[1]):
            flux_values = self.flux[band, i, :]

            if not np.any(np.isfinite(flux_values) * flux_values > 0):
                remove_indices.append(i)

        if len(remove_indices) > 0:
            logging.warn("Invalid orders ({indices}) identified and discarded!".format(indices=remove_indices))

            cleaned_flux = np.delete(self.flux, remove_indices, axis=1)
            cleaned_disp = np.delete(self.disp, remove_indices, axis=0)

            return Spectrum1D(disp=cleaned_disp, flux=cleaned_flux, headers=self.headers)

        return self
        

    @classmethod
    def load_multispec(cls, filename):
        """Reads in a multispec FITS file into a `Spectrum` object.

        Inputs
        ----
        filename : `str`
            Multi-spec FITS filename to load.
        """

        if not os.path.exists(filename):
            raise IOError("filename '{filename}'' does not exist".format(filename=filename))

        with pyfits.open(filename) as image:
            headers = image[0].header
            flux = image[0].data

        headers_dict = {}
        for k, v in headers.iteritems():

            try:
                str(v)
                json.dumps(v)

            except TypeError:
                continue

            if headers_dict.has_key(k):
                headers_dict[k] += v

            else:
                headers_dict[k] = v


        # Determine number of orders
        num_pixels = flux.shape[-1]
        num_orders = 1 if len(flux.shape) == 1 else flux.shape[-2]

        # Try linear dispersion
        try:
            crval = headers['crval1']
            crpix = headers['crpix1']
            cd = headers['cd1_1']
            ctype = headers['ctype1']

            if ctype.strip() == 'LINEAR':
                dispersion = np.zeros((num_orders, num_pixels), dtype=np.float)

                dispersion_base = (np.arange(num_pixels) + 1 - crpix) * cd + crval
                for i in xrange(num_orders):
                    dispersion[i, :] = dispersion_base

                dcflag = headers['dc-flag']
                if dcflag == 1:
                    dispersion = 10.0 ** dispersion

                elif dcflag != 0:
                    raise ValueError("dispersion is not linear or logarithmic (DC-FLAG = {dcflag})".format(dcfloag=dcflag))

                if num_orders == 1:
                    dispersion.shape = (num_pixels, )
                    flux = np.squeeze(flux)

                return cls(dispersion, flux, headers=headers_dict)
        
        except KeyError:
            pass

        # Get multi-spec headers
        try:
            wat = headers['wat2_*']
            num_wat_headers = len(wat)

        except KeyError:
            raise ValueError("cannot decipher header: need either WAT2_* or CRVAL keywords")

        # Concatenate headers
        wat_str = ''
        for i in xrange(num_wat_headers):
            # Apparently this is a hack to fix a problem in
            # older PyFits versions (< 3.1) where trailing blanks are stripped

            value = wat[i]
            if hasattr(value, 'value'): value = value.value
            value = value + (" " * (68 - len(value)))
            wat_str += value

        # Find all the spec#="..." strings
        spec_str = [''] * num_orders
        for i in xrange(num_orders):
            name = 'spec%i' % (i + 1, )

            p0 = wat_str.find(name)
            p1 = wat_str.find('"', p0)
            p2 = wat_str.find('"', p1 + 1)
            if p0 < 0 or p2 < 0 or p2 < 0:
                raise ValueError("cannot find '{name}' in WAT2_* keyword".format(name=name))

            spec_str[i] = wat_str[p1 + 1:p2]

        # Get wavelength calibration information
        z_params = np.zeros(num_orders)
        w_params = np.zeros((num_orders, 9))
        w = np.zeros(9)
        for i in xrange(num_orders):
            w = np.asarray(spec_str[i].split(), dtype=np.float)
            w_params[i, :] = w
            if w[2] == -1:
                raise ValueError("spectrum aperture {index} has no wavelength calibration (type = {index_type})".format(index=i+1, index_type=w[2]))

            elif w[6] != 0:
                z_params[i] = w[6]

        dispersion = np.zeros((num_orders, num_pixels), dtype=np.float)
        disp_fields = [None] * num_orders
        for i in xrange(num_orders):
            if w_params[i, 2] in (0, 1):
                
                # Simple linear or logarithmic spacing
                dispersion[i, :] = np.arange(num_pixels) * w_params[i, 4] + w_params[i, 3]
                if w_params[i, 2] == 1:
                    dispersion[i, :] = 10. ** dispersion[i, :]

            else:
                dispersion[:, i], disp_fields[i] = compute_non_linear_disp(num_pixels, spec_str[i])

            # Apply z-correction for this order
            dispersion[i, :] *= (1 - z_params[i])

        if num_orders == 1:
            flux = np.squeeze(flux)
            dispersion.shape = (num_pixels, )

        # Check blue to red orders
        if np.min(dispersion[0]) > np.min(dispersion[-1]):
            
            dispersion = dispersion[::-1]
            if len(flux.shape) > 2:
                flux = flux[:, ::-1]

            else: flux = flux[::-1]

        return cls(disp=dispersion, flux=flux, headers=headers_dict)


class Spectrum1D(object):
    """A mutable class to represent a one dimensional spectrum."""
    
    def __init__(self, disp, flux, uncertainty=None, headers={}):
        """Initializes a `Spectrum1D` object with the given dispersion and flux
        arrays.
        
        Inputs
        ------
        disp : `np.array`
            Dispersion of the spectrum (i.e. the wavelength points).
            
        flux : `np.array`
            Flux points for each `disp` point.

        uncertainty : `np.array`
            Uncertainty in flux points for each dispersion point.
        """
        
        self.disp = disp
        self.flux = flux
        self.uncertainty = uncertainty
        self.headers = headers
        
        return None
    
    @classmethod
    def load(cls, filename, **kwargs):
        """Load a `Spectrum1D` object from a given filename.
        
        Inputs
        ----
        filename : `str`
            Filename to load. This can be a single dimensional FITS file or an
            ASCII format.

        Notes
        -----
        If you are loading from an non-standard ASCII file, you can pass
        kwargs to `np.loadtxt` through this function.
        """
        
        if not os.path.exists(filename):
            raise IOError("filename '{fileame}' does not exist".format(filename=filename))
        
        uncertainty = None

        if filename.endswith('.fits'):
            image = pyfits.open(filename, **kwargs)
            
            header = image[0].header
            
            # Check for a tabular data structure
            if len(image) > 1 and image[0].data is None:

                names = [name.lower() for name in image[1].data.names]
                dispersion_key = 'wave' if 'wave' in names else 'disp'
                
                disp, flux = image[1].data[dispersion_key], image[1].data['flux']

                if 'error' in names or 'uncertainty' in names:
                    uncertainty_key = 'error' if 'error' in names else 'uncertainty'

                    uncertainty = image[1].data[uncertainty_key]

            else:

                # According to http://iraf.net/irafdocs/specwcs.php ....
                #li = a.headers['LTM1_1'] * np.arange(a.headers['NAXIS1']) + a.headers['LTV1']
                #a.headers['CRVAL1'] + a.headers['CD1_1'] * (li - a.headers['CRPIX1'])

                if np.all([header.has_key(key) for key in ('CDELT1', 'NAXIS1', 'CRVAL1')]):
                    disp = header['CRVAL1'] + np.arange(header['NAXIS1']) * header['CDELT1']
            
                if header.has_key('LTV1'):
                    disp -= header['LTV1'] * header['CDELT1']

                #disp -= header['LTV1'] if header.has_key('LTV1') else 0
                flux = image[0].data
            

            # Add the headers in
            headers = {}
            for row in header.items():
                key, value = row
                
                # Check the value is valid
                try:
                    str(value)
                    json.dumps(value)

                except TypeError:
                    continue

                if len(key) == 0 or len(str(value)) == 0: continue
                
                if headers.has_key(key):
                    if not isinstance(headers[key], list):
                        headers[key] = [headers[key]]
                    
                    headers[key].append(value)

                else:
                    headers[key] = value

            for key, value in headers.iteritems():
                if isinstance(value, list):
                    headers[key] = "\n".join(map(str, value))

        else:
            headers = {}
            disp, flux = np.loadtxt(filename, unpack=True, **kwargs)
            
        return cls(disp, flux, uncertainty=uncertainty, headers=headers)


    def save(self, filename, clobber=True):
        """Saves the `Spectrum1D` object to the specified filename.
        
        Inputs
        ------
        filename : `str`
            The filename to save the `Spectrum1D` object to.
            
        clobber : `bool`, optional
            Whether to overwite the `filename` if it already exists.
        
        Raises
        ------
        IOError
            If the filename exists and we are not asked to clobber it.
            
        ValueError
            If the ``Spectrum1D`` object does not have a linear dispersion map.
        """
        
        if os.path.exists(filename) and not clobber:
            raise IOError("filename '{filename}' already exists and we have been asked not to clobber it".format(filename=filename))
        
        if not filename.endswith('fits'):
            # ASCII
            
            data = np.hstack([self.disp.reshape(len(self.disp), 1), self.flux.reshape(len(self.disp), 1)])
            
            assert len(data.shape) == 2
            assert data.shape[1] == 2
            
            np.savetxt(filename, data)
            
        else:
            # FITS
            crpix1, crval1 = 1, self.disp.min()
            
            cdelt1 = np.mean(np.diff(self.disp))
            
            test_disp = (crval1 + np.arange(len(self.disp), dtype=self.disp.dtype) * cdelt1).astype(self.disp.dtype)
            
            if np.max(self.disp - test_disp) > 10e-2 or self.uncertainty is not None:

                # Non-linear dispersion map, or we have uncertainty information too
                # Create a tabular FITS format.

                col_disp = pyfits.Column(name='disp', format='1D', array=self.disp)
                col_flux = pyfits.Column(name='flux', format='1D', array=self.flux)

                if self.uncertainty is not None:
                    col_uncertainty = pyfits.Column(name='uncertainty', format='1D', array=self.uncertainty)

                    table_hdu = pyfits.new_table([col_disp, col_flux, col_uncertainty])

                else:
                    table_hdu = pyfits.new_table([col_disp, col_flux])

                # Create Primary HDU
                hdu = pyfits.PrimaryHDU()

                # Update primary HDU with headers
                for key, value in self.headers.iteritems():
                    if len(key) > 8:
                        # To deal with ESO compatibility
                        hdu.header.update('HIERARCH %s' % (key, ), value)
                    
                    try:
                        hdu.header.update(key, value)

                    except ValueError:
                        logging.warn("Could not save header key/value combination: %s = %s" % (key, value, ))
                    
                # Create HDU list with our tables
                hdulist = pyfits.HDUList([hdu, table_hdu])

                hdulist.writeto(filename, clobber=clobber)

            else:
                # Linear dispersion map.
                # Create a PrimaryHDU file.

                # Ensure we have an array!
                hdu = pyfits.PrimaryHDU(np.array(self.flux))
                
                headers = self.headers.copy()
                headers.update({
                    'CRVAL1': crval1,
                    'CRPIX1': crpix1,
                    'CDELT1': cdelt1
                })
                
                for key, value in headers.iteritems():
                    if len(key) > 8:
                        # To deal with ESO compatibility
                        hdu.header.update('HIERARCH %s' % (key, ), value)
                    
                    else:
                        try:
                            hdu.header.update(key, value)

                        except ValueError:
                            logging.warn("could not save header key/value combination: {key} = {value}".format(key=key, value=value))
                
                hdu.writeto(filename, clobber=clobber)


    def gaussian_smooth(self, fwhm, **kwargs):
        """Convolves the `Spectrum1D` flux with a Gaussian kernel of `fwhm` (Angstroms).

        Inputs
        ------
        fwhm : `float`
            The full-width-half-maxium of the Gaussian kernel.
        """
        
        profile_sigma = fwhm / (2 * (2*np.log(2))**0.5)
        
        # The requested FWHM is in Angstroms, but the dispersion between each
        # pixel is likely less than an Angstrom, so we must calculate the true
        # smoothing value
        
        true_profile_sigma = profile_sigma / np.median(np.diff(self.disp))

        smoothed_flux = ndimage.gaussian_filter1d(self.flux, true_profile_sigma, **kwargs)
        
        return self.__class__(self.disp, smoothed_flux)
        
    
    def doppler_shift(self, v=None, z=None, interpolate=False):
        """Performs a Doppler shift on the given `Spectrum1D` object by the
        velocity required. Either radial velocities or redshifts can be provided.
        
        Inputs
        ------
        v : `float`
            The velocity (in km/s) to correct the `Spectrum1D` object by.
            
        z : `float`
            The redshift to correct the `Spectrum1D` object by.
            
        Raises
        ------
        ValueError
            When both a velocity `v` and a redshift `z` are provided for Doppler
            corrections.
        """
        
        if isinstance(v, float) and isinstance(z, float):
            raise ValueError('Both a velocity and redshift was supplied')
            
        c = 299792458e-3 # km/s

        if v is not None:
            new_disp = self.disp * np.sqrt((1+v/c)/(1-v/c))
            
        elif z is not None:
            new_disp = self.disp * (1 + z)
            
        else:
            raise TypeError('no velocity or redshift supplied')
        
        rest_spectrum = self.__class__(new_disp, self.flux)
          
        if interpolate:
            return rest_spectrum.interpolate(self.disp)
            
        else:
            return rest_spectrum

    
    def interpolate(self, new_disp, mode='linear', bounds_error=False,
                    fill_value=np.nan):
        """Interpolate the `Spectrum1D` onto a new dispersion map.
        
        Inputs
        ------
        new_disp : `np.array`
            An array of floating-point types containing the new dispersion points.
            
        mode : `str`
            Interpolation mode. See `scipy.interpolate.interp1d` for available
            options.
        
        bounds_error : `bool`
            See `scipy.interpolate.interp1d` for details.
        
        fill_value : `float`-type
            See `scipy.interpolate.interp1d`
        """
        
        f = interpolate.interp1d(self.disp, self.flux, kind=mode, copy=False,
                                 bounds_error=bounds_error, fill_value=fill_value)
        
        return self.__class__(new_disp, f(new_disp))
        

    def cross_correlate(self, template, wl_region, full_output=False):
        """Performs a cross-correlation against the template spectrum, returning
        a radial velocity and associated uncertainty.

        Inputs
        ------
        template : `Spectrum1D`
            A normalised template spectrum.

        wl_region : two length list containing floats [start, end]
            The starting and end wavelength to perform the cross-correlation on.

        full_output : `bool` (optional)
            Whether or not to return the full output of the cross-correlation. If set to True
            then the output is as follows:

            v_rad, v_err, fft, profile

            where fft is a `np.ndarray` of shape (2, *) containing the Fourier transform
            and profile is a length 3 list containing the central peak point, peak height, and
            standard deviation.
        """

        if not isinstance(template, Spectrum1D):
            raise TypeError("template spectrum must be a `specutils.Spectrum1D` object")

        if not isinstance(wl_region, (tuple, list, np.ndarray)) or len(wl_region) != 2:
            raise TypeError("wavelength region must be a two length list-type")

        try:
            wl_region = map(float, wl_region)

        except:
            raise TypeError("wavelength regions must be float-like")

        # The following line of code will be supported until the end of the universe.
        c = 299792458e-3 # km/s

        # Splice the observed spectrum
        idx = np.searchsorted(self.disp, wl_region)
        finite_values = np.isfinite(self.flux[idx[0]:idx[1]])

        observed_slice = Spectrum1D(disp=self.disp[idx[0]:idx[1]][finite_values], flux=self.flux[idx[0]:idx[1]][finite_values])


        # Ensure the template and observed spectra are on the same scale
        template_func = interpolate.interp1d(template.disp, template.flux, bounds_error=False, fill_value=0.0)
        template_slice = Spectrum1D(disp=observed_slice.disp, flux=template_func(observed_slice.disp))

        # Perform the cross-correlation
        padding = observed_slice.flux.size + template_slice.flux.size
        x_norm = (observed_slice.flux - observed_slice.flux[np.isfinite(observed_slice.flux)].mean(axis=None))
        y_norm = (template_slice.flux - template_slice.flux[np.isfinite(template_slice.flux)].mean(axis=None))

        Fx = np.fft.fft(x_norm, padding, )
        Fy = np.fft.fft(y_norm, padding, )
        iFxy = np.fft.ifft(Fx.conj() * Fy).real
        varxy = np.sqrt(np.inner(x_norm, x_norm) * np.inner(y_norm, y_norm))

        fft_result = iFxy/varxy

        # Put around symmetry
        num = len(fft_result) - 1 if len(fft_result) % 2 else len(fft_result)

        fft_y = np.zeros(num)

        fft_y[:num/2] = fft_result[num/2:num]
        fft_y[num/2:] = fft_result[:num/2]

        fft_x = np.arange(num) - num/2

        # Get initial guess of peak
        p0 = np.array([fft_x[np.argmax(fft_y)], np.max(fft_y), 10])

        gaussian_profile = lambda p, x: p[1] * np.exp(-(x - p[0])**2 / (2.0 * p[2]**2))
        errfunc = lambda p, x, y: y - gaussian_profile(p, x)

        try:
            p1, ier = leastsq(errfunc, p0.copy(), args=(fft_x, fft_y))

        except:
            raise

        # Uncertainty
        sigma = np.mean(2.0*(fft_y.real)**2)**0.5

        # Create functions for interpolating back onto the dispersion map
        points = (0, p1[0], sigma)
        interp_x = np.arange(num/2) - num/4

        functions = []
        for point in points:
            idx = np.searchsorted(interp_x, point)
            f = interpolate.interp1d(interp_x[idx-3:idx+3], observed_slice.disp[idx-3:idx+3], bounds_error=True, kind='cubic')

            functions.append(f)

        # 0, p1, sigma
        f, g, h = [func(point) for func, point in zip(functions, points)]

        # Calculate velocity 
        measured_vrad = c * (1 - g/f)

        # Uncertainty
        measured_verr = np.abs(c * (1 - h/f))

        if full_output:
            results = [measured_vrad, measured_verr, np.vstack([fft_x, fft_y])]
            results.extend(p1)

            return results

        return [measured_vrad, measured_verr]


    def remove_telluric_absorption(self, telluric_spectrum, shift=True, scale=True,
        smooth=True, **kwargs):
        """Corrects the observed spectrum for telluric absorption, allowing
        for flux scaling, radial velocity shifts, and instrumental broadening.

        Inputs
        ------
        telluric_spectrum : `Spectrum1D`
            The telluric spectrum to be used for corrections.

        shift : `bool`, optional
            Allow for a radial velocity shift in the telluric spectrum.

        scale : `bool`, optional
            Allow for a scaling factor in the telluric spectrum flux.

        smooth : `bool`, optional
            Allow for additional smoothing in the telluric spectrum flux.
        """

        if not isinstance(telluric_spectrum, Spectrum1D):
            raise TypeError("telluric spectrum must be a Spectrum1D object")

        # If we aren't to use shifting, scaling or smoothing, then we should just
        # be dividing the object spectrum by the telluric flux.
        if not shift and not scale and not smooth:

            # Ensure the telluric spectrum is on the same dispersion scale as the
            # observation
            interpolated_telluric = telluric_spectrum.interpolate(self.disp)
            self.flux /= interpolated_telluric.flux

            return self

        # Depending on the arguments provided, we will generate a function
        # to minimize the difference between the telluric spectrum and the
        # observed spectrum.

        # NOTE: We may want to restrict this comparison to a given
        # wavelength band (e.g. infrared band A)

        def minimize_chi_sq(parameters, telluric_spectrum):
            """Minimizes the \chi^2 difference between the observed and telluric
            spectrum.

            Inputs
            ------
            parameters : list-type of length N where N is the boolean sum of 
                         `shift`, `scale`, `smooth`
                Contains the radial velocity (`shift` [km/s]), flux scaling (`scale` [-])
                and the additional Gaussian broadening kernel (`smooth`, [Angstroms]) to 
                apply to the telluric spectrum.

            telluric_spectrum : `Spectrum1D`
                The telluric spectrum to use for the \chi^2 minimization.

            scale : `bool`, optional
                Allow for a scaling factor in the telluric spectrum flux.

            smooth : `bool`, optional
                Allow for additional smoothing in the telluric spectrum flux.
            """

            telluric_spectrum = telluric_spectrum.doppler_shift(v=parameters[0], interpolate=False)

            if len(parameters) > 1:
                telluric_spectrum.flux *= parameters[1]

            if len(parameters) > 2:
                telluric_spectrum = telluric_spectrum.gaussian_smooth(parameters[2])

            # Interpolate the telluric spectrum onto the observed dispersion
            telluric_spectrum = telluric_spectrum.interpolate(self.disp)

            # Calculate the \chi^2 difference in flux
            difference = (self.flux - telluric_spectrum.flux)**2

            # Do it properly if we have an uncertainty in the observed spectrum, since the
            # telluric spectrum itself may/may not be theoretical. This will be decided by
            # those with a higher pay grade than I.
            if self.uncertainty is not None:
                difference /= self.uncertainty

            # Return the sum of the difference.
            return np.sum(difference)

        # Create the initial parameters for optimization
        parameters = [0.0] # km/s shift in velocity
        if scale: parameters.append(1.0) # Flux scaling
        if smooth: parameters.append(0.0) # Angstroms additional smoothing (i.e. no smoothing)

        best_parameters = scipy.optimize.fmin(minimize_chi_sq, parameters, (telluric_spectrum, ),)

        # Now that we have the "best parameters", we should apply those parameters to the telluric
        # spectrum and correct the observed spectrum.
        if len(best_parameters) < 3:
            best_parameters += [None] * (3 - len(best_parameters))

        best_shift_factor, best_scale_factor, best_smooth_factor = best_parameters

        # Apply those corrections to the telluric spectrum
        telluric_spectrum = telluric_spectrum.doppler_shift(v=best_shift_factor, interpolate=False)
        if best_scale_factor is not None:
            telluric_spectrum.flux *= best_scale_factor

        if best_smooth_factor is not None:
            telluric_spectrum = telluric_spectrum.gaussian_smooth(best_smooth_factor)

        # Correct the observed spectrum
        corrected_observed_flux = self.flux / telluric_spectrum.interpolate(self.disp).flux

        # Return the corrected spectrum
        return self.__class__(self.disp, corrected_observed_flux, uncertainty=self.uncertainty, headers=self.headers)
       


def compute_non_linear_disp(nwave, specstr, verbose=False):
    """Compute non-linear wavelengths from multispec string
    
    Returns wavelength array and dispersion fields.
    Raises a ValueError if it can't understand the dispersion string.
    """

    fields = specstr.split()
    if int(fields[2]) != 2:
        raise ValueError('not a non-linear dispersion: dtype={dtype}'.format(dtype=fields[2]))

    if len(fields) < 12:
        raise ValueError('bad spectrum format (only {nfields} fields)'.format(nfields=len(fields)))

    wt = float(fields[9])
    w0 = float(fields[10])
    ftype = int(fields[11])
    if ftype == 3:

        # cubic spline

        if len(fields) < 15:
            raise ValueError('bad spline format (only {nfields} fields)'.format(nfields=len(fields)))

        npieces = int(fields[12])
        pmin = float(fields[13])
        pmax = float(fields[14])

        if verbose:
            logging.info('Dispersion is order-{npieces} cubic spline'.format(npieces=npieces))

        if len(fields) != 15+npieces+3:
            raise ValueError('bad order-{npieces} spline format ({nfields} fields)'.format(npieces=npieces, nfields=len(fields)))

        coeff = np.asarray(fields[15:],dtype=float)
        # normalized x coordinates
        s = (np.arange(nwave,dtype=float)+1-pmin)/(pmax-pmin)*npieces
        j = s.astype(int).clip(0, npieces-1)
        a = (j+1)-s
        b = s-j
        x0 = a**3
        x1 = 1+3*a*(1+a*b)
        x2 = 1+3*b*(1+a*b)
        x3 = b**3
        wave = coeff[j]*x0 + coeff[j+1]*x1 + coeff[j+2]*x2 + coeff[j+3]*x3

    elif ftype == 1 or ftype == 2:

        # chebyshev or legendre polynomial
        # legendre not tested yet

        if len(fields) < 15:
            raise ValueError('bad polynomial format (only {nfields} fields)'.format(nfields=len(fields)))

        order = int(fields[12])
        pmin = float(fields[13])
        pmax = float(fields[14])
        if verbose:
            if ftype == 1:
                logging.info('Dispersion is order-{n} Chebyshev polynomial'.format(n=order))

            else:
                logging.warn('Dispersion is order-{n} Legendre polynomial (NEEDS TEST)'.format(n=order))

        if len(fields) != 15+order:
            raise ValueError('bad order-{n} polynomial format ({nfields} fields)'.format(n=order, nfields=len(fields)))

        coeff = np.asarray(fields[15:],dtype=float)
        # normalized x coordinates
        pmiddle = (pmax+pmin)/2
        prange = pmax-pmin
        x = (np.arange(nwave,dtype=float)+1-pmiddle)/(prange/2)
        p0 = np.ones(nwave,dtype=float)
        p1 = x
        wave = p0*coeff[0] + p1*coeff[1]
        for i in range(2, order):
            if ftype == 1:
                # chebyshev
                p2 = 2*x*p1 - p0
            else:
                # legendre
                p2 = ((2*i-1)*x*p1-(i-1)*p0) / i
            wave = wave + p2*coeff[i]
            p0 = p1
            p1 = p2

    else:
        raise ValueError('cannot handle dispersion function of type {ftype}'.format(ftype=ftype))

    return wave, fields
