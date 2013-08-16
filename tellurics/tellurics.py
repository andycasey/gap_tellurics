# coding: utf-8

""" Correct observed spectra from HERMES for telluric absorption. """

from __future__ import division, print_function

__author__ = "Andy Casey <acasey@mso.anu.edu.au>"

# Standard library
import json
import os

# Third-party
import numpy as np

# Module
from specutils import Spectrum1D


def perform_telluric_analysis(observed_filename, telluric_filename,
    atomic_rest_wavelengths, initial_fwhm_guess, allow_shift, allow_scale,
    allow_smooth, ew_error_percent=0.05, **kwargs):
    """Measures equivalent widths for atomic absorption lines, corrects the
    HERMES spectra for telluric absorption, and re-measures equivalent widths
    to ascertain a quantitative measurement of telluric contamination for
    every atomic absorption line.

    Inputs
    ------
    observed_filename : `str`
        Filename of the observed, rest spectrum to correct.

    telluric_filename : `str`
        Filename of the spectrum to use for telluric corrections.

    atomic_rest_wavelengths : `list`-type
        Atomic rest wavelengths in Angstroms.

    initial_fwhm_guess : `float`
        Initial guess for the absorption line FWHM in Angstroms

    allow_shift : `bool`
        Allow for small radial velocity shifts in the telluric spectrum.

    allow_scale : `bool`
        Allow for a free scaling factor to be applied to the telluric flux.

    allow_smooth : `bool`
        Allow for additional smoothing to be applied to the telluric flux.

    Notes
    -----
    Allowing for additional smoothing to be applied to the telluric flux has the
    implied assumption that the telluric spectrum is *always* at a higher resolution
    (e.g. under-smoothed) with respect to the observed spectrum.
    """

    if not os.path.exists(observed_filename):
        raise IOError("observed spectrum filename ({filename}) does not exist".format(filename=observed_filename))

    if not os.path.exists(telluric_filename):
        raise IOError("telluric spectrum filename ({filename}) does not exist".format(filename=telluric_filename))

    if not isinstance(initial_fwhm_guess, (float, int, )):
        raise TypeError("initial FWHM guess must be a float-type")

    if not isinstance(atomic_rest_wavelengths, (list, tuple, np.array)):
        raise TypeError("atomic rest wavelengths must be in a list-type")

    # Float-ify and array-ify our atomic rest wavelengths to be sure
    try:
        atomic_rest_wavelengths = np.array(atomic_rest_wavelengths, dtype=float)

    except TypeError:
        raise TypeError("atomic rest wavelengths must be a list of float-type values")

    # Load the spectra
    observed_spectrum = Spectrum1D.load(observed_filename)
    telluric_spectrum = Spectrum1D.load(telluric_filename)

    # Step 1: Measure equivalent widths for all atomic lines
    #------------------------------------------------------#
    
    initial_equivalent_widths = {}
    for rest_wavelength in atomic_rest_wavelengths:        
        initial_equivalent_widths[rest_wavelength] = measure_line(observed_spectrum, rest_wavelength, initial_fwhm_guess, **kwargs)

    # Step 2: Correct for telluric absorption
    #---------------------------------------#
    corrected_observed_spectrum = observed_spectrum.remove_telluric_absorption(telluric_spectrum, allow_shift, allow_scale, allow_smooth)


    # Step 3: Measure equivalent widths for all atomic lines in the telluric-corrected spectrum
    #-----------------------------------------------------------------------------------------#
    corrected_equivalent_widths = {}
    for rest_wavelength in atomic_rest_wavelengths:
        corrected_equivalent_widths[rest_wavelength] = measure_line(corrected_observed_spectrum, rest_wavelength, initial_fwhm_guess, **kwargs)

    # Step 4: Assemble a measurement of telluric absorption in every atomic line
    #--------------------------------------------------------------------------#
    # We need to establish that the two measurements are "consistent" (e.g. FWHM's and rest
    # wavelengths are similar), and if the corrected equivalent width is "significantly"
    # lower than the initial estimate, we should return a % contamination for
    # that line.

    # In some cases we should be able to give lower limits of % contamination.
    contaminated_absorption_lines = {}
    for rest_wavelength in atomic_rest_wavelengths:

        initial_equivalent_width = initial_equivalent_widths[rest_wavelength]
        corrected_equivalent_width = corrected_equivalent_widths[rest_wavelength]

        # Are the two measurements consistent?

        # Is the corrected equivalent width "substantially" (i.e. less than 1-sigma
        # error fraction) less than the initial measurement?
        if corrected_equivalent_width < ew_error_percent * initial_equivalent_width:
            contaminated_absorption_lines[rest_wavelength] = (initial_equivalent_width - corrected_equivalent_width)/initial_equivalent_width

    return contaminated_absorption_lines

