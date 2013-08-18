# coding: utf-8

""" Correct observed spectra from HERMES for telluric absorption. """

from __future__ import division, print_function

__author__ = "Andy Casey <acasey@mso.anu.edu.au>"

# Standard library
import json
import logging
import os

# Third-party
import numpy as np
try:
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

except ImportError:
    logging.warn("No matplotlib module found. Cannot produce debug plots.")

# Module
from specutils import Spectrum1D
from ews import measure_line


def perform_telluric_analysis(observed_filename, telluric_filename,
    atomic_rest_wavelengths, initial_fwhm_guess, allow_shift, allow_scale,
    allow_smooth, ew_error_percent=0.05, output_plots=False, **kwargs):
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

    ew_error_percent : `float`, optional
        An estimate of the error percent on equivalent width measurements.

    output_plots : `bool`, optional
        Whether or not to plot the outputs.


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

    if not isinstance(initial_fwhm_guess, (float, int)):
        raise TypeError("initial FWHM guess must be a float-type")

    if not isinstance(atomic_rest_wavelengths, (list, tuple, np.ndarray)):
        raise TypeError("atomic rest wavelengths must be in a list-type")

    # Float-ify and array-ify our atomic rest wavelengths to be sure
    try:
        atomic_rest_wavelengths = np.array(atomic_rest_wavelengths, dtype=float)

    except TypeError:
        raise TypeError("atomic rest wavelengths must be a list of float-type values")

    logging.debug("Loading observed spectrum from {observed_filename}".format(observed_filename=observed_filename))
    logging.debug("Loading telluric spectrum from {telluric_filename}".format(telluric_filename=telluric_filename))

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

    idx_ew = 6
    idx_profile_x = 9
    idx_profile_y = 10
    #p = [True/False, rest_wavelength, fwhm, trough, wl_start, wl_end, ew, chi_sq, ier, contx, conty, exclusion]
            
    telluric_spectrum = Spectrum1D.load(telluric_filename)

    # In some cases we should be able to give lower limits of % contamination.
    contaminated_absorption_lines = {}
    for i, rest_wavelength in enumerate(atomic_rest_wavelengths):

        initial_equivalent_width = initial_equivalent_widths[rest_wavelength]
        corrected_equivalent_width = corrected_equivalent_widths[rest_wavelength]

        if initial_equivalent_width[0] and corrected_equivalent_width[0] \
        and corrected_equivalent_width[idx_ew] < (1 - ew_error_percent) * initial_equivalent_width[idx_ew]:

            # Should we make a plot?
            if output_plots:

                try: figure
                except NameError:
                    figure = plt.figure()

                    xlim = (rest_wavelength - 2, rest_wavelength + 2)

                    original_ax = figure.add_subplot(211)
                    corrected_ax = figure.add_subplot(212, sharex=original_ax, sharey=original_ax)

                    original_ax.plot([observed_spectrum.disp[0], observed_spectrum.disp[-1]], [1, 1], ':', color='#000000')
                    corrected_ax.plot([observed_spectrum.disp[0], observed_spectrum.disp[-1]], [1, 1], ':', color='#000000')

                    # Plot tellurics
                    idx = np.searchsorted(telluric_spectrum.disp, xlim)
                    plot_original_telluric, = original_ax.plot(telluric_spectrum.disp[idx[0]:idx[1]], telluric_spectrum.flux[idx[0]:idx[1]], c="#666666")

                    # Plot spectrum
                    idx = np.searchsorted(observed_spectrum.disp, xlim)
                    plot_original_spectrum, = original_ax.plot(observed_spectrum.disp[idx[0]:idx[1]], observed_spectrum.flux[idx[0]:idx[1]], c="#000000")
                    plot_corrected_spectrum, = corrected_ax.plot(corrected_observed_spectrum.disp[idx[0]:idx[1]], corrected_observed_spectrum.flux[idx[0]:idx[1]], c="#000000")

                    # Plot rest wavelength
                    plot_rest_wavelength_original, = original_ax.plot([rest_wavelength, rest_wavelength], [0, 1.2], '-.', c="#000000")
                    plot_rest_wavelength_corrected, = corrected_ax.plot([rest_wavelength, rest_wavelength], [0, 1.2], '-.', c="#000000")

                    # Plot actual wavelength
                    plot_actual_wavelength_original, = original_ax.plot([initial_equivalent_width[1], initial_equivalent_width[1]], [0, 1.2], '-', c="#000000")
                    plot_actual_wavelength_corrected, = corrected_ax.plot([corrected_equivalent_width[1], corrected_equivalent_width[1]], [0, 1.2], '-', c="#000000")

                    plot_original_profile, = original_ax.plot(initial_equivalent_width[idx_profile_x], initial_equivalent_width[idx_profile_y], c='b')
                    plot_corrected_profile, = corrected_ax.plot(corrected_equivalent_width[idx_profile_x], corrected_equivalent_width[idx_profile_y], c='g')

                    corrected_ax.set_xlabel('Wavelength ($\AA$)')
                    corrected_ax.set_ylabel('Flux')
                    original_ax.xaxis.set_visible(False)
                    original_ax.set_ylabel('Flux')

                    corrected_ax.get_yaxis().set_major_locator(MaxNLocator(integer=True))


                else:

                    xlim = (rest_wavelength - 2, rest_wavelength + 2)

                    idx = np.searchsorted(telluric_spectrum.disp, xlim)
                    plot_original_telluric.set_data(
                        telluric_spectrum.disp[idx[0]:idx[1]],
                        telluric_spectrum.flux[idx[0]:idx[1]]
                        )
                    
                    idx = np.searchsorted(observed_spectrum.disp, xlim)
                    plot_original_spectrum.set_data(
                        observed_spectrum.disp[idx[0]:idx[1]],
                        observed_spectrum.flux[idx[0]:idx[1]]
                        )
                    
                    plot_corrected_spectrum.set_data(
                        corrected_observed_spectrum.disp[idx[0]:idx[1]],
                        corrected_observed_spectrum.flux[idx[0]:idx[1]]
                        )
                    
                    plot_original_profile.set_data(
                        initial_equivalent_width[idx_profile_x],
                        initial_equivalent_width[idx_profile_y])
                    
                    plot_corrected_profile.set_data(
                        corrected_equivalent_width[idx_profile_x],
                        corrected_equivalent_width[idx_profile_y])

                    plot_rest_wavelength_original.set_data([rest_wavelength, rest_wavelength], [0, 1.2])
                    plot_rest_wavelength_corrected.set_data([rest_wavelength, rest_wavelength], [0, 1.2])

                    plot_actual_wavelength_original.set_data([initial_equivalent_width[1], initial_equivalent_width[1]], [0, 1.2])
                    plot_actual_wavelength_corrected.set_data([corrected_equivalent_width[1], corrected_equivalent_width[1]], [0, 1.2])
                    
                finally:

                    # Do any other texts exist? If so we should hide and delete them.

                    try: plot_texts
                    except NameError: plot_texts = []
                    else:
                        [item.set_visible(False) for item in plot_texts]
                        del plot_texts

                        plot_texts = []

                    diff_equivalent_width = (corrected_equivalent_width[idx_ew] - initial_equivalent_width[idx_ew])/initial_equivalent_width[idx_ew]

                    plot_texts.append(original_ax.text(rest_wavelength + 2 - 0.05 * np.diff(original_ax.get_xlim())[0], 0.5,
                        "EW = {equivalent_width:.2f} mA".format(equivalent_width=initial_equivalent_width[idx_ew] * 1000),
                        horizontalalignment="right"))

                    plot_texts.append(corrected_ax.text(rest_wavelength + 2 - 0.05 * np.diff(corrected_ax.get_xlim())[0], 0.5,
                        "EW = {equivalent_width:.2f}\n and {diff:.2f}".format(equivalent_width=corrected_equivalent_width[idx_ew] * 1000, diff=diff_equivalent_width * 100),
                        horizontalalignment='right'))


                    original_ax.set_xlim(xlim)
                    original_ax.set_ylim(0, 1.2)
                
                    plt.savefig('telluric-{rest_wavelength:.1f}.png'.format(rest_wavelength=rest_wavelength))
                    

            # Could we measure the EW on both occasions?            
            logging.info("Initial EW at rest wavelength {rest_wavelength:.3f} shifted from {initial_equivalent_width:.2f} mA to {corrected_equivalent_width:.2f} mA"
                .format(rest_wavelength=rest_wavelength, initial_equivalent_width=initial_equivalent_width[idx_ew] * 1000, corrected_equivalent_width=corrected_equivalent_width[idx_ew] * 1000))
    
            # Are the two measurements consistent?
            
            # Is the corrected equivalent width "substantially" (i.e. less than 1-sigma
            # error fraction) less than the initial measurement?
            if corrected_equivalent_width[idx_ew] < initial_equivalent_width[idx_ew]:

                contamination_percent = (initial_equivalent_width[idx_ew] - corrected_equivalent_width[idx_ew])/initial_equivalent_width[idx_ew]
                contamination_percent = np.round(contamination_percent * 100, 2)

                logging.info("Absorption line at {rest_wavelength:.3f} identified to be {contamination_percent:.1f}% contamined by telluric absorption"
                    .format(rest_wavelength=rest_wavelength, contamination_percent=contamination_percent))
                contaminated_absorption_lines[rest_wavelength] = contamination_percent

        elif initial_equivalent_width[0] and not corrected_equivalent_width[0]:
            # Maybe the *entire* line was a telluric?
            continue

        else: continue

    plt.close(figure)

    return contaminated_absorption_lines


