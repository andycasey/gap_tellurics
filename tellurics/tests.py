# coding: utf-8

""" Run tests on the analysis of telluric lines, including correcting observed spectra. """

from __future__ import division, print_function

__author__ = "Andy Casey <acasey@mso.anu.edu.au>"

# Standard library
import json
import logging
import os

# Third-party
import numpy as np

# Module
import tellurics

configuration_filename = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../HERMES.json')
logging.basicConfig(level=logging.DEBUG)


def test_load_configuration():

    # Load the configration file
    with open(configuration_filename, 'r') as fp:
        configuration = json.load(fp)

    return configuration


def test_verify_configuration():
    logging.debug("test_verify_configuration() with configuration_filename={configuration_filename}".format(configuration_filename=configuration_filename))

    configuration = test_load_configuration()

    # Check for configuration keywords
    required_configuration_keywords = 'observed_filename, telluric_filename, atomic_line_list_filename, initial_fwhm_guess, allow_shift, allow_scale, allow_smooth'.split(', ')

    for keyword in required_configuration_keywords:
        assert keyword in configuration

'''
def test_perform_telluric_analysis():

    logging.debug("test_perform_telluric_analysis() with configuration_filename={configuration_filename}".format(configuration_filename=configuration_filename))

    configuration = test_load_configuration()

    keywords = 'observed_filename, telluric_filename, atomic_line_list_filename, initial_fwhm_guess, allow_shift, allow_scale, allow_smooth'.split(', ')

    observed_filename, telluric_filename, atomic_line_list_filename, initial_fwhm_guess, allow_shift, allow_scale, allow_smooth = \
        [configuration[keyword] for keyword in 'observed_filename, telluric_filename, atomic_line_list_filename, initial_fwhm_guess, allow_shift, allow_scale, allow_smooth'.split(', ')]

    atomic_rest_wavelengths = np.loadtxt(atomic_line_list_filename, usecols=(0, ))

    # Perform a test
    return tellurics.perform_telluric_analysis(observed_filename, telluric_filename, atomic_rest_wavelengths, initial_fwhm_guess, allow_shift, allow_scale, allow_smooth)#, output_plots=True)
'''