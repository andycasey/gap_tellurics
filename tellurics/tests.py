# coding: utf-8

""" Run tests on the analysis of telluric lines, including correcting observed spectra. """

from __future__ import division, print_function

__author__ = "Andy Casey <acasey@mso.anu.edu.au>"

# Standard library
import json
import logging

# Third-party
import numpy as np

# Module
import tellurics

configuration_filename = '../HERMES.json'


def test_load_configuration(configuration_filename):

    # Load the configration file
    with open(configuration_filename, 'r') as fp:
        configuration = json.load(fp)

    return configuration


def test_verify_configuration(configuration_filename):

    configuration = test_load_configuration(configuration_filename)

    # Check for configuration keywords
    required_configuration_keywords = 'observed_filename, telluric_filename, atomic_rest_wavelengths, initial_fwhm_guess, allow_shift, allow_scale, allow_smooth'.split(', ')

    for keyword in required_configuration_keywords:
        assert keyword in configuration


def test_perform_telluric_analysis(configuration_filename):

    configuration = test_load_configuration(configuration_filename)

    keywords = 'observed_filename, telluric_filename, atomic_rest_wavelengths, initial_fwhm_guess, allow_shift, allow_scale, allow_smooth'.split(', ')

    observed_filename, telluric_filename, atomic_rest_wavelengths, initial_fwhm_guess, allow_shift, allow_scale, allow_smooth = \
        [configuration[keyword] for keyword in 'observed_filename, telluric_filename, atomic_rest_wavelengths, initial_fwhm_guess, allow_shift, allow_scale, allow_smooth'.split(', ')]

    # Perform a test
    tellurics.perform_telluric_analysis(observed_filename, telluric_filename, atomic_rest_wavelengths, initial_fwhm_guess, allow_shift, allow_scale, allow_smooth)
