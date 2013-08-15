# coding: utf-8

""" Correct observed spectra from HERMES for telluric absorption.  """

from __future__ import division, print_function

__author__ = "Andy Casey <acasey@mso.anu.edu.au"

# Standard library
import os

# Third-party
import numpy as np


def correct_tellurics(observed_spectrum_filename, telluric_spectrum_filename,
    output_filename, **kwargs):
    """Corrects an observed spectrum for telluric absorption, allowing
    for flux scaling, radial velocity shifts, and instrumental broadening.

    Inputs
    ------
    observed_spectrum_filename : `str`
        The filename for the observed spectrum to be corrected for tellurics.

    telluric_spectrum_filename : `str`
        The filename for the telluric spectrum to be used for corrections.

    output_filename : `str`
        Output filename to save the corrected spectrum to.
    """


# 1 Measure equivalent widths of all atomic lines
# 2 Correct for telluric absorption
# 3 Measure equivalent widths of all atomic lines in the corrected spectrum.
# 4 Assemble a measurement of telluric absorption in every atomic line.

