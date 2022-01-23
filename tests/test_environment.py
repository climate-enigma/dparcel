# Copyright (c) 2021 Thomas Schanzer.
# Distributed under the terms of the BSD 3-Clause License.
"""Tests for dparcel.environment."""

import pytest
import pandas as pd
import numpy as np
from metpy.testing import assert_almost_equal, assert_array_equal
from metpy.units import units

from dparcel.environment import Environment, idealised_sounding

data = pd.read_csv('test_soundings/sydney_20210716_00Z.csv', header=0)
p_sydney, z_sydney, t_sydney, td_sydney = data.to_numpy().T
p_sydney *= units.mbar
z_sydney *= units.meter
t_sydney *= units.kelvin
td_sydney *= units.kelvin
sydney = Environment(p_sydney, z_sydney, t_sydney, td_sydney)

data = pd.read_csv('test_soundings/idealised_rh50.csv', header=0)
p_rh50, z_rh50, t_rh50, td_rh50 = data.to_numpy().T
p_rh50 *= units.mbar
z_rh50 *= units.meter
t_rh50 *= units.kelvin
td_rh50 *= units.kelvin
idealised_rh50 = Environment(p_rh50, z_rh50, t_rh50, td_rh50)

data = pd.read_csv('test_soundings/idealised_rh80.csv', header=0)
p_rh80, z_rh80, t_rh80, td_rh80 = data.to_numpy().T
p_rh80 *= units.mbar
z_rh80 *= units.meter
t_rh80 *= units.kelvin
td_rh80 *= units.kelvin
idealised_rh80 = Environment(p_rh80, z_rh80, t_rh80, td_rh80)

soundings = [sydney, idealised_rh50, idealised_rh80]
p_orig = [p_sydney, p_rh50, p_rh80]
z_orig = [z_sydney, z_rh50, z_rh80]
t_orig = [t_sydney, t_rh50, t_rh80]
td_orig = [td_sydney, td_rh50, td_rh80]

p_test = [900.875, 538.993, 381.223]*units.mbar
z_test = [528.228, 7883.236, 11275.324]*units.meter

def test_temperature_from_pressure():
    """Test temperature_from_pressure."""
    truth = [
        np.array([  7.81066176,  -9.6252625 , -22.36655   ])*units.celsius,
        np.array([ 10.31748479, -11.82559048, -31.5137718 ])*units.celsius,
        np.array([ 10.31748479, -11.82559048, -31.5137718 ])*units.celsius,
    ]
    for i in range(len(soundings)):
        actual = soundings[i].temperature_from_pressure(p_orig[i])
        assert_almost_equal(actual, t_orig[i], 6)
        actual = soundings[i].temperature_from_pressure(p_test)
        assert_almost_equal(actual, truth[i], 6)

def test_dewpoint_from_pressure():
    """Test dewpoint_from_pressure."""
    truth = [
        np.array([  5.87683824, -55.1740375 , -25.2518    ])*units.celsius,
        np.array([  4.74839688, -20.16110852, -38.51417191])*units.celsius,
        np.array([  4.74839688, -14.57613197, -33.81901624])*units.celsius,
    ]
    for i in range(len(soundings)):
        actual = soundings[i].dewpoint_from_pressure(p_orig[i])
        assert_almost_equal(actual, td_orig[i], 6)
        actual = soundings[i].dewpoint_from_pressure(p_test)
        assert_almost_equal(actual, truth[i], 6)

def test_liquid_ratio_from_pressure():
    """Test dewpoint_from_pressure."""
    for i in range(len(soundings)):
        actual = soundings[i].liquid_ratio_from_pressure(p_orig[i])
        assert_almost_equal(actual, np.zeros(p_orig[i].size), 6)
        actual = soundings[i].liquid_ratio_from_pressure(p_test)
        assert_almost_equal(actual, np.zeros(p_test.size), 6)
