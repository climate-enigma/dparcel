# Copyright (c) 2021 Thomas Schanzer.
# Distributed under the terms of the BSD 3-Clause License.
"""Tests for dparcel.environment."""

import pandas as pd
import numpy as np
from metpy.testing import assert_almost_equal
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
    for i, sounding in enumerate(soundings):
        actual = sounding.temperature_from_pressure(p_orig[i])
        assert_almost_equal(actual, t_orig[i], 6)
        actual = sounding.temperature_from_pressure(p_test)
        assert_almost_equal(actual, truth[i], 6)

def test_temperature_from_pressure_scalar():
    """Test temperature_from_pressure for scalar input."""
    truth = [
        7.81066176*units.celsius,
        10.31748479*units.celsius,
        10.31748479*units.celsius,
    ]
    for i, sounding in enumerate(soundings):
        actual = sounding.temperature_from_pressure(900.875*units.mbar)
        assert_almost_equal(actual, truth[i], 6)
        assert not hasattr(actual, 'size')

def test_dewpoint_from_pressure():
    """Test dewpoint_from_pressure."""
    truth = [
        np.array([  5.87683824, -55.1740375 , -25.2518    ])*units.celsius,
        np.array([  4.74839688, -20.16110852, -38.51417191])*units.celsius,
        np.array([  4.74839688, -14.57613197, -33.81901624])*units.celsius,
    ]
    for i, sounding in enumerate(soundings):
        actual = sounding.dewpoint_from_pressure(p_orig[i])
        assert_almost_equal(actual, td_orig[i], 6)
        actual = sounding.dewpoint_from_pressure(p_test)
        assert_almost_equal(actual, truth[i], 6)

def test_dewpoint_from_pressure_scalar():
    """Test dewpoint_from_pressure for scalar input."""
    truth = [
        5.87683824*units.celsius,
        4.74839688*units.celsius,
        4.74839688*units.celsius,
    ]
    for i, sounding in enumerate(soundings):
        actual = sounding.dewpoint_from_pressure(900.875*units.mbar)
        assert_almost_equal(actual, truth[i], 6)
        assert not hasattr(actual, 'size')

def test_liquid_ratio_from_pressure():
    """Test liquid_ratio_from_pressure."""
    for i, sounding in enumerate(soundings):
        actual = sounding.liquid_ratio_from_pressure(p_orig[i])
        assert_almost_equal(actual, np.zeros(p_orig[i].size), 6)
        actual = sounding.liquid_ratio_from_pressure(p_test)
        assert_almost_equal(actual, np.zeros(p_test.size), 6)

def test_liquid_ratio_from_pressure_scalar():
    """Test liquid_ratio_from_pressure for scalar input."""
    for _, sounding in enumerate(soundings):
        actual = sounding.liquid_ratio_from_pressure(900.875*units.mbar)
        assert_almost_equal(actual, 0*units.dimensionless, 6)
        assert not hasattr(actual, 'size')

def test_pressure():
    """Test pressure."""
    truth = [
        np.array([958.74320596, 374.3210963 , 230.75851117])*units.mbar,
        np.array([952.44815945, 369.86291472, 220.44549574])*units.mbar,
        np.array([952.44815945, 370.11186768, 220.62150132])*units.mbar,
    ]
    for i, sounding in enumerate(soundings):
        actual = sounding.pressure(z_orig[i])
        assert_almost_equal(actual, p_orig[i], 6)
        actual = sounding.pressure(z_test)
        assert_almost_equal(actual, truth[i], 6)

def test_pressure_scalar():
    """Test pressure for scalar input."""
    truth = [
        958.74320596*units.mbar,
        952.44815945*units.mbar,
        952.44815945*units.mbar,
    ]
    for i, sounding in enumerate(soundings):
        actual = sounding.pressure(528.228*units.meter)
        assert_almost_equal(actual, truth[i], 6)
        assert not hasattr(actual, 'size')

def test_height():
    """Test height."""
    truth = [
        np.array([1044.48897059, 5137.848875  , 7750.57775   ])*units.meter,
        np.array([ 996.14274871, 5109.9809523 , 7669.1646355 ])*units.meter,
        np.array([ 996.17674737, 5113.95050956, 7673.87874374])*units.meter,
    ]
    for i, sounding in enumerate(soundings):
        actual = sounding.height(p_orig[i])
        assert_almost_equal(actual, z_orig[i], 6)
        actual = sounding.height(p_test)
        assert_almost_equal(actual, truth[i], 6)

def test_height_scalar():
    """Test height for scalar input."""
    truth = [
        1044.48897059*units.meter,
        996.14274871*units.meter,
        996.17674737*units.meter,
    ]
    for i, sounding in enumerate(soundings):
        actual = sounding.height(900.875*units.mbar)
        assert_almost_equal(actual, truth[i], 6)

def test_temperature():
    """Test temperature."""
    truth = [
        np.array([ 12.44400397, -23.50209778, -44.94899454])*units.celsius,
        np.array([ 14.86233388, -33.3568757 , -65.05019239])*units.celsius,
        np.array([ 14.86233388, -33.31584591, -65.00311827])*units.celsius,
    ]
    for i, sounding in enumerate(soundings):
        actual = sounding.temperature(z_orig[i])
        assert_almost_equal(actual, t_orig[i], 6)
        actual = sounding.temperature(z_test)
        assert_almost_equal(actual, truth[i], 6)

def test_temperature_scalar():
    """Test temperature for scalar input."""
    truth = [
        12.44400397*units.celsius,
        14.86233388*units.celsius,
        14.86233388*units.celsius,
    ]
    for i, sounding in enumerate(soundings):
        actual = sounding.temperature(528.228*units.meter)
        assert_almost_equal(actual, truth[i], 6)
        assert not hasattr(actual, 'size')

def test_dewpoint():
    """Test dewpoint."""
    truth = [
        np.array([  6.87652705, -25.86789037, -64.33208238])*units.celsius,
        np.array([  5.49242708, -40.23805109, -70.03695656])*units.celsius,
        np.array([  5.49242708, -35.58227355, -66.64036528])*units.celsius,
    ]
    for i, sounding in enumerate(soundings):
        actual = sounding.dewpoint(z_orig[i])
        assert_almost_equal(actual, td_orig[i], 6)
        actual = sounding.dewpoint(z_test)
        assert_almost_equal(actual, truth[i], 6)

def test_dewpoint_scalar():
    """Test dewpoint for scalar input."""
    truth = [
        6.87652705*units.celsius,
        5.49242708*units.celsius,
        5.49242708*units.celsius,
    ]
    for i, sounding in enumerate(soundings):
        actual = sounding.dewpoint(528.228*units.meter)
        assert_almost_equal(actual, truth[i], 6)
        assert not hasattr(actual, 'size')

def test_liquid_ratio():
    """Test liquid_ratio."""
    for i, sounding in enumerate(soundings):
        actual = sounding.liquid_ratio(z_orig[i])
        assert_almost_equal(actual, np.zeros(z_orig[i].size), 6)
        actual = sounding.liquid_ratio(z_test)
        assert_almost_equal(actual, np.zeros(z_test.size), 6)

def test_liquid_ratio_scalar():
    """Test liquid_ratio for scalar input."""
    for _, sounding in enumerate(soundings):
        actual = sounding.liquid_ratio(528.228*units.meter)
        assert_almost_equal(actual, 0*units.dimensionless, 6)
        assert not hasattr(actual, 'size')

def test_wetbulb_temperature():
    """Test wetbulb_temperature."""
    truth = [
        np.array([  9.40419086, -24.06432787, -45.6280737 ])*units.celsius,
        np.array([  9.77720608, -34.06208121, -65.08660595])*units.celsius,
        np.array([  9.77720608, -33.59504385, -65.0176237 ])*units.celsius,
    ]
    for i, sounding in enumerate(soundings):
        actual = sounding.wetbulb_temperature(z_test)
        assert_almost_equal(actual, truth[i], 6)

def test_wetbulb_temperature_scalar():
    """Test wetbulb_temperature for scalar input."""
    truth = [
        9.40419086*units.celsius,
        9.77720608*units.celsius,
        9.77720608*units.celsius,
    ]
    for i, sounding in enumerate(soundings):
        actual = sounding.wetbulb_temperature(528.228*units.meter)
        assert_almost_equal(actual, truth[i], 6)
        assert not hasattr(actual, 'size')

def test_specific_humidity():
    """Test specific_humidity."""
    truth = [
        np.array([6.46709357e-03, 1.24420837e-03, 2.89314967e-05]),
        np.array([5.91477562e-03, 3.11050759e-04, 1.37473751e-05]),
        np.array([5.91477562e-03, 4.99396142e-04, 2.21187020e-05]),
    ]
    for i, sounding in enumerate(soundings):
        actual = sounding.specific_humidity(z_test)
        assert_almost_equal(actual, truth[i]*units.dimensionless, 9)

def test_specific_humidity_scalar():
    """Test specific_humidity for scalar input."""
    truth = [
        6.46709357e-03*units.dimensionless,
        5.91477562e-03*units.dimensionless,
        5.91477562e-03*units.dimensionless,
    ]
    for i, sounding in enumerate(soundings):
        actual = sounding.specific_humidity(528.228*units.meter)
        assert_almost_equal(actual, truth[i], 9)
        assert not hasattr(actual, 'size')

def test_density():
    """Test density."""
    truth = [
        np.array([1.16491877, 0.5219565 , 0.35227253])*units('kg/m^3'),
        np.array([1.14793671, 0.53723983, 0.36903892])*units('kg/m^3'),
        np.array([1.14793671, 0.53744796, 0.36924815])*units('kg/m^3'),
    ]
    for i, sounding in enumerate(soundings):
        actual = sounding.density(z_test)
        assert_almost_equal(actual, truth[i], 6)

def test_density_scalar():
    """Test density for scalar input."""
    truth = [
        1.16491877*units('kg/m^3'),
        1.14793671*units('kg/m^3'),
        1.14793671*units('kg/m^3'),
    ]
    for i, sounding in enumerate(soundings):
        actual = sounding.density(528.228*units.meter)
        assert_almost_equal(actual, truth[i], 6)
        assert not hasattr(actual, 'size')

def test_equivalent_potential_temperature():
    """Test equivalent_potential_temperature."""
    truth = [
        np.array([307.69352415, 335.1216174 , 347.08927477])*units.kelvin,
        np.array([309.3839085 , 319.77661193, 320.61334222])*units.kelvin,
        np.array([309.3839085 , 320.43260768, 320.64746956])*units.kelvin,
    ]
    for i, sounding in enumerate(soundings):
        actual = sounding.equivalent_potential_temperature(z_test)
        assert_almost_equal(actual, truth[i], 6)

def test_equivalent_potential_temperature_scalar():
    """Test equivalent_potential_temperature for scalar input."""
    truth = [
        307.69352415*units.kelvin,
        309.3839085*units.kelvin,
        309.3839085*units.kelvin,
    ]
    for i, sounding in enumerate(soundings):
        actual = sounding.equivalent_potential_temperature(
            528.228*units.meter)
        assert_almost_equal(actual, truth[i], 6)
        assert not hasattr(actual, 'size')

def test_potential_temperature():
    """Test potential_temperature."""
    truth = [
        np.array([289.05267954, 330.56595221, 346.95447101])*units.kelvin,
        np.array([292.04946561, 318.60577536, 320.55303921])*units.kelvin,
        np.array([292.04946561, 318.59903439, 320.55244875])*units.kelvin,
    ]
    for i, sounding in enumerate(soundings):
        actual = sounding.potential_temperature(z_test)
        assert_almost_equal(actual, truth[i], 6)

def test_potential_temperature_scalar():
    """Test potential_temperature for scalar input."""
    truth = [
        289.05267954*units.kelvin,
        292.04946561*units.kelvin,
        292.04946561*units.kelvin,
    ]
    for i, sounding in enumerate(soundings):
        actual = sounding.potential_temperature(
            528.228*units.meter)
        assert_almost_equal(actual, truth[i], 6)
        assert not hasattr(actual, 'size')

def test_virtual_temperature():
    """Test virtual_temperature."""
    truth = [
        np.array([286.71664067, 249.83670225, 228.20501846])*units.kelvin,
        np.array([289.04778683, 239.8384609 , 208.1015465 ])*units.kelvin,
        np.array([289.04778683, 239.90695507, 208.14968013])*units.kelvin,
    ]
    for i, sounding in enumerate(soundings):
        actual = sounding.virtual_temperature(z_test)
        assert_almost_equal(actual, truth[i], 6)

def test_virtual_temperature_scalar():
    """Test virtual_temperature for scalar input."""
    truth = [
        286.71664067*units.kelvin,
        289.04778683*units.kelvin,
        289.04778683*units.kelvin,
    ]
    for i, sounding in enumerate(soundings):
        actual = sounding.virtual_temperature(
            528.228*units.meter)
        assert_almost_equal(actual, truth[i], 6)
        assert not hasattr(actual, 'size')

def test_dry_static_energy():
    """Test dry_static_energy."""
    truth = [
        np.array([292.10679509, 328.12095018, 339.8389973 ])*units('kJ/kg'),
        np.array([294.53640945, 318.22018771, 319.64400287])*units('kJ/kg'),
        np.array([294.53640945, 318.26140896, 319.69129665])*units('kJ/kg'),
    ]
    for i, sounding in enumerate(soundings):
        actual = sounding.dry_static_energy(z_test)
        assert_almost_equal(actual, truth[i], 6)

def test_dry_static_energy_scalar():
    """Test dry_static_energy for scalar input."""
    truth = [
        292.10679509*units('kJ/kg'),
        294.53640945*units('kJ/kg'),
        294.53640945*units('kJ/kg'),
    ]
    for i, sounding in enumerate(soundings):
        actual = sounding.dry_static_energy(
            528.228*units.meter)
        assert_almost_equal(actual, truth[i], 6)
        assert not hasattr(actual, 'size')

def test_moist_static_energy():
    """Test moist_static_energy."""
    truth = [
        np.array([308.27996138, 331.23251624, 339.91135034])*units('kJ/kg'),
        np.array([309.32831692, 318.99807589, 319.67838286])*units('kJ/kg'),
        np.array([309.32831692, 319.51031881, 319.74661198])*units('kJ/kg'),
    ]
    for i, sounding in enumerate(soundings):
        actual = sounding.moist_static_energy(z_test)
        assert_almost_equal(actual, truth[i], 6)

def test_moist_static_energy_scalar():
    """Test moist_static_energy for scalar input."""
    truth = [
        308.27996138*units('kJ/kg'),
        309.32831692*units('kJ/kg'),
        309.32831692*units('kJ/kg'),
    ]
    for i, sounding in enumerate(soundings):
        actual = sounding.moist_static_energy(
            528.228*units.meter)
        assert_almost_equal(actual, truth[i], 6)
        assert not hasattr(actual, 'size')

def test_mixing_ratio():
    """Test mixing_ratio."""
    truth = [
        np.array([6.50918911e-03, 1.24575835e-03, 2.89323338e-05]),
        np.array([5.94996835e-03, 3.11147542e-04, 1.37475641e-05]),
        np.array([5.94996835e-03, 4.99645663e-04, 2.21191913e-05]),
    ]
    for i, sounding in enumerate(soundings):
        actual = sounding.mixing_ratio(z_test)
        assert_almost_equal(actual, truth[i]*units.dimensionless, 9)

def test_mixing_ratio_scalar():
    """Test mixing_ratio for scalar input."""
    truth = [
        6.50918911e-03*units.dimensionless,
        5.94996835e-03*units.dimensionless,
        5.94996835e-03*units.dimensionless,
    ]
    for i, sounding in enumerate(soundings):
        actual = sounding.mixing_ratio(528.228*units.meter)
        assert_almost_equal(actual, truth[i], 9)
        assert not hasattr(actual, 'size')

def test_relative_humidity():
    """Test relative_humidity."""
    truth = [
        np.array([0.688105  , 0.80847653, 0.09590761])*units.dimensionless,
        np.array([0.53435553, 0.49999887, 0.49999534])*units.dimensionless,
        np.array([0.53435553, 0.79999937, 0.79999752])*units.dimensionless,
    ]
    for i, sounding in enumerate(soundings):
        actual = sounding.relative_humidity(z_test)
        assert_almost_equal(actual, truth[i], 6)

def test_relative_humidity_scalar():
    """Test relative_humidity for scalar input."""
    truth = [
        0.688105*units.dimensionless,
        0.53435553*units.dimensionless,
        0.53435553*units.dimensionless,
    ]
    for i, sounding in enumerate(soundings):
        actual = sounding.relative_humidity(528.228*units.meter)
        assert_almost_equal(actual, truth[i], 6)
        assert not hasattr(actual, 'size')

def test_dcape_dcin():
    """Test dcape_dcin."""
    truth = [
        (123.01795081346279*units('J/kg'), -450.4350640133004*units('J/kg')),
        (230.41485077770963*units('J/kg'), -34.70579018913559*units('J/kg')),
        (103.99288157775544*units('J/kg'), -54.02795870575373*units('J/kg')),
    ]
    for i, sounding in enumerate(soundings):
        actual_dcape, actual_dcin = sounding.dcape_dcin()
        assert_almost_equal(actual_dcape, truth[i][0], 6)
        assert_almost_equal(actual_dcin, truth[i][1], 6)
        assert not hasattr(actual_dcape, 'size')
        assert not hasattr(actual_dcin, 'size')

def test_idealised_sounding():
    """Test idealised_sounding."""
    actual_p, actual_z, actual_t, actual_td = idealised_sounding(0.5)
    assert_almost_equal(actual_p, p_rh50, 6)
    assert_almost_equal(actual_z, z_rh50, 6)
    assert_almost_equal(actual_t, t_rh50, 6)
    assert_almost_equal(actual_td, td_rh50, 6)

    actual_p, actual_z, actual_t, actual_td = idealised_sounding(0.8)
    assert_almost_equal(actual_p, p_rh80, 6)
    assert_almost_equal(actual_z, z_rh80, 6)
    assert_almost_equal(actual_t, t_rh80, 6)
    assert_almost_equal(actual_td, td_rh80, 6)
