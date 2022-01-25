# Copyright (c) 2021 Thomas Schanzer.
# Distributed under the terms of the BSD 3-Clause License.
"""Tests for dparcel.parcel."""

import pytest
import pandas as pd
import numpy as np
from metpy.testing import assert_almost_equal, assert_array_equal
from metpy.units import units

from dparcel.parcel import Parcel, FastParcel

data = pd.read_csv('tests/test_soundings/sydney_20210716_00Z.csv', header=0)
p_sydney, z_sydney, t_sydney, td_sydney = data.to_numpy().T
p_sydney *= units.mbar
z_sydney *= units.meter
t_sydney *= units.kelvin
td_sydney *= units.kelvin
sydney = Parcel(p_sydney[4:], z_sydney[4:], t_sydney[4:], td_sydney[4:])
sydneyfast = FastParcel(
    p_sydney[4:], z_sydney[4:], t_sydney[4:], td_sydney[4:])

def test_parcel_profile_nonmonotonic_height():
    """Test Parcel.profile for non-monotonic height array."""
    height = [3000, 2000, 1000, 2000]*units.meter
    t_initial = -2*units.celsius
    q_initial = 1e-4*units.dimensionless
    l_initial = 0*units.dimensionless
    rate = 0.5/units.km

    with pytest.raises(ValueError):
        _, _, _ = sydney.profile(
            height, t_initial, q_initial, l_initial, rate)

def test_parcel_profile_height_above_reference():
    """Test Parcel.profile for final height above the reference height."""
    height = [4000, 2000, 1000]*units.meter
    z_init = 3000*units.meter
    t_initial = -2*units.celsius
    q_initial = 1e-4*units.dimensionless
    l_initial = 0*units.dimensionless
    rate = 0.5/units.km

    with pytest.raises(ValueError):
        _, _, _ = sydney.profile(
            height, t_initial, q_initial, l_initial, rate,
            reference_height=z_init)

def test_parcel_profile_no_descent():
    """Test Parcel.profile when no descent is needed."""
    height = 3000*units.meter
    z_init = 3000*units.meter
    t_initial = -2*units.celsius
    q_initial = 1e-4*units.dimensionless
    l_initial = 0*units.dimensionless
    rate = 0.5/units.km

    actual_t, actual_q, actual_l = sydney.profile(
        height, t_initial, q_initial, l_initial, rate, reference_height=z_init)
    assert_almost_equal(actual_t, t_initial, 3)
    assert_almost_equal(actual_q, q_initial, 6)
    assert_array_equal(actual_l, l_initial)

def test_parcel_profile_no_reference():
    """Test Parcel.profile without a reference height."""
    height = [3000, 2000, 1000]*units.meter
    t_initial = -2*units.celsius
    q_initial = 1e-4*units.dimensionless
    l_initial = 0*units.dimensionless
    rate = 0.5/units.km

    truth_t = [-2., 7.59522877, 14.32321658]*units.celsius
    truth_q = [0.0001, 0.00045977, 0.00213945]*units.dimensionless
    truth_l = [0., 0., 0.]*units.dimensionless
    actual_t, actual_q, actual_l = sydney.profile(
        height, t_initial, q_initial, l_initial, rate)
    assert_almost_equal(actual_t, truth_t, 3)
    assert_almost_equal(actual_q, truth_q, 6)
    assert_array_equal(actual_l, truth_l)

def test_parcel_profile_reference_equals_first_height():
    """Test Parcel.profile when reference height equals first final height."""
    height = [3000, 2000, 1000]*units.meter
    z_init = 3000*units.meter
    t_initial = -2*units.celsius
    q_initial = 1e-4*units.dimensionless
    l_initial = 0*units.dimensionless
    rate = 0.5/units.km

    truth_t = [-2., 7.59522877, 14.32321658]*units.celsius
    truth_q = [0.0001, 0.00045977, 0.00213945]*units.dimensionless
    truth_l = [0., 0., 0.]*units.dimensionless
    actual_t, actual_q, actual_l = sydney.profile(
        height, t_initial, q_initial, l_initial, rate, reference_height=z_init)
    assert_almost_equal(actual_t, truth_t, 3)
    assert_almost_equal(actual_q, truth_q, 6)
    assert_array_equal(actual_l, truth_l)

def test_parcel_profile_scalar_height():
    """Test Parcel.profile when the final height is a scalar."""
    height = 1000*units.meter
    z_init = 3000*units.meter
    t_initial = -2*units.celsius
    q_initial = 1e-4*units.dimensionless
    l_initial = 0*units.dimensionless
    rate = 0.5/units.km

    truth_t = 14.32321658*units.celsius
    truth_q = 0.00213945*units.dimensionless
    truth_l = 0.*units.dimensionless
    actual_t, actual_q, actual_l = sydney.profile(
        height, t_initial, q_initial, l_initial, rate, reference_height=z_init)
    assert_almost_equal(actual_t, truth_t, 3)
    assert_almost_equal(actual_q, truth_q, 6)
    assert_array_equal(actual_l, truth_l)
    assert not hasattr(actual_t, 'size')
    assert not hasattr(actual_q, 'size')
    assert not hasattr(actual_l, 'size')

def test_parcel_profile_initially_subsaturated():
    """Test Parcel.profile for a subsaturated parcel."""
    height = [2000, 1000, 0]*units.meter
    z_init = 3000*units.meter
    t_initial = -2*units.celsius
    q_initial = 1e-4*units.dimensionless
    l_initial = 0*units.dimensionless
    rate = 0.5/units.km

    truth_t = [7.59522877, 14.32321658, 20.96021533]*units.celsius
    truth_q = [0.00045977, 0.00213945, 0.00387132]*units.dimensionless
    truth_l = [0., 0., 0.]*units.dimensionless
    actual_t, actual_q, actual_l = sydney.profile(
        height, t_initial, q_initial, l_initial, rate, reference_height=z_init)
    assert_almost_equal(actual_t, truth_t, 3)
    assert_almost_equal(actual_q, truth_q, 6)
    assert_array_equal(actual_l, truth_l)

def test_parcel_profile_pseudoadiabatic_initially_saturated():
    """Test Parcel.profile for a saturated parcel, pseudoadiabatic descent."""
    height = [2000, 1000, 0]*units.meter
    z_init = 3000*units.meter
    t_initial = -2*units.celsius
    q_initial = 0.004751707262581661*units.dimensionless
    l_initial = 5e-3*units.dimensionless
    rate = 0.5/units.km

    truth_t = [1.80559804, 9.55493141, 17.98658756]*units.celsius
    truth_q = [0.0055382, 0.00564526, 0.00598423]*units.dimensionless
    truth_l = [0.00073854, 0., 0.]*units.dimensionless
    actual_t, actual_q, actual_l = sydney.profile(
        height, t_initial, q_initial, l_initial, rate, reference_height=z_init)
    assert_almost_equal(actual_t, truth_t, 3)
    assert_almost_equal(actual_q, truth_q, 6)
    assert_almost_equal(actual_l, truth_l, 6)

def test_parcel_profile_reversible_initially_saturated():
    """Test Parcel.profile, saturated parcel, reversible adiabatic descent."""
    height = [2000, 1000, 0]*units.meter
    z_init = 3000*units.meter
    t_initial = -2*units.celsius
    q_initial = 0.004751707262581661*units.dimensionless
    l_initial = 5e-3*units.dimensionless
    rate = 0.5/units.km

    truth_t = [1.78019707,  9.52454163, 17.96763569]*units.celsius
    truth_q = [0.00552812, 0.00564526, 0.00598423]*units.dimensionless
    truth_l = [0.00074862, 0., 0.]*units.dimensionless
    actual_t, actual_q, actual_l = sydney.profile(
        height, t_initial, q_initial, l_initial, rate,
        reference_height=z_init, kind='reversible')
    assert_almost_equal(actual_t, truth_t, 3)
    assert_almost_equal(actual_q, truth_q, 6)
    assert_almost_equal(actual_l, truth_l, 6)

def test_parcel_parcel_density_initially_subsaturated():
    """Test Parcel.parcel_density for a subsaturated parcel."""
    height = [2000, 1000, 0]*units.meter
    z_init = 3000*units.meter
    t_initial = -2*units.celsius
    q_initial = 1e-4*units.dimensionless
    l_initial = 0*units.dimensionless
    rate = 0.5/units.km

    truth = [0.97304305, 1.07284677, 1.18054049]*units('kg/m^3')
    actual = sydney.parcel_density(
        height, z_init, t_initial, q_initial, l_initial, rate)
    assert_almost_equal(actual, truth, 6)

def test_parcel_parcel_density_pseudoadiabatic_initially_saturated():
    """Test Parcel.parcel_density, saturated parcel, pseudoadiabatic."""
    height = [2000, 1000, 0]*units.meter
    z_init = 3000*units.meter
    t_initial = -2*units.celsius
    q_initial = 0.004751707262581661*units.dimensionless
    l_initial = 5e-3*units.dimensionless
    rate = 0.5/units.km

    truth = [0.99120751, 1.08862533, 1.19107227]*units('kg/m^3')
    actual = sydney.parcel_density(
        height, z_init, t_initial, q_initial, l_initial, rate)
    assert_almost_equal(actual, truth, 6)

def test_parcel_parcel_density_reversible_initially_saturated():
    """Test Parcel.parcel_density, saturated parcel, reversible descent."""
    height = [2000, 1000, 0]*units.meter
    z_init = 3000*units.meter
    t_initial = -2*units.celsius
    q_initial = 0.004751707262581661*units.dimensionless
    l_initial = 5e-3*units.dimensionless
    rate = 0.5/units.km

    truth = [0.99131515, 1.08874236, 1.19114981]*units('kg/m^3')
    actual = sydney.parcel_density(
        height, z_init, t_initial, q_initial, l_initial, rate,
        kind='reversible')
    assert_almost_equal(actual, truth, 6)

def test_parcel_parcel_density_no_liquid_correction():
    """Test Parcel.parcel_density with liquid_correction=False."""
    height = [2000, 1000, 0]*units.meter
    z_init = 3000*units.meter
    t_initial = -2*units.celsius
    q_initial = 0.004751707262581661*units.dimensionless
    l_initial = 5e-3*units.dimensionless
    rate = 0.5/units.km

    truth = [0.99047547, 1.08862533, 1.19107227]*units('kg/m^3')
    actual = sydney.parcel_density(
        height, z_init, t_initial, q_initial, l_initial, rate,
        liquid_correction=False)
    assert_almost_equal(actual, truth, 6)

def test_parcel_buoyancy_initially_subsaturated():
    """Test Parcel.buoyancy for a subsaturated parcel."""
    height = [2000, 1000, 0]*units.meter
    z_init = 3000*units.meter
    t_initial = -2*units.celsius
    q_initial = 1e-4*units.dimensionless
    l_initial = 0*units.dimensionless
    rate = 0.5/units.km

    truth = [0.07574004, 0.24243366, 0.18312808]*units('m/s^2')
    actual = sydney.buoyancy(
        height, z_init, t_initial, q_initial, l_initial, rate)
    assert_almost_equal(actual, truth, 6)

def test_parcel_buoyancy_pseudoadiabatic_initially_saturated():
    """Test Parcel.buoyancy for a saturated parcel, pseudoadiabatic descent."""
    height = [2000, 1000, 0]*units.meter
    z_init = 3000*units.meter
    t_initial = -2*units.celsius
    q_initial = 0.004751707262581661*units.dimensionless
    l_initial = 5e-3*units.dimensionless
    rate = 0.5/units.km

    truth = [-0.10536063, 0.09678205, 0.09479574]*units('m/s^2')
    actual = sydney.buoyancy(
        height, z_init, t_initial, q_initial, l_initial, rate)
    assert_almost_equal(actual, truth, 6)

def test_parcel_buoyancy_reversible_initially_saturated():
    """Test Parcel.buoyancy for a saturated parcel, reversible descent."""
    height = [2000, 1000, 0]*units.meter
    z_init = 3000*units.meter
    t_initial = -2*units.celsius
    q_initial = 0.004751707262581661*units.dimensionless
    l_initial = 5e-3*units.dimensionless
    rate = 0.5/units.km

    truth = [-0.10641401, 0.09571746, 0.0941512]*units('m/s^2')
    actual = sydney.buoyancy(
        height, z_init, t_initial, q_initial, l_initial, rate,
        kind='reversible')
    assert_almost_equal(actual, truth, 6)

def test_parcel_buoyancy_no_liquid_correction():
    """Test Parcel.buoyancy with liquid_correction=False."""
    height = [2000, 1000, 0]*units.meter
    z_init = 3000*units.meter
    t_initial = -2*units.celsius
    q_initial = 0.004751707262581661*units.dimensionless
    l_initial = 5e-3*units.dimensionless
    rate = 0.5/units.km

    truth = [-0.09819057, 0.09678205, 0.09479574]*units('m/s^2')
    actual = sydney.buoyancy(
        height, z_init, t_initial, q_initial, l_initial, rate,
        liquid_correction=False)
    assert_almost_equal(actual, truth, 6)

def test_parcel_motion_hit_ground():
    """Test Parcel.motion for a parcel reaching the ground."""
    time = np.arange(0, 6*60, 60)*units.second
    z_init = 3000*units.meter
    w_init = 0*units('m/s')
    t_initial = -3*units.celsius
    q_initial = 0.004411511498126446*units.dimensionless
    l_initial = 5e-3*units.dimensionless
    rate = 0.5/units.km

    truth = {
        'height': [3000., 2779.40740568, 2111.85174618,
                   1126.13930493, 345.60577653, np.nan]*units.meter,
        'velocity': [0, -7.40141405, -14.86871612,
                     -15.51565616, -10.59514798, np.nan]*units('m/s'),
        'temperature': [-3., -2.07444998, 0.76303326,
                        8.09248224, 14.78024907, np.nan]*units.celsius,
        'specific_humidity': [0.00441151, 0.004598, 0.00520855,
                              0.00546518, 0.00578415, np.nan]*units(''),
        'liquid_ratio': [0.005, 0.0038962, 0.00114186,
                         0., 0., np.nan]*units(''),
        'density': [0.89587838, 0.91641684, 0.98204805,
                    1.0777025, 1.15644317, np.nan]*units('kg/m^3'),
        'buoyancy': [-0.12107617, -0.12425642, -0.12584922,
                     0.08259816, 0.07495148, np.nan]*units('m/s^2'),
        'neutral_buoyancy_time': 146.72595790038056*units.second,
        'hit_ground_time': 277.523717064885*units.second,
        'min_height_time': np.nan*units.second,
        'neutral_buoyancy_height': 1676.8165192484294*units.meter,
        'neutral_buoyancy_velocity': -17.071598366562046*units('m/s'),
        'hit_ground_velocity': -7.773893588250036*units('m/s'),
        'min_height': np.nan*units.meter
    }

    actual = sydney.motion(
        time, z_init, w_init, t_initial, q_initial, l_initial, rate)

    for truth_var, actual_var in zip(truth.values(), actual.__dict__.values()):
        assert_almost_equal(actual_var, truth_var, 4)

def test_parcel_motion_not_hit_ground():
    """Test Parcel.motion for a parcel not reaching the ground."""
    time = np.arange(0, 6*60, 60)*units.second
    z_init = 3000*units.meter
    w_init = 0*units('m/s')
    t_initial = -1*units.celsius
    q_initial = 1e-3*units.dimensionless
    l_initial = 0*units.dimensionless
    rate = 0.5/units.km

    truth = {
        'height': [3000., 2963.91427237, 2867.7678195,
                   2754.49703165, 2682.19699647, np.nan]*units.meter,
        'velocity': [0., -1.16946209, -1.91033584,
                     -1.69044915, -0.61770804, np.nan]*units('m/s'),
        'temperature': [-1., -0.65261826, 0.29346522,
                        1.41235842, 2.10568038, np.nan]*units.celsius,
        'specific_humidity': [0.001, 0.00099383, 0.00098031,
                              0.00096887, 0.00096388, np.nan]*units(''),
        'liquid_ratio': [0., 0., 0., 0., 0., np.nan]*units(''),
        'density': [0.88668191, 0.88938961, 0.89688489,
                    0.90606089, 0.91194679, np.nan]*units('kg/m^3'),
        'buoyancy': [-0.02061958, -0.0172448, -0.00513284,
                     0.01194682, 0.0220604, np.nan]*units('m/s^2'),
        'neutral_buoyancy_time': 136.97895522269684*units.second,
        'hit_ground_time': np.nan*units.second,
        'min_height_time': 267.06600918022554*units.second,
        'neutral_buoyancy_height': 2834.7941061947395*units.meter,
        'neutral_buoyancy_velocity': -1.9540112264154348*units('m/s'),
        'hit_ground_velocity': np.nan*units('m/s'),
        'min_height': 2673.7559561309704*units.meter
    }

    actual = sydney.motion(
        time, z_init, w_init, t_initial, q_initial, l_initial, rate)

    for truth_var, actual_var in zip(truth.values(), actual.__dict__.values()):
        assert_almost_equal(actual_var, truth_var, 4)

def test_fastparcel_parcel_equivalent_potential_temperature():
    """Test FastParcel.parcel_equivalent_potential_temperature."""
    height = [2000, 1000, 0]*units.meter
    z_init = 3000*units.meter
    t_initial = -2*units.celsius
    q_initial = 0.004751707262581661*units.dimensionless
    rate = 0.5/units.km

    actual = sydneyfast.parcel_equivalent_potential_temperature(
        z_init, t_initial, q_initial, rate)(height)
    truth = [311.18426538, 309.50915144, 308.95267974]*units.kelvin
    assert_almost_equal(actual, truth, 3)

def test_fastparcel_parcel_equivalent_potential_temperature_scalar():
    """Test FastParcel.parcel_equivalent_potential_temperature for scalars."""
    height = 2000*units.meter
    z_init = 3000*units.meter
    t_initial = -2*units.celsius
    q_initial = 0.004751707262581661*units.dimensionless
    rate = 0.5/units.km

    actual = sydneyfast.parcel_equivalent_potential_temperature(
        z_init, t_initial, q_initial, rate)(height)
    truth = 311.18426538*units.kelvin
    assert_almost_equal(actual, truth, 3)
    assert not hasattr(actual, 'size')

def test_fastparcel_parcel_equivalent_potential_temperature_rate_func():
    """
    Test FastParcel.parcel_equivalent_potential_temperature.

    Check the case where entrainment rate is callable.
    """
    height = [2000, 1000, 0]*units.meter
    z_init = 3000*units.meter
    t_initial = -2*units.celsius
    q_initial = 0.004751707262581661*units.dimensionless
    rate = lambda z: 0.5/units.km + z/(1*units.km)*(0.1/units.km)

    actual = sydneyfast.parcel_equivalent_potential_temperature(
        z_init, t_initial, q_initial, rate)(height)
    truth = [309.46813541, 306.84084219, 307.32459864]*units.kelvin
    assert_almost_equal(actual, truth, 3)

def test_fastparcel_water_content():
    """Test FastParcel.water_content."""
    height = [2000, 1000, 0]*units.meter
    z_init = 3000*units.meter
    q_initial = 0.004751707262581661*units.dimensionless
    l_initial = 2e-3*units.dimensionless
    rate = 0.5/units.km

    actual = sydneyfast.water_content(
        z_init, q_initial, l_initial, rate)(height)
    truth = [0.00450332, 0.00462289, 0.00537897]*units.dimensionless
    assert_almost_equal(actual, truth, 6)

def test_fastparcel_water_content_scalar():
    """Test FastParcel.water_content for scalar input."""
    height = 2000*units.meter
    z_init = 3000*units.meter
    q_initial = 0.004751707262581661*units.dimensionless
    l_initial = 2e-3*units.dimensionless
    rate = 0.5/units.km

    actual = sydneyfast.water_content(
        z_init, q_initial, l_initial, rate)(height)
    truth = 0.00450332*units.dimensionless
    assert_almost_equal(actual, truth, 6)
    assert not hasattr(actual, 'size')

def test_fastparcel_water_content_rate_func():
    """Test FastParcel.water_content for callable entrainment rate."""
    height = [2000, 1000, 0]*units.meter
    z_init = 3000*units.meter
    q_initial = 0.004751707262581661*units.dimensionless
    l_initial = 2e-3*units.dimensionless
    rate = lambda z: 0.5/units.km + z/(1*units.km)*(0.1/units.km)

    actual = sydneyfast.water_content(
        z_init, q_initial, l_initial, rate)(height)
    truth = [0.0037374, 0.00424625, 0.00521945]*units.dimensionless
    assert_almost_equal(actual, truth, 6)

def test_fastparcel_properties_moist():
    """Test FastParcel._properties_moist."""
    height = [2000, 1000, 0]*units.meter
    z_init = 3000*units.meter
    t_initial = -2*units.celsius
    q_initial = 0.004751707262581661*units.dimensionless
    l_initial = 2e-3*units.dimensionless
    rate = 0.5/units.km
    theta_e = sydneyfast.parcel_equivalent_potential_temperature(
        z_init, t_initial, q_initial, rate)
    total_water = sydneyfast.water_content(
        z_init, q_initial, l_initial, rate)

    actual_t, actual_q, actual_l = sydneyfast._properties_moist(
        height, z_init, t_initial, theta_e, total_water)
    truth_t = [274.97291915, 279.82475884, 284.81223373]*units.kelvin
    truth_q = [0.00554509, 0.00690006, 0.00857783]*units.dimensionless
    truth_l = [-0.00104177, -0.00227717, -0.00319885]*units.dimensionless
    assert_almost_equal(actual_t, truth_t, 3)
    assert_almost_equal(actual_q, truth_q, 6)
    assert_almost_equal(actual_l, truth_l, 6)

def test_fastparcel_properties_moist_scalar():
    """Test FastParcel._properties_moist for scalar input."""
    height = 2000*units.meter
    z_init = 3000*units.meter
    t_initial = -2*units.celsius
    q_initial = 0.004751707262581661*units.dimensionless
    l_initial = 2e-3*units.dimensionless
    rate = 0.5/units.km
    theta_e = sydneyfast.parcel_equivalent_potential_temperature(
        z_init, t_initial, q_initial, rate)
    total_water = sydneyfast.water_content(
        z_init, q_initial, l_initial, rate)

    actual_t, actual_q, actual_l = sydneyfast._properties_moist(
        height, z_init, t_initial, theta_e, total_water)
    truth_t = 274.97291915*units.kelvin
    truth_q = 0.00554509*units.dimensionless
    truth_l = -0.00104177*units.dimensionless
    assert_almost_equal(actual_t, truth_t, 3)
    assert_almost_equal(actual_q, truth_q, 6)
    assert_almost_equal(actual_l, truth_l, 6)
    assert not hasattr(actual_t, 'size')
    assert not hasattr(actual_q, 'size')
    assert not hasattr(actual_l, 'size')

def test_fastparcel_properties_dry():
    """Test FastParcel._properties_dry."""
    height = [2000, 1000, 0]*units.meter
    z_init = 3000*units.meter
    t_initial = -2*units.celsius
    q_initial = 1e-3*units.dimensionless
    l_initial = 0*units.dimensionless
    rate = 0.5/units.km
    theta_e = sydneyfast.parcel_equivalent_potential_temperature(
        z_init, t_initial, q_initial, rate)
    total_water = sydneyfast.water_content(
        z_init, q_initial, l_initial, rate)

    actual_t, actual_q, actual_l = sydneyfast._properties_dry(
        height, z_init, t_initial, theta_e, total_water)
    truth_t = [280.5449438, 286.3411977, 293.36942736]*units.kelvin
    truth_q = [0.00101066, 0.00250059, 0.00409071]*units.dimensionless
    truth_l = [0., 0., 0.]*units.dimensionless
    assert_almost_equal(actual_t, truth_t, 3)
    assert_almost_equal(actual_q, truth_q, 6)
    assert_array_equal(actual_l, truth_l)

def test_fastparcel_properties_dry_scalar():
    """Test FastParcel._properties_dry for scalar input."""
    height = 2000*units.meter
    z_init = 3000*units.meter
    t_initial = -2*units.celsius
    q_initial = 1e-3*units.dimensionless
    l_initial = 0*units.dimensionless
    rate = 0.5/units.km
    theta_e = sydneyfast.parcel_equivalent_potential_temperature(
        z_init, t_initial, q_initial, rate)
    total_water = sydneyfast.water_content(
        z_init, q_initial, l_initial, rate)

    actual_t, actual_q, actual_l = sydneyfast._properties_dry(
        height, z_init, t_initial, theta_e, total_water)
    truth_t = 280.5449438*units.kelvin
    truth_q = 0.00101066*units.dimensionless
    truth_l = 0.*units.dimensionless
    assert_almost_equal(actual_t, truth_t, 3)
    assert_almost_equal(actual_q, truth_q, 6)
    assert_array_equal(actual_l, truth_l)
    assert not hasattr(actual_t, 'size')
    assert not hasattr(actual_q, 'size')
    assert not hasattr(actual_l, 'size')

def test_fastparcel_transition_point_dry():
    """Test FastParcel._transition_point for dry descent."""
    z_init = 3000*units.meter
    t_initial = -2*units.celsius
    q_initial = 1e-3*units.dimensionless
    l_initial = 0*units.dimensionless
    rate = 0.5/units.km
    theta_e = sydneyfast.parcel_equivalent_potential_temperature(
        z_init, t_initial, q_initial, rate)
    total_water = sydneyfast.water_content(
        z_init, q_initial, l_initial, rate)

    actual_z, actual_t = sydneyfast._transition_point(
        z_init, t_initial, l_initial, theta_e, total_water)
    assert_almost_equal(actual_z, z_init)
    assert_almost_equal(actual_t, t_initial)
    assert not hasattr(actual_z, 'size')
    assert not hasattr(actual_t, 'size')

def test_fastparcel_transition_point_mixed():
    """Test FastParcel._transition_point for mixed moist/dry descent."""
    z_init = 3000*units.meter
    t_initial = -2*units.celsius
    q_initial = 0.004751707262581661*units.dimensionless
    l_initial = 2e-3*units.dimensionless
    rate = 0.5/units.km
    theta_e = sydneyfast.parcel_equivalent_potential_temperature(
        z_init, t_initial, q_initial, rate)
    total_water = sydneyfast.water_content(
        z_init, q_initial, l_initial, rate)

    actual_z, actual_t = sydneyfast._transition_point(
        z_init, t_initial, l_initial, theta_e, total_water)
    truth_z = 2391.6137533395254*units.meter
    truth_t = 273.44171068846475*units.kelvin
    assert_almost_equal(actual_z, truth_z, 3)
    assert_almost_equal(actual_t, truth_t, 3)
    assert not hasattr(actual_z, 'size')
    assert not hasattr(actual_t, 'size')

def test_fastparcel_transition_point_moist():
    """Test FastParcel._transition_point for only moist descent."""
    z_init = 500*units.meter
    t_initial = 10*units.celsius
    q_initial = 0.008146560796663203*units.dimensionless
    l_initial = 2e-3*units.dimensionless
    rate = 0.5/units.km
    theta_e = sydneyfast.parcel_equivalent_potential_temperature(
        z_init, t_initial, q_initial, rate)
    total_water = sydneyfast.water_content(
        z_init, q_initial, l_initial, rate)

    actual_z, actual_t = sydneyfast._transition_point(
        z_init, t_initial, l_initial, theta_e, total_water)
    truth_z = 0*units.meter
    truth_t = 285.41367938613286*units.kelvin
    assert_almost_equal(actual_z, truth_z)
    assert_almost_equal(actual_t, truth_t, 3)
    assert not hasattr(actual_z, 'size')
    assert not hasattr(actual_t, 'size')

def test_fastparcel_properties_all_dry():
    """Test FastParcel.properties for only dry descent."""
    height = [2000, 1000, 0]*units.meter
    z_init = 3000*units.meter
    t_initial = -2*units.celsius
    q_initial = 1e-3*units.dimensionless
    l_initial = 0*units.dimensionless
    rate = 0.5/units.km
    theta_e = sydneyfast.parcel_equivalent_potential_temperature(
        z_init, t_initial, q_initial, rate)
    total_water = sydneyfast.water_content(
        z_init, q_initial, l_initial, rate)

    z_switch, t_switch = z_init, t_initial
    actual_t, actual_q, actual_l = sydneyfast.properties(
        height, z_init, t_initial, z_switch, t_switch, theta_e, total_water)
    actual_t_dry, actual_q_dry, actual_l_dry = sydneyfast._properties_dry(
        height, z_init, t_initial, theta_e, total_water)
    truth_t = [280.5449438, 286.3411977, 293.36942736]*units.kelvin
    truth_q = [0.00101066, 0.00250059, 0.00409071]*units.dimensionless
    truth_l = [0., 0., 0.]*units.dimensionless

    assert_almost_equal(actual_t, truth_t, 3)
    assert_almost_equal(actual_q, truth_q, 6)
    assert_array_equal(actual_l, truth_l)
    assert_array_equal(actual_t, actual_t_dry)
    assert_array_equal(actual_q, actual_q_dry)
    assert_array_equal(actual_l, actual_l_dry)

def test_fastparcel_properties_all_moist():
    """Test FastParcel.properties for only moist descent."""
    height = [400, 200, 0]*units.meter
    z_init = 500*units.meter
    t_initial = 10*units.celsius
    q_initial = 0.008146560796663203*units.dimensionless
    l_initial = 2e-3*units.dimensionless
    rate = 0.5/units.km
    theta_e = sydneyfast.parcel_equivalent_potential_temperature(
        z_init, t_initial, q_initial, rate)
    total_water = sydneyfast.water_content(
        z_init, q_initial, l_initial, rate)

    z_switch, t_switch = sydneyfast._transition_point(
        z_init, t_initial, l_initial, theta_e, total_water)
    actual_t, actual_q, actual_l = sydneyfast.properties(
        height, z_init, t_initial, z_switch, t_switch, theta_e, total_water)
    truth_t = [283.6021255, 284.50188453, 284.26175698]*units.kelvin
    truth_q = [0.00829667, 0.00860362, 0.00939808]*units.dimensionless
    truth_l = [0.00167194, 0.00105108, 0.]*units.dimensionless

    assert_almost_equal(actual_t, truth_t, 3)
    assert_almost_equal(actual_q, truth_q, 6)
    assert_almost_equal(actual_l, truth_l, 6)

def test_fastparcel_properties_mixed():
    """Test FastParcel.properties for mixed moist/dry descent."""
    height = [2000, 1000, 0]*units.meter
    z_init = 3000*units.meter
    t_initial = -2*units.celsius
    q_initial = 0.004751707262581661*units.dimensionless
    l_initial = 2e-3*units.dimensionless
    rate = 0.5/units.km
    theta_e = sydneyfast.parcel_equivalent_potential_temperature(
        z_init, t_initial, q_initial, rate)
    total_water = sydneyfast.water_content(
        z_init, q_initial, l_initial, rate)

    z_switch, t_switch = sydneyfast._transition_point(
        z_init, t_initial, l_initial, theta_e, total_water)
    actual_t, actual_q, actual_l = sydneyfast.properties(
        height, z_init, t_initial, z_switch, t_switch, theta_e, total_water)
    truth_t = [277.59502539, 285.61272634, 292.96804134]*units.kelvin
    truth_q = [0.00450332, 0.00462289, 0.00537897]*units.dimensionless
    truth_l = [0., 0., 0.]*units.dimensionless

    assert_almost_equal(actual_t, truth_t, 3)
    assert_almost_equal(actual_q, truth_q, 6)
    assert_array_equal(actual_l, truth_l)

def test_fastparcel_properties_scalar():
    """Test FastParcel.properties for scalar input."""
    height = 2000*units.meter
    z_init = 3000*units.meter
    t_initial = -2*units.celsius
    q_initial = 0.004751707262581661*units.dimensionless
    l_initial = 2e-3*units.dimensionless
    rate = 0.5/units.km
    theta_e = sydneyfast.parcel_equivalent_potential_temperature(
        z_init, t_initial, q_initial, rate)
    total_water = sydneyfast.water_content(
        z_init, q_initial, l_initial, rate)

    z_switch, t_switch = sydneyfast._transition_point(
        z_init, t_initial, l_initial, theta_e, total_water)
    actual_t, actual_q, actual_l = sydneyfast.properties(
        height, z_init, t_initial, z_switch, t_switch, theta_e, total_water)
    truth_t = 277.59502539*units.kelvin
    truth_q = 0.00450332*units.dimensionless
    truth_l = 0*units.dimensionless

    assert_almost_equal(actual_t, truth_t, 3)
    assert_almost_equal(actual_q, truth_q, 6)
    assert_array_equal(actual_l, truth_l)
    assert not hasattr(actual_t, 'size')
    assert not hasattr(actual_q, 'size')
    assert not hasattr(actual_l, 'size')

def test_fastparcel_buoyancy():
    """Test FastParcel.buoyancy."""
    height = [2000, 1000, 0]*units.meter
    z_init = 3000*units.meter
    t_initial = -2*units.celsius
    q_initial = 0.004751707262581661*units.dimensionless
    l_initial = 5e-3*units.dimensionless
    rate = 0.5/units.km
    theta_e = sydneyfast.parcel_equivalent_potential_temperature(
        z_init, t_initial, q_initial, rate)
    total_water = sydneyfast.water_content(
        z_init, q_initial, l_initial, rate)

    z_switch, t_switch = sydneyfast._transition_point(
        z_init, t_initial, l_initial, theta_e, total_water)
    actual = sydneyfast.buoyancy(
        height, z_init, t_initial, z_switch, t_switch, theta_e, total_water)
    truth = [-0.10509032, 0.09942803, 0.0981763]*units('m/s^2')
    assert_almost_equal(actual, truth, 6)

def test_fastparcel_buoyancy_no_liquid_correction():
    """Test FastParcel.buoyancy with liquid_correction=False."""
    height = [2000, 1000, 0]*units.meter
    z_init = 3000*units.meter
    t_initial = -2*units.celsius
    q_initial = 0.004751707262581661*units.dimensionless
    l_initial = 5e-3*units.dimensionless
    rate = 0.5/units.km
    theta_e = sydneyfast.parcel_equivalent_potential_temperature(
        z_init, t_initial, q_initial, rate)
    total_water = sydneyfast.water_content(
        z_init, q_initial, l_initial, rate)

    z_switch, t_switch = sydneyfast._transition_point(
        z_init, t_initial, l_initial, theta_e, total_water)
    actual = sydneyfast.buoyancy(
        height, z_init, t_initial, z_switch, t_switch, theta_e, total_water,
        liquid_correction=False)
    truth = [-0.09753847, 0.09942803, 0.0981763]*units('m/s^2')
    assert_almost_equal(actual, truth, 6)

def test_fastparcel_buoyancy_scalar():
    """Test FastParcel.buoyancy for scalar input."""
    height = 2000*units.meter
    z_init = 3000*units.meter
    t_initial = -2*units.celsius
    q_initial = 0.004751707262581661*units.dimensionless
    l_initial = 5e-3*units.dimensionless
    rate = 0.5/units.km
    theta_e = sydneyfast.parcel_equivalent_potential_temperature(
        z_init, t_initial, q_initial, rate)
    total_water = sydneyfast.water_content(
        z_init, q_initial, l_initial, rate)

    z_switch, t_switch = sydneyfast._transition_point(
        z_init, t_initial, l_initial, theta_e, total_water)
    actual = sydneyfast.buoyancy(
        height, z_init, t_initial, z_switch, t_switch, theta_e, total_water)
    truth = -0.10509032*units('m/s^2')
    assert_almost_equal(actual, truth, 6)
    assert not hasattr(actual, 'size')

def test_fastparcel_motion_hit_ground():
    """Test FastParcel.motion for a parcel reaching the ground."""
    time = np.arange(0, 6*60, 60)*units.second
    z_init = 3000*units.meter
    w_init = 0*units('m/s')
    t_initial = -3*units.celsius
    q_initial = 0.004411511498126446*units.dimensionless
    l_initial = 5e-3*units.dimensionless
    rate = 0.5/units.km

    truth = {
        'height': [3000., 2778.55973069, 2107.58339093,
                   1095.91987609, 238.61726993, np.nan]*units.meter,
        'velocity': [0., -7.43016393, -14.95630254,
                     -16.31856942, -12.30368638, np.nan]*units('m/s'),
        'temperature': [-2.99991446, -2.082978, 0.7695405,
                        7.87594638, 15.43983267, np.nan]*units.celsius,
        'specific_humidity': [0.00441154, 0.00459455, 0.00520821,
                              0.00556812, 0.00587798, np.nan]*units(''),
        'liquid_ratio': [0.00499997, 0.00391575, 0.00117073,
                         0., 0., np.nan]*units(''),
        'density': [0.89587805, 0.91656296, 0.98256973,
                    1.08247004, 1.16842398, np.nan]*units('kg/m^3'),
        'buoyancy': [-0.12107267, -0.12494048, -0.12674655,
                     0.06644796, 0.06328712, np.nan]*units('m/s^2'),
        'neutral_buoyancy_time': 146.34774405984135*units.second,
        'hit_ground_time': 260.583987587335*units.second,
        'min_height_time': np.nan*units.second,
        'neutral_buoyancy_height': 1673.017976975257*units.meter,
        'neutral_buoyancy_velocity':-17.50973367207873*units('m/s'),
        'hit_ground_velocity': -10.87191203650364*units('m/s'),
        'min_height': np.nan*units.meter
    }

    actual = sydneyfast.motion(
        time, z_init, w_init, t_initial, q_initial, l_initial, rate)

    for truth_var, actual_var in zip(truth.values(), actual.__dict__.values()):
        assert_almost_equal(actual_var, truth_var, 4)

def test_fastparcel_motion_not_hit_ground():
    """Test FastParcel.motion for a parcel not reaching the ground."""
    time = np.arange(0, 6*60, 60)*units.second
    z_init = 3000*units.meter
    w_init = 0*units('m/s')
    t_initial = -1*units.celsius
    q_initial = 1e-3*units.dimensionless
    l_initial = 0*units.dimensionless
    rate = 0.5/units.km

    truth = {
        'height': [3000., 2963.87108081, 2867.13723384,
                   2751.59654342, 2673.04402007, np.nan]*units.meter,
        'velocity': [0., -1.17232744, -1.93045885,
                     -1.75348953, -0.76345234, np.nan]*units('m/s'),
        'temperature': [-1., -0.65626859, 0.2839892,
                        1.39450866, 2.11972233, np.nan]*units.celsius,
        'specific_humidity': [0.001, 0.00099521, 0.00098393,
                              0.0009731, 0.00096824, np.nan]*units(''),
        'liquid_ratio': [0., 0., 0., 0., 0., np.nan]*units(''),
        'density': [0.88668191, 0.88940537, 0.89698569,
                    0.90644589, 0.91293152, np.nan]*units('kg/m^3'),
        'buoyancy': [-0.02061958, -0.01737858, -0.00557845,
                     0.0107469, 0.02073522, np.nan]*units('m/s^2'),
        'neutral_buoyancy_time': 138.70476486187766*units.second,
        'hit_ground_time': np.nan*units.second,
        'min_height_time': 274.9106665963633*units.second,
        'neutral_buoyancy_height': 2830.3238593378405*units.meter,
        'neutral_buoyancy_velocity': -1.982695123108468*units('m/s'),
        'hit_ground_velocity': np.nan*units('m/s'),
        'min_height': 2659.5481080974773*units.meter
    }

    actual = sydneyfast.motion(
        time, z_init, w_init, t_initial, q_initial, l_initial, rate)

    for truth_var, actual_var in zip(truth.values(), actual.__dict__.values()):
        assert_almost_equal(actual_var, truth_var, 4)
