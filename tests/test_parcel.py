# Copyright (c) 2021 Thomas Schanzer.
# Distributed under the terms of the BSD 3-Clause License.
"""Tests for dparcel.parcel."""

import pytest
import pandas as pd
import numpy as np
from metpy.testing import assert_almost_equal, assert_array_equal
from metpy.units import units

from dparcel.parcel import (
    Parcel,
    FastParcel,
)

data = pd.read_csv('test_soundings/sydney_20210716_00Z.csv', header=0)
p_sydney, z_sydney, t_sydney, td_sydney = data.to_numpy().T
p_sydney *= units.mbar
z_sydney *= units.meter
t_sydney *= units.kelvin
td_sydney *= units.kelvin
sydney = Parcel(p_sydney[4:], z_sydney[4:], t_sydney[4:], td_sydney[4:])

def test_profile_nonmonotonic_height():
    """Test profile when height array is not monotonically decreasing."""
    height = [3000, 2000, 1000, 2000]*units.meter
    t_initial = -2*units.celsius
    q_initial = 1e-4*units.dimensionless
    l_initial = 0*units.dimensionless
    rate = 0.5/units.km

    with pytest.raises(ValueError):
        _, _, _ = sydney.profile(
            height, t_initial, q_initial, l_initial, rate)

def test_profile_height_above_reference():
    """Test profile when the final height is above the reference height."""
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

def test_profile_no_descent():
    """Test profile when no descent is needed."""
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

def test_profile_no_reference():
    """Test profile without a reference height."""
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

def test_profile_reference_equals_first_height():
    """Test profile when the reference height is the first final height."""
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

def test_profile_scalar_height():
    """Test profile when the final height is a scalar."""
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

def test_profile_initially_subsaturated():
    """Test profile for a subsaturated parcel."""
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

def test_pseudoadiabatic_profile_initially_saturated():
    """Test profile for a saturated parcel, pseudoadiabatic descent."""
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

def test_reversible_profile_initially_saturated():
    """Test profile for a saturated parcel, reversible adiabatic descent."""
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

def test_parcel_density_initially_subsaturated():
    """Test parcel_density for a subsaturated parcel."""
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

def test_pseudoadiabatic_parcel_density_initially_saturated():
    """Test parcel_density for a saturated parcel, pseudoadiabatic descent."""
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

def test_reversible_parcel_density_initially_saturated():
    """Test parcel_density for a saturated parcel, reversible descent."""
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

def test_parcel_density_no_liquid_correction():
    """Test parcel_density with liquid_correction=False."""
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

def test_buoyancy_initially_subsaturated():
    """Test buoyancy for a subsaturated parcel."""
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

def test_pseudoadiabatic_buoyancy_initially_saturated():
    """Test buoyancy for a saturated parcel, pseudoadiabatic descent."""
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

def test_reversible_buoyancy_initially_saturated():
    """Test buoyancy for a saturated parcel, reversible descent."""
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

def test_buoyancy_no_liquid_correction():
    """Test buoyancy with liquid_correction=False."""
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

def test_motion_hit_ground():
    """Test motion for a parcel reaching the ground."""
    time = np.arange(0, 6*60, 60)*units.second
    z_init = 3000*units.meter
    w_init = 0*units('m/s')
    t_initial = -3*units.celsius
    q_initial = 0.004411511498126446*units.dimensionless
    l_initial = 5e-3*units.dimensionless
    rate = 0.5/units.km

    truth = {
        'height': [3000., 2778.67358835, 2109.63439708,
                   1094.71442864, 272.06605591, np.nan]*units.meter,
        'velocity': [0., -7.42085741, -14.89655764,
                     -16.15247819, -11.36522977, np.nan]*units('m/s'),
        'temperature': [-3., -2.07130099, 0.77247491,
                        8.36316623, 15.41359266, np.nan]*units.celsius,
        'specific_humidity': [0.00441151, 0.00459864, 0.00521067,
                              0.00547946, 0.00580908, np.nan]*units(''),
        'liquid_ratio': [0.005, 0.0038927, 0.00113396,
                         0., 0., np.nan]*units(''),
        'density': [0.89587838, 0.91648691, 0.98227343,
                    1.08081409, 1.16398588, np.nan]*units('kg/m^3'),
        'buoyancy': [-0.12107617, -0.12425255, -0.12586574,
                     0.0826686, 0.07285039, np.nan]*units('m/s^2'),
        'neutral_buoyancy_time': 146.27410530315808*units.second,
        'hit_ground_time': 266.09873084685495*units.second,
        'min_height_time': np.nan*units.second,
        'neutral_buoyancy_height': 1676.8167200514147*units.meter,
        'neutral_buoyancy_velocity': -17.674183799733218*units('m/s'),
        'hit_ground_velocity': -9.473355071777094*units('m/s'),
        'min_height': np.nan*units.meter
    }

    actual = sydney.motion(
        time, z_init, w_init, t_initial, q_initial, l_initial, rate)

    for actual_var, truth_var in zip(truth.values(), actual.__dict__.values()):
        assert_almost_equal(actual_var, truth_var, 4)

def test_motion_not_hit_ground():
    """Test motion for a parcel not reaching the ground."""
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

    for actual_var, truth_var in zip(truth.values(), actual.__dict__.values()):
        assert_almost_equal(actual_var, truth_var, 4)
