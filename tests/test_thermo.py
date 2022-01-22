# Copyright (c) 2021 Thomas Schanzer.
# Distributed under the terms of the BSD 3-Clause License.
"""Tests for dparcel.thermo."""

import pytest
import shelve
from metpy.testing import assert_almost_equal, assert_array_equal
from metpy.units import units

from dparcel.thermo import (
    moist_lapse,
    temperature_change,
    saturation_specific_humidity,
    equivalent_potential_temperature,
    saturation_equivalent_potential_temperature,
    dcape_dcin,
    lcl_romps,
    wetbulb_romps,
    wetbulb_potential_temperature,
    wetbulb,
    reversible_lapse_daviesjones,
    reversible_lapse_saunders,
    descend,
    mix,
    equilibrate,
)

def test_moist_lapse_scalar():
    actual = moist_lapse(1000*units.mbar, 0*units.celsius, 900*units.mbar)
    truth = 5.140677691654389*units.celsius
    assert_almost_equal(actual, truth, 3)

def test_moist_lapse_up():
    pressure = [900, 800, 700]*units.mbar
    actual = moist_lapse(pressure, 0*units.celsius, 1000*units.mbar)
    truth = [
        -5.603133863724906, -12.223376120570379, -20.15342938802408,
    ]*units.celsius
    assert_almost_equal(actual, truth, 3)

def test_moist_lapse_down():
    pressure = [800, 900, 1000]*units.mbar
    actual = moist_lapse(pressure, -20*units.celsius, 700*units.mbar)
    truth = [
        -12.08128253224288, -5.472554931170237, 0.12013477440223141,
    ]*units.celsius
    assert_almost_equal(actual, truth, 3)

def test_moist_lapse_no_reference():
    pressure = [800, 900, 1000]*units.mbar
    actual = moist_lapse(pressure, -20*units.celsius)
    truth = [
        -20.0, -12.835349000048097, -6.699583983263153,
    ]*units.celsius
    assert_almost_equal(actual, truth, 3)

def moist_lapse_fast():
    pressure = [800, 900, 1000]*units.mbar
    actual = moist_lapse(
        pressure, -20*units.celsius, 700*units.mbar, method='fast')
    truth = [
        -12.127397603992893, -5.541946232250285, 0.048623197584600486,
    ]*units.celsius
    assert_almost_equal(actual, truth, 3)

def test_moist_lapse_fast_no_reference():
    pressure = [800, 900, 1000]*units.mbar
    actual = moist_lapse(pressure, -20*units.celsius, method='fast')
    truth = [
        -20.000637807046292, -12.873778773870667, -6.760509849394381,
    ]*units.celsius
    assert_almost_equal(actual, truth, 3)

def test_moist_lapse_invalid_method():
    with pytest.raises(ValueError):
        _ = moist_lapse(
            800*units.mbar, 0*units.celsius, 1000*units.mbar, method='hello')

def test_temperature_change():
    actual = temperature_change([1e-3, 2e-3, 3e-3]*units.dimensionless)
    truth = [
        -2.4892247336957456, -4.978449467391491, -7.467674201087237,
    ]*units.delta_degC
    assert_almost_equal(actual, truth, 3)

def test_saturation_specific_humidity():
    actual = saturation_specific_humidity(
        [1000, 500]*units.mbar, [20, -20]*units.celsius)
    truth = [0.01466435884730134, 0.001565585489192481]*units.dimensionless
    assert_almost_equal(actual, truth, 6)

def test_equivalent_potential_temperature():
    actual = equivalent_potential_temperature(
        [800, 700]*units.mbar, [-10, -20]*units.celsius,
        [2e-3, 1e-3]*units.dimensionless)
    truth = [286.32296194967887, 283.3410452993932]*units.kelvin
    assert_almost_equal(actual, truth, 3)

def test_equivalent_potential_temperature_prime():
    actual, actual_prime = equivalent_potential_temperature(
        [800, 700]*units.mbar, [-10, -20]*units.celsius,
        [2e-3, 1e-3]*units.dimensionless, prime=True)
    truth = [286.32296194967887, 283.3410452993932]*units.kelvin
    truth_prime = [1.062969006161107, 1.10586397901637]*units.dimensionless
    assert_almost_equal(actual, truth, 3)
    assert_almost_equal(actual_prime, truth_prime, 6)

def test_saturation_equivalent_potential_temperature():
    actual = saturation_equivalent_potential_temperature(
        [800, 700]*units.mbar, [-10, -20]*units.celsius)
    truth = [286.96464624491335, 283.6778433487204]*units.kelvin
    assert_almost_equal(actual, truth, 3)

def test_saturation_equivalent_potential_temperature_prime():
    actual, actual_prime = saturation_equivalent_potential_temperature(
        [800, 700]*units.mbar, [-10, -20]*units.celsius, prime=True)
    truth = [286.96464624491335, 283.6778433487204]*units.kelvin
    truth_prime = [1.5827669621042701, 1.3981892491580596]*units.dimensionless
    assert_almost_equal(actual, truth, 3)
    assert_almost_equal(actual_prime, truth_prime, 6)

def test_dcape_dcin():
    with shelve.open('tests/test_soundings') as db:
        env_idealised = db['idealised']
        env_real = db['real']

    actual_dcape_idealised, actual_dcin_idealised = dcape_dcin(env_idealised)
    actual_dcape_real, actual_dcin_real = dcape_dcin(env_real)
    truth_dcape_idealised = 230.4148507777093*units.meter**2/units.second**2
    truth_dcin_idealised = -34.705790189135776*units.meter**2/units.second**2
    truth_dcape_real = 123.01795081346282*units.meter**2/units.second**2
    truth_dcin_real = -450.43506401330427*units.meter**2/units.second**2

    assert_almost_equal(actual_dcape_idealised, truth_dcape_idealised, 1)
    assert_almost_equal(actual_dcin_idealised, truth_dcin_idealised, 1)
    assert_almost_equal(actual_dcape_real, truth_dcape_real, 1)
    assert_almost_equal(actual_dcin_real, truth_dcin_real, 1)

def test_lcl_romps():
    actual_p, actual_t = lcl_romps(
        [800, 700]*units.mbar, [-10, -20]*units.celsius,
        [2e-3, 1e-3]*units.dimensionless)
    truth_p = [782.3940262594958, 685.2531404951791]*units.mbar
    truth_t = [261.4853622159218, 251.61718308588488]*units.kelvin
    assert_almost_equal(actual_p, truth_p, 3)
    assert_almost_equal(actual_t, truth_t, 3)

def test_wetbulb_romps():
    actual = wetbulb_romps(
        800*units.mbar, -10*units.celsius, 1e-3*units.dimensionless)
    truth = 260.9764484774397*units.kelvin
    assert_almost_equal(actual, truth, 6)

def test_wetbulb_potential_temperature():
    actual = wetbulb_potential_temperature([300, 320, 340]*units.kelvin)
    truth = [
        8.107715823233654, 15.519649959977588, 21.107019431217317,
    ]*units.celsius
    assert_almost_equal(actual, truth, 3)

def test_wetbulb():
    pressure = [1000, 600, 300, 100]*units.mbar
    theta_e = [363.0, 328.0, 357.0, 431.0]*units.kelvin

    actual = wetbulb(pressure, theta_e)
    actual1 = wetbulb(pressure, theta_e, improve=1)
    actual_true = wetbulb(pressure, theta_e, improve=True)
    truth = [
        26.06076033323872, -3.3895261649780193,
        -24.76237583185447, -50.89530446287793,
    ]*units.celsius

    assert_almost_equal(actual, truth, 3)
    assert_array_equal(actual, actual1)
    assert_array_equal(actual, actual_true)

def test_wetbulb_improve_off():
    pressure = [1000, 600, 300, 100]*units.mbar
    theta_e = [363.0, 328.0, 357.0, 431.0]*units.kelvin

    actual = wetbulb(pressure, theta_e, improve=False)
    actual0 = wetbulb(pressure, theta_e, improve=0)
    truth = [
        25.689995821493326, -3.4649079014170994,
        -24.622202026610076, -50.96126616846111,
    ]*units.celsius

    assert_almost_equal(actual, truth, 3)
    assert_array_equal(actual, actual0)

def test_reversible_lapse_daviesjones_scalar():
    actual = reversible_lapse_daviesjones(
        1000*units.mbar, -20*units.celsius, 5e-3*units.dimensionless,
        reference_pressure=500*units.mbar)
    truth = 13.338636681996844*units.celsius
    assert_almost_equal(actual, truth, 3)

def test_reversible_lapse_daviesjones_up():
    actual = reversible_lapse_daviesjones(
        [600, 200]*units.mbar, 20*units.celsius, 1e-3*units.dimensionless,
        reference_pressure=1000*units.mbar)
    truth = [-0.13362856392554362, -58.00666967248112]*units.celsius
    assert_almost_equal(actual, truth, 3)

def test_reversible_lapse_daviesjones_down():
    actual = reversible_lapse_daviesjones(
        [700, 1000]*units.mbar, -20*units.celsius, 5e-3*units.dimensionless,
        reference_pressure=500*units.mbar)
    truth = [-2.5298259741789906, 13.338636681996844]*units.celsius
    assert_almost_equal(actual, truth, 3)

def test_reversible_lapse_daviesjones_noreference():
    actual = reversible_lapse_daviesjones(
        [500, 700, 1000]*units.mbar, -20*units.celsius,
        5e-3*units.dimensionless)
    truth = [
        -19.999998916160266, -2.5298259741789906, 13.338636681996844,
    ]*units.celsius
    assert_almost_equal(actual, truth, 3)

def test_reversible_lapse_saunders_scalar():
    actual = reversible_lapse_saunders(
        1000*units.mbar, -20*units.celsius, 5e-3*units.dimensionless,
        reference_pressure=500*units.mbar)
    truth = 13.338717604035764*units.celsius
    assert_almost_equal(actual, truth, 3)

def test_reversible_lapse_saunders_up():
    actual = reversible_lapse_saunders(
        [600, 200]*units.mbar, 20*units.celsius, 1e-3*units.dimensionless,
        reference_pressure=1000*units.mbar)
    truth = [-0.13350754454273783, -58.006586468688965]*units.celsius
    assert_almost_equal(actual, truth, 3)

def test_reversible_lapse_saunders_down():
    actual = reversible_lapse_saunders(
        [700, 1000]*units.mbar, -20*units.celsius, 5e-3*units.dimensionless,
        reference_pressure=500*units.mbar)
    truth = [-2.5298305799927334, 13.338717604035764]*units.celsius
    assert_almost_equal(actual, truth, 3)

def test_reversible_lapse_saunders_noreference():
    actual = reversible_lapse_saunders(
        [500, 700, 1000]*units.mbar, -20*units.celsius,
        5e-3*units.dimensionless)
    truth = [
        -20.0, -2.5298305799927334, 13.338717604035764,
    ]*units.celsius
    assert_almost_equal(actual, truth, 3)

def test_descend_dry():
    actual_t, actual_q, actual_l = descend(
        1000*units.mbar, -20*units.celsius, 1e-3*units.dimensionless,
        0*units.dimensionless, 500*units.mbar)
    truth_t = 308.5933065618629*units.kelvin
    truth_q = 1e-3*units.dimensionless
    truth_l = 0*units.dimensionless
    assert_almost_equal(actual_t, truth_t, 3)
    assert actual_q == truth_q
    assert actual_l == truth_l

def test_descent_moist_pseudoadiabatic():
    actual_t, actual_q, actual_l = descend(
        600*units.mbar, -20*units.celsius,
        0.001565585489192481*units.dimensionless,
        4e-3*units.dimensionless, 500*units.mbar)
    truth_t = -10.00587602262874*units.celsius
    truth_q = 0.002976636458009126*units.dimensionless
    truth_l = 0.0025889490311833555*units.dimensionless
    assert_almost_equal(actual_t, truth_t, 3)
    assert_almost_equal(actual_q, truth_q, 6)
    assert_almost_equal(actual_l, truth_l, 6)

def test_descent_moist_reversible():
    actual_t, actual_q, actual_l = descend(
        600*units.mbar, -20*units.celsius,
        0.001565585489192481*units.dimensionless,
        4e-3*units.dimensionless, 500*units.mbar, kind='reversible')
    truth_t = -10.167217087998836*units.celsius
    truth_q = 0.002938882218753567*units.dimensionless
    truth_l = 0.002626703270438914*units.dimensionless
    assert_almost_equal(actual_t, truth_t, 3)
    assert_almost_equal(actual_q, truth_q, 6)
    assert_almost_equal(actual_l, truth_l, 6)

def test_descent_mixed_pseudoadiabatic():
    actual_t, actual_q, actual_l = descend(
        800*units.mbar, -20*units.celsius,
        0.001565585489192481*units.dimensionless,
        4e-3*units.dimensionless, 500*units.mbar)
    truth_t = 278.7669639040529*units.kelvin
    truth_q = 0.005565585489192481*units.dimensionless
    truth_l = 0*units.dimensionless
    assert_almost_equal(actual_t, truth_t, 3)
    assert actual_q == truth_q
    assert actual_l == truth_l

def test_descent_mixed_reversible():
    actual_t, actual_q, actual_l = descend(
        800*units.mbar, -20*units.celsius,
        0.001565585489192481*units.dimensionless,
        4e-3*units.dimensionless, 500*units.mbar, kind='reversible')
    truth_t = 278.7647125951823*units.kelvin
    truth_q = 0.005565585489192481*units.dimensionless
    truth_l = 0*units.dimensionless
    assert_almost_equal(actual_t, truth_t, 3)
    assert actual_q == truth_q
    assert actual_l == truth_l

def test_descend_invalid_method():
    with pytest.raises(ValueError):
        _ = descend(
            800*units.mbar, -20*units.celsius,
            0.001565585489192481*units.dimensionless,
            4e-3*units.dimensionless, 500*units.mbar, kind='hello')

def test_mix():
    parcel = [270, 280, 290]*units.kelvin
    environment = [280, 280, 280]*units.kelvin
    rate = [0.5, 1, 0.1]*(1/units.km)
    dz = 1*units.km
    actual = mix(parcel, environment, rate, dz)
    truth = [275.0, 280.0, 289.0]*units.kelvin
    assert_array_equal(actual, truth)

def test_equilibrate_subsaturated_no_liquid():
    pressure = 1000*units.mbar
    t_initial = 10*units.celsius
    q_initial = 5e-3*units.dimensionless
    l_initial = 0*units.dimensionless
    actual_t, actual_q, actual_l = equilibrate(
        pressure, t_initial, q_initial, l_initial)
    assert actual_t == t_initial
    assert actual_q == q_initial
    assert actual_l == l_initial

def test_equilibrate_saturated_with_liquid():
    pressure = 1000*units.mbar
    t_initial = 10*units.celsius
    q_initial = 0.007668039921398619*units.dimensionless
    l_initial = 5e-3*units.dimensionless
    actual_t, actual_q, actual_l = equilibrate(
        pressure, t_initial, q_initial, l_initial)
    assert actual_t == t_initial
    assert actual_q == q_initial
    assert actual_l == l_initial

def test_equilibrate_subsaturated_with_high_liquid():
    pressure = 1000*units.mbar
    t_initial = 10*units.celsius
    q_initial = 0.006*units.dimensionless
    l_initial = 5e-3*units.dimensionless
    actual_t, actual_q, actual_l = equilibrate(
        pressure, t_initial, q_initial, l_initial)
    truth_t = 281.2471775273162*units.kelvin
    truth_q = 0.006740497179992207*units.dimensionless
    truth_l = 0.004259502820007793*units.dimensionless
    assert_almost_equal(actual_t, truth_t, 3)
    assert_almost_equal(actual_q, truth_q, 6)
    assert_almost_equal(actual_l, truth_l, 6)

def test_equilibrate_subsaturated_with_low_liquid():
    pressure = 1000*units.mbar
    t_initial = 10*units.celsius
    q_initial = 0.003*units.dimensionless
    l_initial = 1e-3*units.dimensionless
    actual_t, actual_q, actual_l = equilibrate(
        pressure, t_initial, q_initial, l_initial)
    truth_t = 280.5424120530777*units.kelvin
    assert_almost_equal(actual_t, truth_t, 3)
    assert actual_q == 4e-3*units.dimensionless
    assert actual_l == 0*units.dimensionless

def test_equilibrate_supersaturated_no_liquid():
    pressure = 1000*units.mbar
    t_initial = 10*units.celsius
    q_initial = 9e-3*units.dimensionless
    l_initial = 0*units.dimensionless
    actual_t, actual_q, actual_l = equilibrate(
        pressure, t_initial, q_initial, l_initial)
    truth_t = 284.5901525593089*units.kelvin
    truth_q = 0.008443611342606088*units.dimensionless
    truth_l = 0.0005563886573939116*units.dimensionless
    assert_almost_equal(actual_t, truth_t, 3)
    assert_almost_equal(actual_q, truth_q, 6)
    assert_almost_equal(actual_l, truth_l, 6)

def test_equilibrate_supersaturated_with_liquid():
    pressure = 1000*units.mbar
    t_initial = 10*units.celsius
    q_initial = 9e-3*units.dimensionless
    l_initial = 2e-3*units.dimensionless
    actual_t, actual_q, actual_l = equilibrate(
        pressure, t_initial, q_initial, l_initial)
    truth_t = 284.5901525593089*units.kelvin
    truth_q = 0.008443611342606088*units.dimensionless
    truth_l = 0.0025563886573939116*units.dimensionless
    assert_almost_equal(actual_t, truth_t, 3)
    assert_almost_equal(actual_q, truth_q, 6)
    assert_almost_equal(actual_l, truth_l, 6)
