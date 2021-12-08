# Copyright (c) 2021 Thomas Schanzer.
# Distributed under the terms of the BSD 3-Clause License.
"""Tools for specifying environmental profiles in dparcel.

The Environment class allows the user to supply real atmospheric
sounding data to use for parcel calculations. Alternatively, the
idealised_sounding function can be used to generate an Environment
instance using an idealised sounding.
"""

import numpy as np

import metpy.calc as mpcalc
from metpy.units import units
from metpy.units import concatenate
import metpy.constants as const

from scipy.interpolate import interp1d
from scipy.optimize import root_scalar
from scipy.integrate import solve_ivp

from thermo import theta_e, wetbulb, temperature_change
from thermo import saturation_specific_humidity


class Environment:
    """
    Class for atmospheric sounding data.
    """

    def __init__(
            self, pressure, height, temperature, dewpoint, liquid_ratio=None,
            info='', name=''):
        """
        Instantiates an Environment.

        Args:
            pressure: Pressure array in the sounding.
            height: Height array in the sounding.
            temperature: Temperature array in the sounding.
            dewpoint: Dewpoint array in the sounding.
            liquid_ratio: Array of liquid water partial density to
                total density in the sounding (optional, defaults to
                all zero).
            info: Information to store with the sounding, e.g. date
                (optional)
            name: Short name for the sounding, e.g. 'Sydney' (optional).
        """

        # record input data as attributes for use by methods
        self._pressure_raw = pressure.to(units.mbar)
        self._height_raw = height.to(units.meter)
        self._height_raw -= np.min(self._height_raw)  # set z=0 at surface
        self._temperature_raw = temperature.to(units.celsius)
        self._dewpoint_raw = dewpoint.to(units.celsius)

        # if no liquid ratio profile is given, assume it is zero
        if liquid_ratio is None:
            self._liquid_ratio_raw = (
                np.zeros(self._pressure_raw.size)*units.dimensionless)
        else:
            if hasattr(liquid_ratio, 'units'):
                self._liquid_ratio_raw = liquid_ratio.to(units.dimensionless)
            else:
                self._liquid_ratio_raw = liquid_ratio*units.dimensionless

        self.info = info
        self.name = name
        
        # functions to interpolate input data, so variables are known
        # at any height

        self._p_to_T_interp = interp1d(
            self._pressure_raw.m, self._temperature_raw.m,
            fill_value='extrapolate')
        self._z_to_T_interp = interp1d(
            self._height_raw.m, self._temperature_raw.m,
            fill_value='extrapolate')

        self._p_to_Td_interp = interp1d(
            self._pressure_raw.m, self._dewpoint_raw.m,
            fill_value='extrapolate')
        self._z_to_Td_interp = interp1d(
            self._height_raw.m, self._dewpoint_raw.m,
            fill_value='extrapolate')

        self._p_to_l_interp = interp1d(
            self._pressure_raw.m, self._liquid_ratio_raw.m,
            fill_value='extrapolate')
        self._z_to_l_interp = interp1d(
            self._height_raw.m, self._liquid_ratio_raw.m,
            fill_value='extrapolate')

        self._p_to_z_interp = interp1d(
            self._pressure_raw.m, self._height_raw.m,
            fill_value='extrapolate')
        self._z_to_p_interp = interp1d(
            self._height_raw.m, self._pressure_raw.m,
            fill_value='extrapolate')

    def temperature_from_pressure(self, pressure):
        """Finds the environmental temperature at a given pressure."""

        temperature = self._p_to_T_interp(pressure.m_as(units.mbar))
        if temperature.size == 1:
            temperature = temperature.item()
        return temperature*units.celsius

    def dewpoint_from_pressure(self, pressure):
        """Finds the environmental dew point at a given pressure."""

        dewpoint = self._p_to_Td_interp(pressure.m_as(units.mbar))
        if dewpoint.size == 1:
            dewpoint = dewpoint.item()
        return dewpoint*units.celsius

    def liquid_ratio_from_pressure(self, pressure):
        """Finds the environmental liquid ratio at a given pressure."""

        liquid_ratio = self._p_to_l_interp(pressure.m_as(units.mbar))
        if liquid_ratio.size == 1:
            liquid_ratio = liquid_ratio.item()
        return liquid_ratio*units.dimensionless

    def pressure(self, height):
        """Finds the environmental pressure at a given height."""

        pressure = self._z_to_p_interp(height.m_as(units.meter))
        if pressure.size == 1:
            pressure = pressure.item()
        return pressure*units.mbar

    def height(self, pressure):
        """Finds the height at a given environmental pressure."""

        height = self._p_to_z_interp(pressure.m_as(units.mbar))
        if height.size == 1:
            height = height.item()
        return height*units.meter

    def temperature(self, height):
        """Finds the environmental temperature at a given height."""

        temperature = self._z_to_T_interp(height.m_as(units.meter))
        if temperature.size == 1:
            temperature = temperature.item()
        return temperature*units.celsius

    def dewpoint(self, height):
        """Finds the environmental dew point at a given height."""

        dewpoint = self._z_to_Td_interp(height.m_as(units.meter))
        if dewpoint.size == 1:
            dewpoint = dewpoint.item()
        return dewpoint*units.celsius

    def liquid_ratio(self, height):
        """Finds the environmental liquid ratio at a given height."""

        liquid_ratio = self._z_to_l_interp(height.m_as(units.meter))
        if liquid_ratio.size == 1:
            liquid_ratio = liquid_ratio.item()
        return liquid_ratio*units.dimensionless

    def wetbulb_temperature(self, height):
        """
        Finds the environmental wet-bulb temperature at a given height.
        
        Uses the approximation of eq. (39) in Bolton (1980) for
        equivalent potential temperature, and the approximation of
        Davies-Jones (2008) to find the wet bulb temperature.
        """

        pressure = self.pressure(height)
        temperature = self.temperature(height)
        dewpoint = self.dewpoint(height)
        specific_humidity = mpcalc.specific_humidity_from_dewpoint(
            pressure, dewpoint)
        ept = theta_e(pressure, temperature, specific_humidity)
        return wetbulb(pressure, ept, improve=True)

    def specific_humidity(self, height):
        """Finds the environmental specific humidity at a given height."""

        pressure = self.pressure(height)
        dewpoint = self.dewpoint(height)
        return mpcalc.specific_humidity_from_dewpoint(pressure, dewpoint)

    def density(self, height):
        """Finds the environmental density at a given height."""

        pressure = self.pressure(height)
        temperature = self.temperature(height)
        mixing_ratio = mpcalc.mixing_ratio_from_specific_humidity(
            self.specific_humidity(height))
        return mpcalc.density(pressure, temperature, mixing_ratio)
    
    
def idealised_sounding(relative_humidity, info='', name=''):
    """
    Creates an idealised sounding.
    
    The sounding has a 160 mbar thick boundary layer with a dry
    adiabatic temperature profile, a 10 mbar thick capping inversion
    and a moist adiabatic temperature profile above the boundary layer.
    The specific humidity is constant in the boundary layer, and the
    relative humidity is constant above the boundary layer.
    
    Args:
        relative_humidity: Relative humidity above the boundary layer.
        info: Information to store with the sounding, e.g. date
            (optional)
        name: Short name for the sounding, e.g. 'Sydney' (optional).
        
    Returns:
        An Environment instance.
    """
    
    # generate discrete temperature profile
    pressure = np.arange(1013.25, 200, -5)*units.mbar
    t_boundary_layer = mpcalc.dry_lapse(
        np.arange(1013.25, 1013.25 - 161, -5)*units.mbar, 20*units.celsius)
    t_capping = t_boundary_layer[-1] + [1.5, 3.0]*units.delta_degC
    t_remaining = mpcalc.moist_lapse(
        np.arange(1013.25 - 175, 200, -5)*units.mbar, t_capping[-1],
        reference_pressure=(1013.25 - 170)*units.mbar)
    temperature = concatenate([
        t_boundary_layer, t_capping, t_remaining,
    ])

    # generate discrete dew point profile
    q_boundary_layer = mpcalc.specific_humidity_from_mixing_ratio(
        6e-3*units.dimensionless)
    dewpoint_boundary_layer = mpcalc.dewpoint_from_specific_humidity(
        np.arange(1013.25, 1013.25 - 161, -5)*units.mbar, t_boundary_layer,
        np.ones(t_boundary_layer.size)*q_boundary_layer)
    dewpoint_remaining = mpcalc.dewpoint_from_relative_humidity(
        t_remaining, np.ones(t_remaining.size)*relative_humidity)
    # ensure the dewpoint is continuous across the capping inversion
    dewpoint_capping_top = mpcalc.dewpoint_from_relative_humidity(
        t_capping[-1], relative_humidity)
    dewpoint_capping = concatenate([
        (dewpoint_boundary_layer[-1].to(units.kelvin)
         + dewpoint_capping_top.to(units.kelvin))/2,
        dewpoint_capping_top,
    ])
    dewpoint = concatenate([
        dewpoint_boundary_layer, dewpoint_capping, dewpoint_remaining,
    ])
    
    # interpolate discrete profiles to give variables at any pressure
    temperature_interp = interp1d(
        pressure.m_as(units.pascal), temperature.m_as(units.kelvin),
        fill_value='extrapolate')
    dewpoint_interp = interp1d(
        pressure.m_as(units.pascal), dewpoint.m_as(units.kelvin),
        fill_value='extrapolate')
    
    # now solve the hydrostatic equation so the variables can be
    # expressed as functions of height
    def dzdp(pressure, height):
        """
        Calculates the rate of height change w.r.t. pressure, dz/dp.
        
        Args:
            pressure: The pressure at the point of interest, in Pa.
            height: The height of the point of interest, in m.
        Returns:
            The derivative dz/dp in m/Pa.
        """

        pressure = pressure*units.pascal
        temperature = temperature_interp(pressure.m)*units.kelvin
        dewpoint = dewpoint_interp(pressure.m)*units.kelvin

        specific_humidity = mpcalc.specific_humidity_from_dewpoint(
            pressure, dewpoint)
        mixing_ratio = mpcalc.mixing_ratio_from_specific_humidity(
            specific_humidity)
        density = mpcalc.density(pressure, temperature, mixing_ratio)
        dzdp = - 1 / (density.to(units.kg/units.meter**3).m * const.g)

        return dzdp

    height = solve_ivp(
        dzdp, (1013.25e2, np.min(pressure.m_as(units.pascal))),
        [0], t_eval=pressure.m_as(units.pascal)).y*units.meter
    
    return Environment(
        pressure, np.squeeze(height), temperature, dewpoint, info=info,
        name=name)
