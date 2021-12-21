# Copyright (c) 2021 Thomas Schanzer.
# Distributed under the terms of the BSD 3-Clause License.
"""Class for parcel theory calculations on real atmospheric soundings."""

import numpy as np

import metpy.calc as mpcalc
from metpy.units import units
from metpy.units import concatenate
import metpy.constants as const

from scipy.interpolate import interp1d
from scipy.integrate import solve_ivp

from .thermo import descend, equilibrate


class Parcel:
    """Class for parcel theory calculations with entrainment."""

    def __init__(self, environment):
        """
        Instantiate an EntrainingParcel.

        Args:
            environment: An instance of Environment on which the
                calculations are to be performed.
        """
        self._env = environment

    def _entrain_discrete(self, height, state, rate, step, kind='pseudo'):
        """
        Find parcel properties after descent/entrainment.

        Only valid for small steps.

        Args:
            height: Initial height.
            state: 3-tuple of initial temperature, specific humidity
                and liquid ratio.
            rate: Entrainment rate.
            step: Size of *downward* step, i.e. initial minus final height.
            kind: 'pseudo' for pseudoadiabats, 'reversible' for reversible
                adiabats.

        Returns:
            3-tuple of final temperature, specific humidity and liquid ratio.
        """
        t_parcel = state[0]
        q_parcel = state[1]
        l_parcel = state[2]
        p_initial = self._env.pressure(height)
        p_final = self._env.pressure(height - step)

        # steps 1 and 2: mixing and phase equilibration
        t_eq, q_eq, l_eq = equilibrate(
            p_initial, t_parcel, q_parcel, l_parcel,
            self._env.temperature(height), self._env.specific_humidity(height),
            self._env.liquid_ratio(height), rate, step)

        # step 3: dry or moist adiabatic descent
        t_final, q_final, l_final = descend(
            p_final, t_eq, q_eq, l_eq, p_initial, kind=kind)

        return (t_final, q_final, l_final)

    def profile(
            self, height, t_initial, q_initial, l_initial, rate,
            step=50*units.meter, reference_height=None, kind='pseudo'):
        """
        Calculate parcel properties for descent with entrainment.

        Valid for arbitrary steps.

        Args:
            height: Array of heights of interest.
            t_initial: Initial parcel temperature.
            q_initial: Initial parcel specific humidity.
            l_initial: Initial parcel liquid ratio.
            rate: Entrainment rate.
            step: Size of *downward* step for computing finite differences.
            kind: 'pseudo' for pseudoadiabats, 'reversible' for reversible
                adiabats.

        Returns:
            3-tuple containing the temperature, specific humidity and
                liquid ratio arrays for the given height array.
        """
        height = np.atleast_1d(height).m_as(units.meter)
        step = step.m_as(units.meter)
        if reference_height is not None:
            reference_height = reference_height.m_as(units.meter)
            if height.size == 1 and height.item() >= reference_height:
                # no descent needed, return initial values
                return t_initial, q_initial, l_initial

        # create height array with correct spacing
        if reference_height is None or reference_height == height[0]:
            all_heights = np.arange(height[0], height[-1], -step)
            all_heights = np.append(all_heights, height[-1])*units.meter
        else:
            all_heights = np.arange(reference_height, height[-1], -step)
            all_heights = np.append(all_heights, height[-1])*units.meter

        # calculate t, q and l one downward step at a time
        sol_states = [(t_initial, q_initial, l_initial)]
        for i in range(all_heights.size - 1):
            next_state = self._entrain_discrete(
                all_heights[i], sol_states[i], rate,
                all_heights[i] - all_heights[i+1], kind=kind)
            sol_states.append(next_state)

        if height.size == 1:
            return sol_states[-1]

        t_sol = concatenate(
            [state[0] for state in sol_states]).m_as(units.celsius)
        q_sol = concatenate([state[1] for state in sol_states]).m
        l_sol = concatenate([state[2] for state in sol_states]).m

        # find the values of t, q and l at the originally specified heights
        t_interp = interp1d(all_heights.m, t_sol)
        t_out = t_interp(height)*units.celsius
        q_interp = interp1d(all_heights.m, q_sol)
        q_out = q_interp(height)*units.dimensionless
        l_interp = interp1d(all_heights.m, l_sol)
        l_out = l_interp(height)*units.dimensionless

        return t_out, q_out, l_out

    def density(
            self, height, initial_height, t_initial, q_initial, l_initial,
            rate, step=50*units.meter, kind='pseudo', liquid_correction=True):
        """
        Calculate parcel density as a function of height.

        Args:
            height: Height of the parcel.
            initial_height: Initial height.
            t_initial: Initial temperature.
            q_initial: Initial specific humidity.
            l_initial: Initial liquid ratio.
            rate: Entrainment rate.
            step: Step size for entrainment calculation.
            kind: 'pseudo' for pseudoadiabats, 'reversible' for reversible
                adiabats.
            liquid_correction: Whether or not to account for the mass
                of liquid water.

        Returns:
            The density of the parcel at <height>.
        """
        t_final, q_final, l_final = self.profile(
            height, t_initial, q_initial, l_initial, rate, step=step,
            reference_height=initial_height, kind=kind)
        r_final = mpcalc.mixing_ratio_from_specific_humidity(q_final)
        p_final = self._env.pressure(height)

        gas_density = mpcalc.density(p_final, t_final, r_final)
        return gas_density/(1 - l_final.m*liquid_correction)

    def buoyancy(
            self, height, initial_height, t_initial, q_initial, l_initial,
            rate, step=50*units.meter, kind='pseudo', liquid_correction=True):
        """
        Calculate parcel buoyancy as a function of height.

        Args:
            height: Height of the parcel.
            initial_height: Initial height.
            t_initial: Initial temperature.
            q_initial: Initial specific humidity.
            l_initial: Initial liquid ratio.
            rate: Entrainment rate.
            step: Step size for entrainment calculation.
            kind: 'pseudo' for pseudoadiabats, 'reversible' for reversible
                adiabats.
            liquid_correction: Whether or not to account for the mass
                of liquid water.

        Returns:
            The buoyancy of the parcel at <height>.
        """
        env_density = self._env.density(height)
        parcel_density = self.density(
            height, initial_height, t_initial, q_initial, l_initial, rate,
            step, kind=kind, liquid_correction=liquid_correction)

        return (env_density - parcel_density)/parcel_density*const.g

    def motion(
            self, time, initial_height, initial_velocity, t_initial,
            q_initial, l_initial, rate, step=50*units.meter,
            kind='pseudo', liquid_correction=True):
        """
        Solve the equation of motion for the parcel.

        Integration stops if the parcel reaches a minimum height or
        the surface.

        Args:
            time: Array of times for which the results will be reported.
            initial_height: Initial height.
            initial_velocity: Initial vertical velocity.
            t_initial: Initial temperature.
            q_initial: Initial specific humidity.
            l_initial: Initial liquid ratio.
            rate: Entrainment rate.
            step: Step size for entrainment calculation.
            kind: 'pseudo' for pseudoadiabats, 'reversible' for reversible
                adiabats.
            liquid_correction: Whether or not to account for the mass
                of liquid water.

        Returns:
            Bunch object with the folliwing fields defined --
                - **height** -- Array of parcel height at each time step.
                - **velocity** -- Array of parcel velocity at each time step.
                - **temperature** -- Array of parcel temperature at each time
                  step.
                - **specific_humidity** -- Array of parcel specific humidity
                  at each time step.
                - **liquid_ratio** -- Array of parcel liquid water mass ratio
                  at each time step.
                - **density** -- Array of parcel density at each time step.
                - **buoyancy** -- Array of parcel buoyancy at each time step.
                - **neutral_buoyancy_time** -- The time at which the parcel
                  reached its neutral buoyancy level (np.nan if this did
                  not occur because the parcel reached the ground before
                  becoming neutrally buoyant).
                - **hit_ground_time** -- The time at which the parcel reached
                  the surface (np.nan if this did not occur because the
                  parcel stopped at some minimum height above the surface).
                - **min_height_time** -- The time at which the parcel reached
                  its minimum height (np.nan if this did not occur because
                  it reached the surface).
                - **neutral_buoyancy_height** -- The height of the neutral
                  buoyancy level (np.nan if it does not exist because the
                  parcel reached the ground before becoming neutrally
                  buoyant).
                - **neutral_buoyancy_velocity** -- The parcel's velocity at its
                  neutral buoyancy level (np.nan if this does not exist
                  because the parcel reached the ground before becoming
                  neutrally buoyant).
                - **hit_ground_velocity** -- The parcel's velocity at the
                  surface (np.nan if the parcel did not reach the surface).
                - **min_height** -- The minimum height reached by the parcel
                  (np.nan if it reached the surface).
        """
        # pre-compute temperature as a function of height to avoid
        # redundant calculations at every time step
        sample_heights = np.arange(
            initial_height.m_as(units.meter), 0,
            -step.m_as(units.meter))*units.meter
        sample_t, sample_q, sample_l = self.profile(
            sample_heights, t_initial, q_initial, l_initial, rate, step,
            kind=kind)

        def motion_ode(_, state):
            """Define the parcel's equation of motion."""
            height = np.max([state[0], 0])*units.meter

            # find the index of the closest height at which the temperature
            # was pre-computed
            closest_index = (
                sample_heights.size - 1
                - np.searchsorted(np.flip(sample_heights), height))

            # start from the pre-computed values and integrate the small
            # remaining distance to the desired level to find the buoyancy
            buoyancy = self.buoyancy(
                height, sample_heights[closest_index], sample_t[closest_index],
                sample_q[closest_index], sample_l[closest_index], rate, step,
                kind, liquid_correction)
            return [state[1], buoyancy.m]

        # event function for solve_ivp, zero when parcel reaches min height
        def min_height(_, state):
            return state[1]
        min_height.direction = 1  # find zero that goes from - to +
        min_height.terminal = True  # stop integration at minimum height

        # event function for solve_ivp, zero when parcel hits ground
        def hit_ground(_, state):
            return state[0]
        hit_ground.terminal = True  # stop integration at ground

        # event function for solve_ivp, zero when parcel is neutrally
        # buoyant
        def neutral_buoyancy(time, state):
            return motion_ode(time, state)[1]

        # solve the equation of motion
        initial_height = initial_height.m_as(units.meter)
        initial_velocity = initial_velocity.m_as(units.meter/units.second)
        time = time.to(units.second).m
        sol = solve_ivp(
            motion_ode,
            [np.min(time), np.max(time)],
            [initial_height, initial_velocity],
            t_eval=time,
            events=[neutral_buoyancy, hit_ground, min_height])

        # record height and velocity
        height = np.full(len(time), np.nan)
        velocity = np.full(len(time), np.nan)
        height[:len(sol.y[0, :])] = sol.y[0, :]
        velocity[:len(sol.y[1, :])] = sol.y[1, :]

        # record times of events
        # sol.t_events[i].size == 0 means the event did not occur
        neutral_buoyancy_time = (  # record only the first instance
            sol.t_events[0][0] if sol.t_events[0].size > 0 else np.nan)
        hit_ground_time = (
            sol.t_events[1][0] if sol.t_events[1].size > 0 else np.nan)
        min_height_time = (
            sol.t_events[2][0] if sol.t_events[2].size > 0 else np.nan)

        # record states at event times
        neutral_buoyancy_height = (  # record only the first instance
            sol.y_events[0][0, 0] if sol.y_events[0].size > 0 else np.nan)
        neutral_buoyancy_velocity = (  # record only the first instance
            sol.y_events[0][0, 1] if sol.y_events[0].size > 0 else np.nan)
        hit_ground_velocity = (
            sol.y_events[1][0, 1] if sol.y_events[1].size > 0 else np.nan)
        min_height_height = (
            sol.y_events[2][0, 0] if sol.y_events[2].size > 0 else np.nan)

        # compute parcel propterties for the solution
        temperature = np.full(len(time), np.nan)
        specific_humidity = np.full(len(time), np.nan)
        liquid_ratio = np.full(len(time), np.nan)
        density = np.full(len(time), np.nan)
        buoyancy = np.full(len(time), np.nan)

        t_profile, q_profile, l_profile = self.profile(
            sol.y[0, :]*units.meter, t_initial, q_initial, l_initial,
            rate, step, kind=kind)
        temperature[:len(sol.y[0, :])] = t_profile.m_as(units.celsius)
        specific_humidity[:len(sol.y[0, :])] = q_profile.m
        liquid_ratio[:len(sol.y[0, :])] = l_profile.m

        r_profile = mpcalc.mixing_ratio_from_specific_humidity(q_profile)
        p_profile = self._env.pressure(sol.y[0, :]*units.meter)
        gas_density = mpcalc.density(p_profile, t_profile, r_profile)
        density_profile = gas_density/(1 - l_profile.m*liquid_correction)
        density[:len(sol.y[0, :])] = (
            density_profile).m_as(units.kilogram/units.meter**3)

        env_density = self._env.density(sol.y[0, :]*units.meter)
        buoyancy[:len(sol.y[0, :])] = (
            (env_density - density_profile)/density_profile*const.g
        ).m_as(units.meter/units.second**2)

        # collect everything in a bunch object
        class MotionResult:
            """Container for calculation results."""

            def __init__(self):
                """Instantiates a MotionResult."""
                self.height = None
                self.velocity = None
                self.temperature = None
                self.specific_humidity = None
                self.liquid_ratio = None
                self.density = None
                self.buoyancy = None
                self.neutral_buoyancy_time = None
                self.hit_ground_time = None
                self.min_height_time = None
                self.neutral_buoyancy_height = None
                self.neutral_buoyancy_velocity = None
                self.hit_ground_velocity = None
                self.min_height = None

        result = MotionResult()
        result.height = height*units.meter
        result.velocity = velocity*units.meter/units.second
        result.temperature = temperature*units.celsius
        result.specific_humidity = specific_humidity*units.dimensionless
        result.liquid_ratio = liquid_ratio*units.dimensionless
        result.density = density*units.kilogram/units.meter**3
        result.buoyancy = buoyancy*units.meter/units.second**2
        result.neutral_buoyancy_time = neutral_buoyancy_time*units.second
        result.hit_ground_time = hit_ground_time*units.second
        result.min_height_time = min_height_time*units.second
        result.neutral_buoyancy_height = neutral_buoyancy_height*units.meter
        result.neutral_buoyancy_velocity = (
            neutral_buoyancy_velocity*units.meter/units.second)
        result.hit_ground_velocity = (
            hit_ground_velocity*units.meter/units.second)
        result.min_height = min_height_height*units.meter
        return result
