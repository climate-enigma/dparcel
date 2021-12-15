Introduction and theory
=========================

Along with updrafts, downdrafts---downward-moving masses of air---are
important features in the dynamics of the Earth's atmosphere;
they transport mass, momentum, heat and moisture vertically
and also generate and maintain storms [knupp_cotton_1985]_.

Indeed, one of the main objectives of present-day research into
downdraft dynamics is to improve the predictions of global climate
models [thayer-calder_2013]_, whose output informs the
understanding of the larger-scale
dynamics, including the pressing issue of anthropogenic climate
change. Specifically, the high computational cost of running a global
climate model over the necessarily large spatial domain and prediction
timescales constrains their maximum resolution, which is still too
coarse to describe convection. The models therefore employ schemes
known as *parametrisations* which estimate the effect of
convection on the state of the model using the information available
at each time step; an accurate estimation requires a strong
understanding of the factors that govern convection.

On a smaller scale, strong downdrafts that reach the Earth's surface
(*downbursts*) are known to cause significant damage to
man-made structures and create hazardous, or even deadly, conditions
for aircraft [thayer-calder_2013]_. Another aim of downdraft
research is therefore to understand the mechanisms that generate
such extreme events and improve the ability to predict them in advance.

Considering these motivations, the goal of this work is to gain
insight into which processes and conditions initiate, and which
maintain or inhibit, downdrafts. The approach will be to construct
a significantly simplified model of a downdraft using *parcel
theory*.

An air parcel is a mass of air with an imaginary flexible (but usually
closed) boundary; under the usual assumptions, its exact size and
shape are irrelevant. The only force assumed to act on the parcel is
the net buoyant force (per unit mass), given in accordance with
Archimedes' principle by

.. math:: b = \frac{\rho_E - \rho_P}{\rho_P} g.
    :label: buoyancy

If the parcel is lowered in the atmosphere to a location with a higher
pressure, the work done to compress it and any heat exchanged will
manifest as a change in its internal energy in accordance with the
first law of thermodynamics. The second key assumption of parcel theory
is that this process is adiabatic; this is valid due to the low
thermal conductivity of air.

The potential presence of water in gas, liquid and solid phases in the
parcel is a major complication; under the assumption that the parcel
remains in phase equilibrium (i.e., changes are slow enough for
excess liquid to evaporate if the vapour pressure is below the
saturation value), there are two modes of adiabatic descent the parcel
may undergo. If no liquid is present, the descent is *dry
adiabatic* and the rate of work on the parcel causes it to warm at
an approximate rate of 9.8 K/km.
If liquid is present, the descent is *moist adiabatic*:
progressive warming of the parcel raises its saturation vapour pressure,
allowing the liquid to progressively evaporate during descent,
with the necessary transfer of latent heat from the air to the water
creating an opposing cooling effect.

Moist adiabatic descent is commonly assumed to be either
*pseudoadiabatic*, in which case liquid water does not contribute
to the heat capacity of the parcel (as if it precipitates from the
parcel immediately upon condensation), or *reversible*, in
which case the liquid does contribute to the heat capacity.
A reversibly descending parcel warms at a slightly slower rate than a
pseudoadiabatically descending one due to its larger heat capacity
[saunders_1957]_.
Both modes were investigated, but reversible descent was ultimately
chosen as it is the more realistic case for a parcel known to retain
liquid water.

If the pressure and temperature of the parcel are thus known at any
point in its descent, its density may be calculated using the ideal
gas law,

.. math:: \rho = \frac{p}{RT_v}
    :label: density

where :math:`T_v` is the *virtual temperature* that contains a small
correction to account for the different density of water vapour.
If an mass :math:`l` of liquid water, per unit total parcel mass, is also
present, it is easily shown that (assuming the liquid occupies
negligible volume) the corrected parcel density is

.. math:: \rho = \frac{p}{RT_v (1 - l)}.

Knowledge of the parcel and environmental densities enable calculation
of the buoyant force per unit mass on the parcel using
:eq:`buoyancy`, and its resulting displacement and velocity may be
obtained by (numerically) solving the ODE

.. math:: \frac{\mathrm{d}^2 z}{\mathrm{d}t^2} = b(z).
    :label: ode
