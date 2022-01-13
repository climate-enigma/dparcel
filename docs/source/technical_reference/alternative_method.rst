An alternative method
=======================

:code:`dparcel.parcel.Parcel` functions well and produces reasonable results, but
the method it is built on has some inherent inefficiencies that would make it
unsuitable for applications requiring large amounts of calculation with low
computational cost, such as convection parametrisation in global climate
models.

:code:`dparcel.parcel.FastParcel` implements the far more elegant and efficient method
described by [sherwood_et_al_2013]_ in their Section 4 for homogeneous
parcels. There is very good agreement between :code:`dparcel.parcel.Parcel` and
:code:`dparcel.parcel.FastParcel`, and :code:`dparcel.parcel.FastParcel.motion` is
approximately twice as fast as :code:`dparcel.parcel.Parcel.motion`. The agreement is
best for small entrainment rates.

The major advantage of the new method is that the parcel's properties at any
height :math:`z` can be found without needing to calculate them at every
intermediate level between :math:`z_0` and :math:`z` in a stepwise fashion.
This eliminates any accumulation of errors and allows us to achieve the same
accuracy with fewer iterations. It is also not necessary to constantly check
whether evaporation or condensation must occur in the parcel at each
entrainment step.

The approach consists of the following steps.


Equivalent potential temperature
----------------------------------

[sherwood_et_al_2013]_ assumed that entrainment mixes equivalent potential
temperature between the parcel and its environment in the same way that it
mixes temperature and moisture, as expressed in their Equation (6):

.. math::

    \frac{\mathrm{d} \theta_e}{\mathrm{d} z}
    = - \epsilon (\theta_e - \bar{\theta_e})
    
where :math:`\theta_e` is the parcel equivalent potential temperature
and :math:`\bar{\theta_e}` is the corresponding environmental value at
the same height. The approach then seeks to use the known :math:`\theta_e`
to determine the temperature at any height.

This works for updrafts, but we must modify it for downdrafts so that the
:math:`\theta_e` approaches :math:`\bar{\theta_e}` regardless of the direction
of motion from the initial height :math:`z_0`:

.. math::

    \frac{\mathrm{d} \theta_e}{\mathrm{d} z}
    = - \epsilon (\theta_e - \bar{\theta_e}) \operatorname{sgn} (z - z_0).

This is implemented in
:code:`dparcel.parcel.FastParcel.equivalent_potential_temperature`.
At this stage, we have not explicitly accounted for the effect of introducing
environmental air on the rate of evaporation in the parcel, but since
:math:`\theta_e` is conserved both for adiabatic descent and evaporation at
constant pressure, any additional evaporation should not invalidate the result.


Total water content
---------------------
Pressure and equivalent potential temperature alone do not define a unique
temperature; we must also determine the specific humidity. As long as the
parcel descends moist adiabatically, we will have :math:`q = q^*(p,T)`, 
enabling the calculation of :math:`T`. However, there may come a point where
the saturation specific humidity becomes equal to the total amount of water
(liquid and vapour) in the parcel (i.e., no liquid is left). 
Beyond this point the descent will be dry adiabatic.

In order to find the transition point between moist and dry descent, we reason 
that since the total amount of water :math:`Q` in the parcel is conserved by
adiabatic descent and evaporation without entrainment, it is mixed by
entrainment in the same manner as :math:`\theta_e`:

.. math::

    \frac{\mathrm{d} Q}{\mathrm{d} z}
    = - \epsilon (Q - \bar{Q}) \operatorname{sgn}(z - z_0).

Once we have found :math:`Q(z)`, we may find the point at which 
:math:`Q(z) = q^*(p,T)` -- this is the transition point. Beyond this point we
will have :math:`q = Q(z) < q^*(p,T)`.


Temperature for moist descent
-------------------------------

We begin by finding temperature as a function of height, assuming moist
descent, by solving

.. math::

    \theta_e(p, T, q^*(p,T)) = \theta_e^{\mathrm{sol}}(z)
    
where :math:`\theta_e^{\mathrm{sol}}(z)` is the previously obtained parcel
value. We use Newton's method, with

.. math::

    T' = T
    - \frac{
        \theta_e(p, T, q^*(p,T)) - \theta_e^{\mathrm{sol}}(z)
    }{
        \frac{\partial}{\partial T} [\theta_e(p, T, q^*(p,T))]
    }.

We will first need to evaluate the saturation equivalent potential temperature
:math:`\theta_e^*(p,T) = \theta_e(p, T, q^*(p,T))` and its partial derivative
with respect to temperature.

We will use the approximation presented by [bolton_1980]_:

.. math::

    \theta_{E} = \theta_{DL}\exp\left[\left(\frac{3036 \text{ K}}{T_{L}}-1.78\right)r(1+0.448r)\right]

where :math:`r` is the mixing ratio and

.. math::

    \theta_{DL}=T_{K}\left(\frac{1000 \text{ mbar}}{p-e}\right)^\kappa \left(\frac{T_{K}}{T_{L}}\right)^{0.28r}

is the potential temperature at the LCL, where :math:`T_K` is the absolute
temperature and

.. math::

    T_{L}=\left( \frac{1}{T_{D}-56 \text{ K}}+\frac{ln(T_{K}/T_{D})}{800 \text{ K}} \right)^{-1}+56 \text{ K}

is the temperature at the LCL.

For a saturated parcel, we can see that :math:`T_K = T_D = T_L`. Thus

.. math::

    \theta_{DL} = T_K \left( \frac{1000 \text{ mbar}}{p - e_s} \right)^\kappa

and

.. math::

    \theta_{E}=\theta_{DL}\exp\left[\left(\frac{3036 \text{ K}}{T_K}-1.78\right)r_s(1+0.448r_s)\right].

We first note that
:math:`\partial \theta_E / \partial T' = \theta_E (\partial \log \theta_E / \partial T_K)`,
and

.. math::

    \begin{align}
        \log \theta_E &= \log \theta_{DL} + \left(\frac{3036 \text{ K}}{T_K}-1.78\right)r_s(1+0.448r_s) \\
        \Rightarrow \qquad
        \frac{\partial \log \theta_E}{\partial T_K}
        &= \frac{\partial \log \theta_{DL}}{\partial T_K}
        - \left( \frac{3036 \text{ K}}{T_K^2} - 1.78 \right) r_s (1+0.448r_s)
        + \left( \frac{3036 \text{ K}}{T_K} - 1.78 \right) \frac{\partial r_s}{\partial T_K} (1+2\times0.448r_s)
    \end{align}

and

.. math::

    \begin{align}
        \log \theta_{DL} &= \log T_K + \kappa \log \left( \frac{1000 \text{ mbar}}{p - e_s} \right) \\
        \Rightarrow \qquad
        \frac{\partial \log \theta_{DL}}{\partial T_K} &= \frac{1}{T_K}
        + \frac{\kappa}{p - e_s} \frac{\mathrm{d}e_s}{\mathrm{d} T_K}
    \end{align}

with

.. math::

    \begin{align}
        r_s &= \frac{\epsilon e_s}{p - e_s} \\
        \Rightarrow \qquad
        \frac{\partial r_s}{\partial T_K} &= \epsilon p \frac{\mathrm{d}e_s}{\mathrm{d} T_K} (p - e_s)^{-2}
    \end{align}

and, lastly,

.. math::

    \begin{align}
        e_s &= e_0 \exp \left( \frac{a(T_K - C)}{T_K - C + b} \right) \\
        \Rightarrow \qquad
        \frac{\mathrm{d}e_s}{\mathrm{d} T_K} &= \frac{ab}{(T_K - C + b)^2} e_s
    \end{align}

where :math:`a = 17.67`,
:math:`b = 243.5` K, :math:`e_0 = 6.112` mbar and :math:`C = 273.15` K.

We implement the calculation in
:code:`dparcel.thermo.saturation_equivalent_potential_temperature`.

We then continue with the Newton's method solution, using the approximation of
[davies-jones_2008]_ for the case of non-entraining moist pseudoadiabatic
descent as a first guess. This is implemented in
:code:`dparcel.parcel.FastParcel.properties_moist`.


Temperature for dry descent
-----------------------------

We may perform a similar computation, assuming dry descent, now solving

.. math::

    \theta_e(p, T, Q(z)) = \theta_e^{\mathrm{sol}}(z)

with Newton's method giving

.. math::

    T' = T - \frac{\theta_e(p, T, Q(z)) - \theta_e^{\mathrm{sol}}(z)}{\frac{\partial}{\partial T} [\theta_e(p, T, Q(z))]}.

We must first find :math:`\frac{\partial}{\partial T} [\theta_e(p, T, Q(z))]`.
Returning to the approximation of [bolton_1980]_, now with specific
humidity independent of temperature, we have

.. math::

    \frac{\partial \log \theta_E}{\partial T_K}
    =  \frac{\partial \log \theta_{DL}}{\partial T_K}
    - \frac{3036 \text{ K}}{T_L^2} r (1+0.448r) \frac{\partial T_L}{\partial T_K}.

Now,

.. math::

    \begin{align}
        \frac{\partial \log \theta_{DL}}{\partial T_K}
        &= \frac{1}{T_K} + \frac{0.28r}{T_K} - \frac{0.28r}{T_L} \frac{\partial T_L}{\partial T_K} \\
        &= \frac{1 + 0.28r}{T_K} - \frac{0.28r}{T_L} \frac{\partial T_L}{\partial T_K}
    \end{align}

and

.. math::
    \frac{\partial T_L}{\partial T_K}
    = - \left( \frac{1}{T_D - 56 \text{ K}} + \frac{\log T_K - \log T_D}{800 \text{ K}} \right)^{-2}
    \left(
        -\frac{1}{(T_D - 56 \text{ K})^2} \frac{\partial T_D}{\partial T_K}
        + \frac{1}{800 \text{ K}} \left( \frac{1}{T_K} - \frac{1}{T_D} \frac{\partial T_D}{\partial T_K} \right)
    \right).

Bolton's (10) for the saturation vapour pressure :math:`e_s` implies that

.. math::

    T_D = \frac{b \log(Ue_s/e_0)}{a - \log(Ue_s/e_0)} + C

with :math:`U` being the relative humidity.
Then

.. math::

    \begin{align}
        \frac{\partial T_D}{\partial T_K}
        &= \frac{(a - \log(Ue_s/e_0)b + b\log(Ue_s/e_0)}{(a - \log(Ue_s/e_0))^2} \frac{\partial \log e_s}{\partial T_K} \\
        &= \frac{ab}{(a - \log(Ue_s/e_0))^2} \frac{\partial \log e_s}{\partial T_K}
    \end{align}

Bolton's (10) in its original form gives

.. math::

    e_s = e_0 \exp \left( \frac{a(T_K - C)}{T_K - C + b} \right)

which implies

.. math::

    \frac{\partial \log e_s}{\partial T_K} = \frac{ab}{(T_K - C + b)^2}.

Finally, we can also substitute the exact relation

.. math::

    U = \frac{q}{1-q} \frac{p - e_s}{\epsilon e_s}

where :math:`\epsilon` is the ratio of the molar mass of dry air to the molar
mass of water vapour.

We use the non-entraining dry adiabatic value as a first guess.
This step is implemented in
:code:`dparcel.parcel.FastParcel.properties_dry`.


Transition between moist and dry descent
------------------------------------------

We then find the transition point where :math:`Q(z) = q^*(p,T)`
(or equivalently :math:`l(z) = Q(z) - q^*(p,T) = 0` where :math:`l` is the
liquid water mass ratio). This is implemented in
:code:`dparcel.parcel.FastParcel.transition_point`.


Temperature for general descent
---------------------------------
At last, we may combine moist and dry descent to find the final temperature,
specific humidity and liquid ratio as a function of height. This is implemented
in :code:`dparcel.parcel.FastParcel.properties`.


Buoyancy
----------

It is then a simple matter to find the buoyancy:

.. math::

    b = \frac{(1 - l) T_v - \bar{T_v}}{\bar{T_v}}g

where :math:`T_v` and :math:`\bar{T_v}` are the parcel and environmental
virtual temperatures. This is implemented in
:code:`dparcel.parcel.FastParcel.buoyancy`.


Motion
--------

Knowing the buoyancy as a function of height, we simply substitute the new
buoyancy function into the existing code from
:code:`dparcel.parcel.Parcel.motion` to simulating the parcel's motion.
This is implemented in :code:`dparcel.parcel.FastParcel.motion`.
