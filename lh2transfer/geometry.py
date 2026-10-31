"""Geometry, nozzle flow, and REFPROP polynomial fits from the MATLAB code."""

from __future__ import annotations

import math


def dsqrt(x: float) -> float:
    """Directed square root, ``sqrt(|x|) * sign(x)``."""
    if x < 0.0:
        return -math.sqrt(-x)
    return math.sqrt(x)


def gas_flow(area: float, gamma: float, rho: float, p_up: float, p_down: float) -> float:
    """Choked or subsonic ideal-gas orifice flow, ``gasFlow.m``.

    Returns kg/s. A downstream pressure higher than upstream reverses the flow.
    ``gamma`` is the isentropic exponent stored on the model (5/3).
    """
    if p_up < p_down:
        return -gas_flow(area, gamma, rho, p_down, p_up)
    if p_down <= 0.0 or rho <= 0.0 or area <= 0.0 or p_up <= 0.0:
        return 0.0
    threshold = ((gamma + 1.0) / 2.0) ** (gamma / (gamma - 1.0))
    if p_up / p_down >= threshold:
        return area * math.sqrt(
            gamma * rho * p_up * (2.0 / (gamma + 1.0)) ** ((gamma + 1.0) / (gamma - 1.0))
        )
    return area * math.sqrt(
        2.0
        * rho
        * p_up
        * (gamma / (gamma - 1.0))
        * ((p_down / p_up) ** (2.0 / gamma) - (p_down / p_up) ** ((gamma + 1.0) / gamma))
    )


def cyl_v_to_h(volume: float, radius: float, length: float) -> float:
    """Liquid height in a horizontal cylinder, ``cylVToH.m``.

    Newton iteration copied from the MATLAB function. ``volume`` is m^3,
    ``radius`` and ``length`` are m, and the return value is m.
    """
    if volume <= 0.0:
        return 0.0
    area = math.pi * radius**2
    if volume >= area * length:
        return 2.0 * radius
    s = volume / length
    x = 0.01
    error = 1.0
    b = 1.0
    if s > area / 2.0:
        sup = abs(s - area)
    else:
        sup = s

    def fun(x_val: float) -> float:
        y = math.sqrt(radius**2 - x_val**2)
        return sup - b * (radius**2) * math.atan(y / x_val) + x_val * y

    # The MATLAB loop stops once the relative update drops under 1e-4.
    # Cap iterations so a bad initial volume cannot hang the ODE right-hand side.
    for _ in range(50):
        if error < 1e-4:
            break
        x_old = x
        y = math.sqrt(max(radius**2 - x**2, 0.0))
        f = fun(x)
        alpha = -((radius**2 - x**2) ** (-0.5)) - ((radius**2 - x**2) ** 0.5) / (x**2)
        supd = (
            -b * alpha * (radius**2) * (1.0 / (((y / x) ** 2) + 1.0))
            + ((radius**2 - x**2) ** 0.5)
            - (x**2) * ((radius**2 - x**2) ** (-0.5))
        )
        if supd == 0.0 or not math.isfinite(supd):
            break
        x = x - f / supd
        if x <= 1e-8:
            x = 1e-8
        if x >= radius:
            x = radius * (1.0 - 1e-8)
        error = abs((x_old - x) / x_old)
    if s >= area / 2.0:
        return radius + x
    return radius - x


def liquid_temperature_from_energy(u: float) -> float:
    """Cubic REFPROP fit used inside the ODE, clamped to [13.804, 32.93] K.

    ``u`` is J/kg. The polynomial is evaluated at ``u/1000`` (kJ/kg), matching
    ``LH2Simulate.m``.
    """
    x = u / 1000.0
    temperature = (
        1.44867559e-07 * x**3
        - 2.53438808e-04 * x**2
        + 1.05449468e-01 * x
        + 2.03423757e01
    )
    if temperature < 13.804:
        return 13.804
    if temperature > 32.93:
        return 32.93
    return temperature


def liquid_temperature_from_energy_quadratic(u: float) -> float:
    """Quadratic fit used only in ``Data_extraction.m``."""
    x = u / 1000.0
    return -0.0002041552 * x**2 + 0.1010598604 * x + 20.3899281428


def liquid_density_from_energy(u: float) -> float:
    """Saturated-liquid density (kg/m^3) cubic fit from REFPROP 9.1."""
    x = u / 1000.0
    rho = (
        -5.12074746e-07 * x**3
        - 1.56628367e-05 * x**2
        - 1.18436797e-01 * x
        + 7.06218354e01
    )
    if rho < 1.0:
        return 1.0
    if rho > 90.0:
        return 90.0
    return rho


def enthalpy_of_vaporization(film_temperature: float) -> float:
    """Latent heat correlation, J/kg, from ``LH2Simulate.m`` (already times 1000)."""
    ts = film_temperature
    return 1000.0 * (
        -0.002445451720487 * ts**6
        + 0.3629946692976 * ts**5
        - 22.28028769483 * ts**4
        + 723.6541112107 * ts**3
        - 13116.31006512 * ts**2
        + 125780.2915522 * ts
        - 498095.5392318
    )


def two_phase_liquid_density_from_temperature(temperature: float) -> float:
    """Liquid density used only to correct ET level when the ullage is two-phase."""
    tv = temperature
    return (
        -5.24588e-05 * tv**6
        + 7.39502e-03 * tv**5
        - 4.29976e-01 * tv**4
        + 1.31922e01 * tv**3
        - 2.25208e02 * tv**2
        + 2.02705e03 * tv
        - 7.43508e03
    )


def saturation_temperature_from_psia(psia: float) -> float:
    """Vapor temperature polynomial from ``inputs_TrailerToDewar.m``, including +0.1 K."""
    p = psia
    return 0.1 + (
        -1.603941638811e-11 * p**6
        + 7.830478134841e-09 * p**5
        - 1.549372675881e-06 * p**4
        + 1.614567978153e-04 * p**3
        - 9.861776990784e-03 * p**2
        + 4.314905904166e-01 * p
        + 1.559843335080e01
    )


def film_temperature(pressure: float, t_c: float, p_c: float, lam: float) -> float:
    """Osipov 2008 film temperature, ``T_c * (p/p_c)^(1/lambda)``."""
    ratio = max(pressure, 1.0) / p_c
    return t_c * ratio ** (1.0 / lam)


def wall_heat_capacity(temperature: float) -> tuple[float, float]:
    """304 stainless specific heat and d(cp)/dT, J/kg/K and J/kg/K^2.

    The MATLAB comment says J/K, but the wall energy balance multiplies by
    vessel mass, so the polynomial is used as a specific heat.
    """
    tw = temperature
    cp = (
        2.516173240451e-11 * tw**6
        - 2.695483209737e-08 * tw**5
        + 0.00001122596286143 * tw**4
        - 0.002261465800734 * tw**3
        + 0.214810433559 * tw**2
        - 5.41715155529 * tw
        + 51.75489930095
    )
    dcp = (
        6 * 2.516173240451e-11 * tw**5
        - 5 * 2.695483209737e-08 * tw**4
        + 4 * 0.00001122596286143 * tw**3
        - 3 * 0.002261465800734 * tw**2
        + 2 * 0.214810433559 * tw
        - 5.41715155529
    )
    return cp, dcp


def environmental_heat_leak(liquid_volume: float) -> float:
    """ET environmental heat leak correlation, W, with liquid volume in m^3."""
    vl = liquid_volume
    return -7.462776654302e-02 * vl**2 + 4.445867251697e00 * vl + 3.108170556297e01


def horizontal_interface_area(height: float, radius: float, length: float) -> float:
    """Free-surface area of a horizontal cylinder, m^2 (chord times length)."""
    if height <= 0.0 or height >= 2.0 * radius:
        return 0.0
    if height > radius:
        offset = height - radius
    else:
        offset = radius - height
    # Guard the sqrt when the height is within a rounding error of the wall.
    inside = 1.0 - (offset / radius) ** 2
    if inside <= 0.0:
        return 0.0
    chord = 2.0 * radius * math.sqrt(inside)
    return chord * length


def real_cbrt(value: float) -> float:
    """Real cube root. MATLAB's ``.^ (1/3)`` can go complex for negatives."""
    if value < 0.0:
        return -((-value) ** (1.0 / 3.0))
    return value ** (1.0 / 3.0)


def vapor_pressure_from_temperature(temperature: float, rho: float, hydrogen) -> float:
    """``vaporpressure.m``: saturation polynomial below Tsat(rho), else CoolProp P(T, rho).

    ``hydrogen`` is a ``ParaHydrogen`` instance. The polynomial branch returns pascal
    (the MATLAB expression is already scaled by 1e3 from a kilopascal fit).
    """
    import CoolProp.CoolProp as CP

    if rho < 0.0:
        rho = 0.0001
    t_sat = (
        -3.9389254667e-09 * rho**6
        + 1.0053641879e-06 * rho**5
        - 1.0304184083e-04 * rho**4
        + 5.3058942923e-03 * rho**3
        - 1.4792439609e-01 * rho**2
        + 2.2234419496 * rho
        + 1.7950995359e01
    )
    tv = temperature
    if tv > t_sat:
        if tv < 32.938 and tv > 32.937:
            tv = 32.937
        hydrogen._update(hydrogen.para, CP.DmassT_INPUTS, max(rho, 1e-8), tv)
        return hydrogen.para.p()
    return (
        1.6133821043e-1 * tv**3 - 6.9432088540 * tv**2 + 1.1373052580e2 * tv - 6.9558797798e2
    ) * 1.0e3
