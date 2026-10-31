"""Parahydrogen properties via CoolProp, with a REFPROP-compatible interface.

MATLAB ``refpropm`` calls in LH2Simulate.m use mass-specific SI quantities
except pressure, which is kilopascal in and out. CoolProp's ``PropsSI`` is
pascal and joule per kilogram. The polynomials shipped with the MATLAB code
were fit to that mass-specific internal energy: saturated liquid at 20 K is
about -4.0 kJ/kg, and ``T = poly(u/1000)`` recovers ~20 K. CoolProp's default
reference state for ParaHydrogen matches those polynomials; this module pins
that default so a later CoolProp change is explicit.

Viscosity (and therefore Prandtl number, and parahydrogen conductivity once
the liquid correlation leaves its low-temperature range) is not implemented
for ParaHydrogen in CoolProp. Those values are taken from CoolProp's normal
Hydrogen fluid at the same temperature and vapor quality, or at the same
temperature and pressure when the state is not on the saturation curve.
Normal-hydrogen liquid viscosity at 20 K is about 1.39e-5 Pa·s versus the
REFPROP parahydrogen constant 1.354e-5 Pa·s used as the MATLAB placeholder.
"""

from __future__ import annotations

import math

import CoolProp.CoolProp as CP

_FLUID = "ParaHydrogen"
_TRANSPORT_FLUID = "Hydrogen"
_T_MIN = 13.81
_T_MAX = 32.93

# Pin the energy reference used by the MATLAB polynomials (u in J/kg).
CP.set_reference_state(_FLUID, "DEF")


def _clamp_T(T: float) -> float:
    return min(_T_MAX, max(_T_MIN, float(T)))


def _trunc_hundredths(value: float) -> float:
    """MATLAB ``fix(100*x)/100`` (truncate toward zero)."""
    return math.trunc(float(value) * 100.0) / 100.0


class ParaHydrogen:
    """Small wrapper around one CoolProp AbstractState pair.

    A single AbstractState is reused for speed. Callers that need several
    properties at one specification should use the bundle helpers rather than
    interleaving updates.
    """

    def __init__(self) -> None:
        self.para = CP.AbstractState("HEOS", _FLUID)
        self.h2 = CP.AbstractState("HEOS", _TRANSPORT_FLUID)
        # CoolProp 6.6 has no ParaHydrogen viscosity model. Conductivity works
        # for cold saturated liquid and raises once viscosity is required.
        self._para_mu_available = False
        self._kappa_unsupported = set()

    def _update(self, state, pair, a, b) -> None:
        state.update(pair, float(a), float(b))

    def density_TQ(self, T: float, Q: float) -> float:
        self._update(self.para, CP.QT_INPUTS, Q, _clamp_T(T))
        return self.para.rhomass()

    def density_TP(self, T: float, P: float) -> float:
        self._update(self.para, CP.PT_INPUTS, P, _clamp_T(T))
        return self.para.rhomass()

    def internal_energy_TQ(self, T: float, Q: float) -> float:
        self._update(self.para, CP.QT_INPUTS, Q, _clamp_T(T))
        return self.para.umass()

    def enthalpy_TQ(self, T: float, Q: float) -> float:
        self._update(self.para, CP.QT_INPUTS, Q, _clamp_T(T))
        return self.para.hmass()

    def enthalpy_TD(self, T: float, rho: float) -> float:
        self._update(self.para, CP.DmassT_INPUTS, rho, _clamp_T(T))
        return self.para.hmass()

    def pressure_TQ(self, T: float, Q: float) -> float:
        self._update(self.para, CP.QT_INPUTS, Q, _clamp_T(T))
        return self.para.p()

    def _flash_DU(self, rho: float, u: float):
        rho = max(float(rho), 1e-8)
        u = float(u)
        attempts = (
            (rho, u),
            (rho, _trunc_hundredths(u)),
            (_trunc_hundredths(rho), _trunc_hundredths(u)),
        )
        last = None
        for rho_i, u_i in attempts:
            try:
                self._update(self.para, CP.DmassUmass_INPUTS, rho_i, u_i)
                return rho_i, u_i
            except ValueError as exc:
                last = exc
        raise ValueError(f"ParaHydrogen flash(D, U) failed for rho={rho}, u={u}") from last

    def _flash_PU(self, P: float, u: float):
        P = max(float(P), 1.0)
        u = float(u)
        attempts = ((P, u), (P, _trunc_hundredths(u)), (P, u / 1.5 if u else u))
        last = None
        for P_i, u_i in attempts:
            try:
                self._update(self.para, CP.PUmass_INPUTS, P_i, u_i)
                return
            except ValueError as exc:
                last = exc
        raise ValueError(f"ParaHydrogen flash(P, U) failed for P={P}, u={u}") from last

    def temperature_DU(self, rho: float, u: float) -> float:
        self._flash_DU(rho, u)
        return _trunc_hundredths(self.para.T())

    def temperature_PU(self, P: float, u: float) -> float:
        self._flash_PU(P, u)
        return self.para.T()

    def quality_DU(self, rho: float, u: float) -> float:
        self._flash_DU(rho, u)
        return self.para.Q()

    def pressure_DU(self, rho: float, u: float) -> float:
        self._flash_DU(rho, u)
        return self.para.p()

    def density_PU(self, P: float, u: float) -> float:
        self._flash_PU(P, u)
        return self.para.rhomass()

    def phase_DU(self, rho: float, u: float) -> int:
        self._flash_DU(rho, u)
        return self.para.phase()

    def vapor_pressure(self, u: float, rho: float) -> float:
        """Nested ``vaporpressure(uv, rhov)`` from LH2Simulate.m, in pascal.

        Two-phase states use the saturation pressure at the flashed temperature.
        Single-phase states use P(density, internal energy). Quality strictly
        between 0 and 1 is the MATLAB two-phase test. CoolProp reports Q = -1
        off the dome.
        """
        self._flash_DU(rho, u)
        quality = self.para.Q()
        if 0.0 < quality < 1.0:
            temperature = self.para.T()
            return self.pressure_TQ(temperature, 1.0)
        return self.para.p()

    def _mu_kappa(self, Q_hint: float | None, P: float, T: float) -> tuple[float, float]:
        """Viscosity and conductivity.

        Parahydrogen viscosity is skipped (not in CoolProp). Conductivity is
        attempted once per rounded temperature; a failure is remembered so the
        ODE right-hand side does not pay for an exception on every call.
        """
        mu = None
        kappa = None
        key = (None if Q_hint is None else round(float(Q_hint), 1), round(float(T), 2))
        if self._para_mu_available:
            try:
                mu = self.para.viscosity()
            except ValueError:
                self._para_mu_available = False
        # Saturated-liquid conductivity is the ParaHydrogen correlation that
        # CoolProp can evaluate. Vapor conductivity asks for the missing
        # viscosity model, so those states go straight to Hydrogen.
        if Q_hint == 0.0 and key not in self._kappa_unsupported:
            try:
                kappa = self.para.conductivity()
            except ValueError:
                self._kappa_unsupported.add(key)
                kappa = None
        if mu is None or kappa is None:
            mu_h2, k_h2 = self._transport_from_hydrogen(T, Q_hint, P)
            if mu is None:
                mu = mu_h2
            if kappa is None:
                kappa = k_h2
        return mu, kappa

    def _transport_from_hydrogen(self, T: float, Q: float | None, P: float | None):
        T = _clamp_T(T)
        try:
            if Q is not None and 0.0 <= Q <= 1.0:
                self._update(self.h2, CP.QT_INPUTS, Q, T)
            else:
                # PT is refused when P sits on the saturation curve.
                self._update(self.h2, CP.PT_INPUTS, max(P or 1e5, 1.0), T)
        except ValueError:
            q_fallback = 0.0 if Q is not None and Q < 0.5 else 1.0
            self._update(self.h2, CP.QT_INPUTS, q_fallback, T)
        return self.h2.viscosity(), self.h2.conductivity()

    def transport_TQ(self, T: float, Q: float) -> dict:
        """Saturated transport. Thermodynamics from parahydrogen; mu/k may fall back."""
        T = _clamp_T(T)
        Q = min(1.0, max(0.0, float(Q)))
        self._update(self.para, CP.QT_INPUTS, Q, T)
        cp = self.para.cpmass()
        cv = self.para.cvmass()
        beta = self.para.isobaric_expansion_coefficient()
        mu, kappa = self._mu_kappa(Q_hint=Q, P=self.para.p(), T=T)
        return {
            "kappa": kappa,
            "mu": mu,
            "cp": cp,
            "cv": cv,
            "beta": beta,
            "Pr": mu * cp / kappa,
        }

    def transport_DU(self, rho: float, u: float) -> dict:
        """Single-phase (or flashed) transport at density and internal energy."""
        self._flash_DU(rho, u)
        T = self.para.T()
        P = self.para.p()
        Q = self.para.Q()
        cp = self.para.cpmass()
        cv = self.para.cvmass()
        beta = self.para.isobaric_expansion_coefficient()
        Q_hint = Q if 0.0 <= Q <= 1.0 else None
        mu, kappa = self._mu_kappa(Q_hint=Q_hint, P=P, T=T)
        return {
            "kappa": kappa,
            "mu": mu,
            "cp": cp,
            "cv": cv,
            "beta": beta,
            "Pr": mu * cp / kappa,
            "T": T,
            "Q": Q,
        }


_DEFAULT = ParaHydrogen()


def fluid() -> ParaHydrogen:
    return _DEFAULT
