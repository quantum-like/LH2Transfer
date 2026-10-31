"""ODE right-hand side and integrator for the LH2 transfer model.

The state vector and balances follow ``LH2Simulate.m``. A few MATLAB details
that change the numbers are kept on purpose:

* Liquid temperature and density come from the REFPROP polynomials of specific
  internal energy, not from an extra CoolProp inversion.
* Trailer vapor boundary-layer temperatures use ``U/1.5`` (ET layers do not).
* Trailer liquid conductivity is evaluated at ``TL2(nL1)``, as in the MATLAB
  source, and the trailer vapor conduction grid uses the constant ``c_v``.
* ST vent logic uses the ``if / elseif`` order from ``LH2Control.m``.
* ET venting is a terminal ODE event at the PRD set pressure and reset
  pressure, toggled between segments. Events at the restart time are ignored,
  matching ode15s.
* Vaporizer and ST-vent commands are remembered from one accepted time to the
  next. ``LH2Simulate.m`` assigns ``P.VapValveState`` and ``P.STVentState``
  "for the next iteration", but the struct is copied into the ODE function, so
  a MATLAB run keeps the initial deadband value (usually 0) and the trailer
  pressure stalls just under the delivery set point. The Python port keeps
  that memory so the proportional vaporizer can carry the trailer through
  60 psia and the transfer can start. Memory is committed only when time
  advances, so finite-difference Jacobian probes do not wipe it.

CoolProp replaces ``refpropm``. Pressures passed to CoolProp stay in pascal;
the MATLAB code divided by 1000 because ``refpropm`` wants kilopascal.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.integrate import solve_ivp

from lh2transfer.control import lh2_control
from lh2transfer.geometry import (
    cyl_v_to_h,
    dsqrt,
    enthalpy_of_vaporization,
    environmental_heat_leak,
    film_temperature,
    gas_flow,
    horizontal_interface_area,
    liquid_density_from_energy,
    liquid_temperature_from_energy,
    real_cbrt,
    two_phase_liquid_density_from_temperature,
    wall_heat_capacity,
)
from lh2transfer.parameters import LH2Model, nominal_model
from lh2transfer.props import fluid

DIAG_NAMES = (
    "Jcd1",
    "Jcd2",
    "AAA",
    "BBB",
    "CCC",
    "DDD",
    "EEE",
    "FFF",
    "GGG",
    "HHH",
    "III",
    "JJJ",
    "KKK",
    "LLL",
    "MMM",
    "NNN",
    "OOO",
    "PPP",
    "QQQ",
    "RRR",
    "SSS",
    "TTT",
    "UUU",
    "VVV",
    "WWW",
    "XXX",
)


def state_indices(nL1: int, nV1: int, nL2: int, nV2: int) -> dict:
    """0-based indices and slices matching the MATLAB column layout."""
    names = ["mL1"]
    names += [f"uL1_{i}" for i in range(nL1)]
    names += ["mv1"]
    names += [f"uv1_{i}" for i in range(nV1)]
    names += ["Ts1", "Jtr", "mVap", "Jboil", "mL2"]
    names += [f"uL2_{i}" for i in range(nL2)]
    names += ["mv2"]
    names += [f"uv2_{i}" for i in range(nV2)]
    names += ["Ts2", "Tw2"]
    names += list(DIAG_NAMES)
    index = {name: i for i, name in enumerate(names)}
    index["uL1"] = slice(index["uL1_0"], index["uL1_0"] + nL1)
    index["uv1"] = slice(index["uv1_0"], index["uv1_0"] + nV1)
    index["uL2"] = slice(index["uL2_0"], index["uL2_0"] + nL2)
    index["uv2"] = slice(index["uv2_0"], index["uv2_0"] + nV2)
    index["diag"] = slice(index["Jcd1"], index["XXX"] + 1)
    index["n"] = len(names)
    index["names"] = names
    return index


def initial_state(model: LH2Model) -> np.ndarray:
    idx = state_indices(model.nL1, model.nV1, model.nL2, model.nV2)
    y = np.zeros(idx["n"])
    y[idx["mL1"]] = model.mL10
    y[idx["uL1"]] = model.UL10
    y[idx["mv1"]] = model.mv10
    y[idx["uv1"]] = model.Uv10
    y[idx["Ts1"]] = model.Ts10
    y[idx["Jtr"]] = model.Jtr0
    y[idx["mVap"]] = model.mVap0
    y[idx["Jboil"]] = model.Jboil0
    y[idx["mL2"]] = model.mL20
    y[idx["uL2"]] = model.UL20
    y[idx["mv2"]] = model.mv20
    y[idx["uv2"]] = model.Uv20
    y[idx["Ts2"]] = model.Ts20
    y[idx["Tw2"]] = model.Tw20
    return y


def _conduction_grid(n: int, lmin: float) -> tuple[np.ndarray, np.ndarray]:
    """Boundary-layer thicknesses counted from the interface, as in the MATLAB loops."""
    l_half = np.empty(n)
    l_node = np.empty(n)
    l_node[0] = lmin / (1.0 + math.exp(math.pi / 2.0 / math.sqrt(n)))
    l_half[0] = lmin
    stretch = math.exp(math.pi / math.sqrt(n))
    for i in range(1, n):
        l_half[i] = l_half[i - 1] * stretch
        l_node[i] = math.sqrt(l_half[i - 1] * l_half[i])
    return l_node, l_half


@dataclass
class RHSDiagnostics:
    """Quantities from one right-hand-side evaluation (not all are integrated)."""

    pv1: float
    pv2: float
    pL1: float
    pL2: float
    hL1: float
    hL2: float
    JL1: float
    Jv1: float
    JL2: float
    Jv2: float
    Jtr: float
    Jvap: float
    Jboil: float
    Jcd1: float
    Jcd2: float
    TL1: np.ndarray
    Tv1: np.ndarray
    TL2: np.ndarray
    Tv2: np.ndarray
    Ts1: float
    Ts2: float
    Tw2: float
    rho_L1: float
    rho_L2: float
    rhov1: float
    rhov2: float
    lambdaE: float
    lambdaV: float
    st_vent: float
    st_ready: float
    et_fill_complete: float
    extra: dict = field(default_factory=dict)


class LH2RHS:
    """Right-hand side with the sticky fill flags from the MATLAB globals."""

    def __init__(self, model: LH2Model):
        self.model = model
        self.idx = state_indices(model.nL1, model.nV1, model.nL2, model.nV2)
        self.eos = fluid()
        self.et_vent_state = float(model.ETVentState)
        self.et_fill_complete = 0.0
        self.st_ready = 0.0
        self.st_vent_complete = 0.0
        self.vap_valve = float(model.VapValveState)
        self.st_vent_memory = float(model.STVentState)
        self._mem_t = -1.0
        self._mem_captured = False
        self._pending_vap = self.vap_valve
        self._pending_st = self.st_vent_memory
        self.last = None

    def et_pressure(self, y: np.ndarray) -> float:
        model = self.model
        idx = self.idx
        u_bulk = float(y[idx["uL2"]][-1])
        rho_l = liquid_density_from_energy(u_bulk)
        m_l = float(y[idx["mL2"]])
        v_l = m_l / rho_l
        ullage = model.VTotal2 - v_l
        if ullage <= 1e-8:
            return model.p_ET_high * 2.0
        rho_v = float(y[idx["mv2"]]) / ullage
        u_v = float(y[idx["uv2"]][-1])
        return self.eos.vapor_pressure(u_v, rho_v)

    def __call__(self, t: float, y: np.ndarray) -> np.ndarray:
        dydt, _diag = self.evaluate(t, y)
        return dydt

    def evaluate(self, t: float, y: np.ndarray) -> tuple[np.ndarray, RHSDiagnostics]:
        model = self.model
        idx = self.idx
        eos = self.eos
        nL1, nV1, nL2, nV2 = model.nL1, model.nV1, model.nL2, model.nV2

        mL1 = float(y[idx["mL1"]])
        uL1 = np.array(y[idx["uL1"]], dtype=float)
        mv1 = float(y[idx["mv1"]])
        uv1 = np.array(y[idx["uv1"]], dtype=float)
        Ts1 = float(y[idx["Ts1"]])
        Jtr = float(y[idx["Jtr"]])
        mVap = float(y[idx["mVap"]])
        Jboil = float(y[idx["Jboil"]])
        mL2 = float(y[idx["mL2"]])
        uL2 = np.array(y[idx["uL2"]], dtype=float)
        mv2 = float(y[idx["mv2"]])
        uv2 = np.array(y[idx["uv2"]], dtype=float)
        Ts2 = float(y[idx["Ts2"]])
        Tw2 = float(y[idx["Tw2"]])

        if Ts1 < 14.0:
            Ts1 = 14.0
        if Ts2 < 14.0:
            Ts2 = 14.0
        if mL1 <= 0.0:
            raise ValueError("liquid mass equal to zero in (ST)")

        rho_L1 = liquid_density_from_energy(uL1[-1])
        VL1 = mL1 / rho_L1
        hL1 = cyl_v_to_h(VL1, model.R1, model.Lcyl)
        Vullage1 = model.VTotal1 - VL1
        if Vullage1 <= 1e-8:
            raise ValueError("trailer ullage volume vanished")
        rhov1 = mv1 / Vullage1

        Tv1 = np.zeros(nV1)
        Tv1[-1] = eos.temperature_DU(rhov1, uv1[-1])
        pv1 = eos.vapor_pressure(uv1[-1], rhov1)
        pL1 = rho_L1 * model.g * hL1
        pTotal1 = pv1 + pL1

        if pv1 > min(model.p_ST_slow, model.p_ST_fast):
            self.st_ready = 1.0
        if pv1 <= model.p_ST_final and self.et_fill_complete == 1.0:
            self.st_vent_complete = 1.0

        TL1 = np.array([liquid_temperature_from_energy(u) for u in uL1])

        for i in range(nV1 - 1):
            u_layer = uv1[i]
            if u_layer < 0.0:
                u_layer = 0.0
            # MATLAB divides the trailer boundary-layer energy by 1.5 before T(P, U).
            Tv1[i] = eos.temperature_PU(pv1, u_layer / 1.5)

        S1 = horizontal_interface_area(hL1, model.R1, model.Lcyl)

        rho_L2 = liquid_density_from_energy(uL2[-1])
        VL2 = mL2 / rho_L2
        Vullage2 = model.VTotal2 - VL2
        if Vullage2 <= 1e-8:
            raise ValueError("receiver ullage volume vanished")
        rhov2 = mv2 / Vullage2

        quality2 = eos.quality_DU(rhov2, uv2[-1])
        pv2 = eos.vapor_pressure(uv2[-1], rhov2)
        if pv2 == 0.0:
            pv2 = eos.vapor_pressure(math.trunc(100.0 * uv2[-1]) / 100.0, rhov2)

        Tv2 = np.zeros(nV2)
        Tv2[-1] = eos.temperature_DU(rhov2, uv2[-1])

        if 0.0 < quality2 < 1.0:
            liquid_density = two_phase_liquid_density_from_temperature(Tv2[-1])
            if liquid_density <= 1.0:
                liquid_density = rho_L2
            hL2 = (VL2 + (1.0 - quality2) * mv2 / liquid_density) / model.A2
        else:
            hL2 = VL2 / model.A2

        if hL2 > model.TopET * model.H:
            self.et_fill_complete = 1.0

        pL2 = rho_L2 * model.g * hL2
        pTotal2 = pv2 + pL2

        TL2 = np.array([liquid_temperature_from_energy(u) for u in uL2])
        for i in range(nV2 - 1):
            u_layer = uv2[i]
            if u_layer < 0.0:
                u_layer = 0.0
            Tv2[i] = eos.temperature_PU(pv2, u_layer)

        Ts10 = film_temperature(pv1, model.T_c, model.p_c, model.lambda_)
        Ts20 = film_temperature(pv2, model.T_c, model.p_c, model.lambda_)
        dTs1dt = (Ts10 - Ts1) / model.tminL1
        dTs2dt = (Ts20 - Ts2) / model.tminL2
        qh1 = enthalpy_of_vaporization(Ts1)
        qh2 = enthalpy_of_vaporization(Ts2)

        if t > self._mem_t + 1e-9:
            # Commit the command from the first evaluation at the previous time.
            self.vap_valve = self._pending_vap
            self.st_vent_memory = self._pending_st
            self._mem_t = t
            self._mem_captured = False
        elif t + 1e-9 < self._mem_t:
            # The stepper rejected a probe and moved backward. Keep the valve
            # committed before that probe and sample this earlier time instead.
            self._mem_t = t
            self._mem_captured = False
        saved_vap = model.VapValveState
        saved_st = model.STVentState
        model.VapValveState = self.vap_valve
        model.STVentState = self.st_vent_memory
        command = lh2_control(
            model, hL2, pv1, pv2, self.et_fill_complete, self.st_vent_complete
        )
        model.VapValveState = saved_vap
        model.STVentState = saved_st
        if not self._mem_captured:
            self._pending_vap = command.lambdaV
            self._pending_st = command.STVentState
            self._mem_captured = True

        apipe = (
            2.0
            * math.pi
            * (model.DPipe / 2.0) ** 2
            * math.sqrt(rho_L1 * model.DPipe / 2.0 / model.LPipe / model.f)
        )
        area_e = 2.0 * math.pi * (model.dE / 2.0) ** 2
        lambda_e = command.lambdaE
        alpha_e = area_e * math.sqrt(2.0 * rho_L1 / model.kE)
        if lambda_e <= 0.0:
            aeff = 0.0
        else:
            aeff = ((lambda_e * alpha_e) ** -2 + apipe**-2) ** -0.5
        Jtr0 = self.st_ready * aeff * dsqrt(pTotal1 - pTotal2)
        dJtrdt = (Jtr0 - Jtr) / model.tau_tr

        Jvap = abs(model.c_vap * command.lambdaV * dsqrt(2.0 * rho_L1 * (pTotal1 - model.p_atm)))
        if mVap <= 0.0:
            Jboil0 = 0.0
            Jboil = max(0.0, Jboil)
            dmVapdt = max(0.0, Jvap - Jboil)
        else:
            Jboil0 = Jvap
            dmVapdt = Jvap - Jboil
        dJboildt = (Jboil0 - Jboil) / model.tau_vap

        Jvvalve1 = command.STVentState * gas_flow(
            model.S_valve1, model.gamma_, rhov1, pv1, model.p_atm
        )
        Jvvalve2 = self.et_vent_state * gas_flow(
            model.S_valve2, model.gamma_, rhov2, pv2, model.p_atm
        )

        # Trailer transport. Conductivity uses TL2(nL1); other liquid properties use TL1.
        q_trailer = eos.quality_DU(rhov1, uv1[-1])
        if 0.0 < q_trailer < 1.0:
            vapor_tr = eos.transport_TQ(Tv1[-1], 1.0)
        else:
            vapor_tr = eos.transport_DU(rhov1, uv1[-1])
        # nL1 and nL2 are equal in the nominal grid; MATLAB indexes TL2(nL1).
        liquid_k_temperature = TL2[min(nL1, nL2) - 1]
        kappa_L_st = eos.transport_TQ(liquid_k_temperature, 0.0)["kappa"]
        liquid_tr = eos.transport_TQ(TL1[-1], 0.0)
        kappa_v = vapor_tr["kappa"]
        mu_v = vapor_tr["mu"]
        cv_v = vapor_tr["cv"]
        cp_v = vapor_tr["cp"]
        beta_v = vapor_tr["beta"]
        cv_L = liquid_tr["cv"]
        cp_L = liquid_tr["cp"]
        beta_L = liquid_tr["beta"]
        mu_L = liquid_tr["mu"]

        lmin = math.sqrt(kappa_v * model.tminV1 / model.c_v / rhov1)
        l_V1, l12_V1 = _conduction_grid(nV1, lmin)
        lmin = math.sqrt(kappa_L_st * model.tminL1 / cv_L / rho_L1)
        l_L1, l12_L1 = _conduction_grid(nL1, lmin)

        hVS1_cond = kappa_v / l12_V1[0]
        hVS1_conv = kappa_v * 0.156 * real_cbrt(
            model.g * beta_v * cp_v * rhov1**2 * (Ts1 - Tv1[-1]) / kappa_v / mu_v
        )
        hLS1_cond = kappa_L_st / l12_L1[0]
        hLS1_conv = kappa_L_st * 0.156 * real_cbrt(
            model.g
            * beta_L
            * cp_L
            * rho_L1**2
            * abs(TL1[-1] - Ts1)
            / kappa_L_st
            / mu_L
        )

        QdotLS1_cond = hLS1_cond * S1 * (TL1[0] - Ts1) - l_L1[0] * cp_L * rho_L1 * dTs1dt
        QdotLS1_conv = hLS1_conv * S1 * (TL1[0] - Ts1) * float(TL1[min(nL2, nL1) - 1] > Ts1)
        if QdotLS1_conv > 0.0:
            QdotLS1 = max(QdotLS1_conv, QdotLS1_cond)
        else:
            QdotLS1 = QdotLS1_cond

        QdotVS1_conv = hVS1_conv * S1 * (Tv1[0] - Ts1) * float(Ts1 > Tv1[-1])
        QdotVS1_cond = hVS1_cond * S1 * (Tv1[0] - Ts1) - l_V1[0] * cv_v * rhov1 * dTs1dt
        if QdotVS1_conv < 0.0:
            QdotVS1 = min(QdotVS1_conv, QdotVS1_cond)
        else:
            QdotVS1 = QdotVS1_cond

        # Receiver wall convection. Overwrites the transport properties, as MATLAB does.
        if 0.0 < quality2 < 1.0:
            vapor_et = eos.transport_TQ(Tv2[-1], 1.0)
        else:
            vapor_et = eos.transport_DU(rhov2, uv2[-1])
        Pr = vapor_et["Pr"]
        kappa_v = vapor_et["kappa"]
        mu_v = vapor_et["mu"]
        cv_v = vapor_et["cv"]
        cp_v = vapor_et["cp"]
        beta_v = vapor_et["beta"]
        nuv2 = mu_v / rhov2
        height_vapor = model.H - hL2
        Ra = abs(model.g * beta_v * (Tw2 - Tv2[-1]) * height_vapor**3 * Pr / nuv2)
        Psi = (1.0 + (0.492 / Pr) ** (9.0 / 16.0)) ** (-16.0 / 9.0)
        Nu = 0.68 + 0.503 * (Ra * Psi) ** 0.25

        liquid_et = eos.transport_TQ(TL2[-1], 0.0)
        Pr_L = liquid_et["Pr"]
        kappa_L = liquid_et["kappa"]
        mu_L = liquid_et["mu"]
        cv_L = liquid_et["cv"]
        cp_L = liquid_et["cp"]
        beta_L = liquid_et["beta"]
        nuL2 = mu_L / rho_L2

        height_liquid = hL2 if abs(hL2) > 1e-6 else 1e-6
        Ra_L_side = abs(model.g * beta_L * (Tw2 - TL2[-1]) * height_liquid**3 * Pr_L / nuL2)
        Psi_L_side = (1.0 + (0.492 / Pr_L) ** (9.0 / 16.0)) ** (-16.0 / 9.0)
        Nu_L_side = 0.68 + 0.503 * (Ra_L_side * Psi_L_side) ** 0.25
        Ra_L_bottom = abs(model.g * beta_L * (Tw2 - TL2[-1]) * (model.R2 / 2.0) ** 3 * Pr_L / nuL2)
        Nu_L_bottom = 0.27 * Ra_L_bottom**0.25

        hWV2 = Nu * kappa_v / height_vapor if abs(height_vapor) > 1e-6 else 0.0
        hWL2_side = Nu_L_side * kappa_L / height_liquid
        hWL2_bottom = Nu_L_bottom * kappa_L / (model.R2 / 2.0)
        QdotWL2 = (Tw2 - TL2[-1]) * (hWL2_bottom * model.A2 + hWL2_side * 2.0 * math.pi * model.R2 * hL2)
        QdotWV2 = hWV2 * (Tw2 - Tv2[-1]) * (model.A2 + 2.0 * math.pi * model.R2 * (model.H - hL2))

        lmin = math.sqrt(max(kappa_L, 1e-8) * model.tminL2 / max(cv_L, 1.0) / rho_L2)
        l_L2, l12_L2 = _conduction_grid(nL2, lmin)
        lmin = math.sqrt(max(kappa_v, 1e-8) * model.tminV2 / max(cv_v, 1.0) / rhov2)
        l_V2, l12_V2 = _conduction_grid(nV2, lmin)

        hVS2_cond = kappa_v / l12_V2[0]
        hVS2_conv = kappa_v * 0.156 * real_cbrt(
            model.g * beta_v * cp_v * rhov2**2 * (Ts2 - Tv2[-1]) / kappa_v / mu_v
        )
        hLS2_cond = kappa_L / l12_L2[0]
        hLS2_conv = kappa_L * 0.156 * real_cbrt(
            model.g * beta_L * cp_L * rho_L2**2 * abs(TL2[-1] - Ts2) / kappa_L / mu_L
        )

        QdotLS2_cond = hLS2_cond * model.A2 * (TL2[0] - Ts2) - l_L2[0] * cp_L * rho_L2 * dTs2dt
        QdotLS2_conv = hLS2_conv * model.A2 * (TL2[0] - Ts2) * float(TL2[-1] > Ts2)
        if QdotLS2_conv > 0.0:
            QdotLS2 = max(QdotLS2_conv, QdotLS2_cond)
        else:
            QdotLS2 = QdotLS2_cond

        QdotVS2_conv = hVS2_conv * model.A2 * (Tv2[0] - Ts2) * float(Ts2 > Tv2[-1])
        QdotVS2_cond = hVS2_cond * model.A2 * (Tv2[0] - Ts2) - l_V2[0] * cv_v * rhov2 * dTs2dt
        if QdotVS2_conv < 0.0:
            QdotVS2 = min(QdotVS2_conv, QdotVS2_cond)
        else:
            QdotVS2 = QdotVS2_cond

        Jcd1 = 0.0 if qh1 < 0.0 else -(QdotLS1 + QdotVS1) / qh1
        if qh2 <= 0.0:
            Jcd2 = 0.0
        else:
            Jcd2 = -(QdotLS2 + QdotVS2) / qh2 - 0.0 * model.ratio_top_bottom * Jtr

        Jv1 = Jboil - Jvvalve1 - Jcd1
        JL1 = -Jtr - Jvap + Jcd1
        Jv2 = model.ratio_top_bottom * Jtr - Jvvalve2 - Jcd2
        JL2 = (1.0 - model.ratio_top_bottom) * Jtr + Jcd2

        pdV1 = -pv1 * (JL1 / rho_L1)
        pdV2 = -pv2 * (JL2 / rho_L2)

        vv1 = Jvvalve1 / model.S_valve1 / rhov1
        vv2 = Jvvalve2 / model.S_valve2 / rhov2 if model.S_valve2 else 0.0

        TL_bulk = TL1[-1]
        if TL_bulk > 32.0:
            htr_L = model.c_L * TL_bulk
        else:
            if TL_bulk < 14.0:
                TL_bulk = 14.0
            htr_L = eos.enthalpy_TQ(TL_bulk, 0.0)

        hcd1 = model.c_p * Ts1 if Ts1 > 32.0 else eos.enthalpy_TQ(Ts1, 1.0)
        hcd2 = model.c_p * Ts2 if Ts2 > 32.0 else eos.enthalpy_TQ(Ts2, 1.0)
        hboil = model.c_p * model.Tboil if model.Tboil > 32.0 else eos.enthalpy_TQ(model.Tboil, 1.0)
        hvalve1 = eos.enthalpy_TD(Tv1[-1], rhov1)
        hvalve2 = eos.enthalpy_TD(Tv2[-1], rhov2)

        rhotr = rho_L1
        vtr = Jtr / (math.pi * (0.5 * model.dE) ** 2) / rhotr

        QdotV1 = (
            model.QdotEV1
            - QdotVS1
            - pdV1
            - Jvvalve1 * (hvalve1 + 0.5 * vv1**2)
            - Jcd1 * hcd1
            + Jboil * hboil
        )
        QdotL1 = (
            model.QdotEL1
            - QdotLS1
            + pdV1
            - Jtr * (htr_L + 0.5 * vtr**2)
            + Jcd1 * hcd1
            - Jvap * htr_L
        )
        QdotV2 = (
            QdotWV2
            - QdotVS2
            - pdV2
            + model.ratio_top_bottom * Jtr * (htr_L + 0.5 * vtr**2 - qh2)
            - Jvvalve2 * (hvalve2 + 0.5 * vv2**2)
            - Jcd2 * hcd2
        )
        QdotL2 = (
            QdotWL2
            - QdotLS2
            + pdV2
            + (1.0 - model.ratio_top_bottom) * Jtr * (htr_L + 0.5 * vtr**2)
            + Jcd2 * hcd2
        )

        duL1dt = np.zeros(nL1)
        for i in range(nL1 - 1):
            TL_im1 = Ts1 if i == 0 else TL1[i - 1]
            rho_i = eos.density_TQ(TL1[i], 0.0)
            duL1dt[i] = (
                ((TL1[i + 1] - TL1[i]) / l12_L1[i + 1] - (TL1[i] - TL_im1) / l12_L1[i])
                * kappa_L
                / (l_L1[i] * rho_i)
            )
        # P.kappa_L has already been overwritten by the ET liquid evaluation.
        duL1dt[-1] = (QdotL1 - JL1 * eos.internal_energy_TQ(TL1[-1], 0.0)) / mL1

        duv1dt = np.zeros(nV1)
        for i in range(nV1 - 1):
            Tv_im1 = Ts1 if i == 0 else Tv1[i - 1]
            u_layer = uv1[i] if uv1[i] >= 0.0 else 0.0
            rho_i = eos.density_PU(pv1, u_layer)
            duv1dt[i] = (
                ((Tv1[i + 1] - Tv1[i]) / l12_V1[i + 1] - (Tv1[i] - Tv_im1) / l12_V1[i])
                * kappa_v
                / (l_V1[i] * rho_i)
            )
        # Vapor-layer conduction uses the ET-updated kappa_v: MATLAB overwrites
        # P.kappa_v before this loop. Bulk energy uses U(T, rho), J/kg.
        if mv1 <= 1e-12:
            duv1dt[-1] = 0.0
        else:
            duv1dt[-1] = (QdotV1 - Jv1 * _umass_TD(eos, Tv1[-1], rhov1)) / mv1

        duL2dt = np.zeros(nL2)
        if mL2 <= 0.9:
            duL2dt[:] = 0.0
        else:
            for i in range(nL2 - 1):
                TL_im1 = Ts2 if i == 0 else TL2[i - 1]
                rho_i = eos.density_TQ(TL2[i], 0.0)
                duL2dt[i] = (
                    ((TL2[i + 1] - TL2[i]) / l12_L2[i + 1] - (TL2[i] - TL_im1) / l12_L2[i])
                    * kappa_L
                    / (l_L2[i] * rho_i)
                )
            duL2dt[-1] = (QdotL2 - JL2 * eos.internal_energy_TQ(TL2[-1], 0.0)) / mL2

        duv2dt = np.zeros(nV2)
        for i in range(nV2 - 1):
            Tv_im1 = Ts2 if i == 0 else Tv2[i - 1]
            u_layer = uv2[i] if uv2[i] >= 0.0 else 0.0
            rho_i = eos.density_PU(pv2, u_layer)
            duv2dt[i] = (
                ((Tv2[i + 1] - Tv2[i]) / l12_V2[i + 1] - (Tv2[i] - Tv_im1) / l12_V2[i])
                * kappa_v
                / (l_V2[i] * rho_i)
            )
        if mv2 <= 1e-12:
            duv2dt[-1] = 0.0
        else:
            duv2dt[-1] = (QdotV2 - Jv2 * _umass_TD(eos, Tv2[-1], rhov2)) / mv2

        cw2, dcw2 = wall_heat_capacity(Tw2)
        q_env = environmental_heat_leak(VL2)
        heat_capacity = model.mw2 * (cw2 + Tw2 * dcw2)
        if abs(heat_capacity) < 1e-6:
            dTw2dt = 0.0
        else:
            dTw2dt = (q_env - QdotWL2 - QdotWV2) / heat_capacity

        # Diagnostic filters. HHH uses MATLAB operator precedence:
        # (-Jtr*htr_L) + 0.5*vtr^2, not -Jtr*(htr_L + 0.5*vtr^2).
        AAA = Jvvalve1
        BBB = -QdotVS1
        CCC = pdV1
        DDD = Jvvalve1 * (hvalve1 + 0.5 * vv1**2)
        EEE = -Jcd1 * hcd1
        FFF = Jboil * hboil
        GGG = -QdotLS1
        HHH = -Jtr * htr_L + 0.5 * vtr**2
        III = Jcd1 * hcd1
        JJJ = -Jvap * htr_L
        KKK = QdotV1
        LLL = QdotL1
        MMM = QdotWV2
        NNN = -QdotVS2
        OOO = pdV2
        PPP = model.ratio_top_bottom * Jtr * (htr_L + 0.5 * vtr**2 - qh2)
        QQQ = -Jvvalve2 * (hvalve2 + 0.5 * vv2**2)
        RRR = -Jcd2 * hcd2
        SSS = QdotWL2
        TTT = -QdotLS2
        UUU = (1.0 - model.ratio_top_bottom) * Jtr * (htr_L + 0.5 * vtr**2)
        VVV = Jcd2 * hcd2
        WWW = QdotV2
        XXX = QdotL2
        diag_values = np.array(
            [
                Jcd1,
                Jcd2,
                AAA,
                BBB,
                CCC,
                DDD,
                EEE,
                FFF,
                GGG,
                HHH,
                III,
                JJJ,
                KKK,
                LLL,
                MMM,
                NNN,
                OOO,
                PPP,
                QQQ,
                RRR,
                SSS,
                TTT,
                UUU,
                VVV,
                WWW,
                XXX,
            ]
        )

        dydt = np.zeros(idx["n"])
        dydt[idx["mL1"]] = JL1
        dydt[idx["uL1"]] = duL1dt
        dydt[idx["mv1"]] = Jv1
        dydt[idx["uv1"]] = duv1dt
        dydt[idx["Ts1"]] = dTs1dt
        dydt[idx["Jtr"]] = dJtrdt
        dydt[idx["mVap"]] = dmVapdt
        dydt[idx["Jboil"]] = dJboildt
        dydt[idx["mL2"]] = JL2
        dydt[idx["uL2"]] = duL2dt
        dydt[idx["mv2"]] = Jv2
        dydt[idx["uv2"]] = duv2dt
        dydt[idx["Ts2"]] = dTs2dt
        dydt[idx["Tw2"]] = dTw2dt
        dydt[idx["diag"]] = diag_values - y[idx["diag"]]

        diag = RHSDiagnostics(
            pv1=pv1,
            pv2=pv2,
            pL1=pL1,
            pL2=pL2,
            hL1=hL1,
            hL2=hL2,
            JL1=JL1,
            Jv1=Jv1,
            JL2=JL2,
            Jv2=Jv2,
            Jtr=Jtr,
            Jvap=Jvap,
            Jboil=Jboil,
            Jcd1=Jcd1,
            Jcd2=Jcd2,
            TL1=TL1,
            Tv1=Tv1,
            TL2=TL2,
            Tv2=Tv2,
            Ts1=Ts1,
            Ts2=Ts2,
            Tw2=Tw2,
            rho_L1=rho_L1,
            rho_L2=rho_L2,
            rhov1=rhov1,
            rhov2=rhov2,
            lambdaE=command.lambdaE,
            lambdaV=command.lambdaV,
            st_vent=command.STVentState,
            st_ready=self.st_ready,
            et_fill_complete=self.et_fill_complete,
            extra={"Jtr0": Jtr0, "dmVapdt": dmVapdt, "dJboildt": dJboildt, "Jboil0": Jboil0},
        )
        self.last = diag
        return dydt, diag


def _umass_TD(eos, temperature: float, rho: float) -> float:
    """Internal energy at temperature and density, J/kg.

    MATLAB calls ``refpropm('U','T',T,'D',rho)``. CoolProp's equivalent is a
    DmassT flash. Density is kg/m^3.
    """
    import CoolProp.CoolProp as CP

    eos._update(eos.para, CP.DmassT_INPUTS, max(rho, 1e-8), min(32.93, max(13.81, temperature)))
    return eos.para.umass()


@dataclass
class SimulationResult:
    name: str
    t: np.ndarray
    y: np.ndarray
    et_vent_state: np.ndarray
    model: LH2Model
    index: dict

    def series(self, name: str) -> np.ndarray:
        slot = self.index[name]
        return self.y[:, slot]


def simulate(
    model: LH2Model | None = None,
    t_final: float | None = None,
    rtol: float | None = None,
    max_step: float = 1.0,
    method: str = "BDF",
) -> SimulationResult:
    """Integrate the nominal (or supplied) model.

    ``t_final`` overrides ``model.tFinal`` (3600 s for the trailer-to-dewar hour).
    The stepper is BDF with relative tolerance 1e-4 and a 1 s maximum step,
    which is the ode15s setup in ``LH2Simulate.m``.
    """
    model = nominal_model() if model is None else model
    if t_final is None:
        t_final = model.tFinal
    if rtol is None:
        rtol = model.relTol

    rhs = LH2RHS(model)
    y0 = initial_state(model)
    t0 = 0.0
    times = [t0]
    states = [y0.copy()]
    vents = [rhs.et_vent_state]
    t_start = t0
    y_start = y0
    # Ignore a terminal event that lands on the restart instant (ode15s behavior).
    min_event_delay = 1e-6

    while t_start < t_final - 1e-8:
        vent_state = rhs.et_vent_state

        def event_low(t, y, _rhs=rhs, _model=model):
            return _rhs.et_pressure(y) - _model.p_ET_low

        def event_high(t, y, _rhs=rhs, _model=model):
            return _rhs.et_pressure(y) - _model.p_ET_high

        event_low.terminal = bool(vent_state > 0.0)
        event_low.direction = -1.0
        event_high.terminal = True
        event_high.direction = 1.0

        sol = solve_ivp(
            rhs,
            (t_start, t_final),
            y_start,
            method=method,
            rtol=rtol,
            atol=1e-6,
            max_step=max_step,
            events=(event_low, event_high),
            dense_output=False,
        )
        if sol.status < 0:
            raise RuntimeError(f"LH2 integrator failed at t={t_start}: {sol.message}")

        # Drop the duplicated restart sample.
        t_seg = sol.t
        y_seg = sol.y
        if len(times) > 0 and abs(t_seg[0] - times[-1]) <= 1e-9:
            t_seg = t_seg[1:]
            y_seg = y_seg[:, 1:]
        for i in range(len(t_seg)):
            times.append(float(t_seg[i]))
            states.append(y_seg[:, i].copy())
            vents.append(vent_state)

        if sol.status == 0 or sol.t[-1] >= t_final - 1e-8:
            break

        t_event = float(sol.t[-1])
        if t_event <= t_start + min_event_delay:
            # Force a short step with events off so a root at t_start cannot spin.
            t_force = min(t_final, t_start + min(0.05, max_step))
            sol_force = solve_ivp(
                rhs,
                (t_start, t_force),
                y_start,
                method=method,
                rtol=rtol,
                atol=1e-6,
                max_step=max_step,
            )
            if sol_force.status < 0:
                raise RuntimeError(sol_force.message)
            t_start = float(sol_force.t[-1])
            y_start = sol_force.y[:, -1].copy()
            if len(times) == 0 or t_start > times[-1] + 1e-9:
                times.append(t_start)
                states.append(y_start.copy())
                vents.append(vent_state)
            continue

        t_start = t_event
        y_start = sol.y[:, -1].copy()
        rhs.et_vent_state = abs(rhs.et_vent_state - 1.0)

    return SimulationResult(
        name="fill from trailer to Dewar",
        t=np.asarray(times, dtype=float),
        y=np.vstack(states),
        et_vent_state=np.asarray(vents, dtype=float),
        model=model,
        index=rhs.idx,
    )
