"""Nominal trailer-to-dewar inputs from ``inputs_TrailerToDewar.m``."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from lh2transfer.geometry import film_temperature, saturation_temperature_from_psia
from lh2transfer.props import fluid


PSI_TO_PA = 6894.75729
GAL_TO_M3 = 0.00378541
IN_TO_M = 0.0254


@dataclass
class LH2Model:
    """Boundary conditions and geometry for one fill scenario.

    Field names follow the MATLAB ``LH2Model`` struct. Vessel 1 (ST) is the
    horizontal trailer. Vessel 2 (ET) is the vertical station storage.
    """

    p_atm: float = 1.01325e5
    g: float = 9.8

    VTotal1: float = 17000 * GAL_TO_M3
    VTotal2: float = 3300 * GAL_TO_M3
    R1: float = 1.5
    R2: float = 1.0

    T_c: float = 32.938
    p_c: float = 186.49 * PSI_TO_PA
    lambda_: float = 5.0
    rho_L: float = 70.9
    c_L: float = 9702.5
    kappa_L: float = 0.10061
    mu_L: float = 13.54e-6

    R_v: float = 4124.0
    c_v: float = 6490.0
    mu_v: float = 0.98e-6
    kappa_v: float = 0.0166

    nL1: int = 3
    tminL1: float = 0.1
    nL2: int = 3
    tminL2: float = 0.1
    nV1: int = 3
    tminV1: float = 0.1
    nV2: int = 4
    tminV2: float = 0.1

    p10: float = 14.7 * PSI_TO_PA
    TL10: float = 20.0
    totalmass10: float = 2000.0

    p20: float = 30.0 * PSI_TO_PA
    TL20: float = 20.4
    Tw20: float = 21.0
    pct_hL20: float = 0.01

    Jboil0: float = 0.0
    Jtr0: float = 0.0

    QdotEL1: float = 200.0
    QdotEV1: float = 40.0
    QdotEW2: float = 50.0
    mw2: float = 2529.0

    mVap0: float = 0.0
    Tboil: float = 22.0
    tau_vap: float = 0.2
    c_vap: float = 4e-4
    VapValveState: float = 0.0

    DPipe: float = 2 * IN_TO_M
    LPipe: float = 10.0
    drPipe: float = 1e-6
    tau_tr: float = 10.0
    dE: float = 0.4 * IN_TO_M
    kE: float = 4.0

    S_valve1: float = 0.002
    STVentState: float = 0.0
    S_valve2: float = math.pi * (0.4 / 10.0) ** 2
    ETVentState: float = 0.0

    p_ST_slow: float = 60.0 * PSI_TO_PA
    p_ST_fast: float = 60.0 * PSI_TO_PA
    p_ST_final: float = 20.0 * PSI_TO_PA
    p_ET_low: float = 43.0 * PSI_TO_PA
    p_ET_high: float = 45.0 * PSI_TO_PA

    TopET: float = 0.9
    ratio_top_bottom: float = 0.0

    tFinal: float = 60.0 * 60.0
    relTol: float = 1e-4

    # Filled by ``finalize_nominal``.
    A1: float = field(init=False)
    A2: float = field(init=False)
    Lcyl: float = field(init=False)
    H: float = field(init=False)
    c_p: float = field(init=False)
    gamma_: float = 5.0 / 3.0
    Gamma_: float = field(init=False)
    f: float = field(init=False)
    Tv10: float = field(init=False)
    Ts10: float = field(init=False)
    rhov10: float = field(init=False)
    rhoL10: float = field(init=False)
    Vullage10: float = field(init=False)
    mL10: float = field(init=False)
    mv10: float = field(init=False)
    Tv20: float = field(init=False)
    Ts20: float = field(init=False)
    hL20: float = field(init=False)
    VL20: float = field(init=False)
    rhoL20: float = field(init=False)
    mL20: float = field(init=False)
    rhov20: float = field(init=False)
    mv20: float = field(init=False)
    UL10: float = field(init=False)
    Uv10: float = field(init=False)
    UL20: float = field(init=False)
    Uv20: float = field(init=False)


def finalize_nominal(model: LH2Model) -> LH2Model:
    """Geometry, friction factor, and CoolProp initial states."""
    model.A1 = math.pi * model.R1**2
    model.A2 = math.pi * model.R2**2
    model.Lcyl = model.VTotal1 / model.A1
    model.H = model.VTotal2 / model.A2
    model.c_p = model.c_v + model.R_v
    model.Gamma_ = ((model.gamma_ + 1.0) / 2.0) ** (
        ((model.gamma_ + 1.0) / 2.0) / (model.gamma_ - 1.0)
    )
    # MATLAB: 1.3/log(D/2/eps)^2 with log = natural log, ^ binding tighter than /.
    model.f = 1.3 / math.log(model.DPipe / 2.0 / model.drPipe) ** 2

    eos = fluid()
    model.Tv10 = saturation_temperature_from_psia(model.p10 / PSI_TO_PA)
    model.Ts10 = film_temperature(model.p10, model.T_c, model.p_c, model.lambda_)
    model.rhov10 = eos.density_TP(model.Tv10, model.p10)
    model.rhoL10 = eos.density_TQ(model.TL10, 0.0)
    model.Vullage10 = (model.totalmass10 - model.rhoL10 * model.VTotal1) / (
        model.rhov10 - model.rhoL10
    )
    model.mL10 = model.rhoL10 * (model.VTotal1 - model.Vullage10)
    model.mv10 = model.totalmass10 - model.mL10
    model.UL10 = eos.internal_energy_TQ(model.TL10, 0.0)
    model.Uv10 = eos.internal_energy_TQ(model.Tv10, 1.0)

    model.Tv20 = saturation_temperature_from_psia(model.p20 / PSI_TO_PA)
    model.Ts20 = film_temperature(model.p20, model.T_c, model.p_c, model.lambda_)
    model.hL20 = model.pct_hL20 * model.H
    model.VL20 = model.hL20 * model.A2
    model.rhoL20 = eos.density_TQ(model.TL20, 0.0)
    model.mL20 = model.rhoL20 * model.VL20
    model.rhov20 = eos.density_TP(model.Tv20, model.p20)
    model.mv20 = model.rhov20 * (model.VTotal2 - model.VL20)
    model.UL20 = eos.internal_energy_TQ(model.TL20, 0.0)
    model.Uv20 = eos.internal_energy_TQ(model.Tv20, 1.0)
    return model


def nominal_model() -> LH2Model:
    """Trailer-to-dewar case solved by ``runNominal.m`` (one hour by default)."""
    return finalize_nominal(LH2Model())
