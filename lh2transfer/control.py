"""Fill-regime logic from ``LH2Control.m``.

Vessel 1 is the trailer (ST) and vessel 2 is station storage (ET).
``lambdaE`` is the transfer-line valve fraction and ``lambdaV`` is the
vaporizer valve fraction.

``getETVentState`` is defined in the MATLAB file but is not called. ET venting
is toggled by the ODE event function instead. The Python port keeps both.
"""

from __future__ import annotations

from dataclasses import dataclass

from lh2transfer.parameters import LH2Model


@dataclass
class ControlInput:
    lambdaE: float
    lambdaV: float
    STVentState: float


def lh2_control(
    model: LH2Model,
    h_l2: float,
    p1: float,
    p2: float,
    et_fill_complete: float,
    st_vent_complete: float,
) -> ControlInput:
    """Return valve commands for the current ET liquid height and pressures.

    ``p2`` is unused, matching the MATLAB function (ET vent is not commanded here).
    """
    del p2  # retained in the signature so the call site matches LH2Control.m
    if h_l2 < 0.05 * model.H:
        return ControlInput(
            lambdaE=0.5,
            lambdaV=vaporizer_valve_state(model, p1, model.p_ST_slow),
            STVentState=st_vent_state(model, p1, model.p_ST_slow),
        )
    if h_l2 < 0.30 * model.H:
        return ControlInput(
            lambdaE=1.0,
            lambdaV=vaporizer_valve_state(model, p1, model.p_ST_fast),
            STVentState=st_vent_state(model, p1, model.p_ST_fast),
        )
    if h_l2 < 0.45 * model.H:
        return ControlInput(
            lambdaE=1.0,
            lambdaV=vaporizer_valve_state(model, p1, model.p_ST_slow),
            STVentState=et_fill_complete * float(p1 > model.p_ST_final),
        )
    return ControlInput(
        lambdaE=(1.0 - et_fill_complete),
        lambdaV=(1.0 - et_fill_complete) * vaporizer_valve_state(model, p1, model.p_ST_slow),
        STVentState=et_fill_complete * (1.0 - st_vent_complete),
    )


def st_vent_state(model: LH2Model, p1: float, threshold: float) -> float:
    """ST vent command.

    The MATLAB ``if / elseif`` tests ``p < 1.05*threshold`` first, so the
    deadband branch is unreachable: the valve is open only at or above
    ``1.05 * threshold``. That order is preserved.
    """
    if p1 < threshold + 0.05 * threshold:
        return 0.0
    if p1 > threshold - 0.05 * threshold:
        return 1.0
    return model.STVentState


def et_vent_state(model: LH2Model, p2: float) -> float:
    """Unused ET vent hysteresis from ``LH2Control.m`` (events drive the valve)."""
    if p2 < model.p_ET_low:
        return 0.0
    if p2 > model.p_ET_high:
        return 1.0
    return model.ETVentState


def vaporizer_valve_state(model: LH2Model, p1: float, p_set: float) -> float:
    """Proportional vaporizer opening below 98% of the trailer pressure setpoint."""
    if p1 < p_set - 0.02 * p_set:
        state = max(0.0, 10.0 * (p_set - p1) / p_set)
        return min(1.0, state)
    if p1 > p_set + 0.02 * p_set:
        return 0.0
    return model.VapValveState
