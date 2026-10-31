"""Post-processing translated from ``Data_extraction.m``.

The public MATLAB repository calls ``vaporpressure_bis(uv, rho)``, which is not
one of the shipped files. The nested energy-based ``vaporpressure`` inside
``LH2Simulate.m`` has that signature, and this module uses it. Temperatures of
the stored layers are recovered with CoolProp ``T(P, U)`` as in the extraction
script (the trailer boundary-layer ``U/1.5`` factor is only inside the ODE).
"""

from __future__ import annotations

import numpy as np

from lh2transfer.geometry import (
    cyl_v_to_h,
    gas_flow,
    liquid_density_from_energy,
    liquid_temperature_from_energy_quadratic,
)
from lh2transfer.props import fluid
from lh2transfer.simulate import SimulationResult


def extract(result: SimulationResult) -> dict:
    """Derived histories keyed like the MATLAB ``nominal`` struct."""
    model = result.model
    idx = result.index
    eos = fluid()
    n = len(result.t)
    nL1, nV1 = model.nL1, model.nV1
    nL2, nV2 = model.nL2, model.nV2

    mL1 = result.y[:, idx["mL1"]]
    mv1 = result.y[:, idx["mv1"]]
    mL2 = result.y[:, idx["mL2"]]
    mv2 = result.y[:, idx["mv2"]]
    uL1 = result.y[:, idx["uL1"]]
    uv1 = result.y[:, idx["uv1"]].copy()
    uL2 = result.y[:, idx["uL2"]]
    uv2 = result.y[:, idx["uv2"]].copy()

    rho_L1 = np.array([liquid_density_from_energy(u) for u in uL1[:, -1]])
    rho_L2 = np.array([liquid_density_from_energy(u) for u in uL2[:, -1]])
    VL1 = mL1 / rho_L1
    VL2 = mL2 / rho_L2
    Vullage1 = model.VTotal1 - VL1
    Vullage2 = model.VTotal2 - VL2
    rhov1 = mv1 / Vullage1
    rhov2 = mv2 / Vullage2
    hL2 = VL2 / model.A2
    pcthL2 = hL2 / model.H

    pv1 = np.empty(n)
    pv2 = np.empty(n)
    TL1 = np.empty((n, nL1))
    TL2 = np.empty((n, nL2))
    Tv1 = np.empty((n, nV1))
    Tv2 = np.empty((n, nV2))
    Jvalve111 = np.empty(n)
    Jvalve222 = np.empty(n)
    hL1 = np.empty(n)

    for z in range(n):
        pv1[z] = eos.vapor_pressure(uv1[z, -1], rhov1[z])
        pv2[z] = eos.vapor_pressure(uv2[z, -1], rhov2[z])
        for ii in range(nV1):
            if uv1[z, ii] < 0.0:
                uv1[z, ii] = 0.0
            Tv1[z, ii] = eos.temperature_PU(pv1[z], uv1[z, ii])
        for ii in range(nL1):
            TL1[z, ii] = liquid_temperature_from_energy_quadratic(uL1[z, ii])
        for ii in range(nV2):
            if uv2[z, ii] < 0.0:
                uv2[z, ii] = 0.0
            Tv2[z, ii] = eos.temperature_PU(pv2[z], uv2[z, ii])
        for ii in range(nL2):
            TL2[z, ii] = liquid_temperature_from_energy_quadratic(uL2[z, ii])
        Jvalve111[z] = gas_flow(model.S_valve1, model.gamma_, rhov1[z], pv1[z], model.p_atm)
        Jvalve222[z] = result.et_vent_state[z] * gas_flow(
            model.S_valve2, model.gamma_, rhov2[z], pv2[z], model.p_atm
        )
        hL1[z] = cyl_v_to_h(VL1[z], model.R1, model.Lcyl)

    pL1 = rho_L1 * model.g * hL1
    pL2 = rho_L2 * model.g * hL2
    boiloff = np.zeros(n)
    for ii in range(1, n):
        boiloff[ii] = boiloff[ii - 1] + (result.t[ii] - result.t[ii - 1]) * Jvalve222[ii]

    Jcd1 = result.series("Jcd1")
    Jboil = result.series("Jboil")
    AAA = result.series("AAA")

    return {
        "t": result.t,
        "mL1": mL1,
        "mv1": mv1,
        "mL2": mL2,
        "mv2": mv2,
        "uL1": uL1,
        "uv1": uv1,
        "uL2": uL2,
        "uv2": uv2,
        "Ts1": result.series("Ts1"),
        "Ts2": result.series("Ts2"),
        "Tw2": result.series("Tw2"),
        "Jtr": result.series("Jtr"),
        "Jboil": Jboil,
        "Jcd1": Jcd1,
        "Jcd2": result.series("Jcd2"),
        "rho_L1": rho_L1,
        "rho_L2": rho_L2,
        "rhov1": rhov1,
        "rhov2": rhov2,
        "VL1": VL1,
        "VL2": VL2,
        "hL1": hL1,
        "hL2": hL2,
        "pcthL2": pcthL2,
        "pv1": pv1,
        "pv2": pv2,
        "pL1": pL1,
        "pL2": pL2,
        "pTotal1": pv1 + pL1,
        "pTotal2": pv2 + pL2,
        "TL1": TL1,
        "TL2": TL2,
        "Tv1": Tv1,
        "Tv2": Tv2,
        "Jvalve111": Jvalve111,
        "Jvalve222": Jvalve222,
        "Boiloff_ET": boiloff,
        "Jv10": Jboil - AAA - Jcd1,
        "Jv20": -Jvalve222 - result.series("Jcd2"),
        "et_vent_state": result.et_vent_state,
        "ETTTVenstate": result.et_vent_state,
        "AAA": AAA,
        "BBB": result.series("BBB"),
        "CCC": result.series("CCC"),
        "DDD": result.series("DDD"),
        "EEE": result.series("EEE"),
        "FFF": result.series("FFF"),
        "GGG": result.series("GGG"),
        "HHH": result.series("HHH"),
        "III": result.series("III"),
        "JJJ": result.series("JJJ"),
        "KKK": result.series("KKK"),
        "LLL": result.series("LLL"),
        "MMM": result.series("MMM"),
        "NNN": result.series("NNN"),
        "OOO": result.series("OOO"),
        "PPP": result.series("PPP"),
        "QQQ": result.series("QQQ"),
        "RRR": result.series("RRR"),
        "SSS": result.series("SSS"),
        "TTT": result.series("TTT"),
        "UUU": result.series("UUU"),
        "VVV": result.series("VVV"),
        "WWW": result.series("WWW"),
        "XXX": result.series("XXX"),
    }


def save_bottom_fill(extracted: dict, path: str, stride: int = 10) -> None:
    """Write the ``runNominal.m`` column layout, subsampled every ``stride`` rows."""
    sl = slice(None, None, stride)
    psi = 6894.75729
    columns = [
        extracted["t"][sl],
        extracted["t"][sl] / 60.0,
        extracted["mL2"][sl],
        extracted["mv2"][sl],
        (extracted["mL2"] + extracted["mv2"])[sl],
        extracted["Boiloff_ET"][sl],
        extracted["TL2"][sl],
        extracted["Ts2"][sl],
        extracted["Tv2"][sl],
        extracted["Tw2"][sl],
        extracted["uL2"][sl],
        extracted["uv2"][sl],
        extracted["rho_L2"][sl],
        extracted["rhov2"][sl],
        extracted["pL2"][sl] / psi,
        extracted["pv2"][sl] / psi,
        extracted["hL2"][sl],
        extracted["Jtr"][sl],
        extracted["Jvalve222"][sl],
        extracted["Jcd2"][sl],
        extracted["MMM"][sl],
        extracted["NNN"][sl],
        extracted["OOO"][sl],
        extracted["PPP"][sl],
        extracted["QQQ"][sl],
        extracted["RRR"][sl],
        extracted["SSS"][sl],
        extracted["TTT"][sl],
        extracted["UUU"][sl],
        extracted["VVV"][sl],
        extracted["WWW"][sl],
        extracted["XXX"][sl],
        extracted["mL1"][sl],
        extracted["mv1"][sl],
        (extracted["mL1"] + extracted["mv1"])[sl],
        extracted["TL1"][sl],
        extracted["Ts1"][sl],
        extracted["Tv1"][sl],
        extracted["uL1"][sl],
        extracted["uv1"][sl],
        extracted["rho_L1"][sl],
        extracted["rhov1"][sl],
        extracted["pL1"][sl] / psi,
        extracted["pv1"][sl] / psi,
        extracted["hL1"][sl],
        extracted["AAA"][sl],
        extracted["Jcd1"][sl],
        extracted["Jboil"][sl],
        extracted["BBB"][sl],
        extracted["CCC"][sl],
        extracted["DDD"][sl],
        extracted["EEE"][sl],
        extracted["FFF"][sl],
        extracted["GGG"][sl],
        extracted["HHH"][sl],
        extracted["III"][sl],
        extracted["JJJ"][sl],
        extracted["KKK"][sl],
        extracted["LLL"][sl],
    ]
    blocks = [np.atleast_2d(col).T if np.ndim(col) == 1 else np.asarray(col) for col in columns]
    table = np.hstack(blocks)
    np.savetxt(path, table, delimiter=",")
