"""Smoke tests for the nominal trailer-to-dewar hour.

Eight checks: geometry and schedule constants, horizontal-cylinder height,
orifice flow, CoolProp versus the REFPROP polynomials, initial inventories,
fill-regime logic, the initial right-hand side, and a short integration of
the nominal-hour scenario (same parameters as ``runNominal.m``).
"""

from __future__ import annotations

import math
import tempfile
import unittest

import numpy as np

from lh2transfer.control import et_vent_state, lh2_control, st_vent_state, vaporizer_valve_state
from lh2transfer.extract import extract, save_bottom_fill
from lh2transfer.geometry import (
    cyl_v_to_h,
    enthalpy_of_vaporization,
    gas_flow,
    liquid_density_from_energy,
    liquid_temperature_from_energy,
)
from lh2transfer.parameters import GAL_TO_M3, PSI_TO_PA, nominal_model
from lh2transfer.props import fluid
from lh2transfer.simulate import LH2RHS, initial_state, simulate, state_indices


def _segment_area(height: float, radius: float) -> float:
    """Analytic liquid cross-section of a horizontal cylinder."""
    if height <= 0.0:
        return 0.0
    if height >= 2.0 * radius:
        return math.pi * radius**2
    return (
        radius**2 * math.acos((radius - height) / radius)
        - (radius - height) * math.sqrt(max(0.0, 2.0 * radius * height - height**2))
    )


class NominalHourSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = nominal_model()

    def test_nominal_parameters_match_matlab_inputs(self):
        model = self.model
        self.assertAlmostEqual(model.VTotal1, 17000 * GAL_TO_M3)
        self.assertAlmostEqual(model.VTotal2, 3300 * GAL_TO_M3)
        self.assertAlmostEqual(model.tFinal, 3600.0)
        self.assertEqual(model.nL1, 3)
        self.assertEqual(model.nV1, 3)
        self.assertEqual(model.nL2, 3)
        self.assertEqual(model.nV2, 4)
        self.assertAlmostEqual(model.p10 / PSI_TO_PA, 14.7, places=6)
        self.assertAlmostEqual(model.p20 / PSI_TO_PA, 30.0, places=6)
        self.assertAlmostEqual(model.p_ST_slow / PSI_TO_PA, 60.0, places=6)
        self.assertAlmostEqual(model.p_ET_high / PSI_TO_PA, 45.0, places=6)
        self.assertAlmostEqual(model.p_ET_low / PSI_TO_PA, 43.0, places=6)
        self.assertAlmostEqual(model.TopET, 0.9)
        self.assertAlmostEqual(model.ratio_top_bottom, 0.0)
        self.assertAlmostEqual(model.relTol, 1e-4)
        self.assertGreater(model.f, 0.0)
        idx = state_indices(model.nL1, model.nV1, model.nL2, model.nV2)
        # 1-based MATLAB indices minus one.
        self.assertEqual(idx["mL1"], 0)
        self.assertEqual(idx["mv1"], 4)
        self.assertEqual(idx["Ts1"], 8)
        self.assertEqual(idx["mL2"], 12)
        self.assertEqual(idx["Tw2"], 22)
        self.assertEqual(idx["Jcd1"], 23)
        self.assertEqual(idx["XXX"], 48)
        self.assertEqual(idx["n"], 49)

    def test_horizontal_cylinder_height(self):
        radius, length = 1.5, 8.0
        for fraction in (0.2, 0.5, 0.8):
            height_target = fraction * 2.0 * radius
            volume = _segment_area(height_target, radius) * length
            height = cyl_v_to_h(volume, radius, length)
            self.assertAlmostEqual(height, height_target, delta=2e-3)
        self.assertEqual(cyl_v_to_h(0.0, radius, length), 0.0)
        full = math.pi * radius**2 * length
        self.assertAlmostEqual(cyl_v_to_h(full, radius, length), 2.0 * radius)

    def test_gas_flow_choked_and_subsonic(self):
        gamma = 5.0 / 3.0
        area = 0.002
        rho = 1.3
        # Symmetry of the MATLAB sign flip.
        forward = gas_flow(area, gamma, rho, 2.0e5, 1.0e5)
        backward = gas_flow(area, gamma, rho, 1.0e5, 2.0e5)
        self.assertAlmostEqual(forward, -backward, places=9)
        self.assertGreater(forward, 0.0)
        # Pressure ratio above ((gamma+1)/2)^(gamma/(gamma-1)) is choked.
        threshold = ((gamma + 1.0) / 2.0) ** (gamma / (gamma - 1.0))
        self.assertGreater(3.0e5 / 1.0e5, threshold)
        choked = gas_flow(area, gamma, rho, 3.0e5, 1.0e5)
        expected = area * math.sqrt(
            gamma * rho * 3.0e5 * (2.0 / (gamma + 1.0)) ** ((gamma + 1.0) / (gamma - 1.0))
        )
        self.assertAlmostEqual(choked, expected, places=8)
        # A mild pressure ratio stays on the subsonic branch and is smaller than choked.
        subsonic = gas_flow(area, gamma, rho, 1.2e5, 1.0e5)
        self.assertGreater(subsonic, 0.0)
        self.assertLess(subsonic, choked)

    def test_coolprop_matches_refprop_polynomials(self):
        eos = fluid()
        u_liquid = eos.internal_energy_TQ(20.0, 0.0)
        rho = eos.density_TQ(20.0, 0.0)
        self.assertAlmostEqual(liquid_temperature_from_energy(u_liquid), 20.0, delta=0.15)
        self.assertAlmostEqual(liquid_density_from_energy(u_liquid), rho, delta=0.15)
        h_l = eos.enthalpy_TQ(20.0, 0.0)
        h_v = eos.enthalpy_TQ(20.0, 1.0)
        self.assertAlmostEqual(enthalpy_of_vaporization(20.0), h_v - h_l, delta=2000.0)
        # 1 bar class vapor pressure for parahydrogen is near 20.3 K.
        p_sat = eos.pressure_TQ(20.3, 1.0)
        self.assertAlmostEqual(p_sat, 101325.0, delta=4000.0)
        self.assertAlmostEqual(self.model.rhoL10, rho, delta=2.0)

    def test_initial_state_nominal_hour(self):
        model = self.model
        self.assertAlmostEqual(model.mL10 + model.mv10, model.totalmass10, places=6)
        self.assertGreater(model.mL10, 0.0)
        self.assertGreater(model.mv10, 0.0)
        self.assertGreater(model.Vullage10, 0.0)
        self.assertLess(model.Vullage10, model.VTotal1)
        self.assertGreater(model.mL20, 0.0)
        self.assertGreater(model.mv20, 0.0)
        self.assertAlmostEqual(model.hL20, 0.01 * model.H)
        self.assertGreater(model.UL10, -1.0e6)
        self.assertGreater(model.Uv10, model.UL10)
        y0 = initial_state(model)
        self.assertEqual(y0.shape, (49,))
        self.assertTrue(np.all(np.isfinite(y0)))
        self.assertAlmostEqual(y0[0], model.mL10)

    def test_fill_control_regimes(self):
        model = self.model
        low = lh2_control(model, 0.01 * model.H, model.p10, model.p20, 0.0, 0.0)
        self.assertEqual(low.lambdaE, 0.5)
        self.assertEqual(low.STVentState, 0.0)
        self.assertEqual(low.lambdaV, 1.0)

        fast = lh2_control(model, 0.10 * model.H, model.p10, model.p20, 0.0, 0.0)
        self.assertEqual(fast.lambdaE, 1.0)

        reduced = lh2_control(model, 0.40 * model.H, model.p_ST_final * 2.0, model.p20, 0.0, 0.0)
        self.assertEqual(reduced.lambdaE, 1.0)
        self.assertEqual(reduced.STVentState, 0.0)

        topping = lh2_control(model, 0.50 * model.H, model.p_ST_slow, model.p20, 1.0, 0.0)
        self.assertEqual(topping.lambdaE, 0.0)
        self.assertEqual(topping.lambdaV, 0.0)
        self.assertEqual(topping.STVentState, 1.0)

        # The written ST-vent deadband is unreachable: open only at >= 1.05 * set.
        self.assertEqual(st_vent_state(model, 1.04 * model.p_ST_slow, model.p_ST_slow), 0.0)
        self.assertEqual(st_vent_state(model, 1.06 * model.p_ST_slow, model.p_ST_slow), 1.0)
        self.assertEqual(vaporizer_valve_state(model, model.p_ST_slow, model.p_ST_slow), 0.0)
        self.assertEqual(et_vent_state(model, model.p_ET_high + 1.0), 1.0)
        self.assertEqual(et_vent_state(model, model.p_ET_low - 1.0), 0.0)

    def test_rhs_at_nominal_initial_condition(self):
        rhs = LH2RHS(self.model)
        y0 = initial_state(self.model)
        dydt, diag = rhs.evaluate(0.0, y0)
        self.assertEqual(dydt.shape, (49,))
        self.assertTrue(np.all(np.isfinite(dydt)))
        self.assertAlmostEqual(diag.pv1 / PSI_TO_PA, 14.7, delta=1.0)
        self.assertAlmostEqual(diag.pv2 / PSI_TO_PA, 30.0, delta=2.0)
        self.assertGreater(diag.hL1, 0.0)
        self.assertLess(diag.hL1, 2.0 * self.model.R1)
        self.assertGreater(diag.hL2, 0.0)
        # Trailer is below the 60 psia delivery pressure, so the transfer line is shut
        # and the vaporizer is commanded fully open.
        self.assertEqual(diag.st_ready, 0.0)
        self.assertEqual(diag.lambdaE, 0.5)
        self.assertEqual(diag.lambdaV, 1.0)
        self.assertEqual(diag.st_vent, 0.0)
        self.assertGreater(diag.Jvap, 0.0)
        self.assertGreater(diag.extra["dmVapdt"], 0.0)
        self.assertAlmostEqual(diag.Jtr, 0.0)
        self.assertLess(diag.JL1, 0.0)
        self.assertGreater(diag.TL1[-1], 13.0)
        self.assertLess(diag.TL1[-1], 33.0)
        self.assertGreater(diag.Tv1[-1], 13.0)
        self.assertLess(diag.Tv2[-1], 40.0)

    def test_nominal_hour_startup_integration(self):
        # Same schedule as the one-hour runNominal case, stopped during pressurization.
        result = simulate(self.model, t_final=30.0, max_step=1.0)
        self.assertGreater(result.t[-1], 29.0)
        self.assertTrue(np.all(np.isfinite(result.y)))
        mL1 = result.series("mL1")
        mv1 = result.series("mv1")
        mL2 = result.series("mL2")
        mv2 = result.series("mv2")
        m_vap = result.series("mVap")
        total = mL1 + mv1 + mL2 + mv2 + m_vap
        # Both vents stay shut while the trailer is still below ~63 psia and the
        # receiver is still below the 45 psia PRD, so hydrogen mass is conserved.
        self.assertLess(abs(total[-1] - total[0]) / total[0], 1e-3)
        self.assertGreater(m_vap[-1], m_vap[0])
        self.assertGreater(mv1[-1] + m_vap[-1], mv1[0])
        self.assertTrue(np.all(result.et_vent_state == 0.0))
        self.assertTrue(np.all(result.series("Tw2") > 10.0))
        self.assertTrue(np.all(result.series("Tw2") < 40.0))
        extracted = extract(result)
        self.assertEqual(len(extracted["pv1"]), len(result.t))
        self.assertTrue(np.all(np.isfinite(extracted["pv1"])))
        self.assertTrue(np.all(np.isfinite(extracted["TL1"])))
        self.assertAlmostEqual(extracted["Boiloff_ET"][-1], 0.0, places=5)
        with tempfile.TemporaryDirectory() as folder:
            path = f"{folder}/output_bottomfill.txt"
            save_bottom_fill(extracted, path, stride=10)
            table = np.loadtxt(path, delimiter=",")
        self.assertEqual(table.shape[0], len(result.t[::10]))
        self.assertGreater(table.shape[1], 40)


if __name__ == "__main__":
    unittest.main()
