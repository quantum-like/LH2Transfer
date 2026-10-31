"""Python port of the LLNL / NASA LH2 transfer and boil-off simulation.

The original model is the MATLAB code in this repository (LLNL-CODE-748554),
which calls REFPROP 9.1 for parahydrogen. This package keeps that lumped
two-vessel formulation and evaluates the equation of state with CoolProp.

Thermodynamic states use CoolProp's ParaHydrogen Helmholtz model. CoolProp
6.x does not ship a parahydrogen viscosity model (and vapor conductivity
depends on it), so those transport properties fall back to CoolProp Hydrogen
at the same temperature and quality or pressure. See ``lh2transfer.props``.
"""

from lh2transfer.parameters import nominal_model
from lh2transfer.simulate import simulate

__all__ = ["nominal_model", "simulate"]
__version__ = "0.1.0"
