"""Run the trailer-to-dewar case from ``runNominal.m``.

Example::

    python -m lh2transfer.run_nominal --t-final 3600 --output output_bottomfill.txt
"""

from __future__ import annotations

import argparse
import time

from lh2transfer.extract import extract, save_bottom_fill
from lh2transfer.parameters import PSI_TO_PA, nominal_model
from lh2transfer.simulate import simulate


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Nominal LH2 trailer-to-dewar hour")
    parser.add_argument(
        "--t-final",
        type=float,
        default=None,
        help="Integration length in seconds (default: model hour, 3600)",
    )
    parser.add_argument(
        "--output",
        default="output_bottomfill.txt",
        help="CSV path in the runNominal.m column order",
    )
    parser.add_argument("--stride", type=int, default=10, help="Row stride when writing output")
    args = parser.parse_args(argv)

    model = nominal_model()
    t_final = model.tFinal if args.t_final is None else args.t_final
    print(
        f"LH2 nominal case: trailer {model.totalmass10:.0f} kg, "
        f"ET fill fraction {model.pct_hL20:.2f}, t_final={t_final:.1f} s"
    )
    started = time.perf_counter()
    result = simulate(model, t_final=t_final)
    elapsed = time.perf_counter() - started
    extracted = extract(result)
    save_bottom_fill(extracted, args.output, stride=args.stride)
    print(
        f"steps={len(result.t)} wall={elapsed:.1f}s "
        f"pv1={extracted['pv1'][-1] / PSI_TO_PA:.2f} psia "
        f"pv2={extracted['pv2'][-1] / PSI_TO_PA:.2f} psia "
        f"mL2={extracted['mL2'][-1]:.2f} kg "
        f"boiloff={extracted['Boiloff_ET'][-1]:.4f} kg"
    )
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
