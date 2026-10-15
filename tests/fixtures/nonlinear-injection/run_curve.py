"""Run one literal injection circuit through the public Monte Carlo interface."""
import argparse
from fractions import Fraction
import json
import os
from pathlib import Path
import tempfile

from magna import ControllerRegistry, run
from tests.fixtures.basic.decoder_plugin.magna_example_decoders import majority


HERE = Path(__file__).resolve().parent
RATES = ("0.01", "0.02", "0.03", "0.05", "0.07", "0.1", "0.2", "0.3", "0.5")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shots", type=int, default=131072)
    parser.add_argument("--seed", type=int, default=20260927)
    parser.add_argument("--rates", nargs="+", default=RATES)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    registry = ControllerRegistry()
    registry.register("majority", majority, version="1.0")
    template = (HERE / "physical.magna").read_text()
    assert template.count("M(0.01)") == 1
    rows = []
    for index, rate in enumerate(args.rates):
        circuit = args.output / f"physical-p{rate}.magna"
        circuit.write_text(template.replace("M(0.01)", f"M({rate})"))
        result = run(
            circuit, task=HERE / "task.json", controllers=registry,
            shots=args.shots, seed=args.seed + index,
        )
        (args.output / f"result-p{rate}.json").write_text(
            json.dumps(result, indent=2) + "\n")
        if result["status"] != "complete":
            raise RuntimeError(result["failures"])
        assert result["completed"] == args.shots
        assert result["estimates"]["acceptance"] == 1
        actual = result["estimates"]["error"]
        standard_error = result["standard_errors"]["error"]
        p = Fraction(rate)
        theory = Fraction(3, 2) * p**2 - p**3
        q = 2 * float(theory)
        predicted_se = (q * (1 - q) / (4 * args.shots)) ** 0.5
        rows.append(dict(p=rate, simulator=actual, standard_error=standard_error,
                         theory=float(theory), theory_exact=str(theory),
                         difference=actual - float(theory),
                         theory_standard_error=predicted_se,
                         standardized_residual=(actual - float(theory)) / predicted_se,
                         shots=result["completed"], seed=result["seed"],
                         total_seconds=result["timings"]["total_seconds"],
                         compilations=result["resources"]["compilations"]))
        print(f"p={rate}: estimate={actual:.9g}, SE={standard_error:.3g}, "
              f"theory={float(theory):.9g}, seconds={rows[-1]['total_seconds']:.3f}", flush=True)
    summary = dict(method="public run; ordinary finite-shot physical Monte Carlo",
                   physical_qubits=2, measurement_rounds=3, report_error_sites=3,
                   theory="3*p**2/2 - p**3", shots_per_point=args.shots,
                   max_absolute_standardized_residual=max(
                       abs(row["standardized_residual"]) for row in rows), points=rows)
    (args.output / "curve.json").write_text(json.dumps(summary, indent=2) + "\n")
    with tempfile.TemporaryDirectory(prefix="injection-matplotlib-") as cache:
        os.environ.setdefault("MPLCONFIGDIR", cache)
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        from matplotlib.ticker import LogFormatterSciNotation

        grid = np.geomspace(min(float(p) for p in args.rates),
                           max(float(p) for p in args.rates), 400)
        fig, ax = plt.subplots(figsize=(6.4, 4.3), layout="constrained")
        ax.loglog(grid, 1.5 * grid**2 - grid**3, color="#26384b", lw=1.8,
                  label=r"Theory: $3p^2/2-p^3$")
        ax.errorbar([float(row["p"]) for row in rows],
                    [row["simulator"] for row in rows],
                    yerr=[1.96 * row["standard_error"] for row in rows],
                    fmt="o", ms=5, color="#dc721d", capsize=3,
                    markerfacecolor="white", markeredgewidth=1.3,
                    label=f"Monte Carlo: {args.shots:,} shots/point (1.96 SE)")
        ax.set(xlabel="Independent report-flip probability $p$",
               ylabel=r"Output infidelity $1-\langle T|\rho|T\rangle$",
               title="Two-qubit injection with nonlinear majority feedback")
        ax.xaxis.set_major_formatter(LogFormatterSciNotation())
        ax.yaxis.set_major_formatter(LogFormatterSciNotation())
        ax.grid(which="major", alpha=0.18)
        ax.legend(loc="upper left", frameon=False)
        for suffix in ("png", "pdf"):
            fig.savefig(args.output / f"curve.{suffix}", dpi=180)
        plt.close(fig)


if __name__ == "__main__":
    main()
