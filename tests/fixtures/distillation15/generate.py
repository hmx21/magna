"""Export the retained RM15 EPR gate sequence to the common file interface."""
import argparse
from fractions import Fraction
import json
from pathlib import Path


def generate(resource_error="0.001"):
    probability = Fraction(resource_error)
    if not 0 <= probability <= 1:
        raise ValueError("resource error probability must be in [0, 1]")
    code = json.loads(Path(__file__).with_name("code.json").read_text())
    lines = [
        "# RM15 EPR distillation: data 0..14, output 15, resources 16..30.",
        "# Independent Z errors on prepared resources; all other operations ideal.",
        "H 0 1 2 3 4",
        "CX 0 15",
    ]
    phases = {2: "S", 4: "Z", 6: "S_DAG"}
    for q, phase in enumerate(code["correction_phase"]):
        if phase:
            lines.append(f"{phases[phase]} {q}")
    lines.extend(f"CZ {a} {b}" for a, b in code["correction_cz"])
    lines.extend(f"CX {a} {b}" for a, b in code["encoder"])
    for q in range(15):
        r = 16 + q
        lines.extend([
            f"# Injection {q}: resource control, data target; output moves to {r}.",
            f"H {r}", f"T {r}", f"Z_ERROR({probability}) {r}",
            f"CX {r} {q}", f"M {q}",
            f"CX rec[-1] {r}",
            "CONTROL[XOR](1) rec[-1] -> ctrl[0]",
            f"CS_DAG ctrl[0] {r}", f"S {r}",
        ])
    lines.append("MX " + " ".join(str(q) for q in range(16, 31)))
    for row in code["rows"][1:]:
        lines.append("DETECTOR " + " ".join(
            f"rec[{q - 15}]" for q in range(15) if row >> q & 1))
    lines.append("CONTROL[XOR] " + " ".join(
        f"rec[{q - 15}]" for q in range(15) if code["rows"][0] >> q & 1)
        + " -> ctrl[1]")
    lines.append("CZ ctrl[1] 15")
    lines.append("POSTSELECT " + " ".join(f"det[{i}]" for i in range(4)))
    task = dict(schema_version=1, outputs=[15], ideal_state="ideal.magna")
    return "\n".join(lines) + "\n", task


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resource-error", default="0.001")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    circuit, task = generate(args.resource_error)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "physical.magna").write_text(circuit)
    (args.output / "task.json").write_text(json.dumps(task, indent=2) + "\n")
    (args.output / "ideal.magna").write_text("H 0\nT 0\n")


if __name__ == "__main__":
    main()
