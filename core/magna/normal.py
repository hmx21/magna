"""Polynomial certificate values and facades for the shared native compiler."""
from dataclasses import dataclass


@dataclass(frozen=True)
class NormalForm:
    n: int
    terms: tuple
    constant: int
    power: int
    boundary: tuple
    output_count: int
    environment: tuple
    zero: bool = False
    certificates: tuple = ()


def normalize_fixed_input(branch, max_path_bits=2048):
    """Export a literal branch's native amplitude tensor without decomposition."""
    from . import _native
    from .phase_tensor import QuadraticPhaseTensor
    k = branch.initial
    raw = _native.normalize_literal_program((k.width,k.base,k.columns,k.linear,k.upper,k.power,k.phase),
        branch.inputs, branch.outputs, branch.operations, max_path_bits)
    return QuadraticPhaseTensor(**raw).to_normal()


def absorb_boundaries(normal):
    from .phase_tensor import absorb
    return absorb(normal).to_normal()
