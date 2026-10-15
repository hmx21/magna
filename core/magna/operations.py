"""Operation-level interface: circuits as Python values and exact outcome probabilities.

Describe a circuit as an ``InstrumentIR``, give one complete outcome as a
``SampleAssignment`` and evaluate it with ``Simulator``; ``BoundarySampler`` and
``NativeBoundaryTrace`` answer event queries; the exact value types are re-exported here.
The file interface in ``magna`` covers ordinary use.
"""
from .boundary_sampling import BoundaryDraw, BoundarySampler, BoundaryTask, NativeBoundaryTrace
from .evaluate import StabilizerExpansion, contract_target, target_weight
from .feedback import ControlExpression, ControlGate, ControlRule, PhaseFeedback
from .ir import ControlNode, InstrumentIR, SampleAssignment
from .native_records import NativeRecordMarginals, ProbabilityMarginal
from .noise import NoiseModel, NoiseSite
from .scalar import ExactScalar
from .simulator import (AffineProposal, CompiledSample, IndependentProposal, Options, SampleResult, Simulator,
                        estimate)
from .stabilizer import StabilizerKet

__all__ = ['InstrumentIR', 'ControlNode', 'SampleAssignment', 'ControlRule', 'ControlGate', 'ControlExpression',
           'PhaseFeedback', 'Simulator', 'Options', 'CompiledSample', 'SampleResult', 'estimate',
           'IndependentProposal', 'AffineProposal', 'BoundaryTask', 'BoundarySampler', 'BoundaryDraw',
           'NativeBoundaryTrace', 'NoiseModel', 'NoiseSite', 'ExactScalar', 'StabilizerKet', 'StabilizerExpansion',
           'contract_target', 'target_weight', 'NativeRecordMarginals', 'ProbabilityMarginal']
