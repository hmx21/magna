"""Explicit mutually exclusive Pauli/report channels at original locations."""
from dataclasses import dataclass
from fractions import Fraction
from . import _native


def _rational(value):
    value = Fraction(value)
    return value.numerator, value.denominator


@dataclass(frozen=True)
class NoiseSite:
    after: str
    qubits: tuple
    channels: tuple  # tuples of Pauli letters, or ('REPORT', record_name)
    probability: Fraction
    weights: tuple | None = None  # conditional probabilities given a nonidentity event

    def __deepcopy__(self, memo):
        return self

    def __post_init__(self):
        object.__setattr__(self, 'qubits', tuple(self.qubits))
        object.__setattr__(self, 'channels', tuple(tuple(c) for c in self.channels))
        object.__setattr__(self, 'probability', Fraction(self.probability))
        if self.weights is not None:
            object.__setattr__(self, 'weights', tuple(Fraction(w) for w in self.weights))
        object.__setattr__(self, '_native_handle', _native.make_noise_site(self.after, self.qubits,
            self.channels, _rational(self.probability),
            None if self.weights is None else tuple(_rational(w) for w in self.weights)))


@dataclass(frozen=True)
class NoiseModel:
    sites: tuple

    def draw(self, rng, *, physical_rate=None, proposal_scale=None):
        """Return faults, flips, exact target/proposal likelihood ratio.

        Independent per-site noise; dependencies on controlled gate choices
        require an explicit joint proposal supplied through SampleAssignment.
        proposal_scale multiplies each stochastic site's effective physical rate;
        probabilities zero and one remain deterministic.
        """
        try:
            scale = None if proposal_scale is None else _rational(str(proposal_scale))
        except (ValueError, ZeroDivisionError) as exc:
            raise ValueError('proposal_scale must be a finite positive rational') from exc
        faults, flips, weight = _native.sample_noise_model(tuple(site._native_handle for site in self.sites), rng,
            None if physical_rate is None else _rational(str(physical_rate)),
            scale)
        return (tuple(dict(after=anchor, qubit=q, pauli=p) for anchor, q, p in faults),
                tuple(flips), Fraction(*weight))
