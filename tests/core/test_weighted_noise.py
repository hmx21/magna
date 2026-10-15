from fractions import Fraction
from random import Random
import unittest
from magna.noise import NoiseSite, NoiseModel
from tests.support.operation_hooks import weighted_choice


class WeightedNoiseTests(unittest.TestCase):
    def test_deterministic_nonuniform_and_importance(self):
        site = NoiseSite('original', (4,), (('X',), ('Z',)), Fraction(1, 3), (0, 1))
        rng = Random(14)
        seen = set()
        for _ in range(40):
            faults, flips, ratio = NoiseModel((site,)).draw(rng, proposal_scale=2)
            self.assertFalse(flips)
            self.assertEqual(ratio, Fraction(1, 2) if faults else 2)
            if faults:
                self.assertEqual(faults, ({'after': 'original', 'qubit': 4, 'pauli': 'Z'},))
            seen.add(bool(faults))
        self.assertEqual(seen, {True, False})

    def test_invalid_weights(self):
        for weights in ((1,), (1, 1), (-1, 2)):
            with self.assertRaises(ValueError):
                NoiseSite('a', (0,), (('X',), ('Z',)), 1, weights)

    def test_zero_and_identity(self):
        rng = Random(3)
        self.assertEqual(weighted_choice(rng, (0, 1)), 1)
        self.assertEqual(weighted_choice(rng, (0, 0)), -1)

    def test_invalid_scales_and_scaled_support(self):
        site = NoiseSite('a', (0,), (('X',),), Fraction(1, 3))
        model = NoiseModel((site,))
        for scale in (0, -1, True, 'nan', 'inf', '1/0'):
            with self.subTest(scale=scale), self.assertRaises(ValueError):
                model.draw(Random(3), proposal_scale=scale)
        for scale in (3, 4):
            with self.subTest(scale=scale), self.assertRaises(ValueError):
                model.draw(Random(3), proposal_scale=scale)
        with self.assertRaises(ValueError):
            model.draw(Random(3), physical_rate=Fraction(1, 2), proposal_scale=2)
        with self.assertRaises(TypeError):
            model.draw(Random(3), proposal_rate=Fraction(1, 2))

    def test_unit_scale_and_effective_deterministic_endpoints(self):
        model = NoiseModel((NoiseSite('zero', (0,), (('X',),), 0),
                            NoiseSite('one', (1,), (('Z',),), 1),
                            NoiseSite('variable', (), (('REPORT', 'm0'),), Fraction(1, 5))))
        for physical in (None, 0, Fraction(1, 7), 1):
            left, right = Random(26), Random(26)
            for _ in range(40):
                self.assertEqual(model.draw(left, physical_rate=physical),
                                 model.draw(right, physical_rate=physical, proposal_scale=1))
                self.assertEqual(left.getstate(), right.getstate())
        for physical in (0, 1):
            left, right = Random(17), Random(17)
            self.assertEqual(model.draw(left, physical_rate=physical),
                             model.draw(right, physical_rate=physical, proposal_scale=100))
            self.assertEqual(left.getstate(), right.getstate())


class NativeNoiseStreamTests(unittest.TestCase):
    def test_original_randrange_stream_and_exact_weights(self):
        # Independently retain the public sequential rational choice contract.
        sites = (NoiseSite('report', (4,), (('I',), ('Y',)), Fraction(1, 7), (Fraction(1, 3), Fraction(2, 3))),
                 NoiseSite('readout', (), (('REPORT', 'm0'),), Fraction(1, 10)),
                 NoiseSite('huge', (9,), (('X',),), Fraction(1, 2**140)),
                 NoiseSite('zero', (1,), (('Z',),), 0),
                 NoiseSite('one', (2,), (('X',),), 1))
        for physical, scale in ((None, None), (None, Fraction(1, 2)), (None, 2),
                                (Fraction(1, 9), 2)):
            rng, reference = Random(814), Random(814)
            for _ in range(120):
                faults, flips, weight = [], [], Fraction(1)
                for site in sites:
                    p = site.probability if physical is None else physical
                    q = p if scale is None or p in (0, 1) else scale*p
                    if reference.randrange(q.denominator) < q.numerator:
                        weight *= p/q
                        if site.weights is None:
                            index = reference.randrange(len(site.channels))
                        else:
                            remaining = Fraction(1)
                            for index, chance in enumerate(site.weights):
                                conditional = chance/remaining
                                if conditional and reference.randrange(conditional.denominator) < conditional.numerator:
                                    break
                                remaining -= chance
                        channel = site.channels[index]
                        if channel[0] == 'REPORT': flips.append(channel[1])
                        else:
                            faults.extend(dict(after=site.after, qubit=q, pauli=p)
                                for q,p in zip(site.qubits,channel) if p != 'I')
                    else: weight *= (1-p)/(1-q)
                self.assertEqual(NoiseModel(sites).draw(rng, physical_rate=physical, proposal_scale=scale),
                                 (tuple(faults),tuple(flips),weight))
                self.assertEqual(rng.getstate(),reference.getstate())
