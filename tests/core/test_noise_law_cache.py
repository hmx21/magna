"""Exact independent channel laws and replay across preparation cache reuse."""
from fractions import Fraction
from random import Random
import unittest

from magna import _native, parse_circuit
from magna.file_api import _NoiseSchedule
from tests.support import file_hooks as hooks
from tests.support import native_testing


def chain_law(probabilities):
    remaining = Fraction(1)
    result = []
    for probability in probabilities:
        probability = Fraction(probability)
        result.append(remaining * probability)
        remaining *= 1 - probability
    return tuple(result)


def chain_text(probabilities, *, guard=''):
    return '\n'.join(
        f'{"CORRELATED_ERROR" if index == 0 else "ELSE_CORRELATED_ERROR"}'
        f'({probability}) {guard}{"XYZ"[index % 3]}1'
        for index, probability in enumerate(probabilities))


def pairs(rows):
    return tuple(tuple((value.numerator, value.denominator) for value in row) for row in rows)


class NoiseLawCacheTests(unittest.TestCase):
    def check_laws(self, circuit, laws, *, guarded=(), scale=None, actions=None):
        laws = tuple(tuple(Fraction(value) for value in law) for law in laws)
        sampling = None if scale is None else {'method': 'importance', 'proposal_scale': str(scale)}
        schedule = _NoiseSchedule(circuit, sampling)
        proposed = []
        factors = []
        tilted = 0
        for index, law in enumerate(laws):
            total = sum(law)
            if scale is not None and index not in guarded and 0 < total < 1:
                proposed.append(tuple(scale * value for value in law))
                factors.append(((1 - total) / (1 - scale * total), 1 / scale))
                tilted += scale != 1
            else:
                proposed.append(law)
                factors.append((Fraction(1), Fraction(1)))
        self.assertEqual(schedule.probabilities, pairs(proposed))
        self.assertEqual(hooks.weight_factors(schedule), pairs(factors))
        self.assertEqual(schedule.category_count, len(laws))
        self.assertEqual(schedule.action_count, sum(map(len, laws)))
        self.assertEqual(schedule.guarded, len(guarded))
        self.assertEqual(schedule.tilted, tilted)
        self.assertEqual(schedule.deterministic, sum(sum(law) in (0, 1) for law in laws))

        # Direct category construction bypasses prepare_file_noise and its cache.
        reference = native_testing.load().make_file_noise(pairs(proposed),
            None if scale is None else pairs(factors), actions)
        seed_rng = Random(90210)
        seeds = [0, 1, (1 << 64) - 1] + [seed_rng.getrandbits(64) for _ in range(61)]
        actual = native_testing.load().draw_file_noise(schedule._native_plan, seeds)
        self.assertEqual(actual, native_testing.load().draw_file_noise(reference, seeds))
        self.assertEqual(actual, native_testing.load().draw_file_noise(schedule._native_plan, seeds[:19])
                         + native_testing.load().draw_file_noise(schedule._native_plan, seeds[19:]))
        if actions is not None:
            self.assertEqual(_native.file_noise_actions(schedule._native_plan), actions)
            self.assertEqual(native_testing.load().draw_file_noise_actions(schedule._native_plan, seeds),
                             native_testing.load().draw_file_noise_actions(reference, seeds))
        batch_rng, single_rng = Random(85), Random(85)
        self.assertEqual(hooks.draw_many(schedule, batch_rng, 37),
                         [hooks.draw(schedule, single_rng) for _ in range(37)])
        self.assertEqual(batch_rng.getstate(), single_rng.getstate())
        return schedule

    def test_chain_and_exclusive_probabilities_have_distinct_keys(self):
        raw = (Fraction(1, 10), Fraction(2, 9), Fraction(0))
        direct = 'PAULI_CHANNEL_1(' + ','.join(map(str, raw)) + ') 1'
        for first_chain in (False, True):
            kinds = (first_chain, not first_chain) * 3
            circuit = parse_circuit('\n'.join(chain_text(raw) if chain else direct for chain in kinds))
            laws = [chain_law(raw) if chain else raw for chain in kinds]
            for scale in (None, Fraction(2)):
                with self.subTest(first_chain=first_chain, scale=scale):
                    self.check_laws(circuit, laws, scale=scale)

    def test_guards_reuse_physical_laws_without_importance_tilting(self):
        raw = (Fraction(1, 10), Fraction(2, 9), Fraction(0))
        direct = 'PAULI_CHANNEL_1(' + ','.join(map(str, raw)) + ') '
        circuit = parse_circuit('\n'.join(('M(1/10) 0',
            chain_text(raw, guard='rec[-1] '), chain_text(raw), direct + '1',
            direct + 'rec[-1] 1', chain_text(raw, guard='rec[-1] '), chain_text(raw))))
        laws = [(Fraction(1, 10),), chain_law(raw), chain_law(raw), raw,
                raw, chain_law(raw), chain_law(raw)]
        for scale in (None, Fraction(1, 2), Fraction(2)):
            with self.subTest(scale=scale):
                self.check_laws(circuit, laws, guarded=(1, 4, 5), scale=scale)

    def test_long_chains_large_rationals_and_unreachable_tails(self):
        denominator = (1 << 80) + 7
        chains = ((0, 0, 0), (0, 1, Fraction(1, 3), 0, 1), (1, 0, 1),
                  (Fraction(1, denominator), Fraction((1 << 79) + 9, denominator), Fraction(1, 3), 0),
                  tuple(Fraction(1, 128 - index) for index in range(65)))
        repeated = chains * 2
        circuit = parse_circuit('\n'.join(chain_text(raw) for raw in repeated))
        for scale in (None, Fraction(1, 2)):
            with self.subTest(scale=scale):
                self.check_laws(circuit, [chain_law(raw) for raw in repeated], scale=scale)

    def test_cache_eviction_keeps_the_original_channel_order(self):
        unique = tuple((Fraction(1, value), Fraction(1, value + 1)) for value in range(200, 235))
        # Immediate hits, more than 32 laws, then misses/hits on older laws.
        chains = (unique[0],) + unique + (unique[0], unique[-1], unique[0])
        circuit = parse_circuit('\n'.join(chain_text(raw) for raw in chains))
        self.check_laws(circuit, [chain_law(raw) for raw in chains], scale=Fraction(3))

    def test_repeated_laws_keep_location_actions_report_flips_and_order(self):
        circuit = parse_circuit('CORRELATED_ERROR(1/3) X2 Y0\n'
            'ELSE_CORRELATED_ERROR(1/2) Z1\nM(1/4) 0 2\n'
            'CORRELATED_ERROR(1/3) Y1 X2\nELSE_CORRELATED_ERROR(1/2) X0 Z2\nM(1/4) 1')
        instructions = circuit.instructions
        actions = (((instructions[0].op_id, (('X', 2), ('Y', 0))),
                    (instructions[1].op_id, (('Z', 1),))),
                   (('report', 'm0'),), (('report', 'm1'),),
                   ((instructions[3].op_id, (('Y', 1), ('X', 2))),
                    (instructions[4].op_id, (('X', 0), ('Z', 2)))),
                   (('report', 'm2'),))
        law = chain_law((Fraction(1, 3), Fraction(1, 2)))
        laws = (law, (Fraction(1, 4),), (Fraction(1, 4),), law, (Fraction(1, 4),))
        for scale in (None, Fraction(1, 2)):
            with self.subTest(scale=scale):
                self.check_laws(circuit, laws, scale=scale, actions=actions)


if __name__ == '__main__':
    unittest.main()
