"""Independent weights, action provenance and independence across packed lanes."""
from collections import Counter
from fractions import Fraction
from random import Random
import math
import unittest

from magna import _native, compile_circuit, parse_circuit
from tests.support import native_testing


class PackedNoiseTests(unittest.TestCase):
    def test_guarded_sites_share_a_group_but_wait_for_their_causal_instruction(self):
        for control in (0, 1):
            text = ('X 0\n' if control else '') + 'M 0\n'
            text += '\n'.join('X_ERROR(1) rec[-1] 1\nX_ERROR(1) 2' for _ in range(65))
            target = parse_circuit(('X 0\n' if control else '') + 'X 1')
            compiled = compile_circuit(parse_circuit(text), task={'outputs': [1, 2]}, ideal_state=target)
            for seed in range(4):
                draw = compiled.sample(Random(seed))
                self.assertEqual(draw['acceptance'], 1)
                self.assertEqual(draw['error'], 0)

    def test_sparse_actions_weights_and_partitioning(self):
        # Interleaved equal totals with different conditional alternatives,
        # report flips, ordered multi-Pauli actions and a partial final block.
        laws = [((1, 12), (1, 6)) if j % 2 else ((1, 4),) for j in range(137)]
        laws += [((0, 1),), ((1, 1),)]
        actions = tuple(tuple(('report', f'm{j}') if j % 3 == 0 else
                              (f'a{j}', (('Y', j), ('X', 0), ('Z', j)))
                              for _ in law) for j, law in enumerate(laws))
        factors = tuple(((7, 5), (2, 3)) if j < 137 else ((1, 1), (1, 1))
                        for j in range(len(laws)))
        plan = native_testing.load().make_file_noise(laws, factors, actions)
        seeds = [Random(19 + j).getrandbits(64) for j in range(64)]
        dense = native_testing.load().draw_file_noise(plan, seeds)
        sparse = native_testing.load().draw_file_noise_actions(plan, seeds)
        self.assertEqual(dense, native_testing.load().draw_file_noise(plan, seeds[:23]) +
                         native_testing.load().draw_file_noise(plan, seeds[23:]))
        for (choices, weight), (selected, sparse_weight) in zip(dense, sparse):
            expected = Fraction(1)
            for j, choice in enumerate(choices):
                expected *= Fraction(*factors[j][choice >= 0])
            self.assertEqual(Fraction(*weight), expected)
            self.assertEqual(weight, sparse_weight)
            self.assertEqual(selected, tuple(actions[j][choice] for j, choice in enumerate(choices)
                                            if choice >= 0))
            self.assertEqual(choices[-2:], (-1, 0))

    def test_large_weight_group_without_count_table(self):
        count = 513
        plan = native_testing.load().make_file_noise([((1, 100),)] * count,
                                      [((101, 100), (1, 2))] * count)
        for choices, weight in native_testing.load().draw_file_noise(plan, range(8)):
            present = sum(choice >= 0 for choice in choices)
            expected = Fraction(101, 100) ** (count - present) * Fraction(1, 2) ** present
            self.assertEqual(Fraction(*weight), expected)

    def test_marginals_and_joint_law_across_lanes_and_blocks(self):
        plan = native_testing.load().make_file_noise([((1, 4),)] * 131)
        shots = 4096
        rng = Random(928)
        draws = native_testing.load().draw_file_noise(plan, [rng.getrandbits(64) for _ in range(shots)])
        counts = Counter()
        for choices, _ in draws:
            for j in (0, 1, 63, 64, 127, 130):
                counts[j] += choices[j] >= 0
            for pair in ((0, 1), (0, 63), (63, 64), (64, 130)):
                counts[pair] += all(choices[j] >= 0 for j in pair)
        for key, observed in counts.items():
            p = 1 / 16 if isinstance(key, tuple) else 1 / 4
            self.assertLess(abs(observed - shots * p), 6 * math.sqrt(shots * p * (1 - p)))
