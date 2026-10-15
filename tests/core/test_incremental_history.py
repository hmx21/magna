"""Replay incremental causal queries against the pre-optimization production stream."""
from hashlib import sha256
from fractions import Fraction
import json
from random import Random
import unittest

from magna import ControllerRegistry, compile_circuit, parse_circuit


# Captured from production core a44eab122f19d0c20d760c3f3194bff0196b099c,
# before incremental-history changes. Each digest includes every returned draw
# field, the complete caller RNG state and all controller inputs/decisions.
BASELINE = {
    (False, 'affine_support'): '69da1d9e326780fe91a782daed2ae278a453fce9bd5bb5743800bb99343dba72',
    (False, 'exact_marginal'): '16a30cdbc947993e0cd31c282b5b848da27217b2ec9f624bff19e9c846b865ae',
    (True, 'affine_support'): '592226dbfc1a8d9cdb4c371fd5bfdbb9bc30ae0e7cdc17218c88b8c32d697003',
    (True, 'exact_marginal'): 'f7dc316f97bb0007ddaa9397971b33e252dce9a6a5552c009cb3bb106ae4f265',
}


def fixture(transcript, magic):
    registry = ControllerRegistry()
    def record(bits, config):
        transcript.append((tuple(bits), dict(config)))
        return bits
    registry.register('record', record, version='1')
    prefix = 'H 0 1 2\n' + ('T 0 2\n' if magic else '')
    physical = parse_circuit(prefix +
        'M(1/5) 0 1\n'
        'DETECTOR rec[-2]\n'
        'CONTROL[record] det[-1] -> ctrl[0]\n'
        'CS ctrl[0] 2\n'
        'MX(1/7) 2\n'
        'DETECTOR rec[-2]\nDETECTOR rec[-1]\n'
        'CONTROL[record] det[-2] det[-1] -> ctrl[1] ctrl[2]\n'
        'CH ctrl[1] 2\n'
        'Z_ERROR(1/3) ctrl[2] 2\n'
        'MX 2\n'
        'DETECTOR rec[-4]\nDETECTOR rec[-1]\n'
        'CONTROL[record] det[-2] det[-1] -> ctrl[3] ctrl[4]\n'
        'CZ ctrl[3] 2\n'
        'X_ERROR(1/4) ctrl[4] 3\n'
        'M 3\n'
        'DETECTOR rec[-1]\n'
        'CONTROL[record] det[-1] -> ctrl[5]\n'
        'CX ctrl[5] 4\nCX 4 5\n'
        'DETECTOR rec[-5] rec[-4]\nPOSTSELECT det[6]\n', controllers=registry)
    return physical, dict(outputs=[4, 5], sample_records=[0, 1, 2, 3, 4])


class IncrementalHistoryTests(unittest.TestCase):
    def test_interference_zero_prefix_completes_controllers_and_guarded_noise(self):
        triples = ((0, 1, 2), (0, 1, 3), (0, 2, 3))
        # Independent Walsh sum for the four measured prefix bits.
        amplitudes = [sum((-1)**(sum(((x >> i) & 1)*((x >> j) & 1)*((x >> k) & 1)
            for i, j, k in triples)+(x & y).bit_count()) for x in range(16)) for y in range(16)]
        probabilities = [Fraction(a*a, 256) for a in amplitudes]
        events = []
        class LoggedRandom(Random):
            def getrandbits(self, width):
                events.append(('rng', width))
                return super().getrandbits(width)
        registry = ControllerRegistry()
        def record(bits, config):
            events.append(('controller', tuple(bits)))
            return bits
        registry.register('record', record, version='1')
        source = ('H 0 1 2 3\n' + ''.join(f'CCZ {i} {j} {k}\n' for i, j, k in triples)
            + 'H 0 1 2 3\nM 0 1 2 3\n'
            'DETECTOR rec[-4]\nDETECTOR rec[-3]\nDETECTOR rec[-2]\nDETECTOR rec[-1]\n'
            'CONTROL[record] det[-4] det[-3] det[-2] det[-1] -> ctrl[0] ctrl[1] ctrl[2] ctrl[3]\n'
            'X_ERROR(1) ctrl[0] 4\n'
            'CONTROL[record] ctrl[0] -> ctrl[4]\n'
            'M 4\nH 5\nT 5\nCZ ctrl[4] 5\nMX 5\n'
            'DETECTOR rec[-2]\nDETECTOR rec[-1]\n'
            'CONTROL[record] det[-2] det[-1] -> ctrl[5] ctrl[6]\n'
            'CX ctrl[5] 6\nCX ctrl[6] 7')
        compiled = compile_circuit(parse_circuit(source, controllers=registry),
            task=dict(outputs=[6, 7], sample_records=list(range(6))),
            ideal_state=parse_circuit('I 0 1'), options=dict(history_sampler='affine_support'))
        zero_prefixes = set()
        for seed in range(256):
            events.clear()
            draw = compiled.sample(LoggedRandom(seed))
            word = draw['sampled_records']
            prefix = sum(bit << j for j, bit in enumerate(word[:4]))
            callbacks = [(j, value) for j, (kind, value) in enumerate(events) if kind == 'controller']
            self.assertEqual([value for _, value in callbacks],
                [tuple(word[:4]), (word[0],), tuple(word[4:])])
            self.assertEqual(draw['decoder_calls'], 3)
            # Between these callbacks there is only the guarded physical
            # channel. Its draw must still execute for a zero-weight prefix.
            self.assertEqual(events[callbacks[0][0]+1:callbacks[1][0]], [('rng', 64)]*word[0])
            raw = draw['exact']['history_mass']
            actual = tuple(Fraction(c)*Fraction(2)**raw['power'] for c in raw['coefficients'])
            p = probabilities[prefix]
            sign = 1-2*(word[5] ^ word[0])
            self.assertEqual(actual, (p/2, sign*p/4, 0, -sign*p/4))
            if p:
                self.assertEqual(word[4], word[0])
                self.assertEqual(draw['error'], draw['acceptance']*int(bool(word[4] or word[5])))
            else:
                zero_prefixes.add(prefix)
                self.assertEqual(draw['acceptance'], 0)
                self.assertEqual(draw['error'], 0)
                self.assertEqual(draw['weight_scientific'], (0.0, 0))
        self.assertEqual(zero_prefixes, {j for j, p in enumerate(probabilities) if not p})
        self.assertTrue(any(word & 1 for word in zero_prefixes))

    def test_complete_draw_rng_and_controller_replay(self):
        # The first callback asks only for m0. The next needs m1, which was
        # traced only in the earlier temporary query, plus a new noncommuting
        # measurement. Later calls revisit m0 after actual CS/CH feedback and
        # guarded noise. T gates additionally exercise the general term path.
        for magic in (False, True):
            for method in ('affine_support', 'exact_marginal'):
                for budgeted in (False, True):
                    with self.subTest(magic=magic, method=method, budgeted=budgeted):
                        transcript = []
                        circuit, task = fixture(transcript, magic)
                        options = dict(history_sampler=method)
                        if budgeted:
                            options.update(max_queries=10000, max_plans=1)
                        compiled = compile_circuit(circuit, task=task,
                            ideal_state=parse_circuit('I 0 1'), options=options)
                        rows = []
                        for seed in range(32):
                            transcript.clear()
                            rng = Random(26090400 + seed)
                            draw = compiled.sample(rng)
                            word = draw['sampled_records']
                            self.assertEqual(transcript, [
                                ((word[0],), {}), ((word[1], word[2]), {}),
                                ((word[0], word[3]), {}), ((word[4],), {}),
                            ])
                            self.assertEqual(draw['decoder_calls'], 4)
                            self.assertEqual(draw['error'], draw['acceptance']*word[4])
                            if word[0] != word[1]:
                                self.assertEqual(draw['acceptance'], 0)
                            rows.append(dict(draw=draw, rng=rng.getstate(), controllers=list(transcript)))
                        payload = json.dumps(rows, sort_keys=True, separators=(',', ':')).encode()
                        self.assertEqual(sha256(payload).hexdigest(), BASELINE[magic, method])
                        state = compiled._execution.state()
                        self.assertEqual(state['persistent_networks'], 3)
                        self.assertEqual(state['reference_network_resource_constructions'], 0)
                        self.assertEqual(state['reference_network_coordinate_constructions'], 0)


if __name__ == '__main__':
    unittest.main()
