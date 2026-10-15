"""Exact fixed-input matrix checks for shared reference-layer phase coordinates."""
import random
import unittest

from magna import _native
from tests.support.exact_scalar import ExactScalar, scalar
from tests.core.test_native_core import dense
from tests.support import file_hooks as hooks
from tests.support import native_testing


def matrix_output(width, operations, faults=()):
    state = {0: scalar((1, 0), 3, 0)}
    for index, (kind, qubits, exponent) in enumerate(operations):
        out = {}
        for word, amplitude in state.items():
            if kind == 'H':
                bit = 1 << qubits[0]
                for value in (0, 1):
                    target = (word & ~bit) | (value * bit)
                    term = amplitude * scalar((1, 0), 4 * value * bool(word & bit), -1)
                    out[target] = out.get(target, ExactScalar()) + term
            else:
                target, phase = word, 0
                if kind == 'PHASE':
                    phase = exponent * (sum((word >> q) & 1 for q in qubits) % 2)
                elif kind == 'CCZ':
                    phase = 4 * all((word >> q) & 1 for q in qubits)
                elif kind == 'CX':
                    if word >> qubits[0] & 1:
                        target ^= 1 << qubits[1]
                elif kind == 'CZ':
                    phase = 4 * all((word >> q) & 1 for q in qubits)
                elif kind in ('X', 'Y', 'Z'):
                    q = qubits[0]
                    if kind in ('Y', 'Z'):
                        phase += 4 * ((word >> q) & 1)
                    if kind == 'Y':
                        phase += 2
                    if kind in ('X', 'Y'):
                        target ^= 1 << q
                out[target] = out.get(target, ExactScalar()) + amplitude * scalar((1, 0), phase, 0)
        state = out
        for location, q, pauli in faults:
            if location == index:
                out = {}
                for word, amplitude in state.items():
                    phase = (2 if pauli == 'Y' else 0) + (4*((word >> q)&1) if pauli in ('Y', 'Z') else 0)
                    target = word ^ ((1 << q) if pauli in ('X', 'Y') else 0)
                    out[target] = amplitude * scalar((1, 0), phase, 0)
                state = out
    return {word: value for word, value in state.items() if value}


def prepare(width, operations):
    raw = dict(initial=(width, 0, (), (), (), 0, 3), inputs=tuple(range(width)),
               outputs=tuple(range(width)), records=0, max_path_bits=256,
               operations=tuple((kind, qs, exponent, -1, 0, index, -1, '', '')
                                for index, (kind, qs, exponent) in enumerate(operations)))
    return native_testing.load().debug_prepare_compiler(raw)


def compiled_output(compiled, faults=()):
    x = z = phase = 0
    histories = dict(compiled['histories'])
    suffixes = dict(compiled['suffix'])
    for location, wire, pauli in faults:
        index, position = compiled['locations'][location]
        internal = [q for at, q in histories[wire] if at <= index][-1]
        _, fx, fz = next(row for row in suffixes[internal] if row[0] >= position)
        if pauli == 'X':
            ax, az, ap = fx
        elif pauli == 'Z':
            ax, az, ap = fz
        else:
            ax, az = fx[0] ^ fz[0], fx[1] ^ fz[1]
            ap = fx[2] + fz[2] + 2*(fx[1]&fz[0]).bit_count() + 1
        phase = (ap + phase + 2*(az&x).bit_count()) % 4
        x ^= ax
        z ^= az
    result = {}
    physical = compiled['physical']
    for word, amplitude in dense(compiled['skeleton']).items():
        shifted = word ^ x
        exponent = 2*phase + 4*(z&word).bit_count()
        exponent += sum(c*((shifted >> (physical+j))&1) for j, c in enumerate(compiled['magic']))
        target = shifted & ((1 << physical)-1)
        result[target] = result.get(target, ExactScalar()) + amplitude*scalar((1, 0), exponent, 0)
    return {word: value for word, value in result.items() if value}


class ReferenceLayerCompilerTests(unittest.TestCase):
    def check_circuit(self, width, operations, *, faults=True):
        compiled = prepare(width, operations)
        self.assertEqual(compiled_output(compiled), matrix_output(width, operations))
        if faults:
            for location in range(len(operations)):
                for q in range(width):
                    for pauli in 'XYZ':
                        fault = ((location, q, pauli),)
                        with self.subTest(location=location, qubit=q, pauli=pauli):
                            self.assertEqual(compiled_output(compiled, fault), matrix_output(width, operations, fault))
            if len(operations)>1:
                faults = ((0, 0, 'Y'), (len(operations)-1, 0, 'X'))
                self.assertEqual(compiled_output(compiled, faults), matrix_output(width, operations, faults))
        return compiled

    def test_first_middle_last_and_tie_layers(self):
        h = ('H', (0,), 0)
        t = ('PHASE', (0,), 1)
        cases = [(0, 2, 2, 1, [h,t,t,h,t]), (1, 2, 2, 1, [h,t,h,t,t]),
                 (1, 3, 3, 2, [h,t,h,t,('PHASE',(1,),-1),t,h,t]),
                 (0, 3, 1, 2, [t,h,t,h,t])]
        for selected, layers, retained, ancillas, circuit in cases:
            with self.subTest(circuit=circuit):
                compiled = self.check_circuit(2, circuit)
                self.assertEqual(compiled['reference_layer'], selected)
                self.assertEqual(compiled['rotation_layers'], layers)
                self.assertEqual(compiled['reference_layer_slots'], retained)
                self.assertEqual(compiled['phase_ancillas'], ancillas)

    def test_dependent_constant_inverse_and_ccz_phases(self):
        for circuit in [ [('PHASE',(0,),1),('PHASE',(0,),-1)],
                         [('H',(0,),0),('PHASE',(0,),1),('PHASE',(0,),-1)],
                         [('H',(q,),0) for q in range(3)] + [('CCZ',(0,1,2),0)] ]:
            compiled = self.check_circuit(3, circuit)
            self.assertEqual(compiled['phase_ancillas'], 0)
            self.assertLessEqual(compiled['shared_phase_coordinates'], 3)
        compiled = self.check_circuit(1, [('H',(0,),0),('X',(0,),0),('PHASE',(0,),2)])
        self.assertEqual(compiled['rotation_layers'], 0)

    def test_target_suffix_preserves_parent_layer_and_prefix(self):
        from random import Random
        from magna import compile_circuit, parse_circuit
        from magna.operations import NativeBoundaryTrace
        
        physical = parse_circuit('H 0\nT 0\nX_ERROR(1/4) 0')
        # The inverse target adds two X-axis rotations after the physical Z slot.
        target = parse_circuit('H 0\nT 0\nT 0\nH 0')
        compiled = compile_circuit(physical, ideal_state=target)
        diagnostics = compiled._execution.state()['compiler_diagnostics']
        self.assertEqual(diagnostics['protocol']['reference_layer'], 0)
        self.assertEqual(diagnostics['target']['reference_layer'], 0)
        self.assertEqual(diagnostics['target']['phase_ancillas'], 2)
        self.assertEqual(diagnostics['target']['layer_slots'], (1, 2))
        self.assertGreater(compiled._execution.state()['shared_path_gates'], 0)
        for noise in hooks.draw_many(compiled.noise_schedule, Random(16), 8):
            actual = compiled.sample(Random(4), noise=noise)
            ir = hooks.lower(physical, (0,), {}, noise)
            target_ir, _ = hooks.attach_target(compiled.ideal_state, ir, physical.num_records)
            fresh = NativeBoundaryTrace(target_ir)
            rows = fresh.rows((record,) for record in compiled.target_records)
            value = fresh.query(hooks.pack(compiled, fresh, noise), event_rows=rows,
                                outcomes=(0,)*len(rows)).total()
            self.assertEqual(actual['exact']['target_success'], value.to_dict())

    def test_clifford_parent_can_select_first_target_layer(self):
        from random import Random
        from magna import compile_circuit, parse_circuit
        compiled = compile_circuit(parse_circuit('H 0'),
                                   ideal_state=parse_circuit('H 0\nT 0'))
        diagnostics = compiled.inspect()['compilation']['compiler_diagnostics']
        self.assertEqual(diagnostics['protocol']['reference_layer'], -1)
        self.assertEqual(diagnostics['target']['reference_layer'], 0)
        self.assertEqual(diagnostics['target']['phase_ancillas'], 0)
        self.assertGreater(compiled._execution.state()['shared_path_gates'], 0)
        value = compiled.sample(Random(3))['exact']['target_success']
        from magna.scalar import ExactScalar as PublicScalar
        found = PublicScalar(tuple(value['coefficients']), value['power'])
        self.assertAlmostEqual(float(found.real_decimal()), (2 + 2**0.5) / 4)

    def test_small_ordered_random_circuits(self):
        rng = random.Random(260930)
        for trial in range(24):
            operations = []
            for _ in range(8):
                kind = rng.choice(('H', 'PHASE', 'CX', 'CZ', 'X', 'Y', 'Z'))
                qs = (0,1) if kind in ('CX','CZ') else (rng.randrange(2),)
                operations.append((kind,qs,rng.choice((1,2,3,5,6,7)) if kind=='PHASE' else 0))
            with self.subTest(trial=trial):
                self.check_circuit(2, operations)

    def test_compact_suffix_preserves_ordered_fault_phases(self):
        # These are exact amplitudes, so a lost common i/sign cannot hide in a
        # probability. Same-location Pauli faults retain their declared order.
        operations = [('H', (0,), 0), ('PHASE', (0, 1), 3),
                      ('Y', (1,), 0), ('CX', (1, 0), 0),
                      ('PHASE', (0,), -2), ('H', (1,), 0),
                      ('PHASE', (0, 1), -1), ('CZ', (0, 1), 0)]
        compiled = prepare(2, operations)
        for faults in (
                ((1, 0, 'X'), (1, 0, 'Y')),
                ((1, 0, 'Y'), (1, 0, 'X')),
                ((1, 1, 'Z'), (1, 1, 'Y'), (1, 1, 'X')),
                ((0, 0, 'Y'), (3, 1, 'X'), (6, 0, 'Z'), (7, 1, 'Y')),
                ((2, 1, 'Y'), (2, 1, 'Y')),
                ((0, 0, 'X'), (0, 1, 'Z'), (7, 0, 'Y'), (7, 1, 'X'))):
            with self.subTest(faults=faults):
                self.assertEqual(compiled_output(compiled, faults),
                                 matrix_output(2, operations, faults))
        first = compiled_output(compiled, ((1, 0, 'X'), (1, 0, 'Y')))
        reverse = compiled_output(compiled, ((1, 0, 'Y'), (1, 0, 'X')))
        self.assertEqual(first, {word: -amplitude for word, amplitude in reverse.items()})


if __name__ == '__main__':
    unittest.main()
