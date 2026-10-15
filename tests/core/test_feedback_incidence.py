"""Prepared incidence agrees with direct parity evaluation on partial decisions."""
import itertools
import unittest

from magna import ControllerRegistry, parse_circuit
from tests.support.feedback_template import FeedbackTemplate


class FeedbackIncidenceTest(unittest.TestCase):
    def test_partial_decisions_xors_versions_constants_and_wire_remap(self):
        registry=ControllerRegistry()
        registry.register('bits',lambda bits,config:(0,0),version='1')
        circuit=parse_circuit('''H 0
M 0
DETECTOR rec[-1]
CONTROL[bits] det[-1] -> ctrl[0] ctrl[1]
CONTROL[XOR](1) ctrl[0] ctrl[1] -> ctrl[2]
CX ctrl[0] 1
CS ctrl[2] 1
CZ ctrl[1] 2
DETECTOR rec[-1]
CONTROL[bits] det[-1] -> ctrl[0] ctrl[1]
CONTROL[XOR] ctrl[0] ctrl[1] ctrl[2] -> ctrl[3]
CY ctrl[3] 2
CONTROL[NOT] ctrl[3] -> ctrl[4]
POSTSELECT ctrl[4] ctrl[2]
''',controllers=registry)
        template=FeedbackTemplate(circuit)
        terms=sorted({t for _,_,_,row in template.updates for t in row},
                     key=lambda t:(t.kind,t.index,t.version))
        self.assertEqual(len(terms),4)
        for choices in itertools.product((None,0,1),repeat=len(terms)):
            values={t:b for t,b in zip(terms,choices) if b is not None}
            for prefix,wiremap in (('',None),('__recovery/',{i:i+17 for i in range(template.wire_count)})):
                expected=[]
                for anchor,wire,pauli,row in template.updates:
                    bit=template.selection_offsets.get(anchor,0)
                    for term in row: bit^=values.get(term,0)
                    if bit:
                        expected.append(dict(after=prefix+anchor,qubit=wire if wiremap is None else wiremap[wire],pauli=pauli))
                self.assertEqual(list(template.faults(values,prefix=prefix,wires=wiremap)),expected)


if __name__=='__main__':
    unittest.main()
