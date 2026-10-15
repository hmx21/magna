"""Literal surface-code 15-to-1 with noisy physical T inputs and ideal encoding.

Five data patches implement the 15 commuting RM phase gadgets; a sixth patch
is reused for the fifteen T inputs. A binary Gray walk synthesizes the parity
rotations using transversal CNOTs. S-dagger feedforward uses Chen's inverse
half-cycle fold, including its ancillas. Output is the entire encoded patch.
"""
import argparse
from fractions import Fraction
import json
from pathlib import Path

from magna import ControllerRegistry, parse_circuit
import stim

from .decoder import decode
from .patch import Patch


class Builder:
    def __init__(self, d, p, q, *, encoded_input_error=None, minimal_syndromes=False):
        self.patch = Patch(d)
        self.p, self.q = Fraction(str(p)), Fraction(str(q))
        if not (0 <= self.p <= 1 and 0 <= self.q <= 1):
            raise ValueError('p and q must be probabilities')
        self.encoded_input_error = (None if encoded_input_error is None
                                    else Fraction(str(encoded_input_error)))
        if self.encoded_input_error is not None:
            if not 0 <= self.encoded_input_error <= 1 or self.p or self.q:
                raise ValueError('encoded-input noise requires a probability and p=q=0')
        if minimal_syndromes and self.encoded_input_error is None:
            raise ValueError('minimal syndromes require encoded-input noise with p=q=0')
        self.minimal_syndromes = minimal_syndromes
        self.lines = ['# Rotated surface-code RM15; patches 0..4 data, 5 resource.']
        self.records = 0
        self.control = 0
        self.bindings = {}
        self.active = set(range(5))
        self.live_ancillas = set()
        self.stats = {'distance':d, 'p':str(self.p), 'q':str(self.q),
                      'physical_qubits':6*self.patch.width, 'data_patches':5,
                      'resource_patches':1, 'magic_inputs':15, 'memory_windows':0,
                      'fold_cycles':0, 'ordinary_cycles':0, 'transversal_cnots':0}

    def wire(self, block, q):
        return block*self.patch.width+q

    def emit(self, name, targets, probability=None, guard=None):
        if not targets:
            return []
        text = name + (f'({probability})' if probability is not None else '')
        if guard is not None:
            text += ' ' + guard
        self.lines.append(text + ' ' + ' '.join(map(str,targets)))
        if name in ('M','MX'):
            old = self.records
            self.records += len(targets)
            return list(range(old,self.records))
        return []

    def noise(self, name, targets, guard=None):
        if self.p:
            self.emit(name,targets,probability=self.p,guard=guard)

    def layer(self, ops, *, guard=None):
        """One physical time step, including p noise on all live idle data."""
        used = set()
        for name, targets in ops:
            targets = list(targets)
            used.update(targets)
            if guard is None:
                self.emit(name,targets)
                self.noise('DEPOLARIZE2' if name in ('CX','CZ') else 'DEPOLARIZE1',targets)
            else:
                conditional = {'S':'CS','S_DAG':'CS_DAG','CZ':'CCZ'}[name]
                stride = 2 if name == 'CZ' else 1
                for i in range(0,len(targets),stride):
                    group = targets[i:i+stride]
                    self.emit(conditional,[guard,*group])
                    self.noise('DEPOLARIZE2' if stride == 2 else 'DEPOLARIZE1',group,guard)
                    complement = guard[1:] if guard.startswith('!') else '!'+guard
                    self.noise('DEPOLARIZE1',group,complement)
        live = {self.wire(b,q) for b in self.active for q in range(self.patch.n)}
        idle = sorted((live | self.live_ancillas) - used)
        self.noise('DEPOLARIZE1',idle)
        self.lines.append('TICK')

    def controls(self, name, records, count, **config):
        definition = {'distance':self.patch.d, **config}
        if name in self.bindings and self.bindings[name]['config'] != definition:
            raise ValueError('controller configuration changed')
        self.bindings[name] = {'import':'decoder.decoder:decode',
                               'version':'1', 'config':definition}
        regs = [f'ctrl[{self.control+i}]' for i in range(count)]
        self.control += count
        # The decoder reads one-record detectors declared directly before its call.
        inputs = list(dict.fromkeys(f'rec[{r-self.records}]' for r in records))
        self.lines.extend(f'DETECTOR {record}' for record in inputs)
        self.lines.append(f'CONTROL[{name}] '
                          + ' '.join(f'det[{inputs.index(f"rec[{r-self.records}]")-len(inputs)}]' for r in records)
                          + ' -> ' + ' '.join(regs))
        return regs

    def corrections(self, block, x, z=()):
        # Pauli-frame updates: exact conditional actions, no physical pulse or noise.
        for name,bits in (('CX',x),('CZ',z)):
            for q,bit in enumerate(bits):
                self.emit(name,[bit,self.wire(block,q)])

    def ideal_recovery(self, block, name):
        records = []
        for ins in self.patch.cycle():
            records.extend(self.emit(ins.name, [self.wire(block, t.value)
                                               for t in ins.targets_copy()]))
        bits = self.controls(name, records, 2*self.patch.n, mode='ideal_recovery')
        self.corrections(block, bits[:self.patch.n], bits[self.patch.n:])

    def input_noise_recovery(self, block):
        if self.encoded_input_error is not None:
            self.emit('DEPOLARIZE1', [self.wire(block,q) for q in range(self.patch.n)],
                      probability=self.encoded_input_error)
            self.ideal_recovery(block, 'ideal_input_recovery')

    def cycle(self, block, fold_control=None):
        a = self.patch
        self.live_ancillas = {self.wire(block,q) for q in a.anc}
        self.stats['fold_cycles' if fold_control else 'ordinary_cycles'] += 1
        self.emit('R',[self.wire(block,q) for q in a.anc])
        self.noise('X_ERROR',[self.wire(block,q) for q in a.anc])
        self.layer([('H',[self.wire(block,q) for q in a.xanc])])
        for i,pairs in enumerate(a.layers()):
            self.layer([('CX',[self.wire(block,q) for pair in pairs for q in pair])])
            if i == 1 and fold_control is not None:
                self.layer([(ins.name,[self.wire(block,t.value) for t in ins.targets_copy()])
                            for ins in a.fold(inverse=True)],guard=fold_control)
        self.layer([('H',[self.wire(block,q) for q in a.xanc])])
        records = self.emit('M',[self.wire(block,q) for q in a.anc],probability=self.p)
        self.live_ancillas = set()
        return records

    def memory(self, blocks):
        if self.minimal_syndromes:
            return
        # Serial block servicing is explicit: other live data receive idle noise.
        for block in blocks:
            self.stats['memory_windows'] += 1
            records = [r for _ in range(self.patch.d) for r in self.cycle(block)]
            bits = self.controls('memory',records,2*self.patch.n,
                                 mode='memory',rounds=self.patch.d,
                                 model_rate=str(self.p) if self.p else '0.001')
            self.corrections(block,bits[:self.patch.n],bits[self.patch.n:])

    def prepare_plus(self, block):
        data = [self.wire(block,q) for q in range(self.patch.n)]
        self.emit('RX',data)
        self.noise('Z_ERROR',data)
        syndrome = self.cycle(block)
        bits = self.controls('gauge',syndrome[len(self.patch.xanc):],self.patch.n,mode='gauge')
        self.corrections(block,bits)
        self.memory([block])

    def resource(self, index):
        self.active.add(5)
        self.lines.append(f'# Physical T input {index}: Z noise before the ideal encoder.')
        self.emit('R',[self.wire(5,q) for q in range(self.patch.width)])
        self.emit('H',[self.wire(5,0)])
        self.emit('T',[self.wire(5,0)])
        self.emit('Z_ERROR',[self.wire(5,0)],probability=self.q)
        for ins in self.patch.encoder:
            self.emit(ins.name,[self.wire(5,t.value) for t in ins.targets_copy()])
        self.lines.append('# End external T supply.')
        self.input_noise_recovery(5)

    def cnot(self, control, target):
        self.stats['transversal_cnots'] += 1
        self.layer([('CX',[self.wire(b,q) for q in range(self.patch.n) for b in (control,target)])])
        self.memory([control,target])

    def readout(self, block, basis):
        records = self.emit('MX' if basis == 'X' else 'M',
                            [self.wire(block,q) for q in range(self.patch.n)],probability=self.p)
        bit, = self.controls('readout_'+basis,records,1,mode='readout',basis=basis)
        self.active.discard(block)
        return bit

    def build(self):
        for b in range(5):
            self.prepare_plus(b)
        previous = 0
        for i in range(1,16):
            column = i ^ (i >> 1)
            delta = previous ^ column
            self.cnot(delta.bit_length(),0)
            self.resource(i-1)
            self.cnot(0,5)
            outcome = self.readout(5,'Z')
            # T input produces T on outcome 0, T-dagger on outcome 1.
            # S-dagger conditioned on outcome 0 therefore gives T-dagger always.
            self.cycle(0,fold_control='!'+outcome)
            self.memory([0])
            previous = column
        self.cnot(previous.bit_length(),0)
        checks = [self.readout(b,'X') for b in range(1,5)]
        self.lines.append('POSTSELECT '+' '.join(checks))
        self.stats.update(measurement_records=self.records,control_bits=self.control,
                          architecture='compact', unitary_ticks=self.lines.count('TICK'),
                          encoded_input_error=(None if self.encoded_input_error is None
                                               else str(self.encoded_input_error)),
                          minimal_syndromes=self.minimal_syndromes,
                          ideal_input_recovery_rounds=15 if self.encoded_input_error is not None else 0)
        target = 'H 0\nT 0\n' + str(self.patch.encoder) + '\n'
        task = {'schema_version':1,'outputs':list(range(self.patch.n)),
                'ideal_state':'ideal.magna'}
        controllers = {'schema_version':1,'bindings':self.bindings}
        return '\n'.join(self.lines)+'\n', target, task, controllers, self.stats


def generate(d=3,p='0',q='0.01', *, encoded_input_error=None, minimal_syndromes=False):
    return Builder(d,p,q,encoded_input_error=encoded_input_error,
                   minimal_syndromes=minimal_syndromes).build()


def parsed(text, controllers):
    registry = ControllerRegistry()
    for name,binding in controllers['bindings'].items():
        registry.register(name,decode,version=binding['version'],config=binding['config'])
    return parse_circuit(text,controllers=registry)


def write(folder, d=3,p='0',q='0.01', *, encoded_input_error=None, minimal_syndromes=False):
    text,target,task,controllers,stats = generate(d,p,q,encoded_input_error=encoded_input_error,
                                              minimal_syndromes=minimal_syndromes)
    folder = Path(folder)
    folder.mkdir(parents=True,exist_ok=True)
    for name,value in [('physical.magna',text),('ideal.magna',target),
                       ('task.json',task),('controllers.json',controllers),('circuit.json',stats)]:
        (folder/name).write_text(value if isinstance(value,str) else json.dumps(value,indent=2)+'\n')
    return text,target,task,controllers,stats


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--distance',type=int,default=3)
    ap.add_argument('--p',default='0')
    ap.add_argument('--q',default='0.01')
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--encoded-input-error',
                    help='Independent data depolarization after each ideal input encoder, then ideal recovery; requires p=q=0.')
    ap.add_argument('--minimal-syndromes', action='store_true',
                    help='For encoded-input noise only: omit storage rounds and retain preparation, input recovery and full fold rounds.')
    a=ap.parse_args()
    print(json.dumps(write(a.output,a.distance,a.p,a.q,encoded_input_error=a.encoded_input_error,
                          minimal_syndromes=a.minimal_syndromes)[-1],indent=2))


if __name__ == '__main__':
    main()
