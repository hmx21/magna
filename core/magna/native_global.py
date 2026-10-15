"""Python data and certificate adapters for native full-record execution."""
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter
from . import _native
from .batch import CompactAssignment
from .ir import fingerprint
from .native_compiler import compiler_snapshot_to_python
from .native_feedback import PreparedFeedback, BranchDecisions
from .scalar import ExactScalar


class GlobalNativePlan:
    def __init__(self, prepared, options):
        self.prepared = prepared; self.options = options
        self.feedback = PreparedFeedback(prepared)
        tick = perf_counter(); database = None; database_sha = None
        if options.backend == 'database':
            raw = (Path(__file__).parent/'data/phase_decompositions.json').read_bytes()
            database_sha = sha256(raw).hexdigest(); database = json.loads(raw)
        ir = prepared.ir; k = ir.initial
        self._handle, self.handle, info, self.metadata = _native.make_full_record(
            self.feedback._handle, (k.width,k.base,k.columns,k.linear,k.upper,k.power,k.phase),
            ir.inputs, ir.outputs, ir.operations,
            tuple((oid, None if node.feedback is None else
                (node.feedback.qubits,node.feedback.odd_axes,node.feedback.even_axes))
                for oid,node in ir.controls.items()), database,
            dict(max_quotient_rank=options.max_quotient_rank, max_terms=options.max_terms,
                 max_pair_contractions=options.max_pair_contractions, max_path_bits=options.max_path_bits,
                 max_structure_bytes=options.max_structure_bytes), options.reuse_structure)
        self.metadata['fingerprint'] = prepared.fingerprint
        self.info = dict(native_prepare_calls=int(self.handle is not None), threads=1,
                         database_sha256=database_sha)
        self.info.update(engine='native', probability_kernel='single-sided-trace')
        self.info.update(info)
        prepared.structure_info = dict(self.info, enabled=self.handle is not None,
                                       seconds=perf_counter()-tick)

    def _rows(self, assignments, certificate):
        for assignment in assignments:
            compact = isinstance(assignment, CompactAssignment)
            if compact and assignment.fingerprint != self.prepared.fingerprint:
                raise ValueError('compact assignment belongs to a different IR snapshot')
            true = assignment.true_word if compact else tuple(assignment.true.items())
            flips = assignment.report_word if compact else assignment.report_flips
            snapshot = (deepcopy(assignment.true), tuple(deepcopy(assignment.faults))) if certificate else None
            yield (true, flips, tuple((f['after'],f['qubit'],f['pauli']) for f in assignment.faults),
                   compact, assignment.faults if compact else None, snapshot)

    def evaluate(self, assignments):
        raw, timings, info = _native.evaluate_full_records(self._handle, self._rows(assignments, False), False)
        self.info.update(info)
        return [ExactScalar(*value) for value in raw], timings, len(raw)

    def batches(self, assignments, compact, size):
        cursor = _native.make_full_record_cursor(self._handle, self._rows(assignments, False), size, compact)
        while (result := _native.next_full_record_cursor(cursor)) is not None:
            raw, timings, info = result
            self.info.update(info)
            yield [ExactScalar(*value) for value in raw], timings, len(raw)

    def snapshot(self):
        return compiler_snapshot_to_python(_native.debug_global_compiler(self.handle), self.metadata)

    def _tensor(self, raw, environment):
        from .phase_tensor import QuadraticPhaseTensor
        from .stabilizer import StabilizerKet
        raw=dict(raw);raw['environment']=environment
        certificates=[]
        for c in raw['certificates']:
            c=dict(c)
            if c.get('contracted_tensor') is not None:
                c['contracted_tensor']=StabilizerKet(*c['contracted_tensor'])
            certificates.append(c)
        raw['certificates']=tuple(certificates)
        return QuadraticPhaseTensor(**raw)

    def compile(self, assignment, simulator):
        from .simulator import CompiledSample
        from .stabilizer import StabilizerKet
        from .evaluate import StabilizerExpansion
        from dataclasses import replace
        raw, timings, info = _native.evaluate_full_records(self._handle, self._rows([assignment], True), True)
        self.info.update(info); data = raw[0]
        handle, branch_data, (true, faults) = data.pop('branch')
        branch = BranchDecisions(self.feedback, true, faults, handle, branch_data).materialize()
        metadata = dict(data.get('metadata', self.metadata), fingerprint=self.prepared.fingerprint)
        snapshot = (compiler_snapshot_to_python(data['snapshot'], metadata)
                    if 'snapshot' in data else self.snapshot())
        environment = snapshot['environment']
        normal = self._tensor(data['normal'], environment)
        absorbed = self._tensor(data['absorbed'], environment)
        data['resource']['core_phase_mod8'] = list(data['resource']['core_phase_mod8'])
        source = dict(stage='native_global_preparation', ir_fingerprint=self.prepared.fingerprint,
            original_faults=branch.faults, true=branch.true, reported=branch.reported,
            snapshot=snapshot, input_phase_retained=True, record_histories_enumerated=0)
        normal = replace(normal, certificates=(source,)+normal.certificates)
        expansion = StabilizerExpansion(tuple((ExactScalar(tuple(c),p),StabilizerKet(*k))
            for (c,p),k in data['states']))
        value = ExactScalar(*data['probability'])
        data['resource'].update(native_global_compile=True,database_sha256=self.info['database_sha256'])
        result = CompiledSample(branch, normal, absorbed, expansion, data['resource'], value,
            timings, fingerprint(('native-global-v1',branch.operations,branch.true,self.prepared.fingerprint,self.options)),
            status='completed' if value else 'zero')
        simulator.execution_info['native_certificates'] = simulator.execution_info.get('native_certificates',0)+1
        return result.materialize()
