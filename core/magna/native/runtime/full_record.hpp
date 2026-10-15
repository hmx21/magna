// Whole-record execution strategies, snapshots and batch workspace ownership.
#pragma once
#include "operation_transport.hpp"
#include <memory>
namespace magic_native {
struct RecordStreamPolicy {
    size_t block_size;
    RecordStreamPolicy(bool compact, size_t requested) : block_size(compact ? requested : 1) {
        if(!requested || requested > 65536) throw std::invalid_argument("native batch size must be in 1..65536");
    }
};
struct FullRecordInput {
    ValidatedAssignment values;
    std::vector<OriginalFault> faults;
    uintptr_t immutable_fault_identity = 0;
};
struct FullRecordResult {
    Scalar probability;
    std::optional<CollectedBranch> branch;
    PhaseTensor normal;
    PhaseResult certificate;
    std::optional<CompilerInput> literal_input;
    std::optional<CompilerPreparation> literal_compiler;
};
inline ControlGate branch_control_gate(const BranchGate& source) {
    const auto& op = source.operation.value;
    ControlGate gate;
    gate.qubits = op.qubits;
    gate.guard = {{0, 1}};
    gate.exponent = {{0, 1}};
    if(op.kind == "I") gate.kind = 0;
    else if(op.kind == "PAULI" || op.kind == "X" || op.kind == "Y" || op.kind == "Z") {
        auto name = op.kind == "PAULI" ? op.pauli : op.kind;
        auto p = std::string("IXZY").find(name);
        if(name.size() != 1 || p == std::string::npos || op.qubits.size() != 1)
            throw ControlMismatch("feedback Pauli arity or name");
        gate.kind = int(p);
    } else if(op.kind == "T" || op.kind == "S" || op.kind == "PHASE") {
        gate.kind = 6;
        gate.exponent = {{0, size_t(mod8(op.exponent))}};
    } else throw ControlMismatch("feedback changes the declared monomial schedule");
    return gate;
}
struct FullRecordExecution {
    Ket initial;
    std::vector<int> inputs, outputs;
    std::vector<ProgramOperation> operations;
    std::vector<int> inline_outcomes;
    BranchSchedule& schedule;
    PhaseDatabase database;
    PhaseLimits limits;
    size_t max_paths;
    bool use_database;
    CompilerInput input;
    std::shared_ptr<GlobalPlan> global;
    std::unique_ptr<OperationTransport> transport;
    std::string literal_reason;
    std::atomic<size_t> literal_compilations{0}, packed_fault_groups{0};
    FullRecordExecution(Ket initial, std::vector<int> inputs, std::vector<int> outputs,
            std::vector<ProgramOperation> operations, std::vector<int> outcomes, BranchSchedule& schedule,
            std::map<std::string, std::optional<NativeControl>> families, PhaseDatabase database, PhaseLimits limits,
            size_t paths, size_t bytes, bool use_database, bool reuse);
    std::shared_ptr<GlobalPlan> prepare(const CompilerInput& source, size_t bytes) const;
    CollectedBranch collect(const FullRecordInput& row, const BranchSchedule::Callback& callback, bool native = false);
    FullRecordResult literal(CollectedBranch branch, bool certificate);
};
struct FullRecordBatch {
    FullRecordExecution& execution;
    bool certificate;
    struct Row {
        FullRecordInput input;
        Assignment quantum;
        std::optional<CollectedBranch> branch;
        std::optional<FullRecordResult> result;
    };
    std::vector<Row> rows;
    uintptr_t previous_fault_identity = 0;
    std::vector<Fault> previous_faults;
    size_t workspace_peak = 0;
    double fallback_seconds = 0., control_seconds = 0.;
    explicit FullRecordBatch(FullRecordExecution& execution, bool certificate = false)
        : execution(execution), certificate(certificate) {}
    void add(FullRecordInput input, const BranchSchedule::Callback& callback);
    std::vector<FullRecordResult> finish();
};
} // namespace magic_native
