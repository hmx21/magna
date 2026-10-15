// Full-record compilation and execution, independent of CPython.
#include "full_record.hpp"
namespace magic_native {
FullRecordExecution::FullRecordExecution(Ket initial, std::vector<int> inputs, std::vector<int> outputs,
        std::vector<ProgramOperation> operations, std::vector<int> outcomes, BranchSchedule& schedule,
        std::map<std::string, std::optional<NativeControl>> families, PhaseDatabase database, PhaseLimits limits,
        size_t paths, size_t bytes, bool use_database, bool reuse)
    : initial(std::move(initial)), inputs(std::move(inputs)), outputs(std::move(outputs)),
      operations(std::move(operations)), inline_outcomes(std::move(outcomes)), schedule(schedule),
      database(std::move(database)), limits(limits), max_paths(paths), use_database(use_database) {
    if(!reuse) return;
    try {
        input = compile_input(this->initial, this->inputs, this->outputs, this->operations, families, paths);
        auto candidate = prepare(input, bytes);
        transport = std::make_unique<OperationTransport>(input, *candidate, schedule, std::move(families), bytes);
        global = std::move(candidate);
    } catch(const UnsupportedFeedback& e) { literal_reason = e.what(); } catch(const PhaseBudgetError& e) {
        literal_reason = e.what();
    }
}

std::shared_ptr<GlobalPlan> FullRecordExecution::prepare(const CompilerInput& source, size_t bytes) const {
    auto plan = std::make_shared<GlobalPlan>();
    plan->ir = source.ir;
    plan->database = database;
    plan->limits = limits;
    plan->use_database = use_database;
    prepare_global(*plan, bytes);
    return plan;
}

CollectedBranch FullRecordExecution::collect(
        const FullRecordInput& row, const BranchSchedule::Callback& callback, bool native) {
    std::optional<std::vector<std::vector<BranchGate>>> emitted;
    if(native) {
        emitted.emplace();
        for(size_t j = 0; j < schedule.controls.size(); j++) {
            std::map<std::string, int> values;
            for(const auto& name :
                    schedule.layout.controls.at(schedule.layout.operations[schedule.controls[j]].id).parents)
                values[name] = (row.values.true_word ^ row.values.flip_word).bit(schedule.layout.record_index.at(name));
            emitted->push_back(evaluated_branch_gates(*schedule.rules[j], values));
        }
    }
    return schedule.collect(row.values, row.faults, callback, emitted);
}

FullRecordResult FullRecordExecution::literal(CollectedBranch branch, bool certificate) {
    std::vector<ProgramOperation> ops;
    std::vector<int> outcomes;
    for(const auto& step : branch.steps) {
        auto op = step.replacement ? step.replacement->operation
                : step.kind == 0   ? operations.at(step.index)
                                   : branch.decisions.at(step.index).gates.at(step.subindex).operation;
        op.id = step.id;
        outcomes.push_back(step.measured.value_or(step.kind == 0 ? inline_outcomes.at(step.index) : 0));
        ops.push_back(std::move(op));
    }
    auto [source, assignment] = compile_literal_input(initial, inputs, outputs, std::move(ops), outcomes, max_paths);
    auto plan = prepare(source, std::numeric_limits<size_t>::max());
    Workspace workspace;
    FullRecordResult result;
    result.branch = std::move(branch);
    if(certificate) {
        result.normal = plan->actual(assignment);
        result.certificate = plan->certificate(assignment, result.normal, workspace);
        result.probability = result.certificate.probability;
        result.literal_input = std::move(source);
        result.literal_compiler = std::move(plan->compiled);
    } else result.probability = plan->evaluate(assignment, workspace);
    literal_compilations++;
    return result;
}

void FullRecordBatch::add(FullRecordInput input, const BranchSchedule::Callback& callback) {
    auto& e = execution;
    Row row;
    row.input = std::move(input);
    if(!e.global) {
        auto tick = std::chrono::steady_clock::now();
        row.result = e.literal(e.collect(row.input, callback), certificate);
        fallback_seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - tick).count();
    } else {
        std::optional<FeedbackOffsets> feedback;
        if(!e.schedule.controls.empty() && !e.transport->native_controls) {
            row.branch = e.collect(row.input, callback);
            try {
                std::vector<std::pair<std::string, std::vector<ControlGate>>> decisions;
                for(const auto& decision : row.branch->decisions) {
                    std::vector<ControlGate> gates;
                    for(const auto& gate : decision.gates) gates.push_back(branch_control_gate(gate));
                    decisions.push_back({decision.id, std::move(gates)});
                }
                feedback = e.transport->resolve(decisions);
            } catch(const ControlMismatch&) {
                auto tick = std::chrono::steady_clock::now();
                row.result = e.literal(std::move(*row.branch), certificate);
                row.branch.reset();
                fallback_seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - tick).count();
            }
        }
        if(!row.result) {
            auto identity = row.input.immutable_fault_identity;
            if(e.schedule.controls.empty() && identity && identity == previous_fault_identity) {
                row.quantum.record = row.input.values.true_word;
                row.quantum.faults = previous_faults;
            } else {
                row.quantum = e.transport->pack(row.input.values, row.input.faults, feedback);
                e.packed_fault_groups++;
            }
            previous_fault_identity = e.schedule.controls.empty() ? identity : 0;
            if(previous_fault_identity) previous_faults = row.quantum.faults;
        }
    }
    rows.push_back(std::move(row));
}

std::vector<FullRecordResult> FullRecordBatch::finish() {
    std::vector<FullRecordResult> results;
    Workspace workspace;
    auto& e = execution;
    for(auto& row : rows) {
        if(row.result) {
            results.push_back(std::move(*row.result));
            continue;
        }
        bool mismatch = false;
        if(e.transport->native_controls) {
            auto tick = std::chrono::steady_clock::now();
            try {
                apply_native_controls(e.global->controls, row.input.values.true_word ^ row.input.values.flip_word,
                        e.global->plan().magic, row.quantum);
            } catch(const ControlMismatch&) { mismatch = true; }
            control_seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - tick).count();
        }
        if(mismatch) {
            auto tick = std::chrono::steady_clock::now();
            results.push_back(e.literal(e.collect(row.input, {}, true), certificate));
            fallback_seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - tick).count();
            continue;
        }
        FullRecordResult result;
        if(certificate) {
            result.branch = row.branch ? std::move(row.branch)
                                       : std::make_optional(e.collect(row.input, {}, e.transport->native_controls));
            result.normal = e.global->actual(row.quantum);
            result.certificate = e.global->certificate(row.quantum, result.normal, workspace);
            result.probability = result.certificate.probability;
        } else result.probability = e.global->evaluate(row.quantum, workspace);
        workspace_peak = std::max(workspace_peak, global_workspace_capacity(workspace));
        results.push_back(std::move(result));
    }
    return results;
}
} // namespace magic_native
