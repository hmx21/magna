// Causal operation-level decisions and literal branch export plans.
#pragma once
#include "../compile/instrument.hpp"
#include "../compile/file_lowering.hpp"
#include "../compile/expressions.hpp"
#include "../compile/program.hpp"
#include <functional>
#include <atomic>
namespace magic_native {
inline std::vector<Operation> evaluated_declared_operations(
        const std::vector<DeclaredGate>& rule, const std::map<std::string, int>& values) {
    std::vector<Operation> out;
    for(const auto& gate : evaluate_declared_rule(rule, values)) {
        Operation value;
        value.kind = gate.kind;
        for(auto q : gate.qubits) value.qubits.push_back(size_t(q));
        if(gate.exponent) value.exponent = gate.exponent->value;
        out.push_back(std::move(value));
    }
    return out;
}
inline ControlResolution resolve_feedback_operations(NativeControl family, const std::vector<Operation>& operations) {
    for(const auto& op : operations) {
        ControlGate gate;
        gate.guard = {{0, 1}};
        gate.exponent = {{0, 1}};
        for(auto q : op.qubits) gate.qubits.push_back(int(q));
        if(op.kind == "I") gate.kind = 0;
        else if(op.kind == "PAULI" || op.kind == "X" || op.kind == "Y" || op.kind == "Z") {
            auto value = op.pauli.value_or(op.kind);
            auto found = std::string("IXZY").find(value);
            if(value.size() != 1 || found == std::string::npos) throw ControlMismatch("feedback Pauli name");
            if(gate.qubits.size() != 1) throw ControlMismatch("feedback Pauli arity");
            gate.kind = int(found);
        } else if(op.kind == "T" || op.kind == "S" || op.kind == "PHASE") {
            gate.kind = 6;
            gate.exponent = {{0, size_t(mod8(op.exponent.value_or(op.kind == "S" ? 2 : 1)))}};
        } else throw ControlMismatch("feedback changes the declared monomial schedule");
        family.gates.push_back(std::move(gate));
    }
    return resolve_native_control(family, Bits());
}
struct BranchGate {
    ProgramOperation operation;
    bool explicit_exponent = false;
};
inline BranchGate branch_gate(std::string kind, std::vector<int> qubits = {}) {
    BranchGate out;
    out.operation.value.kind = std::move(kind);
    out.operation.value.qubits = std::move(qubits);
    return out;
}
inline std::vector<BranchGate> evaluated_branch_gates(
        const std::vector<DeclaredGate>& rule, const std::map<std::string, int>& values) {
    std::vector<BranchGate> out;
    for(const auto& source : evaluate_declared_rule(rule, values)) {
        auto gate = branch_gate(source.kind, source.qubits);
        gate.explicit_exponent = bool(source.exponent);
        gate.operation.value.exponent = source.exponent ? source.exponent->value : source.kind == "S" ? 2 : 1;
        out.push_back(std::move(gate));
    }
    return out;
}
struct BranchDecision {
    std::string id, version;
    std::vector<std::pair<std::string, int>> parents;
    std::vector<BranchGate> gates;
};
struct BranchStep {
    // kind 0: original operation; 1: emitted gate; 2: original fault.
    int kind = 0;
    size_t index = 0, subindex = 0;
    std::string id;
    std::optional<int> measured, reported;
    std::optional<BranchGate> replacement;
};
struct CollectedBranch {
    Bits true_word, reported_word;
    std::vector<BranchDecision> decisions;
    std::vector<size_t> ordered_faults;
    std::vector<BranchStep> steps;
};
struct BranchSchedule {
    InstrumentLayout layout;
    std::vector<size_t> controls;
    std::vector<std::optional<std::vector<DeclaredGate>>> rules;
    struct Statistics {
        std::atomic<size_t> extractions{0}, callbacks{0}, rules{0}, native_rules{0}, materializations{0};
    } statistics;
    BranchSchedule(
            InstrumentLayout layout, std::map<std::string, std::optional<std::vector<DeclaredGate>>> declarations)
        : layout(std::move(layout)) {
        for(size_t j = 0; j < this->layout.operations.size(); j++)
            if(this->layout.operations[j].kind == "CONTROL") {
                controls.push_back(j);
                rules.push_back(declarations.at(this->layout.operations[j].id));
            }
    }
    bool all_native() const {
        return !controls.empty() &&
                std::all_of(rules.begin(), rules.end(), [](const auto& rule) { return bool(rule); });
    }
    using Callback = std::function<std::vector<BranchGate>(size_t, const std::vector<std::pair<std::string, int>>&)>;
    CollectedBranch collect(const ValidatedAssignment& assignment, const std::vector<OriginalFault>& faults,
            const Callback& callback, const std::optional<std::vector<std::vector<BranchGate>>>& emitted = {},
            const Callback& declared = {}) {
        if(emitted && (!all_native() || emitted->size() != controls.size()))
            throw std::invalid_argument("native control decision count");
        CollectedBranch result;
        result.true_word = assignment.true_word;
        result.reported_word = assignment.true_word ^ assignment.flip_word;
        std::map<std::string, std::vector<size_t>> by_id;
        for(const auto& row : assignment.fault_groups) by_id[row.first] = row.second;
        size_t control = 0;
        for(size_t j = 0; j < layout.operations.size(); j++) {
            const auto& op = layout.operations[j];
            if(op.kind == "CONTROL") {
                const auto& node = layout.controls.at(op.id);
                BranchDecision decision;
                decision.id = op.id;
                decision.version = node.version;
                for(const auto& parent : node.parents)
                    decision.parents.push_back({parent, int(result.reported_word.bit(layout.record_index.at(parent)))});
                if(emitted) {
                    statistics.native_rules++;
                    decision.gates = emitted->at(control);
                } else if(rules[control]) {
                    statistics.rules++;
                    std::map<std::string, int> values(decision.parents.begin(), decision.parents.end());
                    decision.gates = declared ? declared(control, decision.parents)
                                              : evaluated_branch_gates(*rules[control], values);
                } else {
                    statistics.callbacks++;
                    decision.gates = callback(control, decision.parents);
                }
                for(size_t k = 0; k < decision.gates.size(); k++) {
                    const auto& gate = decision.gates[k].operation;
                    if(gate.record || gate.value.kind == "CONTROL")
                        throw std::invalid_argument("control expansion must be measurement-free gates");
                    result.steps.push_back({1, control, k, op.id + "/" + std::to_string(k), {}, {}});
                }
                result.decisions.push_back(std::move(decision));
                control++;
            } else {
                BranchStep step{0, j, 0, op.id, {}, {}};
                if(op.record) {
                    size_t index = layout.record_index.at(*op.record);
                    step.measured = result.true_word.bit(index);
                    step.reported = result.reported_word.bit(index);
                }
                result.steps.push_back(std::move(step));
            }
            const auto found = by_id.find(op.id);
            if(found == by_id.end()) continue;
            for(size_t k = 0; k < found->second.size(); k++) {
                auto index = found->second[k];
                result.ordered_faults.push_back(index);
                if(faults.at(index).pauli) {
                    auto id = op.id + "/fault/" + std::to_string(k);
                    auto gate = branch_gate("PAULI", {faults[index].qubit});
                    gate.operation.id = id;
                    gate.operation.value.pauli = std::string(1, "IXZY"[faults[index].pauli]);
                    result.steps.push_back({2, index, k, id, {}, {}, gate});
                }
            }
        }
        statistics.extractions++;
        return result;
    }
};
} // namespace magic_native
