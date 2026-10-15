// Native operation-model metadata and declared-schedule compiler input.
#pragma once
#include "compiler.hpp"
#include "expressions.hpp"
#include "file_lowering.hpp"
#include "instrument.hpp"
#include <optional>
namespace magic_native {
struct LiteralProgramLayout {
    std::vector<OperationIdentity> operations;
    std::vector<std::pair<std::string, int>> truth;
    LiteralProgramLayout(std::vector<OperationIdentity> original, const std::vector<int>& outcomes)
        : operations(std::move(original)) {
        if(operations.size() != outcomes.size()) throw std::invalid_argument("literal operation outcome shape");
        std::set<std::string> used;
        for(const auto& op : operations)
            if(op.record) used.insert(*op.record);
        for(size_t j = 0; j < operations.size(); j++) {
            auto& op = operations[j];
            op.id = "__literal_operation_" + std::to_string(j);
            if(op.kind.rfind("M", 0) != 0) continue;
            if(!op.record) {
                auto name = "__inline_" + std::to_string(j);
                while(used.count(name)) name += "_";
                used.insert(name);
                op.record = std::move(name);
            }
            if(outcomes[j] != 0 && outcomes[j] != 1) throw std::invalid_argument("record must be an integer bit");
            truth.push_back({*op.record, outcomes[j]});
        }
    }
};
struct ProgramOperation {
    std::string id;
    CompilerOp value;
    std::optional<std::string> record;
};
inline ProgramOperation program_operation(const Operation& op) {
    CompilerOp value;
    value.kind = op.kind;
    for(auto q : op.qubits) {
        if(q > size_t(std::numeric_limits<int>::max())) throw std::length_error("wire index capacity");
        value.qubits.push_back(int(q));
    }
    if(op.kind == "T" || op.kind == "S" || op.kind == "PHASE")
        value.exponent = mod8(op.exponent.value_or(op.kind == "S" ? 2 : 1));
    if(op.kind == "MP") {
        value.sign = op.sign.value_or(0);
        value.paulis = op.paulis.value_or("");
    }
    if(op.kind == "PAULI") value.pauli = op.pauli.value_or("");
    return {op.id, std::move(value), op.record};
}
struct CompilerInput {
    CompilerIR ir;
    std::map<std::string, size_t> locations;
    std::vector<std::string> record_names, operation_ids;
    std::vector<std::pair<std::string, size_t>> feedback_slots;
};
inline CompilerInput compile_input(Ket initial, std::vector<int> inputs, std::vector<int> outputs,
        const std::vector<ProgramOperation>& operations,
        const std::map<std::string, std::optional<NativeControl>>& controls, size_t max_path_bits) {
    CompilerInput out;
    out.ir.initial = std::move(initial);
    out.ir.inputs = std::move(inputs);
    out.ir.outputs = std::move(outputs);
    out.ir.max_path_bits = max_path_bits;
    if(max_path_bits > 1000000) throw std::invalid_argument("compiler dimension capacity");
    if(operations.size() > size_t(std::numeric_limits<int>::max())) throw std::length_error("operation index capacity");
    for(const auto& [id, spec] : controls)
        if(!spec) throw UnsupportedFeedback("undeclared control schedules use the branch compiler");
    std::map<std::string, int> records;
    for(size_t j = 0; j < operations.size(); j++) {
        const auto& op = operations[j];
        out.locations[op.id] = j;
        out.operation_ids.push_back(op.id);
        if(op.record) {
            records[*op.record] = int(out.record_names.size());
            out.record_names.push_back(*op.record);
        }
    }
    out.ir.records = records.size();
    if(out.ir.records > 1000000) throw std::invalid_argument("compiler dimension capacity");
    std::set<int> live(out.ir.inputs.begin(), out.ir.inputs.end());
    auto append = [&](const ProgramOperation& source, int location, int slot = -1) {
        auto op = source.value;
        if(op.kind.rfind("M", 0) == 0 && !source.record)
            throw UnsupportedFeedback("fixed inline outcomes use the branch compiler");
        op.location = location;
        op.feedback_slot = slot;
        op.record = source.record ? records.at(*source.record) : -1;
        out.ir.operations.push_back(std::move(op));
    };
    for(const auto& source : operations) {
        const auto& op = source.value;
        if(op.kind == "CONTROL") {
            const auto& spec = *controls.at(source.id);
            for(int q : spec.qubits)
                if(!live.count(q)) throw UnsupportedFeedback("declared control interface outside wire lifetime");
            for(size_t j = 0; j < spec.axes.size(); j++) {
                CompilerOp phase;
                phase.kind = "PHASE";
                phase.qubits = spec.axes[j];
                phase.exponent = spec.baseline[j];
                size_t slot = out.feedback_slots.size();
                out.feedback_slots.push_back({source.id, j});
                append({"", phase, {}}, -1, int(slot));
            }
            CompilerOp identity;
            identity.kind = "I";
            append({"", identity, {}}, int(out.locations.at(source.id)));
        } else {
            append(source, int(out.locations.at(source.id)));
            if(op.kind == "DISCARD" || op.kind == "MZ_DESTRUCT" || op.kind == "MX_DESTRUCT")
                for(int q : op.qubits) live.erase(q);
            else if(op.kind == "PREP_ZERO" || op.kind == "PREP_PLUS" || op.kind == "RESET_ZERO" ||
                    op.kind == "RESET_PLUS")
                live.insert(op.qubits.begin(), op.qubits.end());
        }
    }
    return out;
}
inline std::pair<CompilerInput, Assignment> compile_literal_input(Ket initial, std::vector<int> inputs,
        std::vector<int> outputs, std::vector<ProgramOperation> operations, const std::vector<int>& outcomes,
        size_t max_paths) {
    std::vector<OperationIdentity> identities;
    for(const auto& op : operations) identities.push_back({op.id, op.value.kind, op.record});
    LiteralProgramLayout layout(std::move(identities), outcomes);
    Assignment assignment;
    for(size_t j = 0; j < operations.size(); j++) {
        operations[j].id = layout.operations[j].id;
        operations[j].record = layout.operations[j].record;
    }
    for(size_t j = 0; j < layout.truth.size(); j++)
        if(layout.truth[j].second) assignment.record.set(j);
    return {compile_input(std::move(initial), std::move(inputs), std::move(outputs), operations, {}, max_paths),
            assignment};
}
} // namespace magic_native
