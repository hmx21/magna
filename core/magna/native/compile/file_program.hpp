// Complete fixed-input file compilation, including terminal module namespaces.
#pragma once
#include "program.hpp"
#include "deferred.hpp"
#include "terminal.hpp"
#include <sstream>
namespace magic_native {
struct FileProgram {
    std::vector<size_t> inputs, outputs;
    LoweredFile lowered;
    FileProgram(const CircuitData& circuit, std::vector<size_t> selected, const FeedbackTemplate& feedback,
            const CircuitData* recovery, const FeedbackTemplate* recovery_feedback, const IdealTarget* target,
            const std::string& stage)
        : outputs(std::move(selected)) {
        if(stage != "protocol" && stage != "recovery" && stage != "target")
            throw std::invalid_argument("unknown file query stage");
        lowered = lower_file(circuit, outputs, {}, {}, &feedback);
        std::set<size_t> wires(outputs.begin(), outputs.end());
        for(auto q : circuit.qubits()) wires.insert(q);
        inputs.assign(wires.begin(), wires.end());
        size_t offset = circuit.num_records;
        if(stage != "protocol" && recovery) {
            auto terminal = attach_recovery(
                    *recovery, outputs.size(), outputs, offset, feedback.wire_count, {}, recovery_feedback);
            lowered.operations.insert(lowered.operations.end(), terminal.operations.begin(), terminal.operations.end());
            lowered.controls.insert(lowered.controls.end(), terminal.controls.begin(), terminal.controls.end());
            offset += recovery->num_records;
        }
        if(stage == "target") {
            if(!target) throw std::invalid_argument("target stage requires an ideal state");
            size_t first = feedback.wire_count;
            if(recovery_feedback) first += recovery_feedback->wire_count - outputs.size();
            auto projection = target->attach(outputs, offset, first);
            lowered.operations.insert(lowered.operations.end(), projection.first.begin(), projection.first.end());
        }
    }
    // Length-prefixed exact values, not a digest or a sampled decision identity.
    std::string identity() const {
        std::ostringstream key;
        auto text = [&](const std::string& value) { key << value.size() << ':' << value; };
        auto number = [&](size_t value) { key << value << ';'; };
        auto gate = [&](const Operation& op) {
            text(op.id);
            text(op.kind);
            number(op.qubits.size());
            for(auto q : op.qubits) number(q);
            for(const auto& value : {op.exponent, op.sign}) {
                key << bool(value) << ';';
                if(value) key << *value << ';';
            }
            for(const auto& value : {op.record, op.pauli, op.paulis}) {
                key << bool(value) << ';';
                if(value) text(*value);
            }
        };
        number(inputs.size());
        for(auto q : inputs) number(q);
        number(outputs.size());
        for(auto q : outputs) number(q);
        number(lowered.operations.size());
        for(const auto& op : lowered.operations) gate(op);
        number(lowered.controls.size());
        for(const auto& [anchor, rule] : lowered.controls) {
            text(anchor);
            number(rule.guard.constant);
            number(rule.guard.records.size());
            for(auto r : rule.guard.records) number(r);
            number(rule.gates.size());
            for(const auto& op : rule.gates) gate(op);
        }
        return key.str();
    }
    std::pair<DeferredProgram, CompilerInput> compile(size_t max_path_bits) const {
        auto deferred = defer_instrument(inputs, outputs, lowered.operations, file_declarations(lowered), {}, true, {});
        std::vector<int> in, out;
        for(auto q : inputs) in.push_back(int(q));
        for(auto q : outputs) out.push_back(int(q));
        for(const auto& [name, wire] : deferred.records) out.push_back(int(wire));
        std::vector<ProgramOperation> operations;
        for(const auto& op : deferred.operations) operations.push_back(program_operation(op.value));
        Ket initial;
        initial.width = inputs.size();
        auto compiled = compile_input(initial, std::move(in), std::move(out), operations, {}, max_path_bits);
        return {std::move(deferred), std::move(compiled)};
    }
};
} // namespace magic_native
