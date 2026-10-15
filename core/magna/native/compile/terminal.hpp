// Noiseless terminal recovery and inverse-preparation effect boundaries.
#pragma once
#include "file_lowering.hpp"
#include <numeric>
namespace magic_native {
inline bool file_noise_instruction(const std::string& name) {
    static const std::set<std::string> names{"X_ERROR", "Y_ERROR", "Z_ERROR", "DEPOLARIZE1", "DEPOLARIZE2",
            "PAULI_CHANNEL_1", "PAULI_CHANNEL_2", "CORRELATED_ERROR", "ELSE_CORRELATED_ERROR"};
    return names.count(name);
}
inline void validate_recovery(const CircuitData& circuit) {
    for(const auto& inst : circuit.instructions) {
        auto location = inst.source.filename + ":" + std::to_string(inst.source.line) + ": ";
        if(inst.name == "POSTSELECT")
            throw std::invalid_argument(
                    location + "POSTSELECT belongs in the physical circuit; terminal recovery averages all outcomes");
        bool noisy_report = false;
        for(const auto& value : inst.args) noisy_report |= !value.n.zero();
        if(file_noise_instruction(inst.name) || (!inst.records.empty() && noisy_report))
            throw std::invalid_argument(
                    location + "terminal recovery must be noiseless, including reported measurements");
    }
}
struct IdealTarget {
    size_t width = 0, preparation_width = 0;
    std::vector<Operation> inverse;
    CircuitData preparation;
    std::vector<size_t> selected, traced;
    std::map<size_t, int> postselection;
    IdealTarget(const CircuitData& circuit, size_t width) : width(width) {
        if(circuit.num_qubits > width)
            throw std::invalid_argument("ideal state has qubits outside the selected output boundary");
        for(const auto& inst : circuit.instructions) {
            bool control = false;
            for(const auto& target : inst.targets) control |= target.kind == "rec" || target.kind == "ctrl";
            if(file_noise_instruction(inst.name) || !inst.records.empty() || control || inst.name == "R" ||
                    inst.name == "RX" || inst.name == "RY" || inst.name == "CONTROL" || inst.name == "POSTSELECT")
                throw std::invalid_argument(inst.source.filename + ":" + std::to_string(inst.source.line) +
                        ": ideal state requires a noiseless unitary preparation; got " + inst.name);
        }
        preparation = circuit;
        preparation.num_qubits = width;
        preparation_width = width;
        selected.resize(width);
        std::iota(selected.begin(), selected.end(), 0);
        auto padded = circuit;
        padded.num_qubits = width;
        std::vector<size_t> wires(width);
        std::iota(wires.begin(), wires.end(), 0);
        auto lowered = lower_file(padded, wires, {}, {});
        for(auto it = lowered.operations.rbegin(); it != lowered.operations.rend(); ++it) {
            auto gate = *it;
            if(gate.kind == "I") continue;
            if(gate.kind == "PHASE") gate.exponent = -gate.exponent.value();
            else if(gate.kind != "X" && gate.kind != "Y" && gate.kind != "Z" && gate.kind != "H" && gate.kind != "CX" &&
                    gate.kind != "CZ" && gate.kind != "CCZ")
                throw std::invalid_argument("unsupported ideal-state inverse primitive: " + gate.kind);
            inverse.push_back(std::move(gate));
        }
    }
    IdealTarget(const CircuitData& circuit, size_t width, std::vector<size_t> outputs, std::map<size_t, int> caps)
        : IdealTarget(circuit, circuit.num_qubits) {
        if(outputs.size() != width || std::set<size_t>(outputs.begin(), outputs.end()).size() != width)
            throw std::invalid_argument(
                    "target outputs must match the selected physical output width without duplicates");
        for(auto q : outputs)
            if(q >= preparation_width) throw std::invalid_argument("target output outside preparation");
        for(auto [q, bit] : caps)
            if(q >= preparation_width || std::find(outputs.begin(), outputs.end(), q) != outputs.end() ||
                    (bit != 0 && bit != 1))
                throw std::invalid_argument("target postselection must specify binary outcomes on auxiliary qubits");
        this->width = width;
        selected = std::move(outputs);
        postselection = std::move(caps);
        for(size_t q = 0; q < preparation_width; q++)
            if(std::find(selected.begin(), selected.end(), q) == selected.end() && !postselection.count(q))
                traced.push_back(q);
    }
    std::pair<std::vector<Operation>, std::vector<std::string>> attach(
            const std::vector<size_t>& outputs, size_t offset, size_t first_ancilla) const {
        if(outputs.size() != width) throw std::invalid_argument("ideal state output width mismatch");
        std::vector<Operation> operations;
        std::vector<std::string> records;
        std::vector<size_t> wires(preparation_width);
        for(size_t j = 0; j < width; j++) wires[selected[j]] = outputs[j];
        auto add = [&](Operation op) {
            op.id = "__ideal/boundary/" + std::to_string(operations.size());
            operations.push_back(std::move(op));
        };
        for(size_t q = 0; q < preparation_width; q++)
            if(std::find(selected.begin(), selected.end(), q) == selected.end()) {
                wires[q] = first_ancilla++;
                auto found = postselection.find(q);
                add(operation(found == postselection.end() ? "PREP_PLUS" : "PREP_ZERO", {wires[q]}));
                if(found != postselection.end()) {
                    if(found->second) add(operation("X", {wires[q]}));
                } else {
                    // The adjoint of a traced target leg inserts identity. A Bell
                    // purification supplies I/2; its factor two is retained exactly.
                    auto environment = first_ancilla++;
                    add(operation("PREP_ZERO", {environment}));
                    add(operation("CX", {wires[q], environment}));
                    add(operation("DISCARD", {environment}));
                }
            }
        for(size_t j = 0; j < inverse.size(); j++) {
            auto op = inverse[j];
            op.id = "__ideal/gate/" + std::to_string(j);
            for(auto& wire : op.qubits) wire = wires.at(wire);
            operations.push_back(std::move(op));
        }
        for(size_t j = 0; j < preparation_width; j++) {
            auto record = "m" + std::to_string(offset + j);
            auto op = operation("MP", {wires[j]});
            op.id = "__ideal/project/" + std::to_string(j);
            op.paulis = "Z";
            op.record = record;
            operations.push_back(std::move(op));
            records.push_back(std::move(record));
        }
        for(size_t q = 0; q < preparation_width; q++)
            if(std::find(selected.begin(), selected.end(), q) == selected.end()) add(operation("DISCARD", {wires[q]}));
        return {operations, records};
    }
};
inline LoweredFile attach_recovery(const CircuitData& circuit, size_t width, const std::vector<size_t>& outputs,
        size_t record_offset, size_t first_ancilla, const DecisionValues& values,
        const FeedbackTemplate* feedback = nullptr) {
    if(outputs.size() != width) throw std::invalid_argument("recovery output width mismatch");
    size_t size = std::max(width, circuit.num_qubits);
    auto padded = circuit;
    padded.num_qubits = size;
    std::vector<size_t> local_outputs(width);
    std::iota(local_outputs.begin(), local_outputs.end(), 0);
    auto local = lower_file(padded, local_outputs, values, {}, feedback);
    size_t local_width = feedback ? feedback->wire_count : size;
    auto wires = outputs;
    for(size_t j = width; j < local_width; j++) wires.push_back(first_ancilla + j - width);
    auto record = [&](const std::string& name) {
        if(name.size() < 2 || name[0] != 'm') throw std::invalid_argument("invalid recovery record name");
        return "m" + std::to_string(record_offset + std::stoull(name.substr(1)));
    };
    auto map_gate = [&](Operation op) {
        for(auto& q : op.qubits) q = wires.at(q);
        return op;
    };
    LoweredFile result;
    for(auto j : circuit.qubits()) {
        if(j < width) continue;
        auto op = operation("PREP_ZERO", {wires[j]});
        op.id = "__recovery/prepare/" + std::to_string(j - width);
        result.operations.push_back(std::move(op));
    }
    for(auto op : local.operations) {
        op = map_gate(std::move(op));
        op.id = "__recovery/" + op.id;
        if(op.record) op.record = record(*op.record);
        result.operations.push_back(std::move(op));
    }
    for(auto [anchor, rule] : local.controls) {
        for(auto& index : rule.guard.records) index += record_offset;
        for(auto& gate : rule.gates) gate = map_gate(std::move(gate));
        result.controls.push_back({"__recovery/" + anchor, std::move(rule)});
    }
    return result;
}
} // namespace magic_native
