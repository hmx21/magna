// Fixed nonlinear templates. Actual decisions supply only ordered Pauli offsets.
#pragma once
#include "frontend.hpp"
#include "../foundation/bits.hpp"
#include <map>
namespace magic_native {
struct RecordGuard {
    std::vector<size_t> records;
    int constant = 0;
};
struct VirtualControl {
    size_t wire;
    std::string gate;
    std::vector<size_t> targets;
};
struct SelectionControl {
    size_t wire;
    FileTarget target;
    std::string record;
};
struct FeedbackUpdate {
    std::string anchor;
    size_t wire;
    char pauli;
    std::vector<FileTarget> terms;
};
struct FeedbackFault {
    std::string anchor;
    size_t wire;
    char pauli;
};
using DecisionValues = std::map<FileTarget, int>;
using NoiseActions = std::map<std::string, std::vector<std::pair<char, size_t>>>;
inline std::pair<RecordGuard, std::vector<FileTarget>> split_expression(const AffineExpression& expression) {
    RecordGuard guard;
    guard.constant = expression.constant;
    std::vector<FileTarget> external;
    for(const auto& t : expression.terms)
        if(t.kind == "rec") guard.records.push_back(t.index);
        else external.push_back(t);
    return {guard, external};
}
inline std::optional<RecordGuard> resolve_file_expression(
        const CircuitData& circuit, const FileTarget& target, const DecisionValues& values) {
    auto [guard, external] = split_expression(circuit.expression(target));
    for(const auto& t : external) {
        auto found = values.find(t);
        if(found == values.end()) return {};
        guard.constant ^= found->second;
    }
    return guard;
}
struct FeedbackTemplate {
    size_t wire_count = 0;
    std::map<std::string, RecordGuard> guards;
    std::map<std::string, VirtualControl> virtual_controls;
    std::vector<std::pair<std::string, SelectionControl>> selections;
    std::map<std::string, int> selection_offsets;
    std::vector<FeedbackUpdate> updates;
    // Fixed reverse incidence of external decisions on Pauli offsets. It grows
    // with the declared template, never with observed histories or branches.
    std::map<FileTarget, std::vector<size_t>> consumers;
    Bits constant_updates;
    explicit FeedbackTemplate(const CircuitData& circuit, size_t minimum_width = 0)
        : wire_count(std::max(circuit.num_qubits, minimum_width)) {
        for(const auto& inst : circuit.instructions) {
            if(inst.name == "POSTSELECT")
                for(size_t j = 0; j < inst.targets.size(); j++) {
                    auto [guard, external] = split_expression(circuit.expression(inst.targets[j]));
                    if(external.empty()) continue;
                    auto anchor = inst.op_id + "/selection/" + std::to_string(j);
                    size_t wire = wire_count++;
                    selections.push_back({anchor, {wire, inst.targets[j], "__selection/" + anchor}});
                    selection_offsets[anchor] = guard.constant;
                    guard.constant = 0;
                    guards[anchor] = guard;
                    updates.push_back({anchor, wire, 'X', external});
                }
            const auto& n = inst.name;
            if(n != "CX" && n != "CY" && n != "CZ" && n != "CS" && n != "CS_DAG" && n != "CH" && n != "CCZ" &&
                    n != "CCX")
                continue;
            size_t width = n == "CCZ" || n == "CCX" ? 3 : 2;
            for(size_t j = 0; j < inst.targets.size(); j += width) {
                const auto& a = inst.targets.at(j);
                const auto& b = inst.targets.at(j + 1);
                FileTarget target;
                size_t wire;
                std::vector<size_t> wires;
                if(a.kind == "rec" || a.kind == "ctrl") {
                    target = a;
                    wire = b.index;
                    for(size_t k = 1; k < width; k++) wires.push_back(inst.targets.at(j + k).index);
                } else if(n == "CZ" && (b.kind == "rec" || b.kind == "ctrl")) {
                    target = b;
                    wire = a.index;
                    wires = {wire};
                } else continue;
                auto [guard, external] = split_expression(circuit.expression(target));
                if(external.empty()) continue;
                std::string oid = inst.op_id + "/gate/" + std::to_string(j / width), anchor = oid;
                char pauli = n[1];
                if(n != "CX" && n != "CY" && n != "CZ") {
                    size_t virtual_wire = wire_count++;
                    virtual_controls[oid] = {virtual_wire, n, wires};
                    anchor = oid + "/virtual/control";
                    wire = virtual_wire;
                    pauli = 'X';
                }
                guards[anchor] = guard;
                updates.push_back({anchor, wire, pauli, external});
            }
        }
        for(size_t j = 0; j < updates.size(); j++) {
            const auto& update = updates[j];
            auto offset = selection_offsets.find(update.anchor);
            if(offset != selection_offsets.end() && offset->second) constant_updates.set(j);
            for(const auto& term : update.terms) consumers[term].push_back(j);
        }
    }
    std::vector<FeedbackFault> faults(const DecisionValues& values, const std::string& prefix = "",
            const std::optional<std::map<size_t, size_t>>& wires = {}) const {
        Bits active = constant_updates;
        for(const auto& [decision, bit] : values)
            if(bit) {
                auto found = consumers.find(decision);
                if(found != consumers.end())
                    for(auto j : found->second) active.flip(j);
            }
        std::vector<FeedbackFault> out;
        out.reserve(active.count());
        for(auto j : active.indices()) {
            const auto& update = updates[j];
            out.push_back({prefix + update.anchor, wires ? wires->at(update.wire) : update.wire, update.pauli});
        }
        return out;
    }
};
} // namespace magic_native
