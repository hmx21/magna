// Classical knowledge schedules global queries; it carries no quantum state.
#pragma once
#include "../compile/frontend.hpp"
#include "../foundation/causal_knowledge.hpp"
namespace magic_native {
struct ControllerInput {
    Bits records;
    int constant = 0;
    std::vector<size_t> decisions;
};
struct ControllerCall {
    size_t instruction = 0;
    std::string id;
    std::vector<ControllerInput> inputs;
    std::vector<size_t> outputs;
};
struct ConditionalNoiseCall {
    size_t instruction;
    std::string id;
    ControllerInput guard;
};
struct FileControlMetadata {
    size_t num_records = 0;
    std::map<FileTarget, size_t> indices;
    std::vector<FileTarget> decision_order;
    std::vector<ControllerCall> callbacks;
    std::vector<ConditionalNoiseCall> conditional_noise;
    explicit FileControlMetadata(const CircuitData& circuit) : num_records(circuit.num_records) {
        // Follow exact, already simplified expressions backwards from physical
        // uses. An opaque symbol that cancels has no quantum/classical consumer.
        std::set<FileTarget> required;
        std::set<size_t> live;
        auto depend = [&](const FileTarget& target) {
            if(target.kind != "rec" && target.kind != "ctrl" && target.kind != "det") return;
            for(const auto& term : circuit.expression(target).terms)
                if(term.kind != "rec") required.insert(term);
        };
        for(const auto& inst : circuit.instructions)
            if(inst.name != "CONTROL") {
                for(const auto& target : inst.targets) depend(target);
                if(inst.guard) depend(*inst.guard);
            }
        for(size_t j = circuit.instructions.size(); j-- > 0;) {
            const auto& inst = circuit.instructions[j];
            if(inst.name != "CONTROL" || inst.control_name == "XOR" || inst.control_name == "NOT") continue;
            bool needed = false;
            for(const auto& target : inst.outputs) needed |= required.count(target);
            if(needed) {
                live.insert(j);
                for(const auto& target : inst.targets) depend(target);
            }
        }
        std::vector<size_t> external;
        for(size_t j = 0; j < circuit.instructions.size(); j++) {
            const auto& inst = circuit.instructions[j];
            if(!live.count(j)) continue;
            external.push_back(j);
            for(const auto& target : inst.outputs) {
                indices[target] = decision_order.size();
                decision_order.push_back(target);
            }
        }
        auto input_of = [&](const FileTarget& target) {
            auto expression = circuit.expression(target);
            ControllerInput input;
            input.constant = expression.constant;
            for(const auto& term : expression.terms)
                if(term.kind == "rec") input.records.set(term.index);
                else input.decisions.push_back(indices.at(term));
            return input;
        };
        for(size_t j = 0; j < circuit.instructions.size(); j++) {
            const auto& inst = circuit.instructions[j];
            if(inst.guard) conditional_noise.push_back({j, inst.op_id, input_of(*inst.guard)});
        }
        for(auto index : external) {
            const auto& inst = circuit.instructions[index];
            ControllerCall call;
            call.instruction = index;
            call.id = inst.op_id;
            for(const auto& target : inst.targets) { call.inputs.push_back(input_of(target)); }
            for(const auto& target : inst.outputs) call.outputs.push_back(indices.at(target));
            callbacks.push_back(std::move(call));
        }
    }
};
inline NamedRows record_names(const std::vector<Bits>& masks) {
    NamedRows rows;
    for(const auto& mask : masks) {
        std::vector<std::string> row;
        for(auto index : mask.indices()) row.push_back("m" + std::to_string(index));
        rows.push_back(std::move(row));
    }
    return rows;
}
struct QueryTask {
    NamedRows selection;
    std::vector<std::pair<Bits, Bits>> observables;
    std::vector<size_t> sample_records;
    std::vector<std::string> target_records;
    bool has_target = false;
};
struct FileQueryFamily {
    NamedRows terminal_history;
};
inline FileQueryFamily plan_file_queries(
        const FileControlMetadata& protocol, const FileControlMetadata* recovery, const QueryTask& task) {
    CausalKnowledge knowledge;
    auto observations = [&](const std::vector<Bits>& masks) {
        std::vector<std::pair<Bits, int>> expressions;
        for(const auto& mask : masks) expressions.push_back({mask, 0});
        knowledge.observe(knowledge.request(expressions), Bits());
    };
    auto controllers = [&](const FileControlMetadata& metadata, size_t offset) {
        for(const auto& call : metadata.callbacks) {
            std::vector<Bits> masks;
            for(const auto& input : call.inputs) masks.push_back(input.records << offset);
            observations(masks);
        }
        for(const auto& call : metadata.conditional_noise) observations({call.guard.records << offset});
    };
    controllers(protocol, 0);
    std::vector<Bits> requested;
    for(auto index : task.sample_records) requested.push_back(Bits::unit(index));
    observations(requested);
    if(recovery) controllers(*recovery, protocol.num_records);
    return {record_names(knowledge.rows)};
}
} // namespace magic_native
