#include "file_lowering.hpp"
#include <functional>
namespace magic_native {
LoweredFile lower_file(const CircuitData& circuit, const std::vector<size_t>& outputs, const DecisionValues& values,
        const NoiseActions&, const FeedbackTemplate* feedback) {
    LoweredFile result;
    auto& ops = result.operations;
    auto add = [&](const std::string& id, Operation op) {
        op.id = id;
        ops.push_back(std::move(op));
    };
    auto append = [&](const std::string& id, const std::vector<Operation>& gates) {
        for(size_t j = 0; j < gates.size(); j++) add(id + "/" + std::to_string(j), gates[j]);
        add(id, operation("I"));
    };
    std::function<void(
            const std::string&, const std::vector<Operation>&, const FileTarget&, std::optional<RecordGuard>)>
            controlled;
    controlled = [&](const std::string& id, const std::vector<Operation>& gates, const FileTarget& target,
                         std::optional<RecordGuard> guard) {
        if(feedback) {
            auto found = feedback->virtual_controls.find(id);
            if(found != feedback->virtual_controls.end()) {
                const auto& v = found->second;
                add(id + "/virtual/prepare", operation("PREP_ZERO", {v.wire}));
                controlled(id + "/virtual/control", literal_gate("X", {v.wire}), target, {});
                auto wires = v.targets;
                wires.insert(wires.begin(), v.wire);
                append(id + "/virtual/unitary", literal_gate(v.gate, wires));
                add(id + "/virtual/trace", operation("DISCARD", {v.wire}));
                append(id, {});
                return;
            }
        }
        if(!guard) {
            auto found = feedback ? feedback->guards.find(id) : std::map<std::string, RecordGuard>::const_iterator{};
            if(feedback && found != feedback->guards.end()) guard = found->second;
            else {
                if(feedback && !split_expression(circuit.expression(target)).second.empty())
                    throw std::invalid_argument("external control requires a fixed feedback template: " + id);
                guard = resolve_file_expression(circuit, target, values);
            }
        }
        if(!guard) {
            append(id, {});
            return;
        }
        if(guard->records.empty()) {
            append(id, guard->constant ? gates : std::vector<Operation>{});
            return;
        }
        result.controls.push_back({id, {*guard, gates}});
        add(id, operation("CONTROL"));
    };
    static const std::set<std::string> annotations{
            "TICK", "QUBIT_COORDS", "SHIFT_COORDS", "DETECTOR", "OBSERVABLE_INCLUDE", "POSTSELECT", "CONTROL"};
    static const std::set<std::string> noise_names{"X_ERROR", "Y_ERROR", "Z_ERROR", "DEPOLARIZE1", "DEPOLARIZE2",
            "PAULI_CHANNEL_1", "PAULI_CHANNEL_2", "CORRELATED_ERROR", "ELSE_CORRELATED_ERROR"};
    for(const auto& inst : circuit.instructions) {
        const auto& n = inst.name;
        const auto& id = inst.op_id;
        if(n == "POSTSELECT" && feedback)
            for(size_t j = 0; j < inst.targets.size(); j++) {
                auto anchor = id + "/selection/" + std::to_string(j);
                auto found = std::find_if(feedback->selections.begin(), feedback->selections.end(),
                        [&](const auto& entry) { return entry.first == anchor; });
                if(found != feedback->selections.end()) {
                    auto wire = found->second.wire;
                    add(anchor + "/prepare", operation("PREP_ZERO", {wire}));
                    controlled(anchor, literal_gate("X", {wire}), inst.targets[j], {});
                }
            }
        if(annotations.count(n)) continue;
        if(noise_names.count(n)) {
            // Every physical channel retains only its original assignment anchor.
            // Its definite action is resolved by the causal sampler, never compiled.
            append(id, {});
            continue;
        }
        if(!inst.records.empty()) {
            auto products = inst.products;
            if(products.empty())
                for(const auto& target : inst.targets) {
                    auto t = target;
                    t.kind = "pauli";
                    t.pauli = n == "MX" || n == "MRX" ? "X" : n == "MY" || n == "MRY" ? "Y" : "Z";
                    products.push_back({t});
                }
            for(size_t j = 0; j < inst.records.size(); j++) {
                Operation op = operation("MP");
                op.record = "m" + std::to_string(inst.records[j]);
                op.sign = 0;
                op.paulis = "";
                for(const auto& t : products.at(j)) {
                    op.qubits.push_back(t.index);
                    *op.paulis += t.pauli;
                    *op.sign ^= t.inverted;
                }
                add(id + "/measure/" + std::to_string(j), op);
                if(n == "MR" || n == "MRX" || n == "MRY") {
                    auto q = products[j][0].index;
                    add(id + "/reset/" + std::to_string(j), operation(n == "MRX" ? "RESET_PLUS" : "RESET_ZERO", {q}));
                    if(n == "MRY") {
                        auto gates = literal_gate("H", {q}), extra = literal_gate("S", {q});
                        gates.insert(gates.end(), extra.begin(), extra.end());
                        append(id + "/yprep/" + std::to_string(j), gates);
                    }
                }
            }
            continue;
        }
        if(n == "R" || n == "RX" || n == "RY") {
            for(size_t j = 0; j < inst.targets.size(); j++) {
                auto q = inst.targets[j].index;
                add(id + "/reset/" + std::to_string(j), operation(n == "RX" ? "RESET_PLUS" : "RESET_ZERO", {q}));
                if(n == "RY") {
                    auto gates = literal_gate("H", {q}), extra = literal_gate("S", {q});
                    gates.insert(gates.end(), extra.begin(), extra.end());
                    append(id + "/yprep/" + std::to_string(j), gates);
                }
            }
            continue;
        }
        size_t width = n == "CCX" || n == "CCZ" ? 3
                : std::set<std::string>{"CX", "CY", "CZ", "SWAP", "ISWAP", "ISWAP_DAG", "CS", "CS_DAG", "CH"}.count(n)
                ? 2
                : 1;
        for(size_t j = 0; j < inst.targets.size(); j += width) {
            auto id_gate = id + "/gate/" + std::to_string(j / width);
            const auto& first = inst.targets[j];
            std::vector<size_t> wires;
            for(size_t k = 0; k < width; k++) wires.push_back(inst.targets.at(j + k).index);
            if(first.kind == "rec" || first.kind == "ctrl") {
                static const std::map<std::string, std::string> inner{{"CX", "X"}, {"CY", "Y"}, {"CZ", "Z"},
                        {"CS", "S"}, {"CS_DAG", "S_DAG"}, {"CH", "H"}, {"CCZ", "CZ"}, {"CCX", "CX"}};
                wires.erase(wires.begin());
                controlled(id_gate, literal_gate(inner.at(n), wires), first, {});
            } else if(width == 2 && n == "CZ" &&
                    (inst.targets[j + 1].kind == "rec" || inst.targets[j + 1].kind == "ctrl"))
                controlled(id_gate, literal_gate("Z", {first.index}), inst.targets[j + 1], {});
            else append(id_gate, literal_gate(n, wires));
        }
    }
    if(feedback)
        for(const auto& [anchor, selection] : feedback->selections) {
            auto op = operation("MP", {selection.wire});
            op.record = selection.record;
            op.paulis = "Z";
            add(anchor + "/readout", op);
            add(anchor + "/discard", operation("DISCARD", {selection.wire}));
        }
    for(auto q : circuit.qubits())
        if(std::find(outputs.begin(), outputs.end(), q) == outputs.end())
            add("__file_discard/" + std::to_string(q), operation("DISCARD", {q}));
    return result;
}
} // namespace magic_native
