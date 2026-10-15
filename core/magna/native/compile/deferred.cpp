#include "deferred.hpp"
namespace magic_native {
DeferredProgram defer_instrument(const std::vector<size_t>& inputs, const std::vector<size_t>& outputs,
        const std::vector<Operation>& operations,
        const std::map<std::string, std::optional<std::vector<DeclaredGate>>>& controls,
        const std::vector<std::string>& report_flips, bool expose_report_faults,
        const std::set<std::string>& preserved) {
    std::set<std::string> all_records, flips, ids;
    std::set<size_t> used(inputs.begin(), inputs.end());
    used.insert(outputs.begin(), outputs.end());
    for(const auto& op : operations) {
        ids.insert(op.id);
        if(op.record) all_records.insert(*op.record);
        used.insert(op.qubits.begin(), op.qubits.end());
    }
    for(const auto& name : report_flips)
        if(!flips.erase(name)) flips.insert(name);
    for(const auto& name : flips)
        if(!all_records.count(name)) throw std::invalid_argument("unknown report flip");
    for(const auto& name : preserved)
        if(!all_records.count(name)) throw std::invalid_argument("unknown preserved record");
    size_t fresh = used.empty() ? 0 : *used.rbegin() + 1, serial = 0;
    DeferredProgram result;
    std::map<std::string, size_t> records;
    auto add = [&](Operation op) {
        do { op.id = "__deferred/" + std::to_string(serial++); } while(ids.count(op.id));
        ids.insert(op.id);
        auto id = op.id;
        result.operations.push_back({std::move(op), {}});
        return id;
    };
    auto alloc = [&]() {
        auto q = fresh++;
        add(operation("PREP_ZERO", {q}));
        return q;
    };
    auto phase = [&](const std::vector<size_t>& qs, int exponent) {
        if(mod8(exponent)) add(operation("PHASE", qs, mod8(exponent)));
    };
    for(size_t index = 0; index < operations.size(); index++) {
        const auto& op = operations[index];
        const auto& kind = op.kind;
        const auto& qs = op.qubits;
        if(kind == "MZ" || kind == "MX" || kind == "MXX" || kind == "MZ_DESTRUCT" || kind == "MX_DESTRUCT" ||
                kind == "MP") {
            if(!op.record) throw std::invalid_argument("measurement record required");
            if(preserved.count(*op.record)) {
                result.operations.push_back({op, index});
                continue;
            }
            bool destructive = kind == "MZ_DESTRUCT" || kind == "MX_DESTRUCT";
            if(destructive && qs.size() != 1)
                throw std::invalid_argument("single-wire destructive measurement required");
            auto r = alloc();
            std::string paulis = kind == "MP" ? op.paulis.value_or("")
                                              : std::string(qs.size(), kind.rfind("MX", 0) == 0 ? 'X' : 'Z');
            if(paulis.size() != qs.size() || paulis.find_first_not_of("IXYZ") != std::string::npos)
                throw std::invalid_argument("invalid Pauli measurement body");
            for(size_t j = 0; j < qs.size(); j++) {
                auto q = qs[j];
                char p = paulis[j];
                if(p == 'Y') add(operation("S", {q}, -2));
                if(p == 'X' || p == 'Y') add(operation("H", {q}));
                if(p != 'I') add(operation("CX", {q, r}));
                if(p == 'X' || p == 'Y') add(operation("H", {q}));
                if(p == 'Y') add(operation("S", {q}, 2));
            }
            if(op.sign.value_or(0)) add(operation("X", {r}));
            records[*op.record] = r;
            result.records.push_back({*op.record, r});
            auto anchor = operation(destructive ? "DISCARD" : "I", destructive ? qs : std::vector<size_t>{});
            anchor.id = op.id;
            result.operations.push_back({anchor, {}});
            if(expose_report_faults) {
                auto id = add(operation("I", {r}));
                result.report_anchors.push_back({*op.record, id, r});
            }
        } else if(kind == "CONTROL") {
            const auto& rule = controls.at(op.id);
            if(!rule) throw UnsupportedFeedback("opaque feedback callback cannot be deferred");
            for(const auto& gate : canonical_affine_gates(*rule)) {
                auto [names, offset] = affine_expression(gate.guard);
                for(const auto& name : names) {
                    offset ^= flips.count(name);
                    if(!records.count(name)) throw std::invalid_argument("future record dependency");
                }
                if(gate.kind == "I") continue;
                std::optional<int> exponent;
                if(gate.kind == "T" || gate.kind == "S" || gate.kind == "PHASE") {
                    if(gate.exponent && gate.exponent->opcode != 0)
                        throw UnsupportedFeedback("record-dependent phase exponent");
                    exponent = gate.exponent ? gate.exponent->value : gate.kind == "S" ? 2 : 1;
                }
                std::vector<size_t> wires;
                for(auto q : gate.qubits) {
                    if(q < 0) throw std::invalid_argument("negative wire");
                    wires.push_back(size_t(q));
                }
                if(names.empty()) {
                    if(offset) add(operation(gate.kind, wires, exponent));
                    continue;
                }
                if(exponent && (*exponent & 1))
                    throw UnsupportedFeedback("controlled odd phase needs a richer resource class");
                auto a = alloc();
                std::vector<size_t> controls;
                for(const auto& name : names) controls.push_back(records.at(name));
                if(offset) add(operation("X", {a}));
                for(auto r : controls) add(operation("CX", {r, a}));
                if(gate.kind == "X" || gate.kind == "Z")
                    add(operation(gate.kind == "X" ? "CX" : "CZ", {a, wires.at(0)}));
                else if(gate.kind == "Y") {
                    phase({a}, 2);
                    add(operation("CZ", {a, wires.at(0)}));
                    add(operation("CX", {a, wires.at(0)}));
                } else if(gate.kind == "CX" || gate.kind == "CZ") {
                    if(gate.kind == "CX") add(operation("H", {wires.at(1)}));
                    add(operation("CCZ", {a, wires.at(0), wires.at(1)}));
                    if(gate.kind == "CX") add(operation("H", {wires.at(1)}));
                } else if(gate.kind == "H") {
                    for(auto literal : literal_gate("CH", {a, wires.at(0)})) {
                        if(literal.kind == "PHASE") phase(literal.qubits, *literal.exponent);
                        else add(std::move(literal));
                    }
                } else {
                    if(!exponent) throw UnsupportedFeedback("unsupported controlled gate");
                    int half = *exponent / 2;
                    phase({a}, half);
                    phase(wires, half);
                    wires.insert(wires.begin(), a);
                    phase(wires, -half);
                }
                for(auto it = controls.rbegin(); it != controls.rend(); ++it) add(operation("CX", {*it, a}));
                if(offset) add(operation("X", {a}));
                add(operation("DISCARD", {a}));
            }
            auto anchor = operation("I");
            anchor.id = op.id;
            result.operations.push_back({anchor, {}});
        } else {
            if(op.record) throw UnsupportedFeedback("unsupported measurement instrument");
            result.operations.push_back({op, index});
        }
    }
    return result;
}
} // namespace magic_native
