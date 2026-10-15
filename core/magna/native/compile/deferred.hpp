// Whole-instrument coherent affine controls; records remain explicit trace legs.
#pragma once
#include "file_lowering.hpp"
#include "expressions.hpp"
namespace magic_native {
inline Expression guard_expression(const RecordGuard& guard) {
    std::vector<Expression> leaves;
    for(auto record : guard.records) leaves.push_back(record_expression("m" + std::to_string(record)));
    if(guard.constant) leaves.push_back(constant_expression(1));
    while(leaves.size() > 1) {
        std::vector<Expression> next;
        for(size_t j = 0; j < leaves.size(); j += 2)
            next.push_back(j + 1 < leaves.size() ? composed_expression(3, {leaves[j], leaves[j + 1]}) : leaves[j]);
        leaves = std::move(next);
    }
    return leaves.empty() ? constant_expression(0) : leaves.front();
}
inline std::map<std::string, std::optional<std::vector<DeclaredGate>>> file_declarations(const LoweredFile& file) {
    std::map<std::string, std::optional<std::vector<DeclaredGate>>> result;
    for(const auto& [anchor, rule] : file.controls) {
        std::vector<DeclaredGate> gates;
        auto guard = guard_expression(rule.guard);
        for(const auto& op : rule.gates) {
            DeclaredGate gate;
            gate.kind = op.kind;
            gate.guard = guard;
            if(op.exponent) gate.exponent = constant_expression(*op.exponent);
            for(auto q : op.qubits) {
                if(q > size_t(std::numeric_limits<int>::max())) throw std::length_error("control wire capacity");
                gate.qubits.push_back(int(q));
            }
            gates.push_back(std::move(gate));
        }
        result[anchor] = std::move(gates);
    }
    return result;
}
struct DeferredOperation {
    Operation value;
    std::optional<size_t> original;
};
struct DeferredProgram {
    std::vector<DeferredOperation> operations;
    std::vector<std::pair<std::string, size_t>> records;
    std::vector<std::tuple<std::string, std::string, size_t>> report_anchors;
};
DeferredProgram defer_instrument(const std::vector<size_t>& inputs, const std::vector<size_t>& outputs,
        const std::vector<Operation>& operations,
        const std::map<std::string, std::optional<std::vector<DeclaredGate>>>& controls,
        const std::vector<std::string>& report_flips, bool expose_report_faults,
        const std::set<std::string>& preserve_records);
} // namespace magic_native
