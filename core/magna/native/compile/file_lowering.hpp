// Literal fixed-input lowering from the physical file model to the operation interface.
#pragma once
#include "feedback_template.hpp"
namespace magic_native {
struct Operation {
    std::string id, kind;
    std::vector<size_t> qubits;
    std::optional<int> exponent, sign;
    std::optional<std::string> record, pauli, paulis;
};
inline Operation operation(std::string kind, std::vector<size_t> qubits = {}, std::optional<int> exponent = {}) {
    Operation out;
    out.kind = std::move(kind);
    out.qubits = std::move(qubits);
    out.exponent = exponent;
    return out;
}
inline std::vector<Operation> literal_gate(const std::string& name, const std::vector<size_t>& qs) {
    if(name == "I" || name == "X" || name == "Y" || name == "Z" || name == "H" || name == "CX" || name == "CZ" ||
            name == "CCZ")
        return {operation(name, qs)};
    static const std::map<std::string, int> phases{{"S", 2}, {"S_DAG", -2}, {"T", 1}, {"T_DAG", -1}};
    auto phase = phases.find(name);
    if(phase != phases.end()) return {operation("PHASE", qs, phase->second)};
    std::vector<Operation> out;
    auto append = [&](const std::string& gate, const std::vector<size_t>& wires) {
        auto extra = literal_gate(gate, wires);
        out.insert(out.end(), extra.begin(), extra.end());
    };
    if(name == "CY") {
        append("S_DAG", {qs.at(1)});
        append("CX", qs);
        append("S", {qs.at(1)});
    } else if(name == "SWAP") {
        append("CX", qs);
        append("CX", {qs.at(1), qs.at(0)});
        append("CX", qs);
    } else if(name == "CS" || name == "CS_DAG") {
        int e = name == "CS" ? 1 : -1;
        out = {operation("PHASE", {qs.at(0)}, e), operation("PHASE", {qs.at(1)}, e), operation("PHASE", qs, -e)};
    } else if(name == "CCX") {
        append("H", {qs.at(2)});
        append("CCZ", qs);
        append("H", {qs.at(2)});
    } else if(name == "CH") {
        auto q = qs.at(1);
        for(const auto& gate : {"S_DAG", "H", "T_DAG", "H", "S"}) append(gate, {q});
        append("CZ", qs);
        for(const auto& gate : {"S_DAG", "H", "T", "H", "S"}) append(gate, {q});
    } else if(name == "ISWAP" || name == "ISWAP_DAG") {
        append("SWAP", qs);
        append("CZ", qs);
        auto s = name == "ISWAP" ? "S" : "S_DAG";
        append(s, {qs.at(0)});
        append(s, {qs.at(1)});
    } else {
        static const std::map<std::string, std::vector<std::string>> single{{"SQRT_X", {"H", "S", "H"}},
                {"SQRT_X_DAG", {"H", "S_DAG", "H"}}, {"SQRT_Y", {"Z", "H"}}, {"SQRT_Y_DAG", {"H", "Z"}},
                {"H_XY", {"X", "S"}}, {"H_YZ", {"S_DAG", "H", "S"}}, {"C_XYZ", {"S_DAG", "H"}}, {"C_ZYX", {"H", "S"}}};
        auto found = single.find(name);
        if(found == single.end()) throw std::invalid_argument("unsupported gate lowering: " + name);
        for(const auto& gate : found->second) append(gate, qs);
    }
    return out;
}
struct AffineControlRule {
    RecordGuard guard;
    std::vector<Operation> gates;
};
struct LoweredFile {
    std::vector<Operation> operations;
    std::vector<std::pair<std::string, AffineControlRule>> controls;
};
LoweredFile lower_file(const CircuitData& circuit, const std::vector<size_t>& outputs, const DecisionValues& values,
        const NoiseActions& noise, const FeedbackTemplate* feedback = nullptr);
} // namespace magic_native
