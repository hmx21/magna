// Immutable declared expression semantics. Opaque callbacks are never inspected.
#pragma once
#include "control.hpp"
#include <memory>
#include <set>
namespace magic_native {
struct UnsupportedFeedback : std::invalid_argument {
    using std::invalid_argument::invalid_argument;
};
struct MissingRecord : std::out_of_range {
    explicit MissingRecord(const std::string& name) : std::out_of_range(name) {}
};
struct ExpressionNode;
using Expression = std::shared_ptr<const ExpressionNode>;
struct ExpressionNode {
    int opcode = 0, value = 0;
    std::string record;
    std::vector<Expression> children;
    std::set<std::string> records() const {
        std::set<std::string> out;
        if(opcode == 1) out.insert(record);
        for(const auto& child : children) {
            auto names = child->records();
            out.insert(names.begin(), names.end());
        }
        return out;
    }
    ControlExpression serialize(const std::map<std::string, size_t>& indices) const {
        if(opcode == 0) return {{0, size_t(value)}};
        if(opcode == 1) return {{1, indices.at(record)}};
        ControlExpression out;
        for(const auto& child : children) {
            auto part = child->serialize(indices);
            out.insert(out.end(), part.begin(), part.end());
        }
        out.push_back({opcode, 0});
        return out;
    }
    int evaluate(const std::map<std::string, int>& values) const {
        if(opcode == 0) return value;
        if(opcode == 1) {
            auto found = values.find(record);
            if(found == values.end()) throw MissingRecord(record);
            return found->second;
        }
        auto a = children.at(0)->evaluate(values);
        if(opcode == 2) return !a;
        auto b = children.at(1)->evaluate(values);
        switch(opcode) {
        case 3: return a ^ b;
        case 4: return a & b;
        case 5: return a | b;
        case 6: return mod8(a + b);
        case 7: return mod8(a * b);
        default: throw std::invalid_argument("expression opcode");
        }
    }
};
inline Expression constant_expression(int value) {
    auto out = std::make_shared<ExpressionNode>();
    out->value = mod8(value);
    return out;
}
inline Expression record_expression(std::string record) {
    auto out = std::make_shared<ExpressionNode>();
    out->opcode = 1;
    out->record = std::move(record);
    return out;
}
inline Expression composed_expression(int opcode, std::vector<Expression> children) {
    if(opcode < 2 || opcode > 7 || children.size() != size_t(opcode == 2 ? 1 : 2))
        throw std::invalid_argument("invalid control expression");
    auto out = std::make_shared<ExpressionNode>();
    out->opcode = opcode;
    out->children = std::move(children);
    return out;
}
using AffineNames = std::pair<std::set<std::string>, int>;
inline AffineNames affine_expression(const Expression& expression) {
    const auto& e = *expression;
    if(e.opcode == 0) {
        if(e.value > 1) throw UnsupportedFeedback("affine Boolean constants must be bits");
        return {{}, e.value};
    }
    if(e.opcode == 1) return {{e.record}, 0};
    if(e.opcode == 2) {
        auto a = affine_expression(e.children.at(0));
        a.second ^= 1;
        return a;
    }
    if(e.opcode == 3) {
        auto a = affine_expression(e.children.at(0)), b = affine_expression(e.children.at(1));
        for(const auto& name : b.first)
            if(!a.first.erase(name)) a.first.insert(name);
        a.second ^= b.second;
        return a;
    }
    throw UnsupportedFeedback("only affine Boolean record guards are supported");
}
struct ParityPhase {
    int constant = 0;
    std::map<std::set<std::string>, int> terms;
};
inline ParityPhase phase_expression(const Expression& expression) {
    const auto& e = *expression;
    if(e.opcode == 0) return {e.value, {}};
    if(e.opcode == 6) {
        auto a = phase_expression(e.children.at(0)), b = phase_expression(e.children.at(1));
        a.constant = mod8(a.constant + b.constant);
        for(const auto& [key, value] : b.terms) a.terms[key] = mod8(a.terms[key] + value);
        return a;
    }
    if(e.opcode == 7) {
        auto left = e.children.at(0), right = e.children.at(1);
        if(right->opcode == 0) std::swap(left, right);
        if(left->opcode != 0) throw UnsupportedFeedback("nonlinear phase arithmetic");
        auto out = phase_expression(right);
        out.constant = mod8(out.constant * left->value);
        for(auto& [key, value] : out.terms) value = mod8(value * left->value);
        return out;
    }
    auto [names, bit] = affine_expression(expression);
    ParityPhase out;
    out.constant = bit;
    if(!names.empty()) out.terms[names] = bit ? -1 : 1;
    return out;
}
struct DeclaredGate {
    std::string kind;
    std::vector<int> qubits;
    Expression exponent, guard = constant_expression(1);
};
inline void validate_declared_gate(const DeclaredGate& gate) {
    const auto& kind = gate.kind;
    const auto& qs = gate.qubits;
    if(!std::set<std::string>{"I", "X", "Y", "Z", "H", "CX", "CZ", "T", "S", "PHASE"}.count(kind))
        throw std::invalid_argument("unsupported control rule gate");
    if(std::set<int>(qs.begin(), qs.end()).size() != qs.size())
        throw std::invalid_argument("invalid control gate wires");
    if((kind == "X" || kind == "Y" || kind == "Z" || kind == "H") && qs.size() != 1)
        throw std::invalid_argument("control Pauli arity");
    if((kind == "CX" || kind == "CZ") && qs.size() != 2) throw std::invalid_argument("controlled Clifford arity");
    bool phase = kind == "T" || kind == "S" || kind == "PHASE";
    if(phase && qs.empty()) throw std::invalid_argument("empty control phase axis");
    if(gate.exponent && !phase) throw std::invalid_argument("only phase gates have exponents");
}
inline std::set<std::string> declared_records(const std::vector<DeclaredGate>& gates) {
    std::set<std::string> result;
    for(const auto& gate : gates) {
        for(const auto& expression : {gate.guard, gate.exponent})
            if(expression) {
                auto names = expression->records();
                result.insert(names.begin(), names.end());
            }
    }
    return result;
}
inline std::vector<DeclaredGate> evaluate_declared_rule(
        const std::vector<DeclaredGate>& gates, const std::map<std::string, int>& values) {
    std::vector<DeclaredGate> result;
    for(const auto& gate : gates) {
        if(!gate.guard->evaluate(values)) continue;
        auto value = gate;
        if(gate.exponent) value.exponent = constant_expression(gate.exponent->evaluate(values));
        result.push_back(std::move(value));
    }
    return result;
}
inline std::vector<ControlGate> serialize_declared_rule(
        const std::vector<DeclaredGate>& gates, const std::map<std::string, size_t>& indices) {
    static const std::map<std::string, int> kinds{
            {"I", 0}, {"X", 1}, {"Z", 2}, {"Y", 3}, {"T", 4}, {"S", 5}, {"PHASE", 6}};
    std::vector<ControlGate> out;
    for(const auto& gate : gates) {
        auto found = kinds.find(gate.kind);
        if(found == kinds.end())
            throw UnsupportedFeedback("nonmonomial control uses boundary dilation or literal branch compilation");
        auto exponent = gate.exponent ? gate.exponent : constant_expression(gate.kind == "S" ? 2 : 1);
        out.push_back({found->second, gate.qubits, exponent->serialize(indices), gate.guard->serialize(indices),
                bool(gate.exponent)});
    }
    return out;
}
inline NativeControl feedback_family(const std::vector<int>& wires, const std::vector<std::vector<int>>& odd,
        const std::vector<std::vector<int>>& even) {
    NativeControl family;
    family.qubits = wires;
    std::set<int> qs(wires.begin(), wires.end());
    if(qs.size() != wires.size()) throw std::invalid_argument("duplicate feedback wire");
    std::set<std::vector<int>> seen, axes, odd_axes;
    for(auto q : wires) {
        family.axes.push_back({q});
        axes.insert({q});
    }
    for(const auto* collection : {&odd, &even})
        for(const auto& axis : *collection) {
            auto key = control_axis(axis);
            if(key.empty() || std::adjacent_find(key.begin(), key.end()) != key.end() ||
                    !std::includes(qs.begin(), qs.end(), key.begin(), key.end()) || !seen.insert(key).second)
                throw std::invalid_argument("invalid or duplicate feedback phase axis");
            if(collection == &odd) odd_axes.insert(key);
            if(axes.insert(key).second) family.axes.push_back(axis);
        }
    for(const auto& axis : family.axes) family.baseline.push_back(odd_axes.count(control_axis(axis)));
    return family;
}
inline std::vector<DeclaredGate> canonical_affine_gates(const std::vector<DeclaredGate>& gates) {
    std::vector<DeclaredGate> out;
    for(const auto& gate : gates) {
        if(!gate.exponent || gate.exponent->opcode == 0) {
            out.push_back(gate);
            continue;
        }
        auto [names, active] = affine_expression(gate.guard);
        if(!names.empty()) throw UnsupportedFeedback("record-dependent phase exponent with a variable guard");
        if(!active) continue;
        auto phase = phase_expression(gate.exponent);
        if(phase.constant) {
            auto copy = gate;
            copy.exponent = constant_expression(phase.constant);
            copy.guard = constant_expression(1);
            out.push_back(std::move(copy));
        }
        for(const auto& [terms, raw] : phase.terms) {
            int coefficient = mod8(raw);
            if(!coefficient) continue;
            Expression guard = constant_expression(0);
            for(const auto& name : terms) guard = composed_expression(3, {guard, record_expression(name)});
            auto copy = gate;
            copy.exponent = constant_expression(coefficient);
            copy.guard = guard;
            out.push_back(std::move(copy));
        }
    }
    return out;
}
inline bool is_affine_rule(const std::vector<DeclaredGate>& gates) {
    try {
        for(const auto& gate : canonical_affine_gates(gates)) {
            auto names = affine_expression(gate.guard).first;
            if(gate.kind == "T" || gate.kind == "S" || gate.kind == "PHASE") {
                if(gate.exponent && gate.exponent->opcode != 0) return false;
                int exponent = gate.exponent ? gate.exponent->value : gate.kind == "S" ? 2 : 1;
                if(!names.empty() && (exponent & 1)) return false;
            }
        }
        return true;
    } catch(const UnsupportedFeedback&) { return false; }
}
} // namespace magic_native
