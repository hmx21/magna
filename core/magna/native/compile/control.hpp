// Explicit finite control expressions. Never infer Python callback semantics.
#pragma once
#include "plan.hpp"

namespace magic_native {
struct ControlToken {
    int opcode;
    size_t operand;
};
using ControlExpression = std::vector<ControlToken>;
struct ControlGate {
    int kind; // I=0, X=1, Z=2, Y=3, T=4, S=5, PHASE=6
    std::vector<int> qubits;
    ControlExpression exponent, guard;
    bool explicit_exponent = false;
};
struct NativeControl {
    size_t location = 0;
    std::vector<int> qubits;
    std::vector<std::vector<int>> axes;
    std::vector<int> baseline;
    std::vector<size_t> phase_slots, parents;
    std::vector<ControlGate> gates;
};
struct ControlResolution {
    std::vector<int> phases, flips;
    int phase = 0;
};
struct ControlMismatch : std::invalid_argument {
    using std::invalid_argument::invalid_argument;
};

inline void validate_control_expression(
        const ControlExpression& expression, const NativeControl& control, size_t nrecords) {
    size_t depth = 0;
    for(auto& t : expression) {
        if(t.opcode == 0) {
            if(t.operand > 7) throw std::invalid_argument("control constant outside Z/8Z");
            ++depth;
        } else if(t.opcode == 1) {
            if(t.operand >= nrecords ||
                    std::find(control.parents.begin(), control.parents.end(), t.operand) == control.parents.end())
                throw std::invalid_argument("control reads undeclared record parent");
            ++depth;
        } else if(t.opcode == 2) {
            if(!depth) throw std::invalid_argument("control expression stack underflow");
        } else if(t.opcode >= 3 && t.opcode <= 7) {
            if(depth < 2) throw std::invalid_argument("control expression stack underflow");
            --depth;
        } else throw std::invalid_argument("unknown control expression opcode");
    }
    if(depth != 1) throw std::invalid_argument("control expression result shape");
}
inline int evaluate_control_expression(const ControlExpression& expression, const Bits& reports) {
    std::vector<int> stack;
    stack.reserve(expression.size());
    for(auto& t : expression) {
        if(t.opcode == 0) {
            stack.push_back(int(t.operand));
            continue;
        }
        if(t.opcode == 1) {
            stack.push_back(reports.bit(t.operand));
            continue;
        }
        if(stack.empty()) throw std::invalid_argument("control expression stack underflow");
        if(t.opcode == 2) {
            stack.back() = !stack.back();
            continue;
        }
        if(stack.size() < 2) throw std::invalid_argument("control expression stack underflow");
        int b = stack.back();
        stack.pop_back();
        int a = stack.back();
        switch(t.opcode) {
        case 3: stack.back() = a ^ b; break;
        case 4: stack.back() = a & b; break;
        case 5: stack.back() = a | b; break;
        case 6: stack.back() = mod8(a + b); break;
        case 7: stack.back() = mod8(a * b); break;
        default: throw std::invalid_argument("unknown control expression opcode");
        }
    }
    if(stack.size() != 1) throw std::invalid_argument("control expression result shape");
    return stack.back();
}
inline std::vector<int> control_axis(std::vector<int> axis) {
    std::sort(axis.begin(), axis.end());
    return axis;
}
inline void validate_native_control(const NativeControl& control, size_t nrecords, size_t nlocations, size_t nphases) {
    if(control.location >= nlocations || control.axes.size() != control.baseline.size() ||
            control.axes.size() != control.phase_slots.size())
        throw std::invalid_argument("native control dimensions");
    auto qs = control_axis(control.qubits);
    if(std::adjacent_find(qs.begin(), qs.end()) != qs.end()) throw std::invalid_argument("duplicate control wire");
    std::vector<std::vector<int>> axes;
    for(size_t i = 0; i < control.axes.size(); i++) {
        auto axis = control_axis(control.axes[i]);
        if(axis.empty() || std::adjacent_find(axis.begin(), axis.end()) != axis.end() ||
                !std::includes(qs.begin(), qs.end(), axis.begin(), axis.end()) ||
                std::find(axes.begin(), axes.end(), axis) != axes.end())
            throw std::invalid_argument("invalid control phase axis");
        if(control.baseline[i] < 0 || control.baseline[i] > 1 || control.phase_slots[i] >= nphases)
            throw std::invalid_argument("native control phase slot/baseline");
        axes.push_back(std::move(axis));
    }
    for(int q : qs)
        if(std::find(axes.begin(), axes.end(), std::vector<int>{q}) == axes.end())
            throw std::invalid_argument("control missing single-wire phase axis");
    for(auto parent : control.parents)
        if(parent >= nrecords) throw std::invalid_argument("control parent outside record interface");
    for(auto& gate : control.gates) {
        if(gate.kind < 0 || gate.kind > 6) throw std::invalid_argument("native control gate kind");
        validate_control_expression(gate.guard, control, nrecords);
        validate_control_expression(gate.exponent, control, nrecords);
    }
}
inline ControlResolution resolve_native_control(const NativeControl& control, const Bits& reports) {
    ControlResolution out;
    out.phases.resize(control.axes.size());
    std::vector<std::vector<int>> axes;
    for(auto& axis : control.axes) axes.push_back(control_axis(axis));
    auto wires = control_axis(control.qubits);
    std::vector<int> flips;
    for(auto& gate : control.gates) {
        if(!evaluate_control_expression(gate.guard, reports)) continue;
        auto axis = control_axis(gate.qubits);
        if(std::adjacent_find(axis.begin(), axis.end()) != axis.end() ||
                !std::includes(wires.begin(), wires.end(), axis.begin(), axis.end()))
            throw ControlMismatch("feedback wires outside declared family");
        if(gate.kind == 0) continue;
        auto found = std::find(axes.begin(), axes.end(), axis);
        if(gate.kind <= 3) {
            if(axis.size() != 1) throw ControlMismatch("feedback Pauli arity");
            if(found == axes.end()) throw ControlMismatch("undeclared feedback phase axis");
            int q = axis.front();
            auto flip = std::find(flips.begin(), flips.end(), q);
            bool shifted = flip != flips.end();
            if(gate.kind == 2 || gate.kind == 3) {
                out.phases[size_t(found - axes.begin())] += 4;
                out.phase += 4 * shifted;
            }
            if(gate.kind == 3) out.phase += 2;
            if(gate.kind == 1 || gate.kind == 3) {
                if(shifted) flips.erase(flip);
                else flips.push_back(q);
            }
        } else {
            if(found == axes.end()) throw ControlMismatch("undeclared feedback phase axis");
            int c = evaluate_control_expression(gate.exponent, reports), shifted = 0;
            for(int q : axis) shifted ^= std::find(flips.begin(), flips.end(), q) != flips.end();
            out.phases[size_t(found - axes.begin())] += shifted ? -c : c;
            out.phase += c * shifted;
        }
        out.phase = mod8(out.phase);
        // Keep arbitrarily long controller programs within bounded integers.
        for(auto& c : out.phases) c = mod8(c);
    }
    for(size_t j = 0; j < out.phases.size(); j++)
        if(mod8(out.phases[j] - control.baseline[j]) % 2)
            throw ControlMismatch("feedback changes the odd phase signature");
    for(int q : control.qubits)
        if(std::find(flips.begin(), flips.end(), q) != flips.end()) out.flips.push_back(q);
    out.phase = mod8(out.phase);
    return out;
}
inline void apply_native_controls(const std::vector<NativeControl>& controls, const Bits& reports,
        const std::vector<int>& baseline, Assignment& assignment) {
    if(controls.empty()) return;
    if(!assignment.feedback) {
        assignment.feedback = true;
        assignment.actual_magic = baseline;
    }
    if(assignment.actual_magic.size() != baseline.size()) throw std::invalid_argument("native feedback dimensions");
    std::vector<Fault> frames;
    for(auto& control : controls) {
        auto resolved = resolve_native_control(control, reports);
        for(size_t j = 0; j < resolved.phases.size(); j++)
            assignment.actual_magic.at(control.phase_slots[j]) = resolved.phases[j];
        for(int q : resolved.flips) frames.push_back({control.location, q, 1});
        assignment.feedback_phase = mod8(assignment.feedback_phase + resolved.phase);
    }
    // Feedback X precedes original faults at the same control ID. The kernel's
    // stable chronological ordering retains order among original faults.
    frames.insert(frames.end(), assignment.faults.begin(), assignment.faults.end());
    assignment.faults = std::move(frames);
}
}
