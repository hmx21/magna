// Own the original compact instrument, numerical compiler and branch evaluator.
#pragma once
#include "../compile/compiler.hpp"
#include "phase.hpp"
#include "../compile/control.hpp"
#include "trace_query.hpp"
#include <chrono>

namespace magic_native {
struct GlobalPlan {
    CompilerIR ir;
    CompilerPreparation compiled;
    PhaseDatabase database;
    PhaseLimits limits;
    TraceFactorCache support_cache, pairing_cache;
    std::vector<NativeControl> controls;
    bool use_database = false;
    size_t payload_bytes = 0;
    double compiler_seconds = 0.;

    const Plan& plan() const { return compiled.plan; }
    PhaseTensor actual(const Assignment& a, bool track_global_phase = true) const {
        auto t = actual_phase_tensor(
                plan(), a, compiled.linear, compiled.upper, compiled.magic_terms, track_global_phase);
        t.output_count = compiled.output_count;
        for(auto id : compiled.environment) t.environment.push_back(std::to_string(id));
        return t;
    }
    PhaseResult certificate(const Assignment& a, const PhaseTensor& normal, Workspace& ws) const {
        if(normal.zero) {
            PhaseResult result;
            result.tensor = normal;
            return result;
        }
        PhaseResult result;
        result.tensor = absorb_phase(normal, ws);
        result.resource = phase_resource(result.tensor, limits, use_database ? &database : nullptr);
        result.states = phase_states(result.tensor, result.resource, ws, limits);
        result.resource.constructive_terms = result.states.size();
        result.probability = evaluate(a, ws);
        return result;
    }

    Scalar evaluate(const Assignment& a, Workspace& ws) const {
        auto result = single_trace_query(actual(a, false), {}, {}, {}, {}, {}, false, ws, support_cache, pairing_cache,
                PathBudget(2 * ir.max_path_bits, "2*max_path_bits"), paired_phase_limits(limits),
                use_database ? &database : nullptr);
        auto value = boundary_sum(result.terms, {}, {}, ws);
        if(real_sign(value) < 0 || real_sign(value + Scalar(-1)) > 0)
            throw std::runtime_error("branch probability outside [0,1]");
        return value;
    }
};

// Deterministic retained data payload, excluding allocator/container overhead.
// This is an observable bound on values, not an RSS or transient-workspace cap.
inline size_t global_bits_bytes(const Bits& v) { return 8 * v.w.size(); }
inline size_t global_bits_bytes(const StoredBits& v) { return v.payload_bytes(); }
inline size_t global_bits_bytes(const std::vector<Bits>& v) {
    size_t n = 0;
    for(auto& a : v) n += 8 + global_bits_bytes(a);
    return n;
}
inline size_t global_scalar_bytes(const Scalar& s) {
    size_t n = 8;
    for(auto& c : s.c) n += 1 + 4 * c.d.size();
    return n;
}
inline size_t global_ket_bytes(const Ket& k) {
    return 32 + global_bits_bytes(k.base) + global_bits_bytes(k.columns) + 4 * k.linear.size() +
            global_bits_bytes(k.upper);
}
inline size_t global_phase_terms_bytes(const PhaseTerms& t) {
    size_t n = 0;
    for(auto& [a, c] : t) n += 12 + global_bits_bytes(a);
    return n;
}
inline size_t global_entry_bytes(const PhaseEntryTerm& t) {
    return global_scalar_bytes(t.coefficient) + global_phase_terms_bytes(t.constraints) +
            global_phase_terms_bytes(t.parities);
}
inline size_t global_payload_bytes(const GlobalPlan& g) {
    auto& p = g.plan();
    size_t n = global_ket_bytes(g.ir.initial) + 4 * (g.ir.inputs.size() + g.ir.outputs.size());
    for(auto& o : g.ir.operations) n += 24 + 4 * o.qubits.size() + o.kind.size() + o.pauli.size() + o.paulis.size();
    n += global_ket_bytes(p.skeleton) + global_bits_bytes(p.basis) + global_bits_bytes(p.polar) +
            global_bits_bytes(p.forms) + 4 * p.magic.size();
    for(auto& [j, v] : p.records.pivots) n += 8 + global_bits_bytes(v.first) + global_bits_bytes(v.second);
    n += global_bits_bytes(p.records.zeros) + global_bits_bytes(p.records.kernel) + 8 * p.locations.size();
    for(auto& [q, rows] : p.histories) n += 4 + 8 * rows.size();
    for(auto& [q, rows] : p.suffix) {
        n += 4;
        for(auto& r : rows)
            n += 12 + global_bits_bytes(r.x.x) + global_bits_bytes(r.x.z) + global_bits_bytes(r.z.x) +
                    global_bits_bytes(r.z.z);
    }
    n += 4 *
            (g.compiled.linear.size() + g.compiled.phase_controls.size() + g.compiled.environment.size() +
                    g.compiled.layer_slots.size());
    for(const auto& axis : g.compiled.rotation_axes) n += 4 + global_bits_bytes(axis.x) + global_bits_bytes(axis.z);
    n += global_bits_bytes(g.compiled.upper) + global_phase_terms_bytes(g.compiled.magic_terms);
    for(auto& [core, terms] : g.database.entries) {
        n += 4 * core.size();
        for(auto& t : terms) n += global_entry_bytes(t);
    }
    for(auto& c : g.controls) {
        n += 8 + 4 * c.qubits.size() + 4 * c.baseline.size() + 8 * (c.phase_slots.size() + c.parents.size());
        for(auto& a : c.axes) n += 4 * a.size();
        for(auto& gate : c.gates) n += 4 + 4 * gate.qubits.size() + 12 * (gate.exponent.size() + gate.guard.size());
    }
    return n;
}

inline void finish_global_controls(GlobalPlan& global, std::vector<NativeControl> controls, size_t max_bytes) {
    std::map<int, std::pair<int, const CompilerOp*>> slot_origins;
    std::vector<const CompilerOp*> pending;
    for(auto& o : global.ir.operations) {
        if(o.feedback_slot >= 0) pending.push_back(&o);
        if(o.location >= 0) {
            for(auto* p : pending) {
                if(slot_origins.count(p->feedback_slot))
                    throw std::invalid_argument("duplicate original feedback slot");
                slot_origins[p->feedback_slot] = {o.location, p};
            }
            pending.clear();
        }
    }
    if(!pending.empty()) throw std::invalid_argument("feedback phase missing original control marker");
    for(auto& c : controls) {
        std::set<size_t> earlier;
        bool found = false;
        for(auto& o : global.ir.operations) {
            if(o.location == int(c.location)) {
                found = true;
                break;
            }
            if(o.record >= 0) earlier.insert(size_t(o.record));
        }
        if(!found) throw std::invalid_argument("native control original location missing");
        for(auto r : c.parents)
            if(!earlier.count(r)) throw std::invalid_argument("native control reads future record");
        for(size_t j = 0; j < c.phase_slots.size(); j++) {
            int raw_slot = global.compiled.phase_controls.at(c.phase_slots[j]);
            auto it = slot_origins.find(raw_slot);
            if(it == slot_origins.end() || it->second.first != int(c.location) ||
                    control_axis(it->second.second->qubits) != control_axis(c.axes[j]))
                throw std::invalid_argument("native control phase slot does not belong to original declaration");
        }
    }
    global.controls = std::move(controls);
    global.payload_bytes = global_payload_bytes(global);
    if(global.payload_bytes > max_bytes)
        throw PhaseBudgetError("retained control data exceed the structure budget", "max_structure_bytes",
                double(max_bytes), double(global.payload_bytes));
}

inline void prepare_global(GlobalPlan& g, size_t max_bytes) {
    auto tick = std::chrono::steady_clock::now();
    g.compiled = prepare_compiler(g.ir);
    g.compiler_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - tick).count();
    g.payload_bytes = global_payload_bytes(g);
    if(g.payload_bytes > max_bytes)
        throw PhaseBudgetError("retained compiled data exceed the structure budget", "max_structure_bytes",
                double(max_bytes), double(g.payload_bytes));
}
inline size_t global_workspace_capacity(const Workspace& ws) {
    size_t n = sizeof(Workspace) + sizeof(int) * ws.linear.capacity();
    for(auto* v : {&ws.columns, &ws.adj}) {
        n += sizeof(Bits) * v->capacity();
        for(auto& b : *v) n += 8 * b.w.capacity();
    }
    return n;
}
} // namespace magic_native
