// Bell equalities and task boundaries precede every resource expansion.
#pragma once
#include "../compile/compiler.hpp"
#include "query_compiler.hpp"
#include <mutex>

namespace magic_native {
struct QueryMeasurement {
    TraceDiagnostics diagnostics;
    size_t path_bits = 0, rank = 0, terms = 0, retained = 0;
    bool affine = false, zero = false;
    double seconds = 0;
};
inline QueryMeasurement measure_boundary_query(const BoundaryResult& result, size_t retained, double seconds) {
    return {result.diagnostics, result.tensor.n, phase_quotient(result.tensor).axes.size(), result.terms.size(),
            retained, result.affine, result.tensor.zero, seconds};
}
// Bounds of a boundary engine: the public options max_path_bits,
// max_doubled_path_bits, max_plans and the phase-resource limits.
struct BoundaryLimits {
    size_t paths = 4096, doubled = 8192, plans = 4;
    PhaseLimits phase;
};
struct BoundaryTracePlan {
    CompilerIR ir;
    CompilerPreparation compiled;
    PhaseLimits limits;
    PhaseDatabase database;
    size_t max_doubled = 4096;
    TraceFactorCache pairing_cache;
    bool axes_ready = false, common_axes = false;
    std::mutex axes_mutex;
    std::vector<Frame> axes;
    // A prefix certificate keeps original generator indices, not endpoint
    // frames: a target suffix must transport those generators to its own boundary.
    std::vector<size_t> axis_generators;
    size_t axis_slots = 0, shared_axis_generators = 0;
    bool axis_prefix_matches(const BoundaryTracePlan& prefix) const {
        if(!prefix.axes_ready || prefix.ir.inputs.size() != ir.inputs.size() ||
                prefix.ir.operations.size() > ir.operations.size())
            return false;
        std::map<int, int> wires, inverse;
        auto match_wire = [&](int a, int b) {
            auto found = wires.find(a);
            if(found != wires.end()) return found->second == b;
            if(inverse.count(b)) return false;
            wires[a] = b;
            inverse[b] = a;
            return true;
        };
        for(size_t j = 0; j < ir.inputs.size(); j++)
            if(!match_wire(prefix.ir.inputs[j], ir.inputs[j])) return false;
        for(size_t j = 0; j < prefix.ir.operations.size(); j++) {
            const auto& a = prefix.ir.operations[j];
            const auto& b = ir.operations[j];
            if(a.kind != b.kind || a.exponent != b.exponent || a.sign != b.sign || a.record != b.record ||
                    a.location != b.location || a.feedback_slot != b.feedback_slot || a.pauli != b.pauli ||
                    a.paulis != b.paulis || a.qubits.size() != b.qubits.size())
                return false;
            for(size_t k = 0; k < a.qubits.size(); k++)
                if(!match_wire(a.qubits[k], b.qubits[k])) return false;
        }
        return true;
    }
    static CompilerIR clifford_part(CompilerIR stripped) {
        for(auto& op : stripped.operations) {
            if(((op.kind == "T" || op.kind == "S" || op.kind == "PHASE") && (op.exponent & 1)) || op.kind == "CCZ") {
                op.kind = "I";
                op.qubits.clear();
            }
        }
        return stripped;
    }
    Ket affine_control_support(const std::vector<Bits>& rows) const {
        auto reference = prepare_compiler(clifford_part(ir), false, nullptr, false, CompilerPurpose::Trace);
        const auto& ket = reference.plan.skeleton;
        Ket support;
        support.width = rows.size();
        for(size_t j = 0; j < rows.size(); j++)
            if(rows[j].parity_and(ket.base)) support.base.set(j);
        std::map<size_t, Bits> basis;
        for(const auto& col : ket.columns) {
            Bits projected;
            for(size_t j = 0; j < rows.size(); j++)
                if(rows[j].parity_and(col)) projected.set(j);
            auto residual = projected;
            while(!residual.empty()) {
                auto found = basis.find(residual.size() - 1);
                if(found == basis.end()) break;
                residual ^= found->second;
            }
            if(!residual.empty()) {
                basis[residual.size() - 1] = residual;
                support.columns.push_back(projected);
            }
        }
        support.linear.resize(support.columns.size());
        support.upper.resize(support.columns.size());
        // This tensor contains probabilities, not amplitudes.
        support.power = -2 * int64_t(support.columns.size());
        return support;
    }
    // Compiled query families, least recently used first out. Reference networks
    // own their parents; this bound only limits arbitrary manual query families.
    static constexpr size_t max_networks = 128;
    struct CachedNetwork {
        std::shared_ptr<const CompiledNetwork> network;
        uint64_t used = 0;
    };
    mutable std::mutex networks_mutex;
    mutable std::map<BoundarySpec, CachedNetwork> networks;
    mutable uint64_t network_uses = 0;
    mutable std::optional<TracePreparation> preparation;
    size_t amplitude_path_bits = 0;

    std::shared_ptr<const CompiledNetwork> compile(
            const BoundarySpec& boundary, bool resources = true, bool stable_reference = false) const {
        std::lock_guard<std::mutex> lock(networks_mutex);
        auto found = networks.find(boundary);
        if(found != networks.end() && (!resources || found->second.network->resources_ready) &&
                (!stable_reference || found->second.network->stable_requested)) {
            found->second.used = ++network_uses;
            return found->second.network;
        }
        Workspace ws;
        if(!preparation) {
            auto temporary = prepare_compiler(ir, false, nullptr, false, CompilerPurpose::Trace);
            preparation.emplace(std::move(temporary.plan.skeleton), temporary.plan.physical, temporary.plan.magic);
        }
        auto network = std::make_shared<const CompiledNetwork>(compile_network(*preparation, compiled.plan.magic,
                boundary, ws, pairing_cache, max_doubled, limits, &database, resources, stable_reference));
        networks[boundary] = {network, ++network_uses};
        if(networks.size() > max_networks)
            networks.erase(std::min_element(networks.begin(), networks.end(),
                    [](const auto& a, const auto& b) { return a.second.used < b.second.used; }));
        return network;
    }
    void finish_preparation() const {
        std::lock_guard<std::mutex> lock(networks_mutex);
        preparation.reset();
    }

    mutable std::optional<Plan> manual_transport;
    mutable std::mutex transport_mutex;
    const Plan& assignment_transport() const {
        if(!compiled.plan.locations.empty() || !compiled.plan.continuation) return compiled.plan;
        std::lock_guard<std::mutex> lock(transport_mutex);
        if(!manual_transport)
            manual_transport = prepare_compiler(ir, false, nullptr, true, CompilerPurpose::Trace).plan;
        return *manual_transport;
    }
    void prepare(CompilerPathPrefix* prefix = nullptr, bool branch_transport_only = false) {
        preparation.reset();
        networks.clear();
        manual_transport.reset();
        if(ir.records) throw std::invalid_argument("deferred instrument required");
        for(auto& o : ir.operations)
            if(o.feedback_slot >= 0) throw std::invalid_argument("unresolved feedback slot");
        // A trace query pairs this entire fixed-input amplitude with its conjugate.
        // Its common phase cancels; relative phases and normalization remain exact.
        compiled = prepare_compiler(ir, false, prefix, true, CompilerPurpose::Trace, branch_transport_only);
        // Records have already been deferred, so the amplitude record kernel
        // is the identity on these columns; no basis needs to be materialized.
        amplitude_path_bits = compiled.plan.skeleton.columns.size();
        preparation.emplace(std::move(compiled.plan.skeleton), compiled.plan.physical, compiled.plan.magic);
        // Runtime keeps only original-location transport metadata and B/C.
        // The output-open dilation is temporary compiler work.
        compiled.plan.skeleton = {};
        // This is a linear expansion of the doubled tensor, not a ket whose
        // R-by-R Gram matrix is needed. Only the actual term/table caps apply.
        limits.max_pairs = std::numeric_limits<size_t>::max();
    }

    void prepare_axes(const BoundaryTracePlan* prefix = nullptr) {
        std::lock_guard<std::mutex> lock(axes_mutex);
        if(!axes_ready) {
            std::map<size_t, Bits> pivots;
            bool shared = prefix && axis_prefix_matches(*prefix);
            common_axes = shared ? prefix->common_axes : true;
            axis_slots = compiled.rotation_axes.size();
            size_t next_generator = 0;
            for(size_t j = 0; j < compiled.rotation_axes.size(); j++) {
                bool inherited = shared && j < prefix->axis_slots;
                if(inherited) {
                    if(next_generator == prefix->axis_generators.size() || prefix->axis_generators[next_generator] != j)
                        continue;
                    next_generator++;
                    shared_axis_generators++;
                }
                auto axis = compiled.rotation_axes[j];
                auto residual = axis.x | (axis.z << compiled.plan.physical);
                while(!residual.empty()) {
                    auto found = pivots.find(residual.size() - 1);
                    if(found == pivots.end()) break;
                    residual ^= found->second;
                }
                if(residual.empty()) continue;
                pivots[residual.size() - 1] = std::move(residual);
                // A common Clifford suffix preserves linear dependencies and
                // symplectic products of all prefix axes, including retired legs.
                if(!inherited && common_axes)
                    for(const auto& prior : axes)
                        if(axis.x.parity_and(prior.z) ^ axis.z.parity_and(prior.x)) {
                            common_axes = false;
                            break;
                        }
                axes.push_back(std::move(axis));
                axis_generators.push_back(j);
            }
            axes_ready = true;
        }
    }
    std::vector<Bits> commuting_rows(const std::vector<Bits>& rows, const BoundaryTracePlan* prefix = nullptr) {
        for(auto& r : rows)
            if(r.size() > compiled.plan.physical) throw std::invalid_argument("commuting row outside boundary");
        prepare_axes(prefix);
        if(!common_axes) return {};
        std::vector<Bits> anti;
        for(auto& a : axes) {
            Bits row;
            for(size_t j = 0; j < rows.size(); j++)
                if(a.x.parity_and(rows[j])) row.set(j);
            anti.push_back(row);
        }
        return Factor(anti, rows.size()).kernel;
    }

    BoundaryResult query(const Assignment& a, const std::vector<Bits>& keep, const std::vector<Bits>& events,
            const std::vector<int>& values, const Bits& x, const Bits& z, bool affine_only, Workspace& ws,
            bool validated = false, const Frame* shared_frame = nullptr) const {
        if(!validated) {
            validate_trace_boundary(compiled.plan.physical, keep, events, x, z);
            validate_trace_outcomes(events.size(), values);
        }
        auto network = compile({keep, events, x, z}, !affine_only);
        finish_preparation();
        auto frame = shared_frame ? *shared_frame : assignment_transport().fault_frame(a.faults, false);
        return network->evaluate(
                frame, a.feedback ? a.actual_magic : compiled.plan.magic, values, affine_only, ws, limits, &database);
    }
};
} // namespace magic_native
