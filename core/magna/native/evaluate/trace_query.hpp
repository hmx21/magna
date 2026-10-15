// Shared production probability kernel for full records, events and retained labels.
#pragma once
#include "single_trace.hpp"
#include "resource_certificate.hpp"
namespace magic_native {
struct BoundaryResult {
    PhaseTensor tensor;
    PhaseResource resource;
    std::vector<std::pair<Scalar, Ket>> terms;
    bool affine = false;
    TraceDiagnostics diagnostics;
};

// Integrate a probability tensor. Its entries are already probabilities (or
// signed effects); neither this function nor its callers take a norm squared.
inline Scalar boundary_sum(const std::vector<std::pair<Scalar, Ket>>& terms, const std::vector<Bits>& rows,
        const std::vector<int>& values, Workspace& ws) {
    if(rows.size() != values.size()) throw std::invalid_argument("boundary event shape");
    for(int b : values)
        if(b != 0 && b != 1) throw std::invalid_argument("binary boundary outcome required");
    size_t required_width = 0;
    for(const auto& row : rows) required_width = std::max(required_width, row.size());
    Scalar total;
    for(auto& [coefficient, k] : terms) {
        if(required_width > k.width) throw std::invalid_argument("boundary row outside tensor");
        auto forms = transpose(k.columns, k.width);
        std::vector<Bits> equations;
        Bits rhs;
        for(size_t j = 0; j < rows.size(); j++) {
            equations.push_back(xors(forms, rows[j]));
            if(values[j] ^ (rows[j] & k.base).parity()) rhs.set(j);
        }
        Factor f(equations, k.columns.size());
        if(!f.consistent(rhs)) continue;
        auto shift = f.solve(rhs);
        auto q = restrict_q(k.linear, k.upper, shift, f.kernel);
        Ket scalar;
        scalar.columns.resize(f.kernel.size());
        scalar.linear = std::move(q.linear);
        scalar.upper = std::move(q.upper);
        scalar.phase = mod8(k.phase + 2 * q.constant);
        scalar.power = k.power;
        scalar = contract(std::move(scalar), ws);
        if(!scalar.zero) total = total + coefficient * omega(scalar.phase, scalar.power);
    }
    return total;
}

struct TraceReduction {
    BoundaryResult result;
    TracePairingKernel pairing;
    PhaseTensor source;
    bool has_source = false;
};

inline void validate_trace_boundary(
        size_t width, const std::vector<Bits>& keep, const std::vector<Bits>& events, const Bits& x, const Bits& z) {
    if(x.size() > width || z.size() > width) throw std::invalid_argument("Pauli outside boundary");
    for(const auto* rows : {&keep, &events})
        for(const auto& row : *rows) {
            if(row.size() > width) throw std::invalid_argument("query outside boundary");
            if((row & x).parity()) throw std::invalid_argument("event/record must commute with Pauli");
        }
}
inline void validate_trace_outcomes(size_t count, const std::vector<int>& values) {
    if(count != values.size()) throw std::invalid_argument("boundary event shape");
    for(int bit : values)
        if(bit != 0 && bit != 1) throw std::invalid_argument("binary event required");
}

inline TraceReduction reduce_trace_query(PhaseTensor t, const std::vector<Bits>& keep, const std::vector<Bits>& events,
        const std::vector<int>& values, const Bits& x, const Bits& z, Workspace& ws,
        const TraceFactorCache& support_cache, const TraceFactorCache& pairing_cache, PathBudget max_doubled) {
    TraceReduction reduction;
    auto& result = reduction.result;
    auto& d = result.diagnostics;
    auto& b = result.tensor;
    auto tick = std::chrono::steady_clock::now();
    auto elapsed = [&]() {
        auto now = std::chrono::steady_clock::now();
        double seconds = std::chrono::duration<double>(now - tick).count();
        tick = now;
        return seconds;
    };
    d.actual_seconds = elapsed();
    d.input_bits = t.n;
    d.event_rows = events.size();
    auto form = [&](const Bits& row) {
        Bits mask;
        int bit = 0;
        for(auto j : row.indices()) {
            mask ^= t.boundary[j].first;
            bit ^= t.boundary[j].second;
        }
        return std::make_pair(mask, bit);
    };
    std::vector<Bits> equations;
    Bits rhs;
    for(size_t j = 0; j < events.size(); j++) {
        auto [row, bit] = form(events[j]);
        equations.push_back(row);
        if(bit ^ values[j]) rhs.set(j);
    }
    b.output_count = keep.size();
    b.boundary.resize(keep.size());
    auto support = support_cache.get(equations, t.n, d.support_hit);
    d.support_factorizations = !d.support_hit;
    d.event_rank = support->factor.pivots.size();
    d.support_bits = support->factor.kernel.size();
    if(t.zero || !support->factor.consistent(rhs)) {
        b.zero = true;
        d.support_seconds = elapsed();
        return reduction;
    }
    t = restrict_trace_support(std::move(t), *support, rhs);
    d.support_seconds = elapsed();
    auto side = single_trace_side(std::move(t), ws);
    d.single_seconds = elapsed();
    d.phase_axes = side.phase_axes;
    d.clifford_bits = side.clifford.columns.size();
    reduction.has_source = true;
    reduction.source.n = side.phase_axes;
    reduction.source.linear.resize(side.phase_axes);
    reduction.source.upper.resize(side.phase_axes);
    reduction.source.terms = side.phase;
    auto& pairing = reduction.pairing;
    b = pair_trace_side(side, keep, x, z, pairing_cache, max_doubled, d, pairing, ws);
    d.pairing_seconds = elapsed();
    if(b.zero) return reduction;
    // Only queried labels are resource boundaries. Every trace slot has
    // already entered the implicit pairing kernel, with its phase intact.
    b = absorb_phase(std::move(b), ws, false);
    d.reduce_seconds = elapsed();
    if(b.zero) return reduction;
    return reduction;
}

inline BoundaryResult expand_trace_query(TraceReduction reduction, bool affine_only, Workspace& ws,
        const PhaseLimits& limits, const PhaseDatabase* database, const ResourceCertificate* joint_reference = nullptr,
        const ResourceCertificate* inherited_reference = nullptr, bool reference_only = false) {
    auto& result = reduction.result;
    auto& d = result.diagnostics;
    auto& b = result.tensor;
    auto tick = std::chrono::steady_clock::now();
    auto elapsed = [&]() { return std::chrono::duration<double>(std::chrono::steady_clock::now() - tick).count(); };
    if(b.zero) return std::move(result);
    auto transported =
            joint_reference ? joint_reference->transport(b, nullptr, reference_only) : std::optional<PhaseResource>{};
    // The prepared reference certifies every transported query representation.
    if(reference_only && !transported && !inherited_reference)
        throw std::logic_error("reference network does not cover this phase representation");
    // A prepared inherited certificate already supplies the resource when
    // the direct reduced representation changes. Do not recompute its rank.
    bool non_clifford = transported ? !transported->quotient.axes.empty()
                                    : (inherited_reference && !affine_only ? true : !phase_quotient(b).axes.empty());
    if(affine_only && non_clifford) {
        d.expand_seconds = elapsed();
        return std::move(result);
    }
    // Two descriptions of the SAME reduced pairing: its combined phase,
    // and the inherited r-by-r product. Keep both available so a Fourier
    // coordinate change cannot discard an existing small construction.
    PhaseResource inherited;
    bool joint_ready = false, inherited_ready = false, joint_reused = false, inherited_reused = false;
    const bool prepared_mode = joint_reference || inherited_reference;
    std::optional<PhaseBudgetError> budget_error;
    try {
        auto reused = std::move(transported);
        joint_reused = bool(reused);
        if(reused) {
            result.resource = std::move(*reused);
            joint_ready = true;
        } else if(!prepared_mode && !reference_only) {
            d.resource_constructions++;
            result.resource = phase_resource(b, limits, database);
            joint_ready = true;
        }
    } catch(const PhaseBudgetError& e) { budget_error = e; }
    if(non_clifford || inherited_reference) {
        try {
            auto reused = inherited_reference
                    ? inherited_reference->transport(reduction.source, nullptr, reference_only)
                    : std::optional<PhaseResource>{};
            inherited_reused = bool(reused);
            if(reused) {
                inherited = std::move(*reused);
                inherited_ready = true;
            } else if(!prepared_mode && !reference_only) {
                d.resource_constructions++;
                inherited = phase_resource(reduction.source, limits, database);
                inherited_ready = true;
            }
        } catch(const PhaseBudgetError& e) {
            if(!joint_ready) budget_error = e;
        }
    }
    auto joint = [&]() {
        result.terms = phase_states(b, result.resource, ws, limits);
        d.expanded_entries = resource_entries(result.resource).size();
    };
    auto product = [&]() { result.terms = inherited_trace_states(reduction.pairing, inherited, limits, ws, d); };
    bool evaluated = false;
    auto evaluate_candidates = [&](bool try_joint, bool try_inherited) {
        bool prefer_product = try_inherited &&
                (!try_joint ||
                        (resource_entries(inherited).size() &&
                                resource_entries(inherited).size() <=
                                        resource_entries(result.resource).size() / resource_entries(inherited).size()));
        for(int candidate = 0; candidate < 2 && !evaluated; candidate++) {
            bool use_product = prefer_product ^ (candidate != 0);
            if((use_product && !try_inherited) || (!use_product && !try_joint)) continue;
            try {
                if(use_product) product();
                else joint();
                evaluated = true;
                d.reference_entries_reused = use_product ? inherited_reused : joint_reused;
            } catch(const PhaseBudgetError& e) { budget_error = e; }
        }
    };
    evaluate_candidates(joint_ready, inherited_ready);
    if(!evaluated && prepared_mode && !reference_only) {
        // A different reduced coordinate domain can need the other existing
        // representation. Construct only missing candidates; never replay
        // successful transport or an expansion that already hit its cap.
        bool new_joint = false, new_inherited = false;
        if(!joint_ready) try {
                d.resource_constructions++;
                result.resource = phase_resource(b, limits, database);
                new_joint = true;
            } catch(const PhaseBudgetError& e) { budget_error = e; }
        if(non_clifford && !inherited_ready) try {
                d.resource_constructions++;
                inherited = phase_resource(reduction.source, limits, database);
                new_inherited = true;
            } catch(const PhaseBudgetError& e) {
                if(!new_joint) budget_error = e;
            }
        evaluate_candidates(new_joint, new_inherited);
    }
    if(!evaluated) {
        if(budget_error) throw *budget_error;
        // Without a budget stop, only a reference-only transport can leave no candidate.
        throw std::logic_error("reference network does not cover this phase representation");
    }
    result.resource.constructive_terms = result.terms.size();
    if(result.terms.size() == 1) {
        auto& [c, k] = result.terms[0];
        result.affine = std::all_of(k.linear.begin(), k.linear.end(), [](int v) { return v == 0; }) &&
                std::all_of(k.upper.begin(), k.upper.end(), [](auto& v) { return v.empty(); }) &&
                real_sign(c * omega(k.phase, k.power)) > 0;
    }
    d.expand_seconds = elapsed();
    return std::move(result);
}

inline BoundaryResult single_trace_query(PhaseTensor t, const std::vector<Bits>& keep, const std::vector<Bits>& events,
        const std::vector<int>& values, const Bits& x, const Bits& z, bool affine_only, Workspace& ws,
        const TraceFactorCache& support_cache, const TraceFactorCache& pairing_cache, PathBudget max_doubled,
        const PhaseLimits& limits, const PhaseDatabase* database) {
    validate_trace_boundary(t.boundary.size(), keep, events, x, z);
    validate_trace_outcomes(events.size(), values);
    auto reduction =
            reduce_trace_query(std::move(t), keep, events, values, x, z, ws, support_cache, pairing_cache, max_doubled);
    return expand_trace_query(std::move(reduction), affine_only, ws, limits, database);
}
} // namespace magic_native
