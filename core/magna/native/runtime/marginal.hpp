// Completed quantum contractions and classical record sampling, not time evolution.
#pragma once
#include "../evaluate/algebra.hpp"
#include "../evaluate/record_trace.hpp"
#include "../evaluate/trace_query.hpp"
#include "../foundation/bounded_cache.hpp"
namespace magic_native {
using StabilizerTerms = std::vector<std::pair<Scalar, Ket>>;
struct AffineMixture {
    StabilizerTerms components;
    Scalar total;
    size_t width = 0;
};
using RandomBits = std::function<Bits(size_t)>;
inline Bits sample_mixture(const AffineMixture& mixture, const RandomBits& random) {
    auto remaining = mixture.total;
    for(const auto& [weight, support] : mixture.components) {
        if(exact_bernoulli(weight, remaining, [&]() { return random(1).bit(0); }))
            return support.base ^ xors(support.columns, random(support.columns.size()));
        remaining = remaining + weight * Scalar(-1);
    }
    throw std::runtime_error("nonempty normalized mixture required");
}
struct ProbabilityMarginal {
    StabilizerTerms terms;
    size_t width;
    Scalar total(const std::vector<Bits>& rows = {}, const std::vector<int>& outcomes = {}) const {
        Workspace ws;
        return boundary_sum(terms, rows, outcomes, ws);
    }
    Scalar value(const Bits& word) const {
        Scalar out;
        for(const auto& [c, k] : terms) {
            auto forms = transpose(k.columns, k.width);
            auto rhs = word ^ k.base;
            Factor f(forms, k.columns.size());
            if(f.consistent(rhs)) out = out + c * omega(k.phase + 2 * qvalue(f.solve(rhs), k.linear, k.upper), k.power);
        }
        return out;
    }
    std::optional<AffineMixture> mixture() const {
        AffineMixture out;
        out.width = width;
        // Keep insertion order, as in the operation-level RNG contract.
        for(const auto& [c, k] : terms) {
            if(std::any_of(k.linear.begin(), k.linear.end(), [](int v) { return v != 0; }) ||
                    std::any_of(k.upper.begin(), k.upper.end(), [](const Bits& b) { return !b.empty(); }))
                return {};
            auto basis = solve_low_pivots(k.columns, std::vector<int>(k.columns.size()), k.width)->rows;
            auto base = k.base;
            for(const auto& row : basis)
                if(base.bit(row.indices().front())) base ^= row;
            auto w = c * omega(k.phase, k.power);
            w.power += int64_t(k.columns.size());
            bool found = false;
            for(auto& [weight, support] : out.components)
                if(support.base == base && support.columns == basis) {
                    weight = weight + w;
                    found = true;
                    break;
                }
            if(!found) {
                Ket support;
                support.width = width;
                support.base = base;
                support.columns = basis;
                support.linear.resize(basis.size());
                support.upper.resize(basis.size());
                support.power = -int64_t(basis.size());
                out.components.push_back({w, std::move(support)});
            }
        }
        for(const auto& [w, k] : out.components)
            if(real_sign(w) < 0) return {};
        out.components.erase(std::remove_if(out.components.begin(), out.components.end(),
                                     [](const auto& item) { return item.first.zero(); }),
                out.components.end());
        for(const auto& [w, k] : out.components) out.total = out.total + w;
        return out;
    }
    std::pair<Bits, std::string> sample(const RandomBits& random) const {
        auto affine = mixture();
        if(affine) {
            if(real_sign(affine->total) <= 0) throw std::invalid_argument("cannot sample zero marginal");
            return {sample_mixture(*affine, random), "affine-mixture"};
        }
        auto current = total();
        if(real_sign(current) <= 0) throw std::invalid_argument("cannot sample zero marginal");
        std::vector<Bits> rows;
        std::vector<int> outcomes;
        Bits word;
        for(size_t j = 0; j < width; j++) {
            rows.push_back(Bits::unit(j));
            outcomes.push_back(0);
            auto plus = total(rows, outcomes);
            int bit = !exact_bernoulli(plus, current, [&]() { return random(1).bit(0); });
            current = bit ? current + plus * Scalar(-1) : plus;
            outcomes.back() = bit;
            if(bit) word.set(j);
        }
        return {word, "exact-global-marginals"};
    }
};
struct RecordMarginals {
    std::vector<Bits> rows, events;
    std::vector<int> outcomes;
    size_t max_plans = 12;
    BoundedCache<RecordTracePlan> plans;
    std::mutex mutex;
    Ket pair_unlocked(const Ket& a, const Ket& b, Workspace& ws) {
        auto* plan = plans.find(
                [&](const RecordTracePlan& p) { return p.width == a.width && p.ac == a.columns && p.bc == b.columns; });
        if(!plan) plan = plans.insert(RecordTracePlan(a, b, rows, events, outcomes), max_plans);
        return plan->evaluate(a, b, ws);
    }
    ProbabilityMarginal evaluate(const StabilizerTerms& terms, const Bits& x, const Bits& z) {
        std::lock_guard<std::mutex> lock(mutex);
        for(const auto& row : rows)
            if((row & x).parity()) throw std::invalid_argument("Pauli must commute with record and event rows");
        for(const auto& row : events)
            if((row & x).parity()) throw std::invalid_argument("Pauli must commute with record and event rows");
        ProbabilityMarginal out{{}, rows.size()};
        Workspace ws;
        auto phase = omega(2 * int((x & z).count()), 0);
        for(const auto& [ca, a] : terms)
            for(const auto& [cb, b] : terms) {
                auto k = pair_unlocked(a, pauli_ket(b, x, z), ws);
                if(!k.zero) out.terms.push_back({ca.conjugate() * cb * phase, std::move(k)});
            }
        return out;
    }
};
inline Ket contract_target(const Ket& k, const Ket& target, size_t outputs, Workspace& ws) {
    if(target.width != outputs || outputs > k.width)
        throw std::invalid_argument("target must cover exactly the declared output boundary");
    auto columns = k.columns;
    columns.insert(columns.end(), target.columns.begin(), target.columns.end());
    auto rows = transpose(columns, k.width);
    rows.resize(outputs);
    std::vector<int> rhs;
    for(size_t j = 0; j < outputs; j++) rhs.push_back(k.base.bit(j) ^ target.base.bit(j));
    auto solution = solve_low_pivots(rows, rhs, columns.size());
    if(!solution) {
        Ket out;
        out.zero = true;
        return out;
    }
    auto linear = k.linear;
    for(int v : target.linear) linear.push_back(mod4(-v));
    auto upper = k.upper;
    for(const auto& v : target.upper) upper.push_back(v << k.columns.size());
    auto q = restrict_q(linear, upper, solution->particular, solution->kernel);
    for(auto& col : columns) col = col >> outputs;
    Ket out;
    out.width = k.width - outputs;
    out.base = (k.base >> outputs) ^ xors(columns, solution->particular);
    for(const auto& v : solution->kernel) out.columns.push_back(xors(columns, v));
    out.linear = std::move(q.linear);
    out.upper = std::move(q.upper);
    out.power = k.power + target.power;
    out.phase = k.phase - target.phase + 2 * q.constant;
    return contract(std::move(out), ws);
}
inline Scalar target_weight(const StabilizerTerms& terms, const StabilizerTerms& target, size_t outputs) {
    Workspace ws;
    StabilizerTerms contracted;
    for(const auto& [c, k] : terms)
        for(const auto& [d, bra] : target) {
            auto state = contract_target(k, bra, outputs, ws);
            if(!state.zero) contracted.push_back({c * d.conjugate(), std::move(state)});
        }
    return phase_norm(contracted, ws);
}
inline Scalar expansion_pauli(const StabilizerTerms& terms, const Bits& x, const Bits& z) {
    Workspace ws;
    Scalar out;
    auto phase = omega(2 * int((x & z).count()), 0);
    for(const auto& [ca, a] : terms)
        for(const auto& [cb, b] : terms) out = out + ca.conjugate() * cb * phase * overlap(a, pauli_ket(b, x, z), ws);
    return out;
}
} // namespace magic_native
