#include "evaluate/resource_certificate.hpp"
#include <iostream>
#include <random>
using namespace magic_native;
void require(bool ok, const char* text) {
    if(!ok) throw std::runtime_error(text);
}
bool equal(const Scalar& a, const Scalar& b) { return a.power == b.power && a.c == b.c; }
PhaseDatabase database() {
    PhaseDatabase db;
    db.add({1, {0, 1}, {{Scalar(1), {{Bits(1), 0}}, {}}, {omega(1, 0), {{Bits(1), 1}}, {}}}});
    db.add({2, {0, 1, 1, 2}, {{Scalar(1), {{Bits(3), 0}}, {{Bits(1), 1}}}, {omega(1, 0), {{Bits(3), 1}}, {}}}});
    return db;
}
Scalar entry_value(const PhaseResource& d, const Bits& x) {
    Scalar sum;
    for(const auto& e : resource_entries(d)) {
        bool valid = true;
        for(const auto& [mask, b] : e.constraints) valid &= mask.parity_and(x) == b;
        if(!valid) continue;
        int p = 0;
        for(const auto& [mask, c] : e.parities) p += 2 * c * mask.parity_and(x);
        sum = sum + e.coefficient * omega(p, 0);
    }
    return sum;
}
Scalar amplitude(const std::vector<std::pair<Scalar, Ket>>& states, size_t word) {
    Scalar sum;
    for(const auto& [c, k] : states)
        for(size_t x = 0; x < (size_t(1) << k.columns.size()); x++)
            if((k.base ^ xors(k.columns, Bits(x))) == Bits(word))
                sum = sum + c * omega(k.phase + 2 * qvalue(Bits(x), k.linear, k.upper), k.power);
    return sum;
}
int main() {
    auto db = database();
    PhaseLimits limits{12, 4096, 16777216, 268435456};
    std::mt19937 rng(1729);
    size_t selected = 0, shared_changes = 0;
    // Exhaustive three-coordinate P3 signatures; random signs and Clifford
    // phases. The oracle is the original finite phase sum, not a decomposition.
    for(size_t n = 2; n <= 6; n++)
        for(size_t trial = 0; trial < 160; trial++) {
            PhaseTensor t;
            t.n = n;
            t.linear.resize(n);
            t.upper.resize(n);
            for(size_t j = 0; j < n; j++) {
                t.linear[j] = rng() % 4;
                t.boundary.push_back({Bits::unit(j), int(rng() % 2)});
            }
            for(size_t mask = 1; mask < (size_t(1) << n); mask++)
                if(n == 3 ? (trial & (size_t(1) << (mask - 1))) : (rng() % 4 == 0))
                    t.terms.push_back({Bits(mask), int(2 * (rng() % 4) + 1)});
            t.constant = rng() % 8;
            t.power = -3;
            auto optimized = phase_resource(t, limits, &db);
            selected += optimized.injection_selected;
            if(optimized.entry_plans && optimized.entry_plans->changes.size() < optimized.entry_plans->entries.size())
                shared_changes++;
            for(size_t x = 0; x < (size_t(1) << optimized.quotient.axes.size()); x++)
                require(equal(entry_value(optimized, Bits(x)), omega(optimized.core_phase_mod8[x], 0)),
                        "pointwise core reconstruction");
            if(n > 4) continue;
            ResourceCertificate certificate(t, optimized);
            Workspace ws;
            for(size_t assignment = 0; assignment < 4; assignment++) {
                auto actual = t;
                for(auto& [mask, c] : actual.terms)
                    if(rng() % 2) c = mod8(-c);
                for(size_t j = 0; j < n; j++) {
                    actual.linear[j] = rng() % 4;
                    actual.boundary[j].second = rng() % 2;
                }
                // Include coupled Clifford updates and noninjective final boundaries.
                actual.upper[0] = Bits::unit(n - 1);
                actual.constant = rng() % 8;
                actual.power = -5;
                if(assignment == 3) actual.boundary[0].first = actual.boundary[1].first;
                auto moved = certificate.transport(actual, nullptr, true);
                require(bool(moved), "prepared transport");
                require(moved->prepared_entries.get() == certificate.reference.prepared_entries.get(),
                        "reuse frozen entries");
                require(moved->entry_plans.get() == certificate.reference.entry_plans.get(),
                        "reuse fixed constrained geometry");
                if(optimized.injection_selected)
                    require(bool(moved->entry_plans), "selected injection support prepared");
                auto states = phase_states(actual, *moved, ws, limits);
                for(size_t word = 0; word < (size_t(1) << n); word++) {
                    Scalar direct;
                    for(size_t x = 0; x < (size_t(1) << n); x++) {
                        Bits output;
                        for(size_t j = 0; j < n; j++)
                            if(actual.boundary[j].first.parity_and(Bits(x)) ^ actual.boundary[j].second) output.set(j);
                        if(output != Bits(word)) continue;
                        int p = actual.constant + 2 * qvalue(Bits(x), actual.linear, actual.upper);
                        for(const auto& [mask, c] : actual.terms) p += c * mask.parity_and(Bits(x));
                        direct = direct + omega(p, actual.power);
                    }
                    require(equal(direct, amplitude(states, word)), "assignment amplitude and scalar");
                }
            }
        }
    require(selected > 0, "selected candidate coverage");
    require(shared_changes > 0, "shared change with distinct support shifts coverage");
    // All affine support branches must be contracted before expansion: seven
    // rotations on three bits have only one independent injection constraint.
    PhaseTensor t;
    t.n = 3;
    t.linear.resize(3);
    t.upper.resize(3);
    PhaseResource d;
    for(size_t j = 0; j < 3; j++) d.quotient.axes.push_back(Bits::unit(j));
    for(size_t mask = 1; mask < 8; mask++) d.quotient.core.push_back({Bits(mask), 1});
    d.entries.resize(8);
    d.core_phase_mod8.resize(8);
    prepare_injection_candidate(d, t, limits, &db, 32);
    require(d.injection_rotations == 7 && d.injection_candidate_terms == 2 && d.injection_selected,
            "dependent injection legs");
    for(size_t x = 0; x < 8; x++) require(equal(entry_value(d, Bits(x)), omega(x ? 4 : 0, 0)), "dependent pullback");
    // r=1 and the existing unit/zero cases never allocate a candidate.
    for(size_t n = 0; n <= 1; n++) {
        PhaseTensor small;
        small.n = n;
        small.linear.resize(n);
        small.upper.resize(n);
        if(n) small.terms = {{Bits(1), 7}};
        auto result = phase_resource(small, limits, &db);
        require(!result.injection_constructed && result.injection_rotations == 0, "small early exit");
    }
    PhaseTensor pair;
    pair.n = 2;
    pair.linear.resize(2);
    pair.upper.resize(2);
    pair.terms = {{Bits(1), 1}, {Bits(2), 1}};
    auto tie = phase_resource(pair, limits, &db);
    require(tie.original_terms == 2 && tie.injection_candidate_terms == 2 && !tie.injection_constructed &&
                    !tie.injection_selected,
            "tie keeps database entries");
    PhaseResource unit;
    unit.quotient.axes = {Bits(1), Bits(2)};
    unit.entries = {{Scalar(1), {}, {}}};
    prepare_injection_candidate(unit, pair, limits, &db, 16);
    require(unit.injection_rotations == 0 && !unit.injection_constructed, "unit resource early exit");
    auto tiny = limits;
    t.zero = true;
    d.injection_constructed = false;
    prepare_injection_candidate(d, t, tiny, &db, 32);
    require(!d.injection_constructed, "zero early exit");
    // An optional scratch-budget failure must preserve the old exact resource.
    t.zero = false;
    t.terms = {{Bits(1), 1}, {Bits(2), 1}, {Bits(4), 1}};
    tiny.max_table_bytes = 4096;
    auto fallback = phase_resource(t, tiny, &db);
    require(fallback.injection_budget_skipped, "optional budget fallback");
    for(size_t x = 0; x < 8; x++)
        require(equal(entry_value(fallback, Bits(x)), omega(__builtin_popcountll(x), 0)), "fallback exactness");
    std::cout << "injection checks passed; selected " << selected << " candidates\n";
}
