// Stabilizer decompositions of phase tensors: budgets, database entries, prepared
// contractions, resources and actual values. No Python calls.
#pragma once
#include "phase_tensor.hpp"
#include "../foundation/limits.hpp"
#include <set>
#include <memory>
#include <cassert>

namespace magic_native {
struct PhaseLimits {
    size_t max_quotient_rank = 8, max_terms = 256, max_pairs = 65536, max_table_bytes = 64 * 1024 * 1024;
    // Settings named by budget errors; paired limits name their derivation.
    const char *rank_option = "max_quotient_rank", *terms_option = "max_terms", *pairs_option = "max_pair_contractions";
};
// Operation-level budgets describe r side terms and r^2 paired contractions.
inline PhaseLimits paired_phase_limits(const PhaseLimits& side) {
    auto paired = side;
    paired.max_quotient_rank = 2 * side.max_quotient_rank;
    paired.rank_option = "2*max_quotient_rank";
    if(side.max_terms && side.max_terms > side.max_pairs / side.max_terms) {
        paired.max_terms = side.max_pairs;
        paired.terms_option = side.pairs_option;
    } else {
        paired.max_terms = side.max_terms * side.max_terms;
        paired.terms_option = "max_terms**2";
    }
    paired.max_pairs = std::numeric_limits<size_t>::max();
    return paired;
}
struct PhaseEntryTerm {
    Scalar coefficient;
    PhaseTerms constraints, parities;
};
struct PhaseDatabaseEntry {
    size_t dimension = 0;
    std::vector<int> core;
    std::vector<PhaseEntryTerm> terms;
};
struct PhaseDatabase {
    std::map<std::vector<int>, std::vector<PhaseEntryTerm>> entries;
    void add(PhaseDatabaseEntry entry) {
        if(entry.dimension > 8 || entry.core.size() != (size_t(1) << entry.dimension))
            throw std::invalid_argument("small database dimension invalid");
        if(entries.count(entry.core)) throw std::invalid_argument("duplicate phase database entry");
        for(int e : entry.core)
            if(e < 0 || e > 7) throw std::invalid_argument("database phase convention");
        for(auto& t : entry.terms) {
            for(auto& [a, b] : t.constraints)
                if(a.size() > entry.dimension || (b != 0 && b != 1))
                    throw std::invalid_argument("database support convention");
            for(auto& [a, c] : t.parities)
                if(a.size() > entry.dimension) throw std::invalid_argument("database parity convention");
        }
        for(size_t x = 0; x < entry.core.size(); x++) {
            Scalar value;
            for(auto& t : entry.terms) {
                bool include = true;
                for(auto& [a, b] : t.constraints)
                    if((a & Bits(x)).parity() != b) {
                        include = false;
                        break;
                    }
                if(!include) continue;
                int phase = 0;
                for(auto& [a, c] : t.parities) phase = mod8(phase + 2 * mod4(c) * (a & Bits(x)).parity());
                value = value + t.coefficient * omega(phase, 0);
            }
            Scalar expected = omega(entry.core[x], 0);
            if(value.power != expected.power || value.c != expected.c)
                throw std::invalid_argument("exact phase database reconstruction failed");
        }
        entries.emplace(std::move(entry.core), std::move(entry.terms));
    }
};
// One homogeneous contraction map for an unconstrained character expansion.
// It is independent of entry signs and actual affine boundary offsets. No
// per-entry matrices, assignment table or sampling-prefix objects are stored.
struct PhaseContraction {
    size_t dimension = 0;
    std::vector<Bits> rows, change, columns, axes;
    Quadratic reference, pulled_reference;
    CliffordContraction clifford;
    bool matches(const PhaseTensor& tensor) const {
        if(tensor.n != dimension || tensor.boundary.size() != rows.size()) return false;
        for(size_t j = 0; j < rows.size(); j++)
            if(rows[j] != tensor.boundary[j].first) return false;
        return true;
    }
    Quadratic pullback(const std::vector<int>& linear, const std::vector<Bits>& upper) const {
        if(upper == reference.upper) {
            Bits signs;
            bool even = true;
            for(size_t j = 0; j < linear.size(); j++) {
                int delta = mod4(linear[j] - reference.linear[j]);
                if(delta & 1) {
                    even = false;
                    break;
                }
                if(delta == 2) signs.set(j);
            }
            if(even) {
                auto result = pulled_reference;
                for(size_t j = 0; j < change.size(); j++)
                    result.linear[j] = mod4(result.linear[j] + 2 * change[j].parity_and(signs));
                return result;
            }
        }
        return restrict_q(linear, upper, {}, change);
    }
};
// Fixed support and boundary geometry of a selected constrained entry.
// Assignment transport changes only its pulled-back quadratic and scalar.
struct PhaseEntryPlan {
    Bits shift, base;
    std::vector<Bits> columns;
    Quadratic reference;
    size_t group = 0;
    CliffordContraction clifford;
    bool zero = false;
};
struct PhaseEntryPlans {
    size_t dimension = 0;
    std::vector<Bits> rows;
    std::vector<PhaseEntryPlan> entries;
    Quadratic reference;
    std::vector<std::vector<Bits>> changes;
    bool matches(const PhaseTensor& t) const {
        if(t.n != dimension || t.boundary.size() != rows.size()) return false;
        for(size_t j = 0; j < rows.size(); j++)
            if(rows[j] != t.boundary[j].first) return false;
        return true;
    }
};
struct PhaseResource {
    PolynomialQuotient quotient;
    std::vector<int> core_phase_mod8;
    std::vector<PhaseEntryTerm> entries;
    std::shared_ptr<const std::vector<PhaseEntryTerm>> prepared_entries;
    std::shared_ptr<const PhaseContraction> contraction;
    std::shared_ptr<const PhaseEntryPlans> entry_plans;
    bool database_hit = false, database_product = false;
    size_t constructive_terms = 0;
    size_t original_terms = 0, injection_rotations = 0, injection_candidate_terms = 0;
    bool injection_constructed = false, injection_selected = false, injection_budget_skipped = false;
    int amplitude_denominator_power = 0, constant_mod8 = 0;
};
inline const std::vector<PhaseEntryTerm>& resource_entries(const PhaseResource& resource) {
    return resource.prepared_entries ? *resource.prepared_entries : resource.entries;
}
inline PhaseResource freeze_resource(PhaseResource resource) {
    if(!resource.prepared_entries)
        resource.prepared_entries = std::make_shared<const std::vector<PhaseEntryTerm>>(std::move(resource.entries));
    return resource;
}
inline size_t phase_entry_bytes(const PhaseEntryTerm& t) {
    size_t bytes = sizeof(PhaseEntryTerm);
    for(auto& c : t.coefficient.c) bytes += c.d.size() * sizeof(uint32_t);
    for(auto& [a, c] : t.constraints) bytes += sizeof(std::pair<Bits, int>) + a.w.size() * sizeof(uint64_t);
    for(auto& [a, c] : t.parities) bytes += sizeof(std::pair<Bits, int>) + a.w.size() * sizeof(uint64_t);
    return bytes;
}
inline size_t phase_entries_bytes(const std::vector<PhaseEntryTerm>& terms) {
    size_t bytes = 0;
    for(auto& t : terms) bytes += phase_entry_bytes(t);
    return bytes;
}
inline void phase_check_bytes(size_t used, size_t extra, const PhaseLimits& limits) {
    if(used > limits.max_table_bytes || extra > limits.max_table_bytes - used)
        throw PhaseBudgetError("prepared decomposition tables exceed the memory budget", "max_table_bytes",
                double(limits.max_table_bytes), double(used) + double(extra));
}
inline void phase_check_terms(size_t n, const PhaseLimits& limits) {
    if(n > limits.max_terms)
        throw PhaseBudgetError("stabilizer decomposition exceeds the term budget", limits.terms_option,
                double(limits.max_terms), double(n));
    if(n && n > limits.max_pairs / n)
        throw PhaseBudgetError("stabilizer decomposition exceeds the pair-contraction budget", limits.pairs_option,
                double(limits.max_pairs), double(n) * double(n));
}
inline void prepare_phase_entry_plans(
        const PhaseTensor& t, PhaseResource& resource, size_t table_bytes, const PhaseLimits& limits) {
    const auto& entries = resource_entries(resource);
    auto plans = std::make_shared<PhaseEntryPlans>();
    plans->dimension = t.n;
    plans->reference = {0, resource.quotient.linear, resource.quotient.upper};
    for(const auto& [row, offset] : t.boundary) plans->rows.push_back(row);
    auto columns = transpose(plans->rows, t.n);
    size_t bytes = table_bytes + phase_entries_bytes(entries);
    for(const auto& entry : entries) {
        // Include fixed maps, elimination scratch and retained quadratic data.
        const size_t row = sizeof(Bits) + 8 * ((std::max(t.n, t.boundary.size()) + 63) / 64);
        phase_check_bytes(bytes, (24 * t.n + 4) * row, limits);
        bytes += (24 * t.n + 4) * row;
        PhaseEntryPlan p;
        std::vector<Bits> constraints, change;
        Bits rhs;
        for(size_t j = 0; j < entry.constraints.size(); j++) {
            constraints.push_back(xors(resource.quotient.axes, entry.constraints[j].first));
            if(entry.constraints[j].second) rhs.set(j);
        }
        Factor f(constraints, t.n);
        p.zero = !f.consistent(rhs);
        if(!p.zero) {
            p.shift = f.solve(rhs);
            p.base = xors(columns, p.shift);
            std::vector<Bits> supported;
            for(const auto& axis : f.kernel) supported.push_back(xors(columns, axis));
            for(const auto& axis : column_change(supported)) {
                change.push_back(xors(f.kernel, axis));
                p.columns.push_back(xors(supported, axis));
            }
            std::vector<int> linear(t.n);
            std::vector<Bits> upper(t.n);
            PhaseTerms increments;
            for(const auto& [mask, c] : entry.parities)
                increments.push_back({xors(resource.quotient.axes, mask), 2 * mod4(c)});
            phase_add_even(linear, upper, increments);
            auto fixed = restrict_q(linear, upper, p.shift, change);
            auto q = restrict_q(resource.quotient.linear, resource.quotient.upper, p.shift, change);
            for(size_t j = 0; j < q.linear.size(); j++) {
                q.linear[j] = mod4(q.linear[j] + fixed.linear[j]);
                q.upper[j] ^= fixed.upper[j];
            }
            q.constant = mod4(q.constant + fixed.constant);
            p.reference = q;
            auto group = std::find(plans->changes.begin(), plans->changes.end(), change);
            p.group = size_t(group - plans->changes.begin());
            if(group == plans->changes.end()) plans->changes.push_back(change);
            p.clifford = CliffordContraction(p.columns, q);
        }
        plans->entries.push_back(std::move(p));
    }
    resource.entry_plans = std::move(plans);
}
inline void prepare_phase_contraction(
        const PhaseTensor& tensor, PhaseResource& resource, size_t table_bytes, const PhaseLimits& limits) {
    const auto& entries = resource_entries(resource);
    if(tensor.zero || entries.empty()) return;
    for(const auto& entry : entries) {
        if(!entry.constraints.empty()) {
            try {
                prepare_phase_entry_plans(tensor, resource, table_bytes, limits);
            } catch(const PhaseBudgetError&) {} // Optional geometry obeys the same explicit table budget.
            return;
        }
        for(const auto& [mask, coefficient] : entry.parities)
            if(coefficient & 1) return;
    }
    // Bound both retained maps and temporary column/dual maps before allocation.
    // A tight resource budget simply omits this optional acceleration.
    size_t used = table_bytes + phase_entries_bytes(entries);
    if(used > limits.max_table_bytes || sizeof(PhaseContraction) > limits.max_table_bytes - used) return;
    used += sizeof(PhaseContraction);
    auto reserve = [&](size_t count, size_t bits) {
        size_t row = sizeof(Bits) + 2 * (bits / 64 + (bits % 64 != 0)) * sizeof(uint64_t);
        if(used > limits.max_table_bytes || count > (limits.max_table_bytes - used) / row) return false;
        used += count * row;
        return true;
    };
    const size_t n = tensor.n, width = tensor.boundary.size();
    if(!reserve(width, n) || !reserve(n, width) || !reserve(n, width) || !reserve(n, width) || !reserve(n, width) ||
            !reserve(resource.quotient.axes.size(), n) || !reserve(n, 512))
        return;
    for(int j = 0; j < 20; j++)
        if(!reserve(n, n)) return; // Basis, elimination steps and temporaries.
    auto plan = std::make_shared<PhaseContraction>();
    plan->dimension = n;
    plan->rows.reserve(width);
    plan->columns.reserve(n);
    plan->axes.reserve(resource.quotient.axes.size());
    for(const auto& [mask, offset] : tensor.boundary) plan->rows.push_back(mask);
    auto columns = transpose(plan->rows, n);
    plan->change = column_change(columns);
    for(const auto& v : plan->change) plan->columns.push_back(xors(columns, v));
    auto dual = transpose(plan->change, n);
    for(const auto& axis : resource.quotient.axes) plan->axes.push_back(xors(dual, axis));
    plan->reference = {0, resource.quotient.linear, resource.quotient.upper};
    plan->pulled_reference = restrict_q(plan->reference.linear, plan->reference.upper, {}, plan->change);
    plan->clifford = CliffordContraction(plan->columns, plan->pulled_reference);
    resource.contraction = std::move(plan);
}
// Pull a product of exact one-/two-T decompositions back to the resource
// coordinates. Keep those coordinates open: cancellations after a physical
// boundary or a particular assignment are not valid transport certificates.
inline void prepare_injection_candidate(PhaseResource& d, const PhaseTensor& tensor, const PhaseLimits& limits,
        const PhaseDatabase* database, size_t table_bytes) {
    d.original_terms = d.entries.size();
    const size_t r = d.quotient.axes.size();
    if(tensor.zero || d.original_terms <= 1 || r <= 1 || !database) return;
    const auto one = database->entries.find({0, 1}), two = database->entries.find({0, 1, 1, 2});
    if(one == database->entries.end() || two == database->entries.end()) return;
    // These database entries are disjoint affine partitions, with one entry
    // per parity value. This permits contraction of all injection constraints
    // before the product is expanded, instead of allocating 2^ceil(t/2) terms.
    auto partition = [](const auto& entries, size_t mask) {
        if(entries.size() != 2) return false;
        for(size_t j = 0; j < 2; j++)
            if(entries[j].constraints != PhaseTerms{{Bits(mask), int(j)}} || entries[j].coefficient.zero())
                return false;
        return true;
    };
    if(!partition(one->second, 1) || !partition(two->second, 3)) return;
    std::vector<Bits> remaining;
    PhaseTerms even;
    for(const auto& [mask, c] : d.quotient.core) {
        if(c & 1) remaining.push_back(mask);
        if(c - (c & 1)) even.push_back({mask, c - (c & 1)});
    }
    d.injection_rotations = remaining.size();
    if(remaining.empty()) return;
    const size_t live = table_bytes + phase_entries_bytes(d.entries);
    // The greedy pairing uses at most 2^r difference keys. Bound its maps,
    // factors and quadratic scratch together, including dynamic bit payloads.
    const size_t row_bytes = 256 + 16 * ((r + 63) / 64);
    const size_t scratch = (remaining.size() * 8 + d.core_phase_mod8.size() + r * r + 1) * row_bytes;
    phase_check_bytes(live, scratch, limits);
    std::vector<std::vector<Bits>> groups;
    std::map<size_t, Bits> span;
    auto remainder = [&](Bits mask) {
        for(auto it = span.rbegin(); it != span.rend(); ++it)
            if(mask.bit(it->first)) mask ^= it->second;
        return mask;
    };
    while(remaining.size() > 1) {
        std::map<Bits, std::vector<Bits>, BitsLess> cosets;
        for(const auto& mask : remaining) cosets[remainder(mask)].push_back(mask);
        remaining.clear();
        for(auto& [key, masks] : cosets) {
            size_t j = 0;
            for(; j + 1 < masks.size(); j += 2) groups.push_back({masks[j], masks[j + 1]});
            if(j < masks.size()) remaining.push_back(masks[j]);
        }
        if(remaining.size() < 2) break;
        // Add the difference that joins the most remaining cosets. This is
        // only a deterministic pairing heuristic, not a T-count search.
        std::map<Bits, size_t, BitsLess> counts;
        for(size_t i = 0; i < remaining.size(); i++)
            for(size_t j = i + 1; j < remaining.size(); j++) counts[remainder(remaining[i] ^ remaining[j])]++;
        auto best = counts.begin();
        for(auto it = counts.begin(); it != counts.end(); ++it)
            if(it->second > best->second) best = it;
        span.emplace(best->first.size() - 1, best->first);
    }
    if(!remaining.empty()) groups.push_back({remaining.front()});
    std::vector<Bits> rows;
    for(const auto& group : groups) {
        Bits row;
        for(const auto& mask : group) row ^= mask;
        rows.push_back(std::move(row));
    }
    Factor support(rows, r);
    const size_t k = support.pivots.size();
    const size_t count = size_t(1) << k;
    d.injection_candidate_terms = count;
    // Every compatible partition has nonempty, disjoint support and nonzero
    // amplitude on that support. Thus this is already the exact surviving
    // count: there can be no duplicates or further pointwise cancellations.
    if(count >= d.original_terms) return;
    phase_check_terms(count, limits);
    d.injection_constructed = true;
    std::vector<Bits> independent;
    for(const auto& [pivot, row] : support.pivots) independent.push_back(row.first);
    // Unique reduced support rows, independent of the selected affine RHS.
    for(size_t j = 0; j < independent.size(); j++)
        for(size_t i = j + 1; i < independent.size(); i++)
            if(independent[i].bit(independent[j].size() - 1)) independent[i] ^= independent[j];
    Factor canonical(independent, r);
    std::vector<size_t> free;
    for(size_t j = 0; j < r; j++)
        if(!canonical.pivots.count(j)) free.push_back(j);
    std::vector<int> common_linear(r);
    std::vector<Bits> common_upper(r);
    phase_add_even(common_linear, common_upper, even);
    std::vector<PhaseEntryTerm> candidate;
    size_t bytes = live + scratch;
    for(size_t word = 0; word < count; word++) {
        Bits rhs(word), shift = canonical.solve(rhs);
        PhaseEntryTerm entry{Scalar(1), {}, {}};
        auto linear = common_linear;
        auto upper = common_upper;
        PhaseTerms updates;
        for(const auto& group : groups) {
            Bits row;
            for(const auto& mask : group) row ^= mask;
            const auto& component = (group.size() == 1 ? one->second : two->second)[row.parity_and(shift)];
            entry.coefficient = entry.coefficient * component.coefficient;
            for(const auto& [mask, c] : component.parities) updates.push_back({xors(group, mask), 2 * mod4(c)});
        }
        phase_add_even(linear, upper, updates);
        auto q = restrict_q(linear, upper, shift, canonical.kernel);
        entry.coefficient = entry.coefficient * omega(2 * q.constant, 0);
        for(size_t j = 0; j < independent.size(); j++) entry.constraints.push_back({independent[j], int(rhs.bit(j))});
        // The canonical kernel parameters are precisely the free coordinates.
        // Extend their restricted Clifford phase to the full resource domain.
        PhaseCounter parities;
        for(size_t j = 0; j < free.size(); j++) {
            Bits a = Bits::unit(free[j]);
            parities[a] += q.linear[j];
            for(auto h : q.upper[j].indices()) {
                Bits b = Bits::unit(free[h]);
                parities[a]++;
                parities[b]++;
                parities[a ^ b]--;
            }
        }
        for(const auto& [mask, c] : parities)
            if(mod4(c)) entry.parities.push_back({mask, mod4(c)});
        const size_t extra = phase_entry_bytes(entry);
        phase_check_bytes(bytes, extra, limits);
        bytes += extra;
        candidate.push_back(std::move(entry));
    }
    d.entries = std::move(candidate);
    d.injection_selected = true;
    // These reused database coefficients are integral cyclotomic values.
    d.database_hit = true;
    d.database_product = true;
    d.amplitude_denominator_power = 0;
}

inline PhaseResource phase_resource(const PhaseTensor& t, PolynomialQuotient quotient, const PhaseLimits& limits,
        const PhaseDatabase* database = nullptr) {
    PhaseResource d;
    d.constant_mod8 = t.constant;
    d.quotient = std::move(quotient);
    size_t r = d.quotient.axes.size();
    if(r > limits.max_quotient_rank)
        throw PhaseBudgetError("phase coordinates left after Clifford elimination exceed the rank budget",
                limits.rank_option, double(limits.max_quotient_rank), double(r));
    if(r >= std::numeric_limits<size_t>::digits || (size_t(1) << r) > limits.max_table_bytes / sizeof(int))
        throw PhaseBudgetError("phase coordinate table exceeds the memory budget", "max_table_bytes",
                double(limits.max_table_bytes), std::ldexp(double(sizeof(int)), int(r)));
    size_t count = size_t(1) << r;
    d.core_phase_mod8.resize(count);
    for(size_t x = 0; x < count; x++)
        for(auto& [a, c] : d.quotient.core)
            if((a & Bits(x)).parity()) d.core_phase_mod8[x] = mod8(d.core_phase_mod8[x] + c);
    size_t table_bytes = count * sizeof(int);
    if(database) {
        auto it = database->entries.find(d.core_phase_mod8);
        if(it != database->entries.end()) {
            phase_check_bytes(table_bytes, phase_entries_bytes(it->second), limits);
            d.entries = it->second;
            d.database_hit = true;
        } else {
            bool product = true;
            for(size_t x = 0; x < count; x++)
                if(d.core_phase_mod8[x] != int(__builtin_popcountll(x) % 8)) {
                    product = false;
                    break;
                }
            if(product) {
                PhaseEntryTerm identity{Scalar(1), {}, {}};
                // Rank zero skips the product loop, so its unit entry needs
                // the same payload guard as every nontrivial product entry.
                phase_check_bytes(table_bytes, phase_entry_bytes(identity), limits);
                d.entries.push_back(std::move(identity));
                for(size_t offset = 0; offset < r; offset += 2) {
                    size_t m = std::min(size_t(2), r - offset);
                    std::vector<int> core(size_t(1) << m);
                    for(size_t x = 0; x < core.size(); x++) core[x] = __builtin_popcountll(x);
                    auto component = database->entries.find(core);
                    if(component == database->entries.end()) {
                        product = false;
                        break;
                    }
                    size_t live_bytes = table_bytes + phase_entries_bytes(d.entries), next_bytes = 0;
                    std::vector<PhaseEntryTerm> next;
                    for(auto& a : d.entries)
                        for(auto& b : component->second) {
                            PhaseEntryTerm term = a;
                            term.coefficient = a.coefficient * b.coefficient;
                            for(auto& [v, c] : b.constraints) term.constraints.push_back({v << offset, c});
                            for(auto& [v, c] : b.parities) term.parities.push_back({v << offset, c});
                            size_t bytes = phase_entry_bytes(term);
                            phase_check_bytes(live_bytes + next_bytes, bytes, limits);
                            next_bytes += bytes;
                            next.push_back(std::move(term));
                        }
                    d.entries = std::move(next);
                }
                if(product) {
                    d.database_hit = true;
                    d.database_product = true;
                } else d.entries.clear();
            }
        }
    }
    if(!d.database_hit) {
        // A Walsh coefficient has magnitude at most 2**r in each cyclotomic
        // component. Bound all limb payloads before allocating the transform.
        size_t coefficient_bytes = sizeof(std::array<Integer, 4>) + 4 * ((r + 32) / 32) * sizeof(uint32_t);
        if(count > (limits.max_table_bytes - table_bytes) / coefficient_bytes)
            throw PhaseBudgetError("phase coordinate transform exceeds the memory budget", "max_table_bytes",
                    double(limits.max_table_bytes), double(table_bytes) + double(count) * double(coefficient_bytes));
        std::vector<std::array<Integer, 4>> coefficients(count);
        for(size_t x = 0; x < count; x++) {
            int e = d.core_phase_mod8[x];
            coefficients[x][e % 4] = Integer(e < 4 ? 1 : -1);
        }
        for(size_t stride = 1; stride < count; stride *= 2)
            for(size_t base = 0; base < count; base += 2 * stride)
                for(size_t j = base; j < base + stride; j++) {
                    auto a = coefficients[j], b = coefficients[j + stride];
                    for(size_t k = 0; k < 4; k++) {
                        coefficients[j][k] = a[k] + b[k];
                        coefficients[j + stride][k] = a[k] - b[k];
                    }
                }
        size_t live_bytes = table_bytes + count * sizeof(std::array<Integer, 4>);
        for(auto& row : coefficients)
            for(auto& v : row) live_bytes += v.d.size() * sizeof(uint32_t);
        d.amplitude_denominator_power = int(r);
        for(size_t x = 0; x < count; x++) {
            Scalar c;
            c.c = coefficients[x];
            c.power = -int64_t(r);
            c.normalize();
            if(!c.zero()) {
                PhaseEntryTerm term{c, {}, {{Bits(x), 2}}};
                size_t bytes = phase_entry_bytes(term);
                phase_check_bytes(live_bytes, bytes, limits);
                live_bytes += bytes;
                d.entries.push_back(std::move(term));
            }
        }
    }
    // Optional preparation only. A candidate budget stop leaves the supported
    // existing decomposition intact; no candidate is retained after selection.
    try {
        prepare_injection_candidate(d, t, limits, database, table_bytes);
    } catch(const PhaseBudgetError&) { d.injection_budget_skipped = true; }
    prepare_phase_contraction(t, d, table_bytes, limits);
    return d;
}
inline PhaseResource phase_resource(
        const PhaseTensor& t, const PhaseLimits& limits, const PhaseDatabase* database = nullptr) {
    return phase_resource(t, phase_quotient(t), limits, database);
}
inline std::vector<std::pair<Scalar, Ket>> phase_states(
        const PhaseTensor& t, const PhaseResource& d, Workspace& ws, const PhaseLimits& limits) {
    std::vector<std::pair<Scalar, Ket>> states;
    if(t.zero) return states;
    std::vector<Bits> rows;
    Bits base;
    if(d.contraction && d.contraction->matches(t)) {
        const auto& plan = *d.contraction;
        // The actual Clifford update can change the quadratic polar form.
        // Pull it back once per query, then transport each character as a sign.
        auto quadratic = plan.pullback(d.quotient.linear, d.quotient.upper);
        const CliffordContraction* contraction = &plan.clifford;
        CliffordContraction updated;
        if(!contraction->matches(quadratic)) {
            updated = CliffordContraction(plan.columns, quadratic);
            contraction = &updated;
        }
        // An isolated hidden variable sums to zero when its linear sign is
        // negative. Test these exact cancellations before copying entry tensors.
        // Surviving entries share the compiled elimination and update its RHS.
        Bits coupled, isolated, required;
        for(size_t j = 0; j < t.n; j++) {
            if(!quadratic.upper[j].empty() || (quadratic.linear[j] & 1)) coupled.set(j);
            coupled |= quadratic.upper[j];
        }
        for(size_t j = 0; j < t.n; j++)
            if(plan.columns[j].empty() && !coupled.bit(j)) {
                isolated.set(j);
                if(quadratic.linear[j] == 2) required.set(j);
            }
        for(size_t j = 0; j < t.boundary.size(); j++)
            if(t.boundary[j].second) base.set(j);
        for(const auto& entry : resource_entries(d)) {
            Bits character;
            for(const auto& [mask, coefficient] : entry.parities)
                if(mod4(coefficient) == 2) character ^= xors(plan.axes, mask);
            if((character & isolated) != required) continue;
            auto linear = quadratic.linear;
            for(auto j : character.indices()) linear[j] = mod4(linear[j] + 2);
            auto k = contraction->evaluate(t.boundary.size(), base, std::move(linear), t.power, t.constant);
            if(!k.zero) {
                states.push_back({entry.coefficient, std::move(k)});
                phase_check_terms(states.size(), limits);
            }
        }
        return states;
    }
    if(d.entry_plans && d.entry_plans->matches(t)) {
        for(size_t j = 0; j < t.boundary.size(); j++)
            if(t.boundary[j].second) base.set(j);
        const auto& entries = resource_entries(d);
        const auto& plans = *d.entry_plans;
        // The fixed quotient Clifford is already restricted in each entry.
        // Pull back only its update, once per naturally shared support map.
        std::vector<int> linear(t.n);
        std::vector<Bits> upper(t.n);
        for(size_t j = 0; j < t.n; j++) {
            linear[j] = mod4(d.quotient.linear[j] - plans.reference.linear[j]);
            upper[j] = d.quotient.upper[j] ^ plans.reference.upper[j];
        }
        auto polar = upper;
        for(size_t j = 0; j < t.n; j++) {
            if(linear[j] & 1) polar[j].set(j);
            for(auto k : upper[j].indices()) polar[k].set(j);
        }
        std::vector<Quadratic> pulled;
        pulled.reserve(plans.changes.size());
        for(const auto& change : plans.changes) pulled.push_back(restrict_q(linear, upper, {}, change));
        for(size_t index = 0; index < entries.size(); index++) {
            const auto& p = plans.entries[index];
            if(p.zero) continue;
            auto q = p.reference;
            const auto& delta = pulled[p.group];
            auto shifted = xors(polar, p.shift);
            q.constant = mod4(q.constant + qvalue(p.shift, linear, upper));
            for(size_t j = 0; j < q.linear.size(); j++) {
                q.linear[j] = mod4(q.linear[j] + delta.linear[j] + 2 * plans.changes[p.group][j].parity_and(shifted));
                q.upper[j] ^= delta.upper[j];
            }
            const CliffordContraction* contraction = &p.clifford;
            CliffordContraction changed;
            if(!contraction->matches(q)) {
                changed = CliffordContraction(p.columns, q);
                contraction = &changed;
            }
            auto k = contraction->evaluate(
                    t.boundary.size(), base ^ p.base, std::move(q.linear), t.power, mod8(t.constant + 2 * q.constant));
            if(!k.zero) {
                states.push_back({entries[index].coefficient, std::move(k)});
                phase_check_terms(states.size(), limits);
            }
        }
        return states;
    }
    for(size_t j = 0; j < t.boundary.size(); j++) {
        rows.push_back(t.boundary[j].first);
        if(t.boundary[j].second) base.set(j);
    }
    auto cols = transpose(rows, t.n);
    for(auto& entry : resource_entries(d)) {
        auto linear = d.quotient.linear;
        auto upper = d.quotient.upper;
        PhaseTerms updates;
        std::vector<Bits> constraints;
        Bits rhs;
        for(auto& [a, c] : entry.parities) updates.push_back({xors(d.quotient.axes, a), 2 * mod4(c)});
        phase_add_even(linear, upper, updates);
        for(size_t j = 0; j < entry.constraints.size(); j++) {
            constraints.push_back(xors(d.quotient.axes, entry.constraints[j].first));
            if(entry.constraints[j].second) rhs.set(j);
        }
        Ket k;
        k.width = t.boundary.size();
        k.base = base;
        k.power = t.power;
        k.phase = t.constant;
        if(constraints.empty()) {
            k.columns = cols;
            k.linear = linear;
            k.upper = upper;
        } else {
            Factor f(constraints, t.n);
            if(!f.consistent(rhs)) continue;
            Bits shift = f.solve(rhs);
            auto q = restrict_q(linear, upper, shift, f.kernel);
            k.base ^= xors(cols, shift);
            for(auto& v : f.kernel) k.columns.push_back(xors(cols, v));
            k.linear = q.linear;
            k.upper = q.upper;
            k.phase += 2 * q.constant;
        }
        k = contract(std::move(k), ws);
        if(!k.zero) {
            states.push_back({entry.coefficient, std::move(k)});
            phase_check_terms(states.size(), limits);
        }
    }
    return states;
}
inline Scalar phase_norm(const std::vector<std::pair<Scalar, Ket>>& states, Workspace& ws) {
    Scalar value;
    for(size_t i = 0; i < states.size(); i++) {
        auto& [a, k] = states[i];
        value = value + a.conjugate() * a * norm(k);
        for(size_t j = i + 1; j < states.size(); j++) {
            auto& [b, h] = states[j];
            Scalar cross = a.conjugate() * b * overlap(k, h, ws);
            value = value + cross + cross.conjugate();
        }
    }
    return value;
}
struct PhaseResult {
    PhaseTensor tensor;
    PhaseResource resource;
    std::vector<std::pair<Scalar, Ket>> states;
    Scalar probability;
};
inline PhaseResult decompose_phase(
        PhaseTensor t, const PhaseLimits& limits, const PhaseDatabase* database, Workspace& ws) {
    PhaseResult r;
    r.tensor = std::move(t);
    if(r.tensor.zero) return r;
    r.resource = phase_resource(r.tensor, limits, database);
    r.states = phase_states(r.tensor, r.resource, ws, limits);
    r.resource.constructive_terms = r.states.size();
    r.probability = phase_norm(r.states, ws);
    return r;
}

// Apply the actual fault frame, projector RHS and Clifford increments before absorption.
inline PhaseTensor actual_phase_tensor(const Plan& p, const Assignment& a, const std::vector<int>& fixed_linear,
        const std::vector<Bits>& fixed_upper, const PhaseTerms& resource_terms, bool track_global_phase = true) {
    PhaseTensor t;
    t.n = p.basis.size();
    t.linear = fixed_linear;
    t.upper = fixed_upper;
    t.terms = resource_terms;
    t.power = p.skeleton.power;
    t.output_count = p.physical;
    Frame frame = p.fault_frame(a.faults, track_global_phase);
    Bits base = p.skeleton.base ^ frame.x, rhs = a.record ^ (base >> p.retained);
    if(!p.records.consistent(rhs)) {
        t.zero = true;
        t.n = 0;
        t.power = 0;
        t.linear.clear();
        t.upper.clear();
        t.terms.clear();
        t.boundary.resize(p.physical);
        return t;
    }
    Bits shift = p.records.solve(rhs), character = xors(p.polar, shift);
    for(size_t j = 0; j < p.skeleton.columns.size(); j++)
        if((frame.z & p.skeleton.columns[j]).parity()) character.flip(j);
    for(size_t j = 0; j < p.basis.size(); j++) t.linear[j] = mod4(t.linear[j] + 2 * (character & p.basis[j]).parity());
    if(track_global_phase)
        t.constant = mod8(p.skeleton.phase + 2 * frame.p + 4 * (frame.z & p.skeleton.base).parity() +
                2 * qvalue(shift, p.skeleton.linear, p.skeleton.upper) +
                4 * (frame.z & xors(p.skeleton.columns, shift)).parity() + a.feedback_phase);
    base ^= xors(p.skeleton.columns, shift);
    PhaseTerms increments;
    for(size_t j = 0; j < p.magic.size(); j++) {
        int actual = a.feedback ? a.actual_magic.at(j) : p.magic[j];
        bool bit = base.bit(p.physical + j);
        if(track_global_phase) t.constant = mod8(t.constant + actual * bit);
        int delta = mod8((bit ? -actual : actual) - p.magic[j]);
        if(delta && !p.forms[p.physical + j].empty()) increments.push_back({p.forms[p.physical + j], delta});
    }
    phase_add_even(t.linear, t.upper, increments);
    for(size_t j = 0; j < p.physical; j++) t.boundary.push_back({p.forms[j], int(base.bit(j))});
    return t;
}

inline PhaseTerms phase_as_parities(const PhaseTensor& t) {
    PhaseCounter terms;
    for(const auto& [a, c] : t.terms) terms[a] = mod8(c);
    for(size_t j = 0; j < t.n; j++) {
        auto unit = Bits::unit(j);
        terms[unit] = mod8(terms[unit] + 2 * t.linear[j]);
        for(auto k : t.upper[j].indices()) {
            auto other = Bits::unit(k);
            terms[unit] = mod8(terms[unit] + 2);
            terms[other] = mod8(terms[other] + 2);
            terms[unit | other] = mod8(terms[unit | other] - 2);
        }
    }
    return phase_terms(terms);
}
inline Ket phase_clifford_tensor(const PhaseTensor& t) {
    auto q = phase_quotient(t);
    if(!q.axes.empty()) throw std::invalid_argument("non-Clifford residual remains; scalar not evaluated");
    Ket k;
    k.width = t.boundary.size();
    k.zero = t.zero;
    k.power = t.power;
    k.phase = t.constant;
    if(t.zero) return k;
    std::vector<Bits> rows;
    for(size_t j = 0; j < t.boundary.size(); j++) {
        rows.push_back(t.boundary[j].first);
        if(t.boundary[j].second) k.base.set(j);
    }
    k.columns = transpose(rows, t.n);
    k.linear = std::move(q.linear);
    k.upper = std::move(q.upper);
    Workspace ws;
    return contract(std::move(k), ws);
}

inline Scalar phase_scalar_value(const PhaseTensor& tensor) {
    if(!tensor.boundary.empty()) throw std::invalid_argument("scalar tensor required; event labels remain");
    if(tensor.zero) return {};
    auto ket = phase_clifford_tensor(tensor);
    return ket.zero ? Scalar() : omega(ket.phase, ket.power);
}
inline std::optional<std::pair<Scalar, Ket>> phase_affine_distribution(const PhaseTensor& tensor) {
    if(tensor.zero || !phase_quotient(tensor).axes.empty()) return {};
    auto ket = phase_clifford_tensor(tensor);
    if(ket.zero || std::any_of(ket.linear.begin(), ket.linear.end(), [](int v) { return v != 0; }) ||
            std::any_of(ket.upper.begin(), ket.upper.end(), [](const Bits& b) { return !b.empty(); }))
        return {};
    auto total = omega(ket.phase, ket.power);
    total.power += int64_t(ket.columns.size());
    if(real_sign(total) <= 0) return {};
    ket.phase = 0;
    ket.power = -int64_t(ket.columns.size());
    return std::make_pair(total, ket);
}
struct PublicResourceTerm {
    std::array<Integer, 4> coefficient;
    PhaseTerms constraints, parities;
    Bits axis;
};
inline std::vector<PublicResourceTerm> public_resource_terms(const PhaseResource& resource) {
    std::vector<PublicResourceTerm> out;
    for(const auto& entry : resource_entries(resource)) {
        int64_t power = entry.coefficient.power + (resource.database_hit ? 0 : resource.amplitude_denominator_power);
        if(power < 0) throw std::invalid_argument("nonintegral public resource coefficient");
        PublicResourceTerm term;
        for(size_t j = 0; j < 4; j++) term.coefficient[j] = entry.coefficient.c[j].shift(size_t(power));
        term.constraints = entry.constraints;
        term.parities = entry.parities;
        if(!resource.database_hit && !entry.parities.empty())
            term.axis = xors(resource.quotient.axes, entry.parities.begin()->first);
        out.push_back(std::move(term));
    }
    return out;
}
} // namespace magic_native
