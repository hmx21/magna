// Phase polynomials over GF(2) and their polynomial P3 quotient, with a bounded
// thread-local template cache. No Python calls.
#pragma once
#include "../foundation/bits.hpp"
#include "../foundation/bounded_cache.hpp"
#include "../foundation/scalar.hpp"
#include <cassert>
#include <map>
#include <stdexcept>
#include <utility>
#include <vector>

namespace magic_native {
using PhaseTerms = std::vector<std::pair<Bits, int>>;
using PhaseCounter = std::map<Bits, int, BitsLess>;
inline PhaseTerms phase_terms(const PhaseCounter& c) {
    PhaseTerms out;
    for(auto& [a, v] : c)
        if(!a.empty() && mod8(v)) out.push_back({a, mod8(v)});
    return out;
}
inline std::vector<Bits> phase_image_basis(const std::vector<Bits>& rows) {
    std::map<size_t, Bits> pivots;
    std::vector<Bits> out;
    for(auto& original : rows) {
        Bits a = original;
        while(!a.empty()) {
            auto j = a.size() - 1;
            auto it = pivots.find(j);
            if(it == pivots.end()) {
                pivots[j] = a;
                out.push_back(original);
                break;
            }
            a ^= it->second;
        }
    }
    return out;
}
// Sparse rows of the phase trilinear form, indexed by (i,j), with k as column.
// Repeated indices encode T and controlled-S; distinct indices encode CCZ.
// Keep the dense reference's lexicographic row order.
inline std::map<std::pair<size_t, size_t>, Bits> phase_trilinear_rows(size_t n, const PhaseTerms& terms) {
    std::map<std::pair<size_t, size_t>, Bits> rows;
    for(auto& [a, c] : terms) {
        if(a.size() > n) throw std::invalid_argument("phase mask outside dimension");
        if(!(c & 1)) continue;
        auto js = a.indices();
        for(size_t p = 0; p < js.size(); p++)
            for(size_t q = p; q < js.size(); q++) rows[{js[p], js[q]}] ^= a;
    }
    for(auto it = rows.begin(); it != rows.end();)
        if(it->second.empty()) it = rows.erase(it);
        else ++it;
    return rows;
}
inline std::vector<Bits> phase_right_inverse(std::vector<Bits> rows, size_t n) {
    size_t r = rows.size(), k = 0;
    std::vector<Bits> rhs;
    std::vector<size_t> pivots;
    for(size_t j = 0; j < r; j++) rhs.push_back(Bits::unit(j));
    for(size_t p = 0; p < n && k < r; p++) {
        size_t q = k;
        while(q < r && !rows[q].bit(p)) q++;
        if(q == r) continue;
        std::swap(rows[k], rows[q]);
        std::swap(rhs[k], rhs[q]);
        for(size_t i = 0; i < r; i++)
            if(i != k && rows[i].bit(p)) {
                rows[i] ^= rows[k];
                rhs[i] ^= rhs[k];
            }
        pivots.push_back(p);
        k++;
    }
    if(k != r) throw std::invalid_argument("phase right inverse requires independent rows");
    std::vector<Bits> cols(r);
    for(size_t j = 0; j < r; j++)
        for(auto i : rhs[j].indices()) cols[i].set(pivots[j]);
    return cols;
}
inline void phase_add_even(std::vector<int>& linear, std::vector<Bits>& upper, const PhaseTerms& terms) {
    // Small updates are common in causal sampling and need no packing/scanning.
    // Larger batches accumulate binary outer products in contiguous row tiles;
    // each destination row is resized and canonicalized only once per batch.
    size_t dimension = 0, odd_terms = 0, odd_entries = 0;
    if(terms.size() >= 8)
        for(const auto& term : terms) {
            const auto& a = term.first;
            int value = term.second;
            if(value & 1) throw std::invalid_argument("Clifford update must be even");
            if(a.size() > linear.size() || a.size() > upper.size()) throw std::invalid_argument("phase dimension");
            dimension = std::max(dimension, a.size());
            if(value & 2) {
                odd_terms++;
                odd_entries += a.count();
            }
        }
    if(odd_terms >= 8 && dimension && odd_entries / 2 >= dimension && odd_entries / odd_terms >= 4) {
        const size_t words = (dimension + 63) / 64;
        // At most 32 KiB quadratic scratch, or one row for very wide inputs.
        const size_t tile_rows = std::min(dimension, std::max(size_t(1), size_t(4096) / words));
        std::vector<uint64_t> low(words), high(words), matrix(tile_rows * words);
        for(const auto& term : terms) {
            const auto& a = term.first;
            int c = mod8(term.second) / 2;
            if(c & 1)
                for(size_t k = 0; k < a.w.size(); k++) {
                    high[k] ^= (c == 1 ? low[k] : ~low[k]) & a.w[k];
                    low[k] ^= a.w[k];
                }
            else if(c == 2)
                for(size_t k = 0; k < a.w.size(); k++) high[k] ^= a.w[k];
        }
        for(size_t j = 0; j < dimension; j++)
            linear[j] = mod4(linear[j] + int((low[j / 64] >> (j % 64)) & 1) + 2 * int((high[j / 64] >> (j % 64)) & 1));
        for(size_t first = 0; first < dimension; first += tile_rows) {
            const size_t end = std::min(dimension, first + tile_rows);
            std::fill(matrix.begin(), matrix.end(), 0);
            for(const auto& term : terms) {
                if(!(term.second & 2)) continue;
                const auto& a = term.first;
                for(size_t w = first / 64; w < a.w.size() && w * 64 < end; w++) {
                    auto bits = a.w[w];
                    if(w == first / 64) bits &= ~uint64_t(0) << (first % 64);
                    if(end / 64 == w) bits &= (uint64_t(1) << (end % 64)) - 1;
                    while(bits) {
                        size_t j = 64 * w + __builtin_ctzll(bits);
                        bits &= bits - 1;
                        auto* row = matrix.data() + (j - first) * words;
                        if(j % 64 != 63) row[w] ^= a.w[w] & (~uint64_t(0) << (j % 64 + 1));
                        for(size_t k = w + 1; k < a.w.size(); k++) row[k] ^= a.w[k];
                    }
                }
            }
            for(size_t j = first; j < end; j++) {
                auto& row = upper[j];
                if(row.w.size() < words) row.w.resize(words);
                const auto* update = matrix.data() + (j - first) * words;
                for(size_t k = j / 64; k < words; k++) row.w[k] ^= update[k];
                row.trim();
            }
        }
        return;
    }
    for(auto& [a, value] : terms) {
        if(value & 1) throw std::invalid_argument("Clifford update must be even");
        if(a.size() > linear.size() || a.size() > upper.size()) throw std::invalid_argument("phase dimension");
        int c = mod8(value) / 2;
        if(!c) continue;
        for(size_t word = 0; word < a.w.size(); word++) {
            auto bits = a.w[word];
            while(bits) {
                size_t j = 64 * word + __builtin_ctzll(bits);
                bits &= bits - 1;
                linear[j] = mod4(linear[j] + c);
                if((c & 1) && j + 1 < a.size()) upper[j].xor_above(a, j);
            }
        }
    }
}
struct PolynomialQuotient {
    std::vector<Bits> axes;
    PhaseTerms core;
    std::vector<int> linear;
    std::vector<Bits> upper, lifts;
};
// Contract the full trilinear form to the section once, then read its nonzero
// weighted monomial coefficients. Repeated indices retain T and controlled-S.
inline PhaseTerms phase_section_core(size_t n, const PhaseTerms& odd, const std::vector<Bits>& lifts) {
    auto dual = transpose(lifts, n);
    PhaseTerms projected;
    projected.reserve(odd.size());
    for(auto& [a, c] : odd) projected.push_back({xors(dual, a), 1});
    auto trilinear_form = phase_trilinear_rows(lifts.size(), projected);
    PhaseCounter core;
    auto monomial = [&](const std::vector<size_t>& js) {
        // Canonical nonzero degree-d coefficients are 2^(d-1).
        for(size_t sub = 1; sub < (size_t(1) << js.size()); sub++) {
            Bits mask;
            for(size_t j = 0; j < js.size(); j++)
                if((sub >> j) & 1) mask.set(js[j]);
            core[mask] = mod8(core[mask] + ((mask.count() & 1) ? 1 : -1));
        }
    };
    for(auto& [ij, row] : trilinear_form) {
        auto [i, j] = ij;
        if(i == j) {
            if(row.bit(i)) monomial({i});
            for(auto k : row.above(i).indices()) monomial({i, k});
        } else
            for(auto k : row.above(j).indices()) monomial({i, j, k});
    }
    return phase_terms(core);
}
inline PolynomialQuotient phase_polynomial_template(size_t n, const PhaseTerms& odd) {
    auto trilinear_form = phase_trilinear_rows(n, odd);
    std::vector<Bits> rows;
    for(auto& [key, row] : trilinear_form) rows.push_back(row);
    PolynomialQuotient q;
    q.axes = phase_image_basis(rows);
    q.lifts = phase_right_inverse(q.axes, n);
    q.core = phase_section_core(n, odd, q.lifts);
    PhaseCounter residual;
    for(auto& [a, c] : odd) residual[a] = 1;
    // The section preserves the trilinear form; independent phase-value tests
    // verify this construction without rebuilding its tensor in production.
    for(auto& [a, c] : q.core) {
        Bits b = xors(q.axes, a);
        residual[b] = mod8(residual[b] - c);
    }
    std::vector<int> linear(n);
    std::map<std::pair<size_t, size_t>, int> pairs;
    for(auto& [a, c] : residual) {
        auto js = a.indices();
        for(auto j : js) linear.at(j) = mod8(linear.at(j) + c);
        for(size_t i = 0; i < js.size(); i++)
            for(size_t j = i + 1; j < js.size(); j++) pairs[{js[i], js[j]}] = mod8(pairs[{js[i], js[j]}] - 2 * c);
    }
    q.upper.resize(n);
    for(int c : linear) {
        assert(!(c & 1));
        q.linear.push_back(c / 2);
    }
    for(auto& [ij, c] : pairs) {
        assert(c % 4 == 0);
        if(c) q.upper[ij.first].set(ij.second);
    }
    return q;
}

// Only the normalized odd masks and dimension determine this template.
// Fault signs/even increments are applied to a returned copy, never cached.
// Thread-local ownership permits concurrent GIL-free native queries without a
// shared lock; retained payload is bounded separately in every worker thread.
struct PhasePolynomialCache {
    using Builder = PolynomialQuotient (*)(size_t, const PhaseTerms&);
    Builder build;
    explicit PhasePolynomialCache(Builder builder = phase_polynomial_template) : build(builder) {}
    struct Entry {
        size_t n;
        PhaseTerms odd;
        PolynomialQuotient quotient;
    };
    static constexpr size_t max_entries = 8;
    BoundedCache<Entry> entries;
    static size_t payload_bytes(const PhaseTerms& odd, const PolynomialQuotient& q) {
        size_t size = sizeof(Entry) + odd.capacity() * sizeof(PhaseTerms::value_type) +
                q.core.capacity() * sizeof(PhaseTerms::value_type) + q.linear.capacity() * sizeof(int);
        for(auto& [a, c] : odd) size += a.w.capacity() * sizeof(uint64_t);
        for(auto& [a, c] : q.core) size += a.w.capacity() * sizeof(uint64_t);
        for(auto* v : {&q.axes, &q.upper, &q.lifts}) {
            size += v->capacity() * sizeof(Bits);
            for(auto& a : *v) size += a.w.capacity() * sizeof(uint64_t);
        }
        return size;
    }
    PolynomialQuotient get(size_t n, const PhaseTerms& odd) {
        if(auto* entry = entries.find([&](const Entry& e) { return e.n == n && e.odd == odd; })) return entry->quotient;
        auto q = build(n, odd);
        if(payload_bytes(odd, q) <= default_cache_bytes) {
            Entry entry{n, odd, q};
            auto bytes = payload_bytes(entry.odd, entry.quotient);
            entries.insert(std::move(entry), max_entries, bytes, default_cache_bytes);
        }
        return q;
    }
};
inline PolynomialQuotient polynomial_quotient(size_t n, const PhaseTerms& input) {
    PhaseCounter combined;
    for(auto& [a, c] : input) combined[a] = mod8(combined[a] + mod8(c));
    PhaseTerms odd, even;
    for(auto& [a, c] : phase_terms(combined)) {
        if(a.size() > n) throw std::invalid_argument("phase mask outside dimension");
        if(c & 1) odd.push_back({a, 1});
        if(c - (c & 1)) even.push_back({a, c - (c & 1)});
    }
    static thread_local PhasePolynomialCache cache;
    auto q = cache.get(n, odd);
    phase_add_even(q.linear, q.upper, even);
    return q;
}
} // namespace magic_native
