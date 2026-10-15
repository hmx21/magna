// Affine quadratic kets, GF(2) factorizations and exact stabilizer contraction.
// Python-independent and single-threaded; workspaces are invocation-local.
#pragma once
#include "bits.hpp"
#include "scalar.hpp"
#include <algorithm>
#include <cstdint>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace magic_native {
inline int qvalue(const Bits& x, const std::vector<int>& l, const std::vector<Bits>& b) {
    int q = 0;
    x.each([&](size_t j) {
        if(j >= l.size() || j >= b.size()) throw std::invalid_argument("quadratic coordinate outside dimension");
        q = mod4(q + l[j] + 2 * b[j].parity_and(x));
    });
    return q;
}
struct Quadratic {
    int constant;
    std::vector<int> linear;
    std::vector<Bits> upper;
};
inline Quadratic restrict_q(const std::vector<int>& l, const std::vector<Bits>& b, const Bits& shift,
        const std::vector<Bits>& basis, bool track_constant = true) {
    auto polar = b;
    for(size_t j = 0; j < l.size(); j++) {
        if(l[j] & 1) polar[j].set(j);
        b[j].each([&](size_t k) { polar[k].set(j); });
    }
    Quadratic q;
    q.constant = track_constant ? qvalue(shift, l, b) : 0;
    auto dual = transpose(basis, l.size());
    q.linear.reserve(basis.size());
    q.upper.reserve(basis.size());
    for(size_t j = 0; j < basis.size(); j++) {
        auto image = xors(polar, basis[j]);
        q.linear.push_back(mod4(qvalue(basis[j], l, b) + 2 * image.parity_and(shift)));
        auto row = xors(dual, image);
        for(size_t k = 0; k < std::min(j / 64, row.w.size()); k++) row.w[k] = 0;
        if(j / 64 < row.w.size()) row.w[j / 64] &= j % 64 == 63 ? 0 : ~uint64_t(0) << (j % 64 + 1);
        row.trim();
        q.upper.push_back(std::move(row));
    }
    return q;
}
struct Factor {
    std::map<size_t, std::pair<Bits, Bits>> pivots;
    std::vector<Bits> zeros, kernel;
    Factor() = default;
    Factor(const std::vector<Bits>& rows, size_t n) {
        // The public factor keeps ordered pivots and full row provenance.
        // Construction resolves each highest pivot through a temporary index;
        // map nodes are stable, including while new pivots are inserted.
        std::vector<const std::pair<Bits, Bits>*> index(n, nullptr);
        for(size_t i = 0; i < rows.size(); i++) {
            Bits r = rows[i], t = Bits::unit(i);
            while(!r.empty()) {
                auto j = r.size() - 1;
                if(j >= index.size() || !index[j]) break;
                r ^= index[j]->first;
                t ^= index[j]->second;
            }
            if(r.empty()) zeros.push_back(std::move(t));
            else {
                auto j = r.size() - 1;
                if(j >= index.size()) index.resize(j + 1, nullptr);
                auto inserted = pivots.emplace(j, std::make_pair(std::move(r), std::move(t)));
                index[j] = &inserted.first->second;
            }
        }
        for(size_t j = 0; j < n; j++)
            if(!index[j]) kernel.push_back(solve(Bits(), Bits::unit(j)));
    }
    bool consistent(const Bits& rhs) const {
        for(auto& t : zeros)
            if(t.parity_and(rhs)) return false;
        return true;
    }
    Bits solve(const Bits& rhs, Bits v = Bits()) const {
        for(auto& [j, rt] : pivots)
            if(rt.first.parity_and(v) ^ rt.second.parity_and(rhs)) v.flip(j);
        return v;
    }
};
struct Ket {
    size_t width = 0;
    Bits base;
    std::vector<Bits> columns;
    std::vector<int> linear;
    std::vector<Bits> upper;
    int64_t power = 0;
    int phase = 0;
    bool zero = false;
};
struct Workspace {
    std::vector<Bits> columns, adj;
    std::vector<int> linear;
};
inline size_t pivot(const Bits& candidates, const std::vector<size_t>& degrees) {
    size_t p = 0, degree = std::numeric_limits<size_t>::max();
    // Degrees count only active neighbors. Descending traversal keeps the
    // original highest-index tie rule and makes a zero-degree pivot final.
    for(size_t word = candidates.w.size(); word-- > 0;) {
        auto bits = candidates.w[word];
        while(bits) {
            size_t bit = 63 - __builtin_clzll(bits), j = 64 * word + bit;
            bits ^= uint64_t(1) << bit;
            size_t d = degrees[j];
            if(d < degree) {
                degree = d;
                p = j;
                if(!d) return p;
            }
        }
    }
    return p;
}
template <class F> inline void each_union(const Bits& a, const Bits& b, F&& f) {
    for(size_t k = 0; k < std::max(a.w.size(), b.w.size()); k++) {
        auto bits = (k < a.w.size() ? a.w[k] : 0) | (k < b.w.size() ? b.w[k] : 0);
        while(bits) {
            f(64 * k + __builtin_ctzll(bits));
            bits &= bits - 1;
        }
    }
}
// Add a symmetric elimination update and maintain its active degree without
// constructing a temporary row or a unit mask. Source rows contain only active
// coordinates and are snapshots distinct from the destination.
inline void clifford_row_update(Bits& row, const Bits& a, const Bits& b, size_t diagonal, size_t& degree) {
    size_t count = std::max(a.w.size(), b.w.size());
    if(row.w.size() < count) row.w.resize(count);
    for(size_t k = 0; k < count; k++) {
        auto v = (k < a.w.size() ? a.w[k] : 0) ^ (k < b.w.size() ? b.w[k] : 0);
        if(k == diagonal / 64) v &= ~(uint64_t(1) << (diagonal % 64));
        degree += __builtin_popcountll(v);
        degree -= 2 * __builtin_popcountll(v & row.w[k]);
        row.w[k] ^= v;
    }
    row.trim();
}
inline std::vector<Bits> compact_clifford(const std::vector<Bits>& adj, const std::vector<size_t>& keep) {
    const size_t absent = keep.size();
    std::vector<size_t> index(adj.size(), absent);
    for(size_t j = 0; j < keep.size(); j++) index[keep[j]] = j;
    std::vector<Bits> upper(keep.size());
    for(size_t j = 0; j < keep.size(); j++) {
        auto& row = upper[j];
        row.w.resize((keep.size() + 63) / 64);
        adj[keep[j]].each([&](size_t k) {
            auto p = index[k];
            if(p > j && p != absent) row.w[p / 64] |= uint64_t(1) << (p % 64);
        });
        row.trim();
    }
    return upper;
}
inline Ket contract_reduced(Ket k, Workspace& ws, bool track_global_phase = true) {
    const size_t n = k.columns.size();
    ws.columns = k.columns;
    ws.linear = k.linear;
    ws.adj = k.upper;
    auto& cols = ws.columns;
    auto& l = ws.linear;
    auto& adj = ws.adj;
    Bits hidden, active = Bits::ones(n);
    for(size_t j = 0; j < n; j++) {
        if(cols[j].empty()) hidden.set(j);
        k.upper[j].each([&](size_t h) { adj[h].set(j); });
    }
    std::vector<size_t> degrees(n);
    for(size_t j = 0; j < n; j++) degrees[j] = adj[j].count();
    while(!hidden.empty()) {
        auto h = pivot(hidden, degrees);
        int lh = l[h];
        active.flip(h);
        hidden.flip(h);
        Bits v = adj[h] & active;
        v.each([&](size_t j) { degrees[j]--; });
        if(lh & 1) {
            int sign = lh == 1 ? -1 : 1;
            v.each([&](size_t j) {
                l[j] = mod4(l[j] + sign);
                clifford_row_update(adj[j], v, {}, j, degrees[j]);
            });
            k.power++;
            if(track_global_phase) k.phase += lh == 1 ? 1 : -1;
            continue;
        }
        k.power += 2;
        int rhs = lh / 2;
        if(v.empty()) {
            if(rhs) {
                k.zero = true;
                return k;
            }
            continue;
        }
        Bits candidates = v & hidden;
        if(candidates.empty()) candidates = v;
        auto p = pivot(candidates, degrees);
        active.flip(p);
        if(hidden.bit(p)) hidden.flip(p);
        Bits support = v & active, neighbors = adj[p] & active;
        neighbors.each([&](size_t j) { degrees[j]--; });
        const auto& cp = cols[p];
        int lp = l[p];
        if(track_global_phase) k.phase += 2 * rhs * lp;
        if(rhs) k.base ^= cp;
        const Bits empty;
        each_union(support, neighbors, [&](size_t j) {
            bool sj = support.bit(j), nj = neighbors.bit(j);
            l[j] = mod4(l[j] + lp * sj + 2 * sj * nj + 2 * rhs * (nj ^ ((lp & 1) * sj)));
            clifford_row_update(
                    adj[j], (nj ^ (sj && (lp & 1))) ? support : empty, sj ? neighbors : empty, j, degrees[j]);
            if(sj) cols[j] ^= cp;
        });
    }
    auto keep = active.indices();
    auto upper = compact_clifford(adj, keep);
    k.columns.clear();
    k.linear.clear();
    k.upper.clear();
    for(auto j : keep) {
        k.columns.push_back(cols[j]);
        k.linear.push_back(l[j]);
    }
    k.upper = std::move(upper);
    k.phase = track_global_phase ? mod8(k.phase) : 0;
    return k;
}
// Compile the structural part of contract_reduced once. Even linear increments
// change only its scalar/affine right-hand sides, never its elimination pivots.
struct CliffordContraction {
    struct Step {
        size_t h, p;
        bool odd;
        Bits support, neighbors, column;
    };
    std::vector<int> parity;
    std::vector<Bits> input_upper, columns, upper;
    std::vector<size_t> keep;
    std::vector<Step> steps;
    CliffordContraction() = default;
    CliffordContraction(std::vector<Bits> cols, const Quadratic& q) : input_upper(q.upper) {
        const size_t n = cols.size();
        auto adj = q.upper;
        auto l = q.linear;
        parity.reserve(n);
        steps.reserve(n);
        for(auto v : l) parity.push_back(v & 1);
        Bits hidden, active = Bits::ones(n);
        for(size_t j = 0; j < n; j++) {
            if(cols[j].empty()) hidden.set(j);
            q.upper[j].each([&](size_t h) { adj[h].set(j); });
        }
        std::vector<size_t> degrees(n);
        for(size_t j = 0; j < n; j++) degrees[j] = adj[j].count();
        while(!hidden.empty()) {
            auto h = pivot(hidden, degrees);
            active.flip(h);
            hidden.flip(h);
            Bits v = adj[h] & active;
            v.each([&](size_t j) { degrees[j]--; });
            if(l[h] & 1) {
                steps.push_back({h, n, true, {}, v, {}});
                v.each([&](size_t j) {
                    l[j] ^= 1;
                    clifford_row_update(adj[j], v, {}, j, degrees[j]);
                });
                continue;
            }
            if(v.empty()) {
                steps.push_back({h, n, false, {}, {}, {}});
                continue;
            }
            Bits candidates = v & hidden;
            if(candidates.empty()) candidates = v;
            auto p = pivot(candidates, degrees);
            active.flip(p);
            if(hidden.bit(p)) hidden.flip(p);
            Bits support = v & active, neighbors = adj[p] & active;
            neighbors.each([&](size_t j) { degrees[j]--; });
            const auto& cp = cols[p];
            steps.push_back({h, p, false, support, neighbors, cp});
            const Bits empty;
            each_union(support, neighbors, [&](size_t j) {
                bool sj = support.bit(j), nj = neighbors.bit(j);
                l[j] ^= (l[p] & 1) * sj;
                clifford_row_update(
                        adj[j], (nj ^ (sj && (l[p] & 1))) ? support : empty, sj ? neighbors : empty, j, degrees[j]);
                if(sj) cols[j] ^= cp;
            });
        }
        keep = active.indices();
        upper = compact_clifford(adj, keep);
        columns.reserve(keep.size());
        for(auto j : keep) columns.push_back(std::move(cols[j]));
    }
    bool matches(const Quadratic& q) const {
        if(q.linear.size() != parity.size() || q.upper != input_upper) return false;
        for(size_t j = 0; j < parity.size(); j++)
            if((q.linear[j] & 1) != parity[j]) return false;
        return true;
    }
    // The same prepared elimination supplies support without evaluating a
    // scalar probability or rebuilding its homogeneous geometry.
    Ket support(size_t width, Bits base, std::vector<int> l) const {
        Ket k;
        k.width = width;
        k.base = std::move(base);
        for(const auto& s : steps) {
            int lh = l[s.h];
            if(s.odd) {
                for(auto j : s.neighbors.indices()) l[j] = mod4(l[j] + (lh == 1 ? -1 : 1));
                continue;
            }
            int rhs = lh / 2;
            if(s.p == l.size()) {
                if(rhs) {
                    k.zero = true;
                    return k;
                }
                continue;
            }
            int lp = l[s.p];
            if(rhs) k.base ^= s.column;
            for(auto j : (s.support | s.neighbors).indices()) {
                bool sj = s.support.bit(j), nj = s.neighbors.bit(j);
                l[j] = mod4(l[j] + lp * sj + 2 * sj * nj + 2 * rhs * (nj ^ ((lp & 1) * sj)));
            }
        }
        k.columns = columns;
        return k;
    }
    Ket evaluate(size_t width, Bits base, std::vector<int> l, int64_t power, int phase) const {
        Ket k;
        k.width = width;
        k.base = std::move(base);
        k.power = power;
        k.phase = phase;
        for(const auto& s : steps) {
            int lh = l[s.h];
            if(s.odd) {
                int sign = lh == 1 ? -1 : 1;
                for(auto j : s.neighbors.indices()) l[j] = mod4(l[j] + sign);
                k.power++;
                k.phase += lh == 1 ? 1 : -1;
                continue;
            }
            k.power += 2;
            int rhs = lh / 2;
            if(s.p == l.size()) {
                if(rhs) {
                    k.zero = true;
                    return k;
                }
                continue;
            }
            int lp = l[s.p];
            k.phase += 2 * rhs * lp;
            if(rhs) k.base ^= s.column;
            for(auto j : (s.support | s.neighbors).indices()) {
                bool sj = s.support.bit(j), nj = s.neighbors.bit(j);
                l[j] = mod4(l[j] + lp * sj + 2 * sj * nj + 2 * rhs * (nj ^ ((lp & 1) * sj)));
            }
        }
        k.columns = columns;
        k.upper = upper;
        k.linear.reserve(keep.size());
        for(auto j : keep) k.linear.push_back(l[j]);
        k.phase = mod8(k.phase);
        return k;
    }
};
inline std::vector<Bits> column_change(const std::vector<Bits>& cols) {
    std::map<size_t, std::pair<Bits, Bits>> piv;
    std::vector<Bits> visible, hidden;
    for(size_t j = 0; j < cols.size(); j++) {
        Bits c = cols[j], b = Bits::unit(j);
        while(!c.empty()) {
            auto p = c.size() - 1;
            auto it = piv.find(p);
            if(it == piv.end()) {
                piv[p] = {c, b};
                visible.push_back(Bits::unit(j));
                break;
            }
            c ^= it->second.first;
            b ^= it->second.second;
        }
        if(c.empty()) hidden.push_back(b);
    }
    visible.insert(visible.end(), hidden.begin(), hidden.end());
    return visible;
}
inline Ket contract(Ket k, Workspace& ws, bool track_global_phase = true) {
    auto change = column_change(k.columns);
    auto q = restrict_q(k.linear, k.upper, Bits(), change, track_global_phase);
    // This by-value ket is private working storage. Keep its old columns alive
    // only until the basis change has read them; do not snapshot each bitset.
    {
        auto old = std::move(k.columns);
        k.columns.clear();
        k.columns.reserve(change.size());
        for(size_t j = 0; j < change.size(); j++) k.columns.push_back(xors(old, change[j]));
    }
    k.linear = std::move(q.linear);
    k.upper = std::move(q.upper);
    if(track_global_phase) k.phase += 2 * q.constant;
    return contract_reduced(std::move(k), ws, track_global_phase);
}
inline Scalar norm(const Ket& k) { return k.zero ? Scalar() : Scalar(1, k.power + int64_t(k.columns.size())); }
inline Scalar overlap(const Ket& a, const Ket& b, Workspace& ws) {
    if(a.zero || b.zero) return Scalar();
    if(a.width != b.width) throw std::invalid_argument("overlap width mismatch");
    std::vector<Bits> cols = a.columns;
    cols.insert(cols.end(), b.columns.begin(), b.columns.end());
    Factor f(transpose(cols, a.width), cols.size());
    Bits rhs = a.base ^ b.base;
    if(!f.consistent(rhs)) return Scalar();
    Bits shift = f.solve(rhs);
    auto linear = a.linear;
    for(auto& x : linear) x = mod4(-x);
    linear.insert(linear.end(), b.linear.begin(), b.linear.end());
    auto upper = a.upper;
    for(auto& x : b.upper) upper.push_back(x << a.columns.size());
    auto q = restrict_q(linear, upper, shift, f.kernel);
    Ket k;
    k.linear = q.linear;
    k.upper = q.upper;
    k.columns.resize(f.kernel.size());
    k.power = a.power + b.power;
    k.phase = b.phase - a.phase + 2 * q.constant;
    k = contract_reduced(std::move(k), ws);
    return k.zero ? Scalar() : omega(k.phase, k.power);
}
} // namespace magic_native
