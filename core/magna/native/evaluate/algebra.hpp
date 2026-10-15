// Public exact-value operations and witness conventions, independent of bindings.
#pragma once
#include "phase.hpp"
#include "../foundation/numeric.hpp"
#include <functional>
#include <optional>
#include <memory>
namespace magic_native {
struct LinearSolution {
    Bits particular;
    std::vector<Bits> kernel, rows;
};
inline std::optional<LinearSolution> solve_low_pivots(std::vector<Bits> rows, const std::vector<int>& rhs, size_t n) {
    if(rows.size() != rhs.size()) throw std::invalid_argument("shape");
    validate_masks(rows, n);
    for(size_t i = 0; i < rows.size(); i++)
        if(rhs[i] & 1) rows[i].set(n);
    size_t k = 0;
    std::vector<size_t> pivots;
    for(size_t p = 0; p < n; p++) {
        size_t q = k;
        while(q < rows.size() && !rows[q].bit(p)) q++;
        if(q == rows.size()) continue;
        std::swap(rows[k], rows[q]);
        for(size_t i = 0; i < rows.size(); i++)
            if(i != k && rows[i].bit(p)) rows[i] ^= rows[k];
        pivots.push_back(p);
        k++;
    }
    for(size_t i = k; i < rows.size(); i++)
        if(rows[i] == Bits::unit(n)) return {};
    LinearSolution out;
    for(size_t i = 0; i < k; i++)
        if(rows[i].bit(n)) out.particular.set(pivots[i]);
    for(size_t j = 0; j < n; j++)
        if(std::find(pivots.begin(), pivots.end(), j) == pivots.end()) {
            Bits v = Bits::unit(j);
            for(size_t i = 0; i < k; i++)
                if(rows[i].bit(j)) v.set(pivots[i]);
            out.kernel.push_back(std::move(v));
        }
    rows.resize(k);
    out.rows = std::move(rows);
    return out;
}
inline void validate_ket(const Ket& k) {
    auto n = k.columns.size();
    if(k.base.size() > k.width || k.linear.size() != n || k.upper.size() != n)
        throw std::invalid_argument("invalid affine ket shape");
    validate_masks(k.columns, k.width);
    validate_masks(k.upper, n);
    if(phase_image_basis(k.columns).size() != n) throw std::invalid_argument("independent support required");
    for(size_t j = 0; j < n; j++)
        if(k.columns[j].empty() || k.linear[j] < 0 || k.linear[j] > 3 || k.upper[j] != k.upper[j].above(j))
            throw std::invalid_argument("invalid support or quadratic convention");
}
inline Ket project_ket(const Ket& k, const std::vector<Bits>& rows, const std::vector<int>& outcomes) {
    validate_masks(rows, k.width);
    std::vector<Bits> forms;
    std::vector<int> rhs;
    auto dual = transpose(k.columns, k.width);
    if(rows.size() != outcomes.size()) throw std::invalid_argument("shape");
    for(size_t j = 0; j < rows.size(); j++) {
        forms.push_back(xors(dual, rows[j]));
        rhs.push_back(outcomes[j] ^ (rows[j] & k.base).parity());
    }
    auto solution = solve_low_pivots(forms, rhs, k.columns.size());
    if(!solution) {
        Ket out;
        out.zero = true;
        return out;
    }
    auto q = restrict_q(k.linear, k.upper, solution->particular, solution->kernel);
    Ket out = k;
    out.base ^= xors(k.columns, solution->particular);
    out.columns.clear();
    for(const auto& b : solution->kernel) out.columns.push_back(xors(k.columns, b));
    out.linear = std::move(q.linear);
    out.upper = std::move(q.upper);
    out.phase = mod8(k.phase + 2 * q.constant);
    return out;
}
inline Ket pauli_ket(Ket k, const Bits& x, const Bits& z) {
    if(x.size() > k.width || z.size() > k.width) throw std::invalid_argument("Pauli outside boundary");
    for(size_t j = 0; j < k.columns.size(); j++) k.linear[j] = mod4(k.linear[j] + 2 * (z & k.columns[j]).parity());
    k.phase = mod8(k.phase + 4 * (z & k.base).parity());
    k.base ^= x;
    return k;
}
inline std::vector<int> sample_ket_rows(const Ket& k, const std::vector<Bits>& rows, const std::function<int()>& bit) {
    auto point = k.base;
    for(const auto& col : k.columns)
        if(bit()) point ^= col;
    std::vector<int> out;
    for(const auto& row : rows) out.push_back((row & point).parity());
    return out;
}
// Operation-level RNG policy: preserve calls to the caller-owned one-bit stream.
inline bool exact_bernoulli(const Scalar& weight, const Scalar& total, const std::function<int()>& bit) {
    if(real_sign(total) <= 0 || real_sign(weight) < 0 || real_sign(total + weight * Scalar(-1)) < 0)
        throw std::invalid_argument("require 0 <= weight <= total and total > 0");
    Integer prefix;
    auto scaled = weight;
    for(;;) {
        Scalar lower;
        lower.c[0] = prefix;
        lower.normalize();
        Scalar upper;
        upper.c[0] = prefix + Integer(1);
        upper.normalize();
        if(real_sign(total * lower + scaled * Scalar(-1)) >= 0) return false;
        if(real_sign(total * upper + scaled * Scalar(-1)) <= 0) return true;
        prefix = prefix.shift(1) + Integer(bit());
        scaled = scaled * Scalar(2);
    }
}
} // namespace magic_native
