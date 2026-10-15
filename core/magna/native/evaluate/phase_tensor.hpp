// Phase tensors and actual-branch boundary absorption. No Python calls.
#pragma once
#include "phase_quotient.hpp"
#include "../compile/plan.hpp"
#include <functional>
#include <string>

namespace magic_native {
struct PhaseCertificate {
    std::string stage;
    size_t input_variables = 0;
    PolynomialQuotient quotient;
    Ket contracted;
    std::vector<Bits> rows, inverse_columns;
};
struct PhaseTensor {
    size_t n = 0;
    std::vector<int> linear;
    std::vector<Bits> upper;
    PhaseTerms terms;
    int constant = 0;
    int64_t power = 0;
    PhaseTerms boundary;
    size_t output_count = 0;
    std::vector<std::string> environment;
    bool zero = false;
    std::vector<PhaseCertificate> certificates;
};
inline PolynomialQuotient phase_quotient(const PhaseTensor& t) {
    if(t.linear.size() != t.n || t.upper.size() != t.n) throw std::invalid_argument("phase quadratic dimensions");
    auto q = polynomial_quotient(t.n, t.terms);
    for(size_t j = 0; j < t.n; j++) {
        q.linear[j] = mod4(q.linear[j] + t.linear[j]);
        q.upper[j] ^= t.upper[j];
    }
    return q;
}
inline PhaseTensor canonicalize_phase(PhaseTensor t, bool certificates = true) {
    if(t.zero) return t;
    std::vector<Bits> rows, identity;
    for(auto& [a, b] : t.boundary) rows.push_back(a);
    for(size_t j = 0; j < t.n; j++) identity.push_back(Bits::unit(j));
    rows.insert(rows.end(), identity.begin(), identity.end());
    rows = phase_image_basis(rows);
    if(rows == identity) return t;
    auto lifts = phase_right_inverse(rows, t.n), dual = transpose(lifts, t.n);
    auto q = restrict_q(t.linear, t.upper, Bits(), lifts);
    PhaseCounter terms;
    for(auto& [a, c] : t.terms) terms[xors(dual, a)] = mod8(terms[xors(dual, a)] + c);
    t.terms = phase_terms(terms);
    for(auto& [a, b] : t.boundary) a = xors(dual, a);
    t.linear = q.linear;
    t.upper = q.upper;
    t.constant = mod8(t.constant + 2 * q.constant);
    if(certificates) {
        PhaseCertificate c;
        c.stage = "boundary_coordinate_change";
        c.rows = rows;
        c.inverse_columns = lifts;
        t.certificates.push_back(std::move(c));
    }
    return t;
}
using PhaseReductionAlgebra = std::function<PolynomialQuotient(const PhaseTensor&, size_t)>;
inline PhaseTensor absorb_phase(
        PhaseTensor t, Workspace& ws, bool certificates = true, const PhaseReductionAlgebra& algebra = {}) {
    size_t stage = 0;
    while(!t.zero) {
        auto q = algebra ? algebra(t, stage++) : phase_quotient(t);
        std::vector<Bits> rows;
        Bits base;
        for(size_t j = 0; j < t.boundary.size(); j++) {
            rows.push_back(t.boundary[j].first);
            if(t.boundary[j].second) base.set(j);
        }
        rows.insert(rows.end(), q.axes.begin(), q.axes.end());
        auto columns = transpose(rows, t.n), change = column_change(columns);
        bool injective = true;
        for(auto& v : change)
            if(xors(columns, v).empty()) {
                injective = false;
                break;
            }
        PhaseCertificate c;
        if(certificates) {
            c.input_variables = t.n;
            c.quotient = q;
        }
        if(injective) {
            if(certificates) {
                c.stage = "injective_resource_boundary";
                t.certificates.push_back(std::move(c));
            }
            return canonicalize_phase(std::move(t), certificates);
        }
        Ket k;
        k.width = rows.size();
        k.base = base;
        k.columns = columns;
        k.linear = q.linear;
        k.upper = q.upper;
        k.power = t.power;
        k.phase = t.constant;
        k = contract(std::move(k), ws);
        if(certificates) {
            c.stage = "global_radical_contraction";
            c.contracted = k;
            t.certificates.push_back(std::move(c));
        }
        if(k.zero) {
            t.zero = true;
            t.n = 0;
            t.terms.clear();
            t.linear.clear();
            t.upper.clear();
            t.constant = 0;
            t.power = 0;
            for(auto& b : t.boundary) b = {Bits(), 0};
            return t;
        }
        size_t old_n = t.n, nb = t.boundary.size();
        t.n = k.columns.size();
        if(t.n > old_n) throw std::runtime_error("contraction increased path dimension");
        auto forms = transpose(k.columns, k.width);
        std::vector<Bits> magic(forms.begin() + nb, forms.end());
        Bits offset = k.base >> nb;
        PhaseCounter terms;
        t.constant = k.phase;
        for(auto& [a, c0] : q.core) {
            Bits mask = xors(magic, a);
            int bit = (a & offset).parity();
            terms[mask] = mod8(terms[mask] + c0 * (bit ? -1 : 1));
            t.constant = mod8(t.constant + c0 * bit);
        }
        t.terms = phase_terms(terms);
        t.linear = k.linear;
        t.upper = k.upper;
        t.power = k.power;
        for(size_t j = 0; j < nb; j++) t.boundary[j] = {forms[j], int(k.base.bit(j))};
        if(t.n == old_n) return canonicalize_phase(std::move(t), certificates);
    }
    return t;
}
} // namespace magic_native
