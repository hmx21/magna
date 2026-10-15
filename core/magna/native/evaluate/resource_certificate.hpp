// The supported phase certificate in a declared common coordinate domain.
// Matching is exact algebra, never inference from sampled assignments or ranks.
#pragma once
#include "phase.hpp"
#include <optional>
namespace magic_native {
struct CliffordResourceUpdate {
    std::vector<int> linear;
    std::vector<Bits> upper;
    int constant = 0;
};
struct ResourceCertificate {
    size_t dimension = 0;
    PhaseResource reference;
    PhaseTerms terms;
    std::vector<int> linear;
    std::vector<Bits> upper;
    // These are phase-domain coordinates. Actual output maps and contraction
    // scalars remain in the assigned PhaseTensor, including noninjective maps.
    explicit ResourceCertificate(const PhaseTensor& tensor, PhaseResource resource)
        : dimension(tensor.n), reference(freeze_resource(std::move(resource))), terms(tensor.terms),
          linear(tensor.linear), upper(tensor.upper) {}
    std::optional<PhaseResource> transport(
            const PhaseTensor& actual, CliffordResourceUpdate* update = nullptr, bool prepared_only = false) const {
        if(actual.linear.size() != actual.n || actual.upper.size() != actual.n)
            throw std::invalid_argument("phase quadratic dimensions");
        if(actual.n != dimension) {
            // Eliminating a Clifford boundary can change its path dimension.
            // Its prepared unit entry has no phase coordinates to transport.
            if(!reference.quotient.axes.empty()) return {};
            for(const auto& [a, c] : actual.terms)
                if(c & 1) return {};
            auto resource = reference;
            auto& q = resource.quotient;
            q.linear = actual.linear;
            q.upper = actual.upper;
            phase_add_even(q.linear, q.upper, actual.terms);
            resource.constant_mod8 = actual.constant;
            if(update) {
                update->linear = q.linear;
                update->upper = q.upper;
                update->constant = mod8(actual.constant - reference.constant_mod8);
            }
            return resource;
        }
        auto quotient = reference.quotient;
        PhaseTerms increments;
        bool even = true;
        if(actual.terms != terms) {
            PhaseCounter difference;
            for(const auto& [a, c] : actual.terms) difference[a] = mod8(difference[a] + c);
            for(const auto& [a, c] : terms) difference[a] = mod8(difference[a] - c);
            increments = phase_terms(difference);
            for(const auto& [a, c] : increments) even &= !(c & 1);
        }
        if(even) {
            // The old Pauli path already supplies only even phase increments
            // and actual Clifford/boundary data. Transport the prepared quotient.
            for(size_t j = 0; j < dimension; j++) {
                quotient.linear[j] = mod4(quotient.linear[j] + actual.linear[j] - linear[j]);
                quotient.upper[j] ^= actual.upper[j] ^ upper[j];
            }
            phase_add_even(quotient.linear, quotient.upper, increments);
        } else {
            if(prepared_only) return {};
            // An alternative parity representation can have the same signature.
            quotient = phase_quotient(actual);
            if(quotient.axes != reference.quotient.axes || quotient.core != reference.quotient.core ||
                    quotient.lifts != reference.quotient.lifts)
                return {};
        }
        if(update) {
            update->linear.resize(dimension);
            update->upper.resize(dimension);
            for(size_t j = 0; j < dimension; j++) {
                update->linear[j] = mod4(quotient.linear[j] - reference.quotient.linear[j]);
                update->upper[j] = quotient.upper[j] ^ reference.quotient.upper[j];
            }
            update->constant = mod8(actual.constant - reference.constant_mod8);
        }
        auto resource = reference;
        resource.quotient = std::move(quotient);
        resource.constant_mod8 = actual.constant;
        return resource;
    }
    size_t term_bound() const { return resource_entries(reference).size(); }
};
} // namespace magic_native
