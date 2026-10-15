// Existing real-record proposal distributions with caller-owned exact RNG streams.
#pragma once
#include "../compile/instrument.hpp"
#include "noise_model.hpp"
#include "marginal.hpp"
namespace magic_native {
struct RecordProposalDraw {
    Bits true_word;
    Rational probability{Integer(1)};
};
inline RecordProposalDraw independent_proposal(const std::vector<std::string>& records,
        const std::map<std::string, Rational>& probabilities, const RandomIndex& random) {
    const Rational zero, one(Integer(1));
    for(const auto& [name, p] : probabilities) {
        if(!(zero < p && p < one))
            throw std::invalid_argument("full-support probabilities strictly between zero and one required");
        if(std::find(records.begin(), records.end(), name) == records.end())
            throw std::invalid_argument("unknown proposal record");
    }
    RecordProposalDraw out;
    for(size_t j = 0; j < records.size(); j++) {
        auto found = probabilities.find(records[j]);
        auto p = found == probabilities.end() ? Rational(Integer(1), Integer(2)) : found->second;
        bool bit = (random(p.d) - p.n).neg;
        if(bit) out.true_word.set(j);
        out.probability = out.probability * (bit ? p : one - p);
    }
    return out;
}
struct AffineProposal {
    std::vector<std::string> records;
    LinearSolution solution;
    AffineProposal(std::vector<std::string> records, const std::vector<Bits>& rows, const std::vector<int>& rhs)
        : records(std::move(records)) {
        if(std::set<std::string>(this->records.begin(), this->records.end()).size() != this->records.size())
            throw std::invalid_argument("distinct true records and a physical coverage certificate are required");
        for(int bit : rhs)
            if(bit != 0 && bit != 1) throw std::invalid_argument("invalid affine record constraints");
        auto value = solve_low_pivots(rows, rhs, this->records.size());
        if(!value) throw std::invalid_argument("empty proposal support");
        solution = std::move(*value);
    }
    RecordProposalDraw draw(const std::vector<std::string>& current, const RandomBits& random) const {
        if(current != records) throw std::invalid_argument("proposal record ordering mismatch");
        return {solution.particular ^ xors(solution.kernel, random(solution.kernel.size())),
                Rational(Integer(1), Integer(1).shift(solution.kernel.size()))};
    }
};
} // namespace magic_native
