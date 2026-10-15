// Operation-interface noise retains the caller's exact randrange stream.
#pragma once
#include "noise.hpp"
#include "../foundation/rational.hpp"
#include <set>
#include <functional>
namespace magic_native {
using RandomIndex = std::function<Integer(const Integer&)>;
struct NoiseSite {
    std::string anchor;
    std::vector<size_t> qubits;
    std::vector<FileNoiseAction> channels;
    Rational probability;
    std::optional<std::vector<Rational>> weights;
    void validate() const {
        const Rational zero, one(Integer(1));
        if(probability < zero || probability > one || channels.empty() ||
                std::set<size_t>(qubits.begin(), qubits.end()).size() != qubits.size())
            throw std::invalid_argument("invalid mutually exclusive noise site");
        if(weights) {
            Rational total;
            for(const auto& p : *weights) {
                if(p < zero) throw std::invalid_argument("channel weights must be nonnegative and sum to one");
                total = total + p;
            }
            if(weights->size() != channels.size() || total != one)
                throw std::invalid_argument("channel weights must be nonnegative and sum to one");
        }
    }
};
inline int64_t weighted_choice(const std::vector<Rational>& probabilities, const RandomIndex& randrange) {
    Rational remaining(Integer(1)), zero;
    for(size_t j = 0; j < probabilities.size(); j++) {
        const auto& p = probabilities[j];
        if(p < zero || p > remaining) throw std::invalid_argument("invalid categorical probability");
        if(remaining == zero) break;
        auto conditional = p / remaining;
        if(conditional != zero) {
            auto value = randrange(conditional.d);
            if((value - conditional.n).neg) return int64_t(j);
        }
        remaining = remaining - p;
    }
    return -1;
}
struct NoiseModelDraw {
    std::vector<std::pair<bool, FileNoiseAction>> actions;
    Rational weight{Integer(1)};
};
inline NoiseModelDraw draw_noise_model(const std::vector<const NoiseSite*>& sites, const RandomIndex& randrange,
        const std::optional<Rational>& physical = {}, const std::optional<Rational>& scale = {}) {
    NoiseModelDraw out;
    const Rational zero, one(Integer(1));
    if(physical && (*physical < zero || *physical > one)) throw std::invalid_argument("invalid physical rate");
    if(scale && !(*scale > zero)) throw std::invalid_argument("proposal_scale must be positive");
    for(const auto* site : sites) {
        const auto& p = physical ? *physical : site->probability;
        const auto q = scale && p > zero && p < one ? p * (*scale) : p;
        if(p < one && !(q < one))
            throw std::invalid_argument(
                    "proposal_scale makes noise probability reach or exceed one at " + site->anchor);
        const bool occurs = (randrange(q.d) - q.n).neg;
        if(occurs) {
            if(scale) out.weight = out.weight * (p / q);
            int64_t index;
            if(site->weights) index = weighted_choice(*site->weights, randrange);
            else {
                auto selected = randrange(Integer(site->channels.size()));
                index = selected.zero() ? 0 : int64_t(selected.d.at(0));
            }
            if(index < 0) index = int64_t(site->channels.size()) - 1;
            out.actions.push_back({site->qubits.empty(), site->channels.at(size_t(index))});
        } else if(scale) out.weight = out.weight * ((one - p) / (one - q));
    }
    return out;
}
} // namespace magic_native
