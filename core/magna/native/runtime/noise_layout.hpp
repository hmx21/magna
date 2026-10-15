// Immutable operation-interface noise-code layouts and record assignments.
#pragma once
#include "../compile/instrument.hpp"
#include "noise_model.hpp"
namespace magic_native {
struct NoiseCodeAction {
    std::vector<OriginalFault> faults;
    Bits flips;
};
inline bool same_noise_action(const NoiseCodeAction& a, const NoiseCodeAction& b) {
    if(a.flips != b.flips || a.faults.size() != b.faults.size()) return false;
    for(size_t j = 0; j < a.faults.size(); j++)
        if(std::tie(a.faults[j].after, a.faults[j].qubit, a.faults[j].pauli) !=
                std::tie(b.faults[j].after, b.faults[j].qubit, b.faults[j].pauli))
            return false;
    return true;
}
struct DecodedNoiseRow {
    std::vector<OriginalFault> faults;
    Bits flips;
    std::vector<Bits> true_words;
};
struct NoiseCodeLayout {
    size_t records = 0;
    std::vector<std::map<Bits, NoiseCodeAction, BitsLess>> entries;
    NoiseCodeLayout(const InstrumentLayout& layout, const std::vector<const NoiseSite*>& sites)
        : records(layout.record_names.size()) {
        for(const auto* site : sites) {
            if(!layout.locations.count(site->anchor)) throw std::invalid_argument("noise site outside original IR");
            std::map<Bits, NoiseCodeAction, BitsLess> codes;
            for(const auto& channel : site->channels) {
                Bits code;
                NoiseCodeAction action;
                if(!channel.report.empty()) {
                    auto found = layout.record_index.find(channel.report);
                    if(found == layout.record_index.end()) throw std::invalid_argument("unknown flipped record");
                    action.flips.set(found->second);
                    code = Bits(1);
                } else {
                    if(site->channels.size() == 1) code = Bits(1);
                    for(size_t j = 0; j < site->qubits.size(); j++) {
                        int pauli = 0;
                        for(const auto& [letter, wire] : channel.gates)
                            if(wire == site->qubits[j]) {
                                pauli = int(std::string("IXZY").find(letter));
                                break;
                            }
                        if(site->channels.size() != 1) {
                            if(pauli & 1) code.set(2 * j);
                            if(pauli & 2) code.set(2 * j + 1);
                        }
                        if(pauli) action.faults.push_back({site->anchor, int(site->qubits[j]), pauli});
                    }
                }
                auto previous = codes.find(code);
                if(previous != codes.end() && !same_noise_action(previous->second, action))
                    throw std::invalid_argument("ambiguous noise code");
                codes[code] = std::move(action);
            }
            entries.push_back(std::move(codes));
        }
    }
    void validate_words(const std::vector<Bits>& words) const {
        for(const auto& word : words)
            if(word.size() > records) throw std::invalid_argument("reported word outside owned record interface");
    }
    DecodedNoiseRow decode(
            const std::vector<std::pair<size_t, Bits>>& selected, const std::vector<Bits>& reported) const {
        validate_words(reported);
        DecodedNoiseRow result;
        for(const auto& [index, code] : selected) {
            if(index >= entries.size()) throw std::invalid_argument("invalid original noise index/code");
            auto found = entries[index].find(code);
            if(found == entries[index].end())
                throw std::invalid_argument("code is not a channel at this original site");
            const auto& action = found->second;
            result.faults.insert(result.faults.end(), action.faults.begin(), action.faults.end());
            result.flips ^= action.flips;
        }
        for(const auto& word : reported) result.true_words.push_back(word ^ result.flips);
        return result;
    }
};
inline void validate_compact_assignment(const InstrumentLayout& layout, const Bits& word, const Bits& report,
        const std::vector<OriginalFault>& faults) {
    if(word.size() > layout.record_names.size() || report.size() > layout.record_names.size())
        throw std::invalid_argument("record word outside owned record interface");
    for(const auto& fault : faults)
        if(!layout.locations.count(fault.after) || fault.pauli < 0 || fault.pauli > 3)
            throw std::invalid_argument("invalid original fault");
}
} // namespace magic_native
