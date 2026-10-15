// Original physical noise locations and importance proposals; no quantum evaluation.
#pragma once
#include "../compile/frontend.hpp"
#include "../foundation/bounded_cache.hpp"
#include "noise.hpp"
namespace magic_native {
inline FileNoisePlan prepare_file_noise(const CircuitData& circuit, const std::optional<Rational>& scale = {}) {
    const Rational zero, one(Integer(1));
    if(scale && !(*scale > zero)) throw std::invalid_argument("proposal_scale must be positive");
    FileNoisePlan plan;
    plan.has_actions = true;
    // One preparation call fixes the proposal. Cache only exact channel laws;
    // original anchors, guard metadata and ordered actions remain per location.
    struct Law {
        std::vector<Rational> input;
        bool guarded, chained, deterministic, tilted = false;
        std::array<FileFraction, 2> factors;
        std::shared_ptr<const FileNoiseCategory> category;
        Law(std::vector<Rational> input, bool guarded, bool chained, std::vector<FileFraction> probabilities,
                std::array<FileFraction, 2> factors, bool deterministic, bool tilted)
            : input(std::move(input)), guarded(guarded), chained(chained), deterministic(deterministic), tilted(tilted),
              factors(std::move(factors)),
              category(std::make_shared<const FileNoiseCategory>(std::move(probabilities))) {}
        size_t bytes() const {
            const auto& probabilities = category->probabilities;
            size_t size = sizeof(Law) + sizeof(FileNoiseCategory) + input.capacity() * sizeof(Rational) +
                    probabilities.capacity() * sizeof(FileFraction) + category->nonzero.capacity() * sizeof(size_t) +
                    category->conditional.capacity() * sizeof(RationalCoin);
            auto count = [&](const auto& v) { size += (v.n.d.capacity() + v.d.d.capacity()) * sizeof(uint32_t); };
            for(const auto& p : input) count(p);
            for(const auto& p : probabilities) count(p);
            for(const auto& p : factors) count(p);
            return size;
        }
    };
    constexpr size_t max_laws = 32;
    BoundedCache<Law> laws;
    size_t instruction = 0;
    auto add = [&](std::vector<Rational> probabilities, std::vector<FileNoiseAction> actions, bool guarded = false,
                       bool chained = false, std::optional<FileNoiseActions::Template> channel = {}, size_t first = 0,
                       size_t second = 0, std::string report = {}) {
        auto append = [&](const Law& law) {
            plan.guarded += guarded;
            plan.deterministic += law.deterministic;
            plan.tilted += law.tilted;
            if(guarded) plan.conditional_sites[instruction].push_back(plan.categories.size());
            plan.conditional.push_back(guarded);
            plan.categories.push_back(law.category);
            if(scale) plan.factors.push_back(law.factors);
            plan.offsets.push_back(plan.actions.size());
            if(channel)
                plan.actions.add(*channel,
                        *channel == FileNoiseActions::Template::Report ? "report"
                                                                       : circuit.instructions[instruction].op_id,
                        first, second, report);
            else
                for(auto& action : actions) plan.actions.push_back(std::move(action));
        };
        if(auto* law = laws.find([&](const Law& law) {
               return law.guarded == guarded && law.chained == chained && law.input == probabilities;
           })) {
            append(*law);
            return;
        }
        auto input = probabilities;
        // Compare conditional chains before expanding their exclusive masses.
        // The same input vector has a different law when it is already exclusive.
        if(chained) {
            Rational survival = one;
            for(auto& p : probabilities) {
                const auto conditional = p;
                p = survival * conditional;
                survival = survival * (one - conditional);
            }
        }
        Rational total;
        for(const auto& p : probabilities) total = total + p;
        Rational absent = one, present = one;
        bool tilted = false;
        if(scale && !guarded && total > zero && total < one) {
            const auto proposal = total * (*scale);
            if(!(proposal < one))
                throw std::invalid_argument("proposal_scale makes noise probability reach or exceed one at " +
                        circuit.instructions.at(instruction).op_id);
            absent = (one - total) / (one - proposal);
            present = one / (*scale);
            for(auto& p : probabilities) p = p * (*scale);
            tilted = *scale != one;
        }
        std::vector<FileFraction> raw;
        for(const auto& p : probabilities) raw.emplace_back(p.n, p.d);
        Law law(std::move(input), guarded, chained, std::move(raw),
                {FileFraction(absent.n, absent.d), FileFraction(present.n, present.d)}, total == zero || total == one,
                tilted);
        append(law);
        auto bytes = law.bytes();
        laws.insert(std::move(law), max_laws, bytes, default_cache_bytes);
    };
    auto action = [](const FileInstruction& inst, size_t start, const std::string& letters) {
        FileNoiseAction out;
        out.anchor = inst.op_id;
        for(size_t j = 0; j < letters.size(); j++)
            if(letters[j] != 'I') out.gates.push_back({letters[j], inst.targets.at(start + j).index});
        return out;
    };
    for(size_t index = 0; index < circuit.instructions.size(); index++) {
        instruction = index;
        const auto& inst = circuit.instructions[index];
        const auto& n = inst.name;
        if(!inst.args.empty())
            for(auto record : inst.records)
                add(inst.args, {}, false, false, FileNoiseActions::Template::Report, 0, 0,
                        "m" + std::to_string(record));
        if(n == "CORRELATED_ERROR") {
            std::vector<Rational> probabilities;
            std::vector<FileNoiseAction> actions;
            for(size_t j = index; j < circuit.instructions.size(); j++) {
                const auto& other = circuit.instructions[j];
                if(j != index && other.name != "ELSE_CORRELATED_ERROR") break;
                probabilities.push_back(other.args.at(0));
                std::string letters;
                for(const auto& t : other.targets) letters += t.pauli;
                actions.push_back(action(other, 0, letters));
            }
            add(std::move(probabilities), std::move(actions), inst.guard.has_value(), true);
            continue;
        }
        if(n != "X_ERROR" && n != "Y_ERROR" && n != "Z_ERROR" && n != "DEPOLARIZE1" && n != "DEPOLARIZE2" &&
                n != "PAULI_CHANNEL_1" && n != "PAULI_CHANNEL_2")
            continue;
        size_t width = n == "DEPOLARIZE2" || n == "PAULI_CHANNEL_2" ? 2 : 1;
        using Template = FileNoiseActions::Template;
        auto channel = n == "X_ERROR" ? Template::X
                : n == "Y_ERROR"      ? Template::Y
                : n == "Z_ERROR"      ? Template::Z
                : width == 1          ? Template::One
                                      : Template::Two;
        const size_t choices = channel == Template::Two ? 15 : channel == Template::One ? 3 : 1;
        auto probabilities = inst.args;
        if(n == "DEPOLARIZE1" || n == "DEPOLARIZE2")
            probabilities.assign(choices, inst.args.at(0) / Rational(Integer(choices)));
        for(size_t start = 0; start < inst.targets.size(); start += width)
            add(probabilities, {}, inst.guard.has_value(), false, channel, inst.targets[start].index,
                    width == 2 ? inst.targets[start + 1].index : 0);
    }
    plan.prepare_weights();
    return plan;
}
} // namespace magic_native
