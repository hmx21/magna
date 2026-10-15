// The exact route traverses original categorical assignments and global histories.
#pragma once
#include "file_execution.hpp"
#include "low_weight.hpp"
namespace magic_native {
struct HistoryBranchRequest {
    std::vector<Bits> words;
};
struct ExactHistoryResponse {
    std::vector<std::optional<Scalar>> values;
    size_t branches = 0;
};
inline ExactHistoryResponse enumerate_histories(
        CompiledExperiment& run, const ActualNoise& noise, ExecutionHooks hooks, size_t cap, size_t max_record_bits) {
    std::vector<std::vector<Bits>> pending{{}};
    ExactHistoryResponse result;
    size_t attempts = 0;
    while(!pending.empty()) {
        auto path = std::move(pending.back());
        pending.pop_back();
        size_t index = 0;
        hooks.branch = [&](const StabilizerTerms& terms, size_t width) {
            if(index < path.size()) return path[index++];
            if(width > max_record_bits)
                throw ResourceLimitError("decoder input has more unresolved record bits than enumeration allows",
                        "max_record_bits", double(max_record_bits), double(width));
            throw HistoryBranchRequest{enumerate_boundary(terms, width, cap)};
        };
        try {
            auto values = run.sample(hooks, noise).masses.exact_row();
            if(!result.branches) result.values = std::move(values);
            else {
                if(result.values.size() != values.size()) throw std::runtime_error("exact history row shape");
                for(size_t j = 0; j < values.size(); j++) {
                    if(bool(values[j]) != bool(result.values[j]))
                        throw std::runtime_error("exact history column mismatch");
                    if(values[j]) result.values[j] = *result.values[j] + *values[j];
                }
            }
            result.branches++;
        } catch(const HistoryBranchRequest& request) {
            if(result.branches + pending.size() + request.words.size() > cap)
                throw ResourceLimitError("decoder-history enumeration exceeds the branch budget", "max_branches",
                        double(cap), double(result.branches + pending.size() + request.words.size()));
            for(auto it = request.words.rbegin(); it != request.words.rend(); ++it) {
                auto next = path;
                next.push_back(*it);
                pending.push_back(std::move(next));
            }
        }
        attempts++;
        if(attempts > cap *
                        (1 + run.source.protocol.instructions.size() +
                                (run.source.recovery ? run.source.recovery->instructions.size() : 0)))
            throw ResourceLimitError("decoder-history traversal exceeds its step bound of max_branches per instruction",
                    "max_branches", double(cap));
    }
    return result;
}
struct EnumeratedPattern {
    NoisePattern pattern;
    ExactHistoryResponse response;
};
struct FileEnumeration {
    std::shared_ptr<CompiledExperiment> run;
    size_t cap = 0, max_record_bits = 0, next_index = 0, branches = 0;
    Integer denominator{1};
    std::vector<NoisePattern> patterns;
    std::vector<bool> missing;
    std::vector<WeightedExactRow> rows;
    std::map<size_t, std::vector<WeightedExactRow>> sectors;
    FileEnumeration(std::shared_ptr<CompiledExperiment> run, size_t max_weight, size_t max_patterns, size_t cap,
            size_t max_record_bits)
        : run(std::move(run)), cap(cap), max_record_bits(max_record_bits) {
        if(!cap || cap > 1000000 || !max_record_bits || max_record_bits > 1000000)
            throw std::invalid_argument("invalid record enumeration budget");
        std::vector<std::vector<Integer>> units;
        for(const auto& law : this->run->noise.categories) {
            const auto& category = law->probabilities;
            Integer d(1);
            for(const auto& p : category) d = integer_divmod(d, integer_gcd(d, p.d)).first * p.d;
            std::vector<Integer> values{d};
            for(const auto& p : category) {
                auto n = p.n * integer_divmod(d, p.d).first;
                values[0] = values[0] - n;
                values.push_back(std::move(n));
            }
            denominator = denominator * d;
            units.push_back(std::move(values));
        }
        patterns = enumerate_noise(units, max_weight, max_patterns);
        const auto& task = this->run->task;
        missing = {false, false, !this->run->source.target};
        missing.resize(3 + task.boundaries.observables.size(), false);
    }
    std::optional<EnumeratedPattern> next(const ExecutionHooks& hooks) {
        if(next_index == patterns.size()) return {};
        run->context.check(run->limits);
        const auto& pattern = patterns[next_index];
        FileNoiseDraw selected;
        for(const auto& [category, channel] : pattern.selected)
            selected.selected.push_back(int64_t(run->noise.offsets.at(category) + channel));
        auto response = enumerate_histories(*run, noise_actions(run->noise, selected), hooks, cap, max_record_bits);
        WeightedExactRow row;
        row.probability = pattern.numerator;
        for(size_t j = 0; j < response.values.size(); j++) {
            if(response.values[j].has_value() == missing.at(j))
                throw std::runtime_error("inconsistent exact row columns");
            row.values.push_back(response.values[j].value_or(Scalar()));
        }
        rows.push_back(row);
        sectors[pattern.weight].push_back(std::move(row));
        branches += response.branches;
        next_index++;
        return EnumeratedPattern{pattern, std::move(response)};
    }
    ExactAggregation aggregate() const {
        if(!rows.empty()) return aggregate_exact(rows, denominator, missing, run->target_normalization);
        std::vector<Scalar> zero(missing.size());
        zero[0] = Scalar(1);
        return aggregate_exact({{Integer(), std::move(zero)}}, denominator, missing, run->target_normalization);
    }
};
} // namespace magic_native
