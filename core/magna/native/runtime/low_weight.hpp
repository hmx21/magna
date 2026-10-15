// Exact bounded enumeration independent of Python.
#pragma once
#include "sampling.hpp"
namespace magic_native {
struct NoisePattern {
    std::vector<std::pair<size_t, size_t>> selected;
    Integer numerator;
    size_t weight = 0;
};
inline std::vector<NoisePattern> enumerate_noise(
        const std::vector<std::vector<Integer>>& units, size_t requested, uint64_t cap) {
    if(!cap || cap > 1000000) throw std::invalid_argument("max_patterns must be 1..1000000");
    size_t n = units.size(), k = std::min(requested, n);
    if(n > 100000 || (k && n > 10000000 / k))
        throw std::length_error(
                "low-weight enumeration supports at most 100000 noise sites and 10^7 site-weight pairs");
    for(const auto& group : units) {
        if(group.empty()) throw std::invalid_argument("empty category");
        for(const auto& value : group)
            if(value.neg) throw std::invalid_argument("negative category mass");
    }
    // Saturated dynamic count, before allocating any assignment or quantum query.
    std::vector<uint64_t> counts(k + 1, 0);
    counts[0] = 1;
    auto satadd = [&](uint64_t a, uint64_t b) { return a > cap - b ? cap + 1 : a + b; };
    for(auto& u : units) {
        uint64_t alternatives = 0;
        for(size_t j = 1; j < u.size(); j++) alternatives += !u[j].zero();
        auto old = counts;
        for(size_t w = 0; w <= k; w++) {
            auto extra = w ? old[w - 1] : 0;
            extra = alternatives && extra > cap / alternatives ? cap + 1 : extra * alternatives;
            auto base = u[0].zero() ? 0 : old[w];
            counts[w] = (base > cap || extra > cap) ? cap + 1 : satadd(base, extra);
        }
    }
    uint64_t count = 0;
    for(auto v : counts) {
        if(v > cap || count > cap - v)
            throw ResourceLimitError(
                    "low-weight enumeration has more noise patterns than allowed; nothing was enumerated",
                    "max_patterns", double(cap), double(cap) + 1);
        count += v;
    }
    std::vector<NoisePattern> output;
    if(!count) return output;
    std::vector<size_t> mandatory(n + 1, 0);
    for(size_t i = n; i > 0; i--) mandatory[i - 1] = mandatory[i] + units[i - 1][0].zero();
    // Iterative traversal keeps stack depth independent of the number of sites.
    std::vector<size_t> next(n + 1, 0), weight(n + 1, 0), choice(n, 0);
    std::vector<Integer> probability(n + 1);
    probability[0] = Integer(1);
    size_t depth = 0;
    while(true) {
        if(depth == n) {
            NoisePattern pattern;
            pattern.numerator = probability[depth];
            pattern.weight = weight[depth];
            for(size_t i = 0; i < n; i++)
                if(choice[i]) pattern.selected.push_back({i, choice[i] - 1});
            output.push_back(std::move(pattern));
            if(!depth) break;
            depth--;
            continue;
        }
        size_t c = next[depth]++;
        if(c == units[depth].size()) {
            next[depth] = 0;
            if(!depth) break;
            depth--;
            continue;
        }
        if(units[depth][c].zero() || weight[depth] + (c != 0) + mandatory[depth + 1] > k) continue;
        choice[depth] = c;
        weight[depth + 1] = weight[depth] + (c != 0);
        probability[depth + 1] = probability[depth] * units[depth][c];
        depth++;
    }
    return output;
}
inline std::vector<Bits> enumerate_boundary(
        const std::vector<std::pair<Scalar, Ket>>& terms, size_t width, size_t cap) {
    if(!cap || cap > 1000000 || width > 1000000) throw std::invalid_argument("invalid record enumeration budget");
    struct Node {
        Bits word;
        size_t depth;
        Scalar mass;
    };
    Workspace ws;
    auto total = boundary_sum(terms, {}, {}, ws);
    if(real_sign(total) <= 0) throw std::invalid_argument("nonpositive record marginal");
    auto branch_budget = [&](size_t needed) {
        return ResourceLimitError(
                "decoder-history enumeration exceeds the branch budget", "max_branches", double(cap), double(needed));
    };
    std::vector<Node> stack{{{}, 0, total}};
    std::vector<Bits> words;
    while(!stack.empty()) {
        auto node = std::move(stack.back());
        stack.pop_back();
        if(node.depth == width) {
            words.push_back(std::move(node.word));
            if(words.size() > cap) throw branch_budget(words.size());
            continue;
        }
        std::vector<Bits> rows;
        std::vector<int> bits;
        for(size_t j = 0; j <= node.depth; j++) {
            rows.push_back(Bits::unit(j));
            bits.push_back(j == node.depth ? 0 : node.word.bit(j));
        }
        auto zero = boundary_sum(terms, rows, bits, ws), one = node.mass + zero * Scalar(-1);
        if(real_sign(zero) < 0 || real_sign(one) < 0) throw std::runtime_error("negative record marginal");
        if(!one.zero()) {
            auto w = node.word;
            w.set(node.depth);
            stack.push_back({std::move(w), node.depth + 1, one});
        }
        if(!zero.zero()) stack.push_back({node.word, node.depth + 1, zero});
        if(words.size() + stack.size() > cap) throw branch_budget(words.size() + stack.size());
    }
    return words;
}

struct WeightedExactRow {
    Integer probability;
    std::vector<Scalar> values;
};
struct ExactAggregation {
    std::vector<Scalar> sums;
    Scalar error_normalization{1};
    Integer covered;
    using Scientific = std::pair<double, int64_t>;
    std::map<std::string, std::pair<Scientific, Scientific>> bounds;
    bool has_conditional = false, conditional_bound_missing = false;
    Scientific conditional;
};
inline ExactAggregation aggregate_exact(const std::vector<WeightedExactRow>& rows, const Integer& denominator,
        const std::vector<bool>& missing, const Scalar& error_normalization = Scalar(1)) {
    if(denominator.neg || denominator.zero()) throw std::invalid_argument("invalid weight denominator");
    ExactAggregation out;
    out.error_normalization = error_normalization;
    if(real_sign(error_normalization) <= 0) throw std::invalid_argument("positive target normalization required");
    for(const auto& row : rows) {
        if(row.probability.neg) throw std::invalid_argument("negative noise probability");
        if(row.values.size() != missing.size()) throw std::invalid_argument("exact row width mismatch");
        if(row.values.size() < 3 || missing[0] || missing[1])
            throw std::invalid_argument("history and acceptance required");
        if(!((row.values[0] + Scalar(-1)).zero()))
            throw std::runtime_error("feedback enumeration did not sum to unit mass");
        if(out.sums.empty()) out.sums.resize(row.values.size());
        out.covered = out.covered + row.probability;
        for(size_t i = 0; i < out.sums.size(); i++)
            out.sums[i] = out.sums[i] + row.values[i] * file_integer_scalar(row.probability);
    }
    if(Integer::cmp(out.covered, denominator) > 0) throw std::runtime_error("noise mass exceeds one");
    auto tail = file_integer_scalar(denominator - out.covered), d = file_integer_scalar(denominator);
    const auto& sums = out.sums;
    if(sums.size() >= 3) {
        out.bounds["acceptance"] = {scaled_real_ratio(sums[1], d), scaled_real_ratio(sums[1] + tail, d)};
        if(!missing[2]) {
            out.bounds["error"] = {scaled_real_ratio(sums[2], d * error_normalization),
                    scaled_real_ratio(sums[2] + tail * error_normalization, d * error_normalization)};
            if(!sums[1].zero()) {
                out.has_conditional = true;
                out.conditional = scaled_real_ratio(sums[2], sums[1] * error_normalization);
            }
            auto high_a = sums[1] + tail;
            if(!high_a.zero())
                out.bounds["conditional_error"] = {scaled_real_ratio(sums[2], high_a * error_normalization),
                        scaled_real_ratio(sums[2] + tail * error_normalization, high_a * error_normalization)};
            else out.conditional_bound_missing = true;
        }
    }
    return out;
}
} // namespace magic_native
