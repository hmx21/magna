// Independent finite assignment/span oracles for the online classical basis.
#include "runtime/causal.hpp"
#include <iostream>
#include <random>
using namespace magic_native;

void require(bool condition, const char* message) {
    if(!condition) throw std::runtime_error(message);
}
int scalar_parity(const Bits& mask, const Bits& value, size_t width) {
    int parity = 0;
    for(size_t j = 0; j < width; j++) parity ^= mask.bit(j) && value.bit(j);
    return parity;
}
bool same_request(const CausalRequest& a, const CausalRequest& b) {
    return a.pending == b.pending && a.outputs == b.outputs;
}

void check_pending_semantics() {
    for(size_t width : {size_t(2), size_t(63), size_t(64), size_t(65), size_t(129)}) {
        auto a = Bits::unit(0), b = Bits::unit(width - 1);
        CausalKnowledge known;
        std::vector<std::pair<Bits, int>> expressions = {{b, 0}, {a, 1}, {a ^ b, 0}, {b, 1}, {Bits(), 1}};
        auto request = known.request(expressions);
        require(request.pending == std::vector<Bits>({b, a, a ^ b}),
                "dependent pending labels retain first-seen order");
        require(request.outputs ==
                        std::vector<std::pair<std::optional<size_t>, int>>(
                                {{0, 0}, {1, 1}, {2, 0}, {0, 1}, {std::nullopt, 1}}),
                "duplicate labels preserve independent offsets");
        require(same_request(request, known.request(expressions)), "request cannot observe its pending labels");
        require(known.rows.empty() && known.values.empty(), "requests leave observed history untouched");
        auto result = known.observe(request, Bits(5));
        require(result == std::vector<int>({1, 1, 1, 0, 1}), "observed output order is unchanged");
        require(known.rows == std::vector<Bits>({b, a}) && known.values == std::vector<int>({1, 0}),
                "only newly independent observations enter ordered history");
        auto after = known.request(expressions);
        require(after.pending.empty(), "dependent observations become known");
        require(known.observe(after, {}) == result, "known affine constants reproduce outputs");
        CausalKnowledge inconsistent;
        bool rejected = false;
        try {
            inconsistent.observe(request, Bits(1));
        } catch(const std::runtime_error&) { rejected = true; }
        require(rejected, "contradictory dependent pending labels are rejected");
        CausalKnowledge out_of_range;
        rejected = false;
        try {
            out_of_range.observe(request, Bits::unit(3));
        } catch(const std::invalid_argument&) { rejected = true; }
        require(rejected && out_of_range.rows.empty(), "out-of-range words are rejected before observing");
    }
}

void check_finite_assignments() {
    std::mt19937_64 random(202610041);
    // Enumerate possible assignments on a small set of physical positions.
    // Embedding crosses packed boundaries without using a GF(2) solver oracle.
    for(size_t width : {size_t(0), size_t(1), size_t(7), size_t(63), size_t(64), size_t(65), size_t(137)}) {
        std::vector<size_t> positions;
        for(size_t j : {size_t(0), size_t(1), size_t(2), size_t(62), size_t(63), size_t(64), size_t(65),
                    width ? width - 1 : 0})
            if(j < width && std::find(positions.begin(), positions.end(), j) == positions.end()) positions.push_back(j);
        for(size_t trial = 0; trial < 24; trial++) {
            std::vector<Bits> possible;
            for(size_t word = 0; word < (size_t(1) << positions.size()); word++) {
                Bits value;
                for(size_t j = 0; j < positions.size(); j++)
                    if((word >> j) & 1) value.set(positions[j]);
                possible.push_back(std::move(value));
            }
            Bits truth = possible[random() % possible.size()];
            CausalKnowledge known;
            for(size_t batch = 0; batch < 12; batch++) {
                std::vector<std::pair<Bits, int>> expressions;
                for(size_t j = 0; j < 1 + batch % 7; j++) {
                    Bits mask;
                    for(auto position : positions)
                        if(random() & 1) mask.set(position);
                    expressions.push_back({std::move(mask), int(random() & 1)});
                }
                if(batch % 3 == 0) expressions.push_back(expressions.front());
                auto request = known.request(expressions);
                require(same_request(request, known.request(expressions)), "repeated unobserved request is identical");
                for(size_t j = 0; j < expressions.size(); j++) {
                    const auto& [mask, offset] = expressions[j];
                    int first = scalar_parity(mask, possible[0], width) ^ offset;
                    bool variable = false;
                    for(const auto& world : possible) variable |= (scalar_parity(mask, world, width) ^ offset) != first;
                    require(bool(request.outputs[j].first) == variable,
                            "known status agrees with finite assignment set");
                }
                for(const auto& world : possible) {
                    std::vector<int> pending;
                    for(const auto& mask : request.pending) pending.push_back(scalar_parity(mask, world, width));
                    for(size_t j = 0; j < expressions.size(); j++) {
                        const auto& [index, constant] = request.outputs[j];
                        int result = constant ^ (index ? pending[*index] : 0);
                        require(result == (scalar_parity(expressions[j].first, world, width) ^ expressions[j].second),
                                "requested affine outputs agree for every possible assignment");
                    }
                }
                auto expected_rows = known.rows;
                auto expected_values = known.values;
                Bits observed;
                for(size_t j = 0; j < request.pending.size(); j++) {
                    const auto& mask = request.pending[j];
                    int value = scalar_parity(mask, truth, width);
                    if(value) observed.set(j);
                    bool variable = false;
                    for(const auto& world : possible) variable |= scalar_parity(mask, world, width) != value;
                    if(variable) {
                        expected_rows.push_back(mask);
                        expected_values.push_back(value);
                    }
                    possible.erase(
                            std::remove_if(possible.begin(), possible.end(),
                                    [&](const Bits& world) { return scalar_parity(mask, world, width) != value; }),
                            possible.end());
                }
                auto output = known.observe(request, observed);
                require(known.rows == expected_rows && known.values == expected_values,
                        "observed independent rows retain finite-oracle order");
                for(size_t j = 0; j < expressions.size(); j++)
                    require(output[j] == (scalar_parity(expressions[j].first, truth, width) ^ expressions[j].second),
                            "reported output equals actual assignment parity");
            }
        }
    }
}

void check_large_request_order() {
    // Descending independent unit rows have an analytical first-seen order.
    // The 64/65 cases straddle the bounded sorted-index/tree dispatch; extra
    // duplicate expressions test equal pointer keys with distinct offsets.
    for(size_t width : {size_t(63), size_t(64), size_t(65), size_t(257)})
        for(bool duplicates : {false, true}) {
            std::vector<std::pair<Bits, int>> expressions;
            std::vector<Bits> expected_pending;
            std::vector<std::pair<std::optional<size_t>, int>> expected_outputs;
            Bits truth, word;
            std::vector<int> expected_values, expected_result;
            for(size_t j = 0; j < width; j++)
                if(j % 5 < 2) truth.set(j);
            for(size_t j = width; j-- > 0;) {
                int offset = j & 1, value = truth.bit(j);
                size_t index = width - 1 - j;
                auto mask = Bits::unit(j);
                expected_pending.push_back(mask);
                expected_values.push_back(value);
                expressions.push_back({mask, offset});
                expected_outputs.push_back({index, offset});
                expected_result.push_back(value ^ offset);
                if(value) word.set(index);
                if(duplicates && j % 3 == 0) {
                    expressions.push_back({mask, offset ^ 1});
                    expected_outputs.push_back({index, offset ^ 1});
                    expected_result.push_back(value ^ offset ^ 1);
                }
            }
            if(duplicates) {
                expressions.push_back({Bits(), 1});
                expected_outputs.push_back({std::nullopt, 1});
                expected_result.push_back(1);
            }
            CausalKnowledge known;
            auto request = known.request(expressions);
            require(request.pending == expected_pending && request.outputs == expected_outputs,
                    "large reverse requests preserve first-seen labels, duplicate indices, and offsets");
            require(same_request(request, known.request(expressions)), "large pending request does not observe labels");
            require(known.observe(request, word) == expected_result,
                    "large reversed output order matches scalar truth pattern");
            require(known.rows == expected_pending && known.values == expected_values,
                    "large observed history retains pending order");
            auto after = known.request(expressions);
            require(after.pending.empty(), "large observed labels become known");
            require(known.observe(after, {}) == expected_result, "large duplicate outputs preserve known constants");
        }
}

void check_dense_spans() {
    std::mt19937_64 random(202610042);
    for(size_t width : {size_t(0), size_t(1), size_t(63), size_t(64), size_t(65), size_t(137)}) {
        CausalKnowledge known;
        std::vector<Bits> independent;
        Bits truth;
        for(size_t j = 0; j < width; j++)
            if(random() & 1) truth.set(j);
        for(size_t round = 0; round < 9; round++) {
            Bits next;
            for(size_t j = 0; j < width; j++)
                if(random() & 1) next.set(j);
            auto added = known.insert(next, scalar_parity(next, truth, width));
            if(added.first) independent.push_back(next);
            // Scalar coordinate sums enumerate the span independently of the
            // production pivot order, packed XOR, and reduction procedure.
            for(size_t choice = 0; choice < (size_t(1) << independent.size()); choice++) {
                Bits row;
                for(size_t j = 0; j < width; j++) {
                    int parity = 0;
                    for(size_t k = 0; k < independent.size(); k++)
                        if((choice >> k) & 1) parity ^= independent[k].bit(j);
                    if(parity) row.set(j);
                }
                int expected = scalar_parity(row, truth, width);
                auto reduced = known.reduce(row);
                require(reduced.first.empty() && reduced.second == expected,
                        "dense enumerated span has exact affine value");
                require(known.insert(row, expected) == std::make_pair(false, 0),
                        "dependent consistent insert adds no row");
                require(known.insert(row, expected ^ 1) == std::make_pair(false, 1),
                        "dependent inconsistent insert exposes contradiction");
            }
        }
        auto outside = Bits::unit(width + 64);
        auto residual = known.reduce(outside);
        require(residual.first == outside && residual.second == 0,
                "unseen higher pivot stops reduction without index access");
    }
}

int main() {
    check_pending_semantics();
    check_finite_assignments();
    check_large_request_order();
    check_dense_spans();
    std::cout << "native causal checks passed\n";
}
