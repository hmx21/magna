// Immutable scalar-query layout; plan lifetime belongs to the caller.
#pragma once
#include "boundary_trace.hpp"
namespace magic_native {
struct BoundaryTotalsData {
    struct Query {
        std::vector<Bits> events;
        std::vector<int> outcomes;
        Bits x, z;
    };
    BoundaryTracePlan* plan = nullptr;
    std::vector<Bits> prefix;
    std::vector<Query> queries;
};

struct BoundaryTotalsResult {
    std::vector<std::vector<Scalar>> values;
    std::vector<BoundaryResult> last;
    std::vector<double> last_seconds;
};
inline BoundaryTotalsResult evaluate_totals(const BoundaryTotalsData& totals, const std::vector<Assignment>& input,
        const std::vector<std::vector<int>>& outcomes) {
    if(!totals.plan || outcomes.size() != totals.queries.size()) throw std::invalid_argument("scalar query shape");
    for(size_t k = 0; k < outcomes.size(); k++) {
        if(outcomes[k].size() != totals.prefix.size() + totals.queries[k].events.size())
            throw std::invalid_argument("boundary event shape");
        for(int bit : outcomes[k])
            if(bit != 0 && bit != 1) throw std::invalid_argument("binary event required");
    }
    BoundaryTotalsResult out;
    Workspace ws;
    auto events = totals.prefix;
    for(size_t j = 0; j < input.size(); j++) {
        std::vector<Scalar> values;
        auto frame = totals.plan->assignment_transport().fault_frame(input[j].faults, false);
        for(size_t k = 0; k < totals.queries.size(); k++) {
            const auto& q = totals.queries[k];
            events.resize(totals.prefix.size());
            events.insert(events.end(), q.events.begin(), q.events.end());
            auto start = std::chrono::steady_clock::now();
            auto result = totals.plan->query(input[j], {}, events, outcomes[k], q.x, q.z, false, ws, true, &frame);
            values.push_back(boundary_sum(result.terms, {}, {}, ws));
            if(j + 1 == input.size()) {
                out.last_seconds.push_back(
                        std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count());
                out.last.push_back(std::move(result));
            }
        }
        out.values.push_back(std::move(values));
    }
    return out;
}
} // namespace magic_native
