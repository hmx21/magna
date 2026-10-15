// Named boundary coordinates, bounded layouts and existing commuting-label sampling.
#pragma once
#include "causal.hpp"
#include "../compile/instrument.hpp"
#include "../evaluate/boundary_totals.hpp"
#include "marginal.hpp"
#include "../foundation/bounded_cache.hpp"
namespace magic_native {
inline Bits named_record_row(const std::map<std::string, size_t>& indices, const std::vector<std::string>& names) {
    Bits row;
    for(const auto& name : names) {
        auto found = indices.find(name);
        if(found == indices.end()) throw std::invalid_argument("unknown boundary record: " + name);
        row.flip(found->second);
    }
    return row;
}
struct BoundaryCoordinates {
    std::vector<Bits> transform;
    std::vector<int> values;
    size_t selection = 0, feedback = 0;
    bool compatible = true;
    BoundaryCoordinates(const std::vector<std::string>& records, const NamedRows& rows,
            const std::vector<int>& outcomes, const std::vector<std::string>& required) {
        if(rows.size() != outcomes.size()) throw std::invalid_argument("postselection shape");
        std::map<std::string, size_t> indices;
        for(size_t j = 0; j < records.size(); j++) indices[records[j]] = j;
        CausalKnowledge knowledge;
        auto insert = [&](const Bits& mask, int bit) {
            auto residual = knowledge.reduce(mask, bit);
            auto result = knowledge.insert(mask, bit);
            if(result.first) transform.push_back(residual.first);
            return result;
        };
        for(size_t j = 0; j < rows.size(); j++) {
            if(outcomes[j] != 0 && outcomes[j] != 1) throw std::invalid_argument("binary postselection");
            auto [added, rhs] = insert(named_record_row(indices, rows[j]), outcomes[j]);
            if(added) values.push_back(rhs);
            else if(rhs) compatible = false;
        }
        selection = transform.size();
        for(const auto& name : required) insert(named_record_row(indices, {name}), 0);
        feedback = transform.size() - selection;
        for(size_t j = 0; j < records.size(); j++) insert(Bits::unit(j), 0);
    }
};
struct BoundarySampleRows {
    using Query = QueryMeasurement;
    Bits word;
    Scalar mass;
    bool common = false;
    size_t commuting = 0, sampled_affine = 0;
    std::optional<Query> affine;
    Query final;
    std::string method;
};
inline BoundarySampleRows sample_boundary_rows(BoundaryTracePlan& plan, const Assignment& assignment,
        const std::vector<Bits>& rows, std::vector<Bits> events, std::vector<int> outcomes, const RandomBits& random) {
    BoundarySampleRows out;
    auto sector = plan.commuting_rows(rows);
    out.common = plan.common_axes;
    out.commuting = sector.size();
    Workspace ws;
    auto query = [&](const std::vector<Bits>& kept, bool affine) {
        auto start = std::chrono::steady_clock::now();
        auto result = plan.query(assignment, kept, events, outcomes, {}, {}, affine, ws);
        double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        return std::make_pair(std::move(result), seconds);
    };
    if(!sector.empty()) {
        std::vector<Bits> selected;
        for(const auto& mask : sector) selected.push_back(xors(rows, mask));
        auto [result, seconds] = query(selected, true);
        out.affine = measure_boundary_query(result, selected.size(), seconds);
        if(result.affine) {
            auto sample = ProbabilityMarginal{std::move(result.terms), selected.size()}.sample(random);
            events.insert(events.end(), selected.begin(), selected.end());
            for(size_t j = 0; j < selected.size(); j++) outcomes.push_back(sample.first.bit(j));
            out.sampled_affine = selected.size();
        }
    }
    auto [result, seconds] = query(rows, false);
    out.final = measure_boundary_query(result, rows.size(), seconds);
    ProbabilityMarginal marginal{std::move(result.terms), rows.size()};
    auto sample = marginal.sample(random);
    out.word = sample.first;
    out.method = sample.second;
    std::vector<Bits> points;
    std::vector<int> bits;
    for(size_t j = 0; j < rows.size(); j++) {
        points.push_back(Bits::unit(j));
        bits.push_back(out.word.bit(j));
    }
    out.mass = marginal.total(points, bits);
    return out;
}
struct BoundaryService {
    BoundaryTracePlan* plan;
    std::map<std::string, size_t> locations, records;
    std::map<std::string, std::pair<std::string, int>> report_anchors;
    struct Rows {
        NamedRows names;
        std::vector<Bits> values;
    };
    static constexpr size_t max_row_sets = 128, max_total_bundles = 32;
    BoundedCache<Rows> row_cache;
    BoundedCache<std::shared_ptr<const BoundaryTotalsData>> total_cache;
    mutable std::mutex mutex;
    explicit BoundaryService(BoundaryTracePlan* plan) : plan(plan) {}
    Assignment pack(const std::vector<OriginalFault>& original, const std::vector<std::string>& flips) const {
        auto out = pack_original_faults(locations, original);
        for(const auto& name : flips) {
            auto found = report_anchors.find(name);
            if(found == report_anchors.end()) throw std::invalid_argument("unknown report flip");
            const auto& [id, wire] = found->second;
            out.faults.push_back({locations.at(id), wire, 1});
        }
        return out;
    }
    std::vector<Bits> rows(const NamedRows& names) {
        std::lock_guard<std::mutex> lock(mutex);
        if(auto* cached = row_cache.find([&](const Rows& r) { return r.names == names; })) return cached->values;
        std::vector<Bits> result;
        for(const auto& row : names) result.push_back(named_record_row(records, row));
        row_cache.insert({names, result}, max_row_sets);
        return result;
    }
    std::shared_ptr<const BoundaryTotalsData> totals(const std::vector<BoundaryTotalsData::Query>& queries) {
        if(queries.size() > 4096) throw std::length_error("scalar query bundle limit");
        auto data = std::make_shared<BoundaryTotalsData>();
        data->plan = plan;
        if(!queries.empty()) data->prefix = queries.front().events;
        for(const auto& query : queries) {
            size_t j = 0;
            while(j < data->prefix.size() && j < query.events.size() && data->prefix[j] == query.events[j]) j++;
            data->prefix.resize(j);
        }
        size_t width = plan->compiled.plan.physical;
        for(auto query : queries) {
            if(query.x.size() > width || query.z.size() > width) throw std::invalid_argument("Pauli outside boundary");
            for(const auto& event : query.events)
                if(event.size() > width || (event & query.x).parity())
                    throw std::invalid_argument("invalid scalar event/Pauli boundary");
            query.events.erase(query.events.begin(), query.events.begin() + data->prefix.size());
            query.outcomes.clear();
            data->queries.push_back(std::move(query));
        }
        std::lock_guard<std::mutex> lock(mutex);
        auto same = [&](const std::shared_ptr<const BoundaryTotalsData>& item) {
            const auto& cached = *item;
            if(cached.prefix != data->prefix || cached.queries.size() != data->queries.size()) return false;
            for(size_t j = 0; j < cached.queries.size(); j++) {
                const auto& a = cached.queries[j];
                const auto& b = data->queries[j];
                if(a.events != b.events || a.x != b.x || a.z != b.z) return false;
            }
            return true;
        };
        if(auto* cached = total_cache.find(same)) return *cached;
        total_cache.insert(data, max_total_bundles);
        return data;
    }
};
} // namespace magic_native
