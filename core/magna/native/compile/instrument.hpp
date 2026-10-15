// Original operation/record identities and assignment validation.
#pragma once
#include "plan.hpp"
#include <optional>
#include <set>
namespace magic_native {
struct OperationIdentity {
    std::string id, kind;
    std::optional<std::string> record;
};
struct ControlIdentity {
    std::vector<std::string> parents;
    std::string version;
};
struct OriginalFault {
    std::string after;
    int qubit = 0, pauli = 0;
};
inline Assignment pack_original_faults(
        const std::map<std::string, size_t>& locations, const std::vector<OriginalFault>& faults) {
    Assignment out;
    for(const auto& fault : faults) {
        auto found = locations.find(fault.after);
        if(found == locations.end() || fault.pauli < 0 || fault.pauli > 3)
            throw std::invalid_argument("invalid original fault");
        out.faults.push_back({found->second, fault.qubit, fault.pauli});
    }
    return out;
}
struct ValidatedAssignment {
    Bits true_word, flip_word;
    std::vector<std::pair<std::string, std::vector<size_t>>> fault_groups;
};
struct InstrumentLayout {
    std::vector<OperationIdentity> operations;
    std::map<std::string, ControlIdentity> controls;
    std::map<std::string, size_t> locations, record_index;
    std::vector<std::string> record_names, ids;
    InstrumentLayout(const std::vector<int>& inputs, const std::vector<int>& outputs, size_t initial_width,
            int64_t initial_power, size_t initial_columns, std::vector<OperationIdentity> operations,
            std::map<std::string, ControlIdentity> controls)
        : operations(std::move(operations)), controls(std::move(controls)) {
        if(std::set<int>(inputs.begin(), inputs.end()).size() != inputs.size() ||
                std::set<int>(outputs.begin(), outputs.end()).size() != outputs.size())
            throw std::invalid_argument("duplicate boundary wire");
        if(initial_width != inputs.size() || initial_power != -int64_t(initial_columns))
            throw std::invalid_argument("a normalized stabilizer ket on exactly the declared inputs is required");
        for(size_t j = 0; j < this->operations.size(); j++) {
            const auto& op = this->operations[j];
            if(!locations.emplace(op.id, j).second) throw std::invalid_argument("stable unique operation IDs required");
            ids.push_back(op.id);
            if(op.kind == "CONTROL") {
                auto found = this->controls.find(op.id);
                if(found == this->controls.end()) throw std::invalid_argument("missing control declaration");
                const auto& node = found->second;
                if(node.version.empty() || std::any_of(node.parents.begin(), node.parents.end(), [&](const auto& name) {
                       return !record_index.count(name);
                   }))
                    throw std::invalid_argument("control must declare version and earlier record parents");
            }
            if(op.record) {
                if(!record_index.emplace(*op.record, record_names.size()).second)
                    throw std::invalid_argument("duplicate record");
                record_names.push_back(*op.record);
            }
        }
    }
    ValidatedAssignment validate(const std::vector<std::pair<std::string, int>>& true_values,
            const std::vector<std::string>& report_flips, const std::vector<OriginalFault>& faults) const {
        ValidatedAssignment out;
        std::set<std::string> seen;
        for(const auto& [name, bit] : true_values) {
            auto found = record_index.find(name);
            if(found == record_index.end() || !seen.insert(name).second)
                throw std::invalid_argument("one complete assignment of true records required");
            if(bit != 0 && bit != 1) throw std::invalid_argument("record must be an integer bit");
            if(bit) out.true_word.set(found->second);
        }
        if(seen.size() != record_index.size())
            throw std::invalid_argument("one complete assignment of true records required");
        for(const auto& name : report_flips) {
            auto found = record_index.find(name);
            if(found == record_index.end()) throw std::invalid_argument("unknown flipped record");
            out.flip_word.flip(found->second);
        }
        std::map<std::string, size_t> groups;
        for(size_t j = 0; j < faults.size(); j++) {
            const auto& fault = faults[j];
            if(!locations.count(fault.after) || fault.pauli < 0 || fault.pauli > 3)
                throw std::invalid_argument("invalid original fault location or Pauli");
            auto found = groups.find(fault.after);
            if(found == groups.end()) {
                size_t k = out.fault_groups.size();
                groups[fault.after] = k;
                out.fault_groups.push_back({fault.after, {j}});
            } else out.fault_groups[found->second].second.push_back(j);
        }
        return out;
    }
};
} // namespace magic_native
