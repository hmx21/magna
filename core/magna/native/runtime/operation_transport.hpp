// Original record/fault identities and declared phase coordinates share one transport.
#pragma once
#include "../evaluate/global.hpp"
#include "../compile/program.hpp"
#include "branch.hpp"
namespace magic_native {
struct FeedbackOffsets {
    std::map<std::pair<std::string, size_t>, int> phases;
    std::vector<OriginalFault> faults;
    int constant = 0;
};
struct OperationTransport {
    InstrumentLayout layout;
    std::map<std::string, size_t> locations;
    std::map<std::string, std::optional<NativeControl>> families;
    std::vector<int> magic;
    std::vector<std::optional<std::pair<std::string, size_t>>> phase_slots;
    bool native_controls = false;
    OperationTransport(const CompilerInput& input, GlobalPlan& global, const BranchSchedule& schedule,
            std::map<std::string, std::optional<NativeControl>> families, size_t max_bytes)
        : layout(schedule.layout), locations(input.locations), families(std::move(families)),
          magic(global.plan().magic), native_controls(schedule.all_native()) {
        std::map<std::pair<std::string, size_t>, size_t> slots;
        for(size_t j = 0; j < global.compiled.phase_controls.size(); j++) {
            int index = global.compiled.phase_controls[j];
            if(index < 0) phase_slots.push_back({});
            else {
                auto slot = input.feedback_slots.at(size_t(index));
                phase_slots.push_back(slot);
                slots[slot] = j;
            }
        }
        std::vector<NativeControl> controls;
        if(native_controls)
            for(size_t j = 0; j < schedule.controls.size(); j++) {
                const auto& id = layout.operations[schedule.controls[j]].id;
                auto family = this->families.at(id);
                if(!family) throw std::invalid_argument("native rule requires a phase feedback family");
                family->location = locations.at(id);
                for(size_t k = 0; k < family->axes.size(); k++) family->phase_slots.push_back(slots.at({id, k}));
                for(const auto& parent : layout.controls.at(id).parents)
                    family->parents.push_back(layout.record_index.at(parent));
                family->gates = serialize_declared_rule(*schedule.rules[j], layout.record_index);
                validate_native_control(*family, global.plan().nrecords, global.plan().locations.size(), magic.size());
                controls.push_back(std::move(*family));
            }
        finish_global_controls(global, std::move(controls), max_bytes);
    }
    FeedbackOffsets resolve(const std::vector<std::pair<std::string, std::vector<ControlGate>>>& decisions) const {
        FeedbackOffsets out;
        for(const auto& [id, gates] : decisions) {
            auto family = families.at(id);
            if(!family) throw ControlMismatch("missing declared feedback family");
            family->gates = gates;
            auto value = resolve_native_control(*family, Bits());
            for(size_t j = 0; j < value.phases.size(); j++) out.phases[{id, j}] = value.phases[j];
            for(int q : value.flips) out.faults.push_back({id, q, 1});
            out.constant = mod8(out.constant + value.phase);
        }
        return out;
    }
    Assignment pack(const ValidatedAssignment& values, const std::vector<OriginalFault>& faults,
            const std::optional<FeedbackOffsets>& feedback = {}) const {
        Assignment out;
        out.record = values.true_word;
        auto append = [&](const std::vector<OriginalFault>& originals) {
            for(const auto& fault : originals)
                out.faults.push_back({locations.at(fault.after), fault.qubit, fault.pauli});
        };
        if(feedback) append(feedback->faults);
        append(faults);
        if(feedback && !native_controls) {
            out.feedback = true;
            out.actual_magic = magic;
            for(size_t j = 0; j < phase_slots.size(); j++)
                if(phase_slots[j]) out.actual_magic[j] = feedback->phases.at(*phase_slots[j]);
            out.feedback_phase = feedback->constant;
        }
        return out;
    }
};
} // namespace magic_native
