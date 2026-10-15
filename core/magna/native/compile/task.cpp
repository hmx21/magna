// File task semantic expansion, independent of language conversion.
#include "task.hpp"
#include <cctype>
namespace magic_native {
NamedRows task_record_rows(size_t count, const IndexedRows& rows) {
    NamedRows out;
    for(const auto& row : rows) {
        std::vector<std::string> names;
        for(int index : row) {
            if(index < 0 || size_t(index) >= count) throw std::invalid_argument("task references unknown record index");
            names.push_back("m" + std::to_string(index));
        }
        out.push_back(std::move(names));
    }
    return out;
}

RecordSuccessTask lower_record_success(const CircuitData& circuit, const std::vector<int>& observables) {
    // Each observable becomes one fresh output wire, flipped by its records and register bits.
    std::vector<std::vector<FileTarget>> parities;
    for(int index : observables) {
        if(index < 0) throw std::invalid_argument("unknown observable annotation");
        bool found = false;
        std::vector<FileTarget> parity;
        for(const auto& inst : circuit.instructions) {
            if(inst.name != "OBSERVABLE_INCLUDE" || inst.args.at(0) != Rational(index)) continue;
            found = true;
            parity.insert(parity.end(), inst.targets.begin(), inst.targets.end());
        }
        if(!found) throw std::invalid_argument("unknown observable annotation " + std::to_string(index));
        parities.push_back(std::move(parity));
    }
    RecordSuccessTask result;
    result.circuit = circuit;
    auto& out = result.circuit;
    while(!out.text.empty() && std::isspace(static_cast<unsigned char>(out.text.back()))) out.text.pop_back();
    out.text += "\n# Noiseless quantum parity outputs for the declared ideal target.\n";
    auto append = [&](const std::string& name, std::vector<FileTarget> targets, const std::string& text) {
        FileInstruction inst;
        inst.name = name;
        inst.targets = std::move(targets);
        inst.op_id = "__record_success/" + std::to_string(out.instructions.size());
        inst.source = {"<record-success>", out.instructions.size() + 1, 1, {}};
        out.instructions.push_back(std::move(inst));
        out.text += text + "\n";
    };
    for(const auto& parity : parities) {
        size_t q = out.num_qubits++;
        result.outputs.push_back(q);
        append("I", {{"qubit", q}}, "I " + std::to_string(q));
        for(const auto& bit : parity) {
            // Register targets keep their SSA version, so a later overwrite does not change the parity.
            auto name = bit.kind == "rec" ? "rec[-" + std::to_string(circuit.num_records - bit.index) + "]"
                                          : "ctrl[" + std::to_string(bit.index) + "]";
            append("CX", {bit, {"qubit", q}}, "CX " + name + " " + std::to_string(q));
        }
    }
    result.target.source = "<record-success-zero>";
    result.target.text = "# All-zero record-success target.\n";
    return result;
}

FileTaskEvents expand_file_task(const CircuitData& circuit) {
    // Acceptance is declared only by POSTSELECT: affine bits become record rows, decoder bits virtual selections.
    IndexedRows selection;
    std::vector<int> outcomes;
    for(const auto& inst : circuit.instructions)
        if(inst.name == "POSTSELECT")
            for(const auto& target : inst.targets) {
                auto expression = circuit.expression(target);
                std::vector<int> row;
                bool affine = true;
                for(const auto& term : expression.terms) {
                    if(term.kind != "rec") {
                        affine = false;
                        break;
                    }
                    row.push_back(int(term.index));
                }
                if(affine) {
                    selection.push_back(std::move(row));
                    outcomes.push_back(expression.constant);
                }
            }
    FileTaskEvents result{task_record_rows(circuit.num_records, selection), std::move(outcomes)};
    FeedbackTemplate feedback(circuit);
    for(const auto& [anchor, entry] : feedback.selections) {
        result.selection.push_back({entry.record});
        result.outcomes.push_back(0);
    }
    return result;
}

} // namespace magic_native
