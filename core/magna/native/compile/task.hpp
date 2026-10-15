// File task annotations and public record coordinates have one native owner.
#pragma once
#include "../foundation/causal_knowledge.hpp"
#include "feedback_template.hpp"
#include "../foundation/rational.hpp"
namespace magic_native {
using IndexedRows = std::vector<std::vector<int>>;
NamedRows task_record_rows(size_t count, const IndexedRows& rows);
struct RecordSuccessTask {
    CircuitData circuit, target;
    std::vector<size_t> outputs;
};
RecordSuccessTask lower_record_success(const CircuitData& circuit, const std::vector<int>& observables);
struct FileTaskEvents {
    NamedRows selection;
    std::vector<int> outcomes;
};
FileTaskEvents expand_file_task(const CircuitData& circuit);
} // namespace magic_native
