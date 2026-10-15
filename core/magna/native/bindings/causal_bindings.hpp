// Static callback metadata and query-family values; callbacks are not invoked here.
FileControlMetadata* file_control_handle(PyObject* cap) {
    return static_cast<FileControlMetadata*>(PyCapsule_GetPointer(cap, "magic-file-controls-v1"));
}
void file_control_destructor(PyObject* cap) { delete file_control_handle(cap); }
PyObject* make_file_controls(PyObject*, PyObject* cap) {
    try {
        auto* circuit = circuit_handle(cap);
        if(!circuit) return nullptr;
        auto metadata =
                std::make_unique<FileControlMetadata>(native_work([&]() { return FileControlMetadata(*circuit); }));
        Object indices(PyTuple_New(metadata->decision_order.size())), calls(PyTuple_New(metadata->callbacks.size()));
        for(size_t j = 0; j < metadata->decision_order.size(); j++)
            PyTuple_SET_ITEM(indices.p, j, output_target(metadata->decision_order[j]));
        for(size_t j = 0; j < metadata->callbacks.size(); j++) {
            const auto& call = metadata->callbacks[j];
            Object inputs(PyTuple_New(call.inputs.size())), outputs(output_sizes(call.outputs));
            for(size_t k = 0; k < call.inputs.size(); k++) {
                const auto& input = call.inputs[k];
                Object mask(output_bits(input.records)), decisions(output_sizes(input.decisions));
                PyTuple_SET_ITEM(inputs.p, k, Py_BuildValue("(OiO)", mask.p, input.constant, decisions.p));
            }
            PyTuple_SET_ITEM(calls.p, j, Py_BuildValue("(nOO)", Py_ssize_t(call.instruction), inputs.p, outputs.p));
        }
        Object handle(PyCapsule_New(metadata.release(), "magic-file-controls-v1", file_control_destructor));
        return PyTuple_Pack(3, handle.p, indices.p, calls.p);
    } catch(...) { return error(); }
}
NamedRows read_named_rows(PyObject* raw) {
    Seq rows(raw);
    NamedRows out;
    for(size_t j = 0; j < rows.size(); j++) {
        Seq names(rows[j]);
        std::vector<std::string> row;
        for(size_t k = 0; k < names.size(); k++) row.push_back(file_text(names[k]));
        out.push_back(std::move(row));
    }
    return out;
}
QueryTask read_query_task(PyObject* raw) {
    QueryTask task;
    task.selection = read_named_rows(field(raw, "postselection"));
    Seq observables(field(raw, "observables"));
    for(size_t j = 0; j < observables.size(); j++) {
        Seq row(observables[j]);
        if(row.size() != 2) throw std::invalid_argument("observable shape");
        task.observables.push_back({bits(row[0]), bits(row[1])});
    }
    task.sample_records = read_sizes(field(raw, "sample_records"));
    return task;
}
