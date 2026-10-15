IndexedRows read_indexed_rows(PyObject* raw) {
    Seq rows(raw);
    IndexedRows out;
    for(size_t j = 0; j < rows.size(); j++) out.push_back(intvec(rows[j]));
    return out;
}
PyObject* output_named_rows(const NamedRows& rows) {
    Object out(PyTuple_New(rows.size()));
    for(size_t j = 0; j < rows.size(); j++) PyTuple_SET_ITEM(out.p, j, output_names(rows[j]));
    return Py_NewRef(out.p);
}
PyObject* task_record_rows_binding(PyObject*, PyObject* args) {
    try {
        PyObject *count, *raw;
        if(!PyArg_ParseTuple(args, "OO", &count, &raw)) return nullptr;
        return output_named_rows(task_record_rows(natural(count), read_indexed_rows(raw)));
    } catch(...) { return error(); }
}
PyObject* lower_record_success_binding(PyObject*, PyObject* args) {
    try {
        PyObject *cap, *observables;
        if(!PyArg_ParseTuple(args, "OO", &cap, &observables)) return nullptr;
        auto* source = circuit_handle(cap);
        if(!source) return nullptr;
        auto indices = intvec(observables);
        auto result = native_work([&]() { return lower_record_success(*source, indices); });
        Object circuit(output_circuit(std::make_unique<CircuitData>(std::move(result.circuit))));
        Object target(output_circuit(std::make_unique<CircuitData>(std::move(result.target))));
        Object outputs(PyTuple_New(result.outputs.size()));
        for(size_t j = 0; j < result.outputs.size(); j++)
            PyTuple_SET_ITEM(outputs.p, j, PyLong_FromSize_t(result.outputs[j]));
        return PyTuple_Pack(3, circuit.p, outputs.p, target.p);
    } catch(...) { return error(); }
}
PyObject* expand_file_task_binding(PyObject*, PyObject* args) {
    try {
        PyObject* cap;
        if(!PyArg_ParseTuple(args, "O", &cap)) return nullptr;
        auto* source = circuit_handle(cap);
        if(!source) return nullptr;
        auto task = native_work([&]() { return expand_file_task(*source); });
        Object a(output_named_rows(task.selection)), b(output_int_tuple(task.outcomes));
        return PyTuple_Pack(2, a.p, b.p);
    } catch(...) { return error(); }
}
