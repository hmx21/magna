PyObject* literal_program_layout(PyObject*, PyObject* raw) {
    try {
        Seq rows(raw);
        std::vector<OperationIdentity> operations;
        std::vector<int> outcomes;
        for(size_t j = 0; j < rows.size(); j++) {
            auto kind = file_text(field(rows[j], "kind"));
            auto* record = PyDict_GetItemString(rows[j], "record");
            operations.push_back({"", kind, record ? std::make_optional(file_text(record)) : std::nullopt});
            int outcome = 0;
            if(kind.rfind("M", 0) == 0) {
                auto* value = PyDict_GetItemString(rows[j], "true");
                if(!value) value = PyDict_GetItemString(rows[j], "outcome");
                if(value) outcome = index_int(value);
            }
            outcomes.push_back(outcome);
        }
        LiteralProgramLayout layout(std::move(operations), outcomes);
        Object out(PyTuple_New(rows.size())), truth(output_named_bits(layout.truth));
        for(size_t j = 0; j < rows.size(); j++) {
            Object op(PyDict_Copy(rows[j]));
            phase_set(op.p, "id", PyUnicode_FromString(layout.operations[j].id.c_str()));
            if(layout.operations[j].record)
                phase_set(op.p, "record", PyUnicode_FromString(layout.operations[j].record->c_str()));
            PyTuple_SET_ITEM(out.p, j, Py_NewRef(op.p));
        }
        return PyTuple_Pack(2, out.p, truth.p);
    } catch(...) { return error(); }
}
