IdealTarget* ideal_target_handle(PyObject* cap) {
    return static_cast<IdealTarget*>(PyCapsule_GetPointer(cap, "magic-ideal-target-v1"));
}
void ideal_target_destructor(PyObject* cap) { delete ideal_target_handle(cap); }
PyObject* make_ideal_target(PyObject*, PyObject* args) {
    try {
        PyObject *cap, *raw_width, *raw_outputs = Py_None, *raw_caps = Py_None;
        if(!PyArg_ParseTuple(args, "OO|OO", &cap, &raw_width, &raw_outputs, &raw_caps)) return nullptr;
        auto* circuit = circuit_handle(cap);
        if(!circuit) return nullptr;
        auto width = natural(raw_width);
        std::vector<size_t> outputs;
        std::map<size_t, int> caps;
        if(raw_outputs != Py_None) {
            outputs = read_sizes(raw_outputs);
            Seq rows(raw_caps);
            for(size_t j = 0; j < rows.size(); j++) {
                Seq row(rows[j]);
                if(row.size() != 2) throw std::invalid_argument("target postselection entry shape");
                if(!caps.emplace(natural(row[0]), index_int(row[1])).second)
                    throw std::invalid_argument("duplicate target postselection qubit");
            }
        }
        auto target = std::make_unique<IdealTarget>(native_work([&]() {
            return raw_outputs == Py_None ? IdealTarget(*circuit, width) : IdealTarget(*circuit, width, outputs, caps);
        }));
        Object ops(PyTuple_New(target->inverse.size()));
        for(size_t j = 0; j < target->inverse.size(); j++)
            PyTuple_SET_ITEM(ops.p, j, output_operation(target->inverse[j]));
        Object handle(PyCapsule_New(target.release(), "magic-ideal-target-v1", ideal_target_destructor));
        return PyTuple_Pack(2, handle.p, ops.p);
    } catch(...) { return error(); }
}
PyObject* validate_terminal_recovery(PyObject*, PyObject* cap) {
    try {
        auto* circuit = circuit_handle(cap);
        if(!circuit) return nullptr;
        native_work([&]() {
            validate_recovery(*circuit);
            return 0;
        });
        Py_RETURN_NONE;
    } catch(...) { return error(); }
}
