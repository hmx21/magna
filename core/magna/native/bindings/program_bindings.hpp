PyObject* output_compiler_metadata(const CompilerInput& input) {
    Object locations(PyDict_New()), records(PyTuple_New(input.record_names.size())),
            ids(PyTuple_New(input.operation_ids.size())), slots(PyTuple_New(input.feedback_slots.size()));
    for(const auto& name : input.operation_ids)
        phase_set(locations.p, name.c_str(), PyLong_FromSize_t(input.locations.at(name)));
    for(size_t j = 0; j < input.record_names.size(); j++)
        PyTuple_SET_ITEM(records.p, j, PyUnicode_FromString(input.record_names[j].c_str()));
    for(size_t j = 0; j < input.operation_ids.size(); j++)
        PyTuple_SET_ITEM(ids.p, j, PyUnicode_FromString(input.operation_ids[j].c_str()));
    for(size_t j = 0; j < input.feedback_slots.size(); j++) {
        const auto& [id, slot] = input.feedback_slots[j];
        PyTuple_SET_ITEM(slots.p, j, Py_BuildValue("(sn)", id.c_str(), Py_ssize_t(slot)));
    }
    Object metadata(Py_BuildValue("{s:O,s:O,s:O,s:O}", "locations", locations.p, "record_names", records.p,
            "operation_ids", ids.p, "feedback_slots", slots.p));
    return Py_NewRef(metadata.p);
}
std::vector<ProgramOperation> read_program_operations(PyObject* operations, bool original = true) {
    Seq raw(operations);
    std::vector<ProgramOperation> ops;
    for(size_t j = 0; j < raw.size(); j++) {
        auto* item = raw[j];
        ProgramOperation source;
        if(original) source.id = file_text(field(item, "id"));
        auto& op = source.value;
        op.kind = file_text(field(item, "kind"));
        if(auto* qs = PyDict_GetItemString(item, "qubits")) op.qubits = intvec(qs);
        if(auto* record = PyDict_GetItemString(item, "record")) source.record = file_text(record);
        if(op.kind == "T" || op.kind == "S" || op.kind == "PHASE") {
            auto* value = PyDict_GetItemString(item, "actual_exponent");
            if(!value) value = PyDict_GetItemString(item, "exponent");
            op.exponent = op.kind == "S" ? 2 : 1;
            if(value) {
                if(!PyLong_CheckExact(value)) throw std::invalid_argument("integer phase exponent required");
                op.exponent = phase_exponent(value);
            }
        }
        if(op.kind == "MP") {
            if(auto* sign = PyDict_GetItemString(item, "sign")) op.sign = index_int(sign);
            auto* paulis = field(item, "paulis");
            if(PyUnicode_Check(paulis)) op.paulis = file_text(paulis);
            else {
                Seq letters(paulis);
                for(size_t p = 0; p < letters.size(); p++) op.paulis += file_text(letters[p]);
            }
        }
        if(op.kind == "PAULI")
            if(auto* pauli = PyDict_GetItemString(item, "pauli")) op.pauli = file_text(pauli);
        ops.push_back(std::move(source));
    }
    return ops;
}
// Adapt the operation interface once; native input owns all compiler values.
PyObject* serialize_instrument(PyObject*, PyObject* args) {
    try {
        PyObject *initial, *inputs, *outputs, *operations, *controls, *limit;
        if(!PyArg_ParseTuple(args, "OOOOOO", &initial, &inputs, &outputs, &operations, &controls, &limit))
            return nullptr;
        auto k = ket(initial);
        auto in = intvec(inputs), out = intvec(outputs);
        size_t max_path = natural(limit);
        auto ops = read_program_operations(operations);
        Seq rules(controls);
        std::map<std::string, std::optional<NativeControl>> declarations;
        for(size_t j = 0; j < rules.size(); j++) {
            Seq row(rules[j]);
            if(row.size() != 2) throw std::invalid_argument("feedback declaration shape");
            auto id = file_text(row[0]);
            if(row[1] == Py_None) {
                declarations[id] = {};
                continue;
            }
            Seq spec(row[1]);
            if(spec.size() != 3) throw std::invalid_argument("feedback family shape");
            auto axes = [](PyObject* raw) {
                Seq rows(raw);
                std::vector<std::vector<int>> out;
                for(size_t j = 0; j < rows.size(); j++) out.push_back(intvec(rows[j]));
                return out;
            };
            declarations[id] = feedback_family(intvec(spec[0]), axes(spec[1]), axes(spec[2]));
        }
        auto input = std::make_unique<CompilerInput>(
                native_work([&]() { return compile_input(k, in, out, ops, declarations, max_path); }));
        Object metadata(output_compiler_metadata(*input));
        Object handle(PyCapsule_New(input.release(), "magic-compiler-input-v1", compiler_input_destructor));
        return PyTuple_Pack(2, handle.p, metadata.p);
    } catch(const UnsupportedFeedback& e) {
        PyErr_SetString(PyExc_NotImplementedError, e.what());
        return nullptr;
    } catch(...) { return error(); }
}
