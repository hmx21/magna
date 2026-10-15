InstrumentLayout* instrument_layout(PyObject* cap) {
    return static_cast<InstrumentLayout*>(PyCapsule_GetPointer(cap, "magic-instrument-layout-v1"));
}
void instrument_layout_destructor(PyObject* cap) { delete instrument_layout(cap); }
PyObject* make_instrument_layout(PyObject*, PyObject* args) {
    try {
        PyObject *raw_inputs, *raw_outputs, *raw_initial, *raw_ops, *raw_controls;
        if(!PyArg_ParseTuple(args, "OOOOO", &raw_inputs, &raw_outputs, &raw_initial, &raw_ops, &raw_controls))
            return nullptr;
        auto inputs = intvec(raw_inputs), outputs = intvec(raw_outputs);
        Seq initial(raw_initial), ops(raw_ops), controls(raw_controls);
        if(initial.size() != 3) throw std::invalid_argument("initial shape");
        auto width = natural(initial[0]);
        auto power = integer(initial[1]);
        auto columns = natural(initial[2]);
        std::vector<OperationIdentity> operations;
        std::map<std::string, ControlIdentity> declarations;
        for(size_t j = 0; j < ops.size(); j++) {
            Seq row(ops[j]);
            if(row.size() != 3) throw std::invalid_argument("operation identity shape");
            operations.push_back({file_text(row[0]), file_text(row[1]),
                    row[2] == Py_None ? std::nullopt : std::make_optional(file_text(row[2]))});
        }
        for(size_t j = 0; j < controls.size(); j++) {
            Seq row(controls[j]);
            if(row.size() != 3) throw std::invalid_argument("control identity shape");
            ControlIdentity node;
            node.version = file_text(row[2]);
            Seq parents(row[1]);
            for(size_t k = 0; k < parents.size(); k++) node.parents.push_back(file_text(parents[k]));
            declarations[file_text(row[0])] = std::move(node);
        }
        auto layout = std::make_unique<InstrumentLayout>(native_work(
                [&]() { return InstrumentLayout(inputs, outputs, width, power, columns, operations, declarations); }));
        Object records(PyTuple_New(layout->record_names.size())), ids(PyTuple_New(layout->ids.size()));
        for(size_t j = 0; j < layout->record_names.size(); j++)
            PyTuple_SET_ITEM(records.p, j, PyUnicode_FromString(layout->record_names[j].c_str()));
        for(size_t j = 0; j < layout->ids.size(); j++)
            PyTuple_SET_ITEM(ids.p, j, PyUnicode_FromString(layout->ids[j].c_str()));
        Object cap(PyCapsule_New(layout.release(), "magic-instrument-layout-v1", instrument_layout_destructor));
        return PyTuple_Pack(3, cap.p, records.p, ids.p);
    } catch(...) { return error(); }
}
PyObject* validate_instrument_assignment(PyObject*, PyObject* args) {
    try {
        PyObject *cap, *raw_true, *raw_flips, *raw_faults;
        if(!PyArg_ParseTuple(args, "OOOO", &cap, &raw_true, &raw_flips, &raw_faults)) return nullptr;
        auto* layout = instrument_layout(cap);
        if(!layout) return nullptr;
        Seq values(raw_true), flips(raw_flips), faults(raw_faults);
        std::vector<std::pair<std::string, int>> truth;
        std::vector<std::string> flip_names;
        std::vector<OriginalFault> original;
        for(size_t j = 0; j < values.size(); j++) {
            Seq row(values[j]);
            if(row.size() != 2) throw std::invalid_argument("true record shape");
            truth.push_back({file_text(row[0]), index_int(row[1])});
        }
        for(size_t j = 0; j < flips.size(); j++) flip_names.push_back(file_text(flips[j]));
        for(size_t j = 0; j < faults.size(); j++) {
            Seq row(faults[j]);
            if(row.size() != 3) throw std::invalid_argument("original fault shape");
            auto pauli = file_text(row[2]);
            auto p = std::string("IXZY").find(pauli);
            original.push_back(
                    {file_text(row[0]), index_int(row[1]), pauli.size() == 1 && p != std::string::npos ? int(p) : -1});
        }
        auto result = native_work([&]() { return layout->validate(truth, flip_names, original); });
        Object groups(PyTuple_New(result.fault_groups.size())), names(PyTuple_New(result.flip_word.count()));
        size_t k = 0;
        for(auto index : result.flip_word.indices())
            PyTuple_SET_ITEM(names.p, k++, PyUnicode_FromString(layout->record_names[index].c_str()));
        for(size_t j = 0; j < result.fault_groups.size(); j++) {
            const auto& [name, indices] = result.fault_groups[j];
            Object values(output_sizes(indices));
            PyTuple_SET_ITEM(groups.p, j, Py_BuildValue("(sO)", name.c_str(), values.p));
        }
        Object word(output_bits(result.true_word)), flip_word(output_bits(result.flip_word));
        return PyTuple_Pack(4, word.p, flip_word.p, names.p, groups.p);
    } catch(...) { return error(); }
}
