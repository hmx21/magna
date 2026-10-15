NoiseCodeLayout* noise_layout_handle(PyObject* cap) {
    return static_cast<NoiseCodeLayout*>(PyCapsule_GetPointer(cap, "magic-noise-layout-v1"));
}
void noise_layout_destructor(PyObject* cap) { delete noise_layout_handle(cap); }
std::vector<OriginalFault> read_original_faults(PyObject* raw) {
    Seq rows(raw);
    std::vector<OriginalFault> out;
    for(size_t j = 0; j < rows.size(); j++) {
        Seq row(rows[j]);
        if(row.size() != 3) throw std::invalid_argument("fault shape");
        auto p = file_text(row[2]);
        auto code = std::string("IXZY").find(p);
        out.push_back(
                {file_text(row[0]), index_int(row[1]), p.size() == 1 && code != std::string::npos ? int(code) : -1});
    }
    return out;
}
PyObject* output_original_faults(const std::vector<OriginalFault>& faults) {
    Object out(PyTuple_New(faults.size()));
    for(size_t j = 0; j < faults.size(); j++) {
        const auto& f = faults[j];
        char p[2] = {"IXZY"[f.pauli], 0};
        PyTuple_SET_ITEM(out.p, j, Py_BuildValue("(sis)", f.after.c_str(), f.qubit, p));
    }
    return Py_NewRef(out.p);
}
PyObject* make_noise_layout(PyObject*, PyObject* args) {
    try {
        PyObject *cap, *raw;
        if(!PyArg_ParseTuple(args, "OO", &cap, &raw)) return nullptr;
        auto* layout = instrument_layout(cap);
        if(!layout) return nullptr;
        Seq rows(raw);
        std::vector<const NoiseSite*> sites;
        for(size_t j = 0; j < rows.size(); j++) {
            auto* site = noise_site_handle(rows[j]);
            if(!site) return nullptr;
            sites.push_back(site);
        }
        auto result = std::make_unique<NoiseCodeLayout>(native_work([&]() { return NoiseCodeLayout(*layout, sites); }));
        return PyCapsule_New(result.release(), "magic-noise-layout-v1", noise_layout_destructor);
    } catch(...) { return error(); }
}
PyObject* noise_layout_words(PyObject*, PyObject* args) {
    try {
        PyObject *cap, *raw;
        if(!PyArg_ParseTuple(args, "OO", &cap, &raw)) return nullptr;
        auto* layout = noise_layout_handle(cap);
        if(!layout) return nullptr;
        layout->validate_words(bitvec(raw));
        Py_RETURN_NONE;
    } catch(...) { return error(); }
}
PyObject* decode_noise_layout(PyObject*, PyObject* args) {
    try {
        PyObject *cap, *raw, *reported;
        if(!PyArg_ParseTuple(args, "OOO", &cap, &raw, &reported)) return nullptr;
        auto* layout = noise_layout_handle(cap);
        if(!layout) return nullptr;
        Seq rows(raw);
        std::vector<std::pair<size_t, Bits>> selected;
        for(size_t j = 0; j < rows.size(); j++) {
            Seq row(rows[j]);
            if(row.size() != 2) throw std::invalid_argument("noise index/code shape");
            selected.push_back({natural(row[0]), bits(row[1])});
        }
        auto words = bitvec(reported);
        auto result = native_work([&]() { return layout->decode(selected, words); });
        Object faults(output_original_faults(result.faults)), flips(output_bits(result.flips)),
                true_words(output_bit_tuple(result.true_words));
        return PyTuple_Pack(3, faults.p, flips.p, true_words.p);
    } catch(...) { return error(); }
}
PyObject* check_compact_assignment(PyObject*, PyObject* args) {
    try {
        PyObject *cap, *raw_word, *raw_report, *raw_faults;
        if(!PyArg_ParseTuple(args, "OOOO", &cap, &raw_word, &raw_report, &raw_faults)) return nullptr;
        auto* layout = instrument_layout(cap);
        if(!layout) return nullptr;
        auto word = bits(raw_word), report = bits(raw_report);
        auto faults = read_original_faults(raw_faults);
        validate_compact_assignment(*layout, word, report, faults);
        Py_RETURN_NONE;
    } catch(...) { return error(); }
}
PyObject* record_names_from_word(PyObject*, PyObject* args) {
    try {
        PyObject *raw, *value;
        if(!PyArg_ParseTuple(args, "OO", &raw, &value)) return nullptr;
        Seq names(raw);
        auto word = bits(value);
        if(word.size() > names.size()) throw std::invalid_argument("record word width");
        Object out(PyTuple_New(word.count()));
        size_t j = 0;
        for(auto index : word.indices()) PyTuple_SET_ITEM(out.p, j++, Py_NewRef(names[index]));
        return Py_NewRef(out.p);
    } catch(...) { return error(); }
}
