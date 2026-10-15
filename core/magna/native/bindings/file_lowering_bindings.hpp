// Python inspection values and callback-free native lowering.
PyObject* output_sizes(const std::vector<size_t>& values) {
    Object out(PyTuple_New(values.size()));
    for(size_t j = 0; j < values.size(); j++) PyTuple_SET_ITEM(out.p, j, PyLong_FromSize_t(values[j]));
    return Py_NewRef(out.p);
}
std::vector<size_t> read_sizes(PyObject* raw) {
    Seq row(raw);
    std::vector<size_t> out;
    for(size_t j = 0; j < row.size(); j++) out.push_back(natural(row[j]));
    return out;
}
NoiseActions read_noise_actions(PyObject* raw) {
    Seq rows(raw);
    NoiseActions out;
    for(size_t j = 0; j < rows.size(); j++) {
        Seq row(rows[j]);
        if(row.size() != 2) throw std::invalid_argument("noise action shape");
        auto& group = out[file_text(row[0])];
        Seq gates(row[1]);
        for(size_t k = 0; k < gates.size(); k++) {
            Seq gate(gates[k]);
            if(gate.size() != 2) throw std::invalid_argument("Pauli shape");
            auto name = file_text(gate[0]);
            if(name.size() != 1 || std::string("IXYZ").find(name[0]) == std::string::npos)
                throw std::invalid_argument("Pauli required");
            group.push_back({name[0], natural(gate[1])});
        }
    }
    return out;
}
PyObject* output_operation(const Operation& op, bool include_id = true) {
    Object out(PyDict_New()), qs(PyList_New(op.qubits.size()));
    for(size_t j = 0; j < op.qubits.size(); j++) PyList_SET_ITEM(qs.p, j, PyLong_FromSize_t(op.qubits[j]));
    if(include_id) phase_set(out.p, "id", PyUnicode_FromString(op.id.c_str()));
    phase_set(out.p, "kind", PyUnicode_FromString(op.kind.c_str()));
    if(PyDict_SetItemString(out.p, "qubits", qs.p) < 0) throw std::bad_alloc();
    if(op.exponent) phase_set(out.p, "exponent", PyLong_FromLong(*op.exponent));
    if(op.sign) phase_set(out.p, "sign", PyLong_FromLong(*op.sign));
    if(op.record) phase_set(out.p, "record", PyUnicode_FromString(op.record->c_str()));
    if(op.pauli) phase_set(out.p, "pauli", PyUnicode_FromString(op.pauli->c_str()));
    if(op.paulis) phase_set(out.p, "paulis", PyUnicode_FromString(op.paulis->c_str()));
    return Py_NewRef(out.p);
}
