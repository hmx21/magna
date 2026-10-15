// CPython conversion only; include inside bindings.cpp after standard helpers.
std::string compiler_string(PyObject* p) {
    if(!PyUnicode_Check(p)) throw std::invalid_argument("compiler text required");
    Py_ssize_t length = 0;
    const char* t = PyUnicode_AsUTF8AndSize(p, &length);
    if(!t) throw std::invalid_argument("compiler text encoding");
    return std::string(t, size_t(length));
}
CompilerInput* compiler_input_handle(PyObject* cap) {
    return static_cast<CompilerInput*>(PyCapsule_GetPointer(cap, "magic-compiler-input-v1"));
}
void compiler_input_destructor(PyObject* cap) { delete compiler_input_handle(cap); }
CompilerIR parse_compiler(PyObject* d) {
    if(PyCapsule_CheckExact(d)) {
        auto* input = compiler_input_handle(d);
        if(!input) throw std::invalid_argument("compiler input handle required");
        return input->ir;
    }
    if(!PyDict_Check(d)) throw std::invalid_argument("compiler dictionary required");
    CompilerIR ir;
    ir.initial = ket(field(d, "initial"));
    ir.inputs = intvec(field(d, "inputs"));
    ir.outputs = intvec(field(d, "outputs"));
    ir.records = natural(field(d, "records"));
    ir.max_path_bits = natural(field(d, "max_path_bits"));
    if(ir.records > 1000000 || ir.max_path_bits > 1000000) throw std::invalid_argument("compiler dimension capacity");
    Seq ops(field(d, "operations"));
    for(size_t j = 0; j < ops.size(); j++) {
        Seq r(ops[j]);
        if(r.size() != 9) throw std::invalid_argument("compiler operation shape");
        CompilerOp op;
        op.kind = compiler_string(r[0]);
        op.qubits = intvec(r[1]);
        op.exponent = index_int(r[2]);
        op.record = index_int(r[3]);
        op.sign = index_int(r[4]);
        op.location = index_int(r[5]);
        op.feedback_slot = index_int(r[6]);
        op.pauli = compiler_string(r[7]);
        op.paulis = compiler_string(r[8]);
        if(op.record < -1 || op.location < -1 || op.feedback_slot < -1)
            throw std::invalid_argument("compiler metadata index");
        ir.operations.push_back(std::move(op));
    }
    return ir;
}
PyObject* compiler_bitvec(const std::vector<Bits>& vs) {
    Object out(PyTuple_New(vs.size()));
    for(size_t j = 0; j < vs.size(); j++) PyTuple_SET_ITEM(out.p, j, output_bits(vs[j]));
    Py_INCREF(out.p);
    return out.p;
}
PyObject* compiler_intvec(const std::vector<int>& vs) {
    Object out(PyTuple_New(vs.size()));
    for(size_t j = 0; j < vs.size(); j++) PyTuple_SET_ITEM(out.p, j, PyLong_FromLong(vs[j]));
    Py_INCREF(out.p);
    return out.p;
}
PyObject* compiler_frame(const Frame& v) {
    Object x(output_bits(v.x)), z(output_bits(v.z));
    return Py_BuildValue("(OOi)", x.p, z.p, v.p);
}
PyObject* compiler_frame(const StoredFrame& v) { return compiler_frame(v.materialize()); }
PyObject* output_compiler(const CompilerPreparation& c) {
    auto& p = c.plan;
    Object out(PyDict_New());
    auto set = [&](const char* k, PyObject* v) {
        Object ref(v);
        if(PyDict_SetItemString(out.p, k, v) < 0) throw std::bad_alloc();
    };
    set("skeleton", output_ket(p.skeleton));
    set("retained", PyLong_FromSize_t(p.retained));
    set("physical", PyLong_FromSize_t(p.physical));
    set("nrecords", PyLong_FromSize_t(p.nrecords));
    set("basis", compiler_bitvec(p.basis));
    set("polar", compiler_bitvec(p.polar));
    set("forms", compiler_bitvec(p.forms));
    set("magic", compiler_intvec(p.magic));
    set("linear", compiler_intvec(c.linear));
    set("upper", compiler_bitvec(c.upper));
    set("phase_controls", compiler_intvec(c.phase_controls));
    set("environment", compiler_intvec(c.environment));
    set("path_variables", PyLong_FromSize_t(c.path_variables));
    set("output_count", PyLong_FromSize_t(c.output_count));
    set("gate_count", PyLong_FromSize_t(c.gate_count));
    set("injective_physical", PyBool_FromLong(c.injective_physical));
    set("reference_layer", PyLong_FromLong(c.reference_layer));
    set("layer_slots", compiler_intvec(c.layer_slots));
    set("rotation_layers", PyLong_FromSize_t(c.rotation_layers));
    set("reference_layer_slots", PyLong_FromSize_t(c.reference_layer_slots));
    set("phase_ancillas", PyLong_FromSize_t(c.ancilla_count));
    set("shared_phase_coordinates", PyLong_FromSize_t(c.phase_coordinates));
    Object terms(PyTuple_New(c.magic_terms.size()));
    for(size_t j = 0; j < c.magic_terms.size(); j++) {
        Object a(output_bits(c.magic_terms[j].first));
        PyTuple_SET_ITEM(terms.p, j, Py_BuildValue("(Oi)", a.p, c.magic_terms[j].second));
    }
    if(PyDict_SetItemString(out.p, "magic_terms", terms.p) < 0) throw std::bad_alloc();
    Object loc(PyTuple_New(p.locations.size()));
    for(size_t j = 0; j < p.locations.size(); j++)
        PyTuple_SET_ITEM(loc.p, j, Py_BuildValue("(ii)", p.locations[j].index, p.locations[j].position));
    if(PyDict_SetItemString(out.p, "locations", loc.p) < 0) throw std::bad_alloc();
    Object hs(PyTuple_New(p.histories.size()));
    size_t j = 0;
    for(auto& [wire, values] : p.histories) {
        Object row(PyTuple_New(values.size()));
        for(size_t i = 0; i < values.size(); i++)
            PyTuple_SET_ITEM(row.p, i, Py_BuildValue("(ii)", values[i].first, values[i].second));
        PyTuple_SET_ITEM(hs.p, j++, Py_BuildValue("(iO)", wire, row.p));
    }
    if(PyDict_SetItemString(out.p, "histories", hs.p) < 0) throw std::bad_alloc();
    Object ss(PyTuple_New(p.suffix.size()));
    j = 0;
    for(auto& [wire, values] : p.suffix) {
        Object row(PyTuple_New(values.size()));
        for(size_t i = 0; i < values.size(); i++) {
            Object x(compiler_frame(values[i].x)), z(compiler_frame(values[i].z));
            PyTuple_SET_ITEM(row.p, i, Py_BuildValue("(iOO)", values[i].position, x.p, z.p));
        }
        PyTuple_SET_ITEM(ss.p, j++, Py_BuildValue("(iO)", wire, row.p));
    }
    if(PyDict_SetItemString(out.p, "suffix", ss.p) < 0) throw std::bad_alloc();
    Object times(Py_BuildValue("{s:d,s:d,s:d,s:d,s:d}", "dilation_seconds", c.dilation_seconds, "path_seconds",
            c.path_seconds, "contraction_seconds", c.contraction_seconds, "projection_seconds", c.projection_seconds,
            "fault_map_seconds", c.fault_map_seconds));
    if(PyDict_SetItemString(out.p, "timings", times.p) < 0) throw std::bad_alloc();
    Py_INCREF(out.p);
    return out.p;
}
