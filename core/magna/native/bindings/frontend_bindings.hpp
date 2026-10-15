// Only source/value conversion and opaque native data ownership live here.
std::string file_text(PyObject* raw) {
    if(!PyUnicode_Check(raw)) throw std::invalid_argument("string required");
    Py_ssize_t size;
    const char* value = PyUnicode_AsUTF8AndSize(raw, &size);
    if(!value) throw std::invalid_argument("UTF-8 conversion failed");
    return {value, size_t(size)};
}
PyObject* output_source(const SourceLocation& s) {
    Object repeat(PyTuple_New(s.repeat.size()));
    for(size_t j = 0; j < s.repeat.size(); j++) PyTuple_SET_ITEM(repeat.p, j, PyLong_FromSize_t(s.repeat[j]));
    return Py_BuildValue("(snnO)", s.filename.c_str(), Py_ssize_t(s.line), Py_ssize_t(s.column), repeat.p);
}
PyObject* output_target(const FileTarget& t) {
    return Py_BuildValue("(snLOs)", t.kind.c_str(), Py_ssize_t(t.index), (long long)t.version,
            t.inverted ? Py_True : Py_False, t.pauli.c_str());
}
FileTarget read_target(PyObject* raw) {
    Seq value(raw);
    if(value.size() != 5) throw std::invalid_argument("target shape");
    return {file_text(value[0]), natural(value[1]), integer(value[2]), PyObject_IsTrue(value[3]) != 0,
            file_text(value[4])};
}
PyObject* output_targets(const std::vector<FileTarget>& targets) {
    Object out(PyTuple_New(targets.size()));
    for(size_t j = 0; j < targets.size(); j++) PyTuple_SET_ITEM(out.p, j, output_target(targets[j]));
    Py_INCREF(out.p);
    return out.p;
}
PyObject* output_expression(const AffineExpression& expr) {
    Object terms(output_targets(std::vector<FileTarget>(expr.terms.begin(), expr.terms.end())));
    return Py_BuildValue("(iO)", expr.constant, terms.p);
}
AffineExpression read_expression(PyObject* raw) {
    Seq expr(raw);
    if(expr.size() != 2) throw std::invalid_argument("expression shape");
    AffineExpression out;
    out.constant = index_int(expr[0]);
    Seq terms(expr[1]);
    for(size_t j = 0; j < terms.size(); j++) out.terms.insert(read_target(terms[j]));
    return out;
}
PyObject* output_instruction(const FileInstruction& inst) {
    Object args(PyTuple_New(inst.args.size()));
    for(size_t j = 0; j < inst.args.size(); j++) {
        Object n(output_big(inst.args[j].n)), d(output_big(inst.args[j].d));
        PyTuple_SET_ITEM(args.p, j, PyTuple_Pack(2, n.p, d.p));
    }
    Object targets(output_targets(inst.targets)), source(output_source(inst.source)),
            outputs(output_targets(inst.outputs));
    Object records(PyTuple_New(inst.records.size()));
    for(size_t j = 0; j < inst.records.size(); j++) PyTuple_SET_ITEM(records.p, j, PyLong_FromSize_t(inst.records[j]));
    Object control(inst.control_name.empty() ? Py_NewRef(Py_None) : PyUnicode_FromString(inst.control_name.c_str()));
    Object guard(inst.guard ? output_target(*inst.guard) : Py_NewRef(Py_None));
    Object products(PyTuple_New(inst.products.size()));
    for(size_t j = 0; j < inst.products.size(); j++) PyTuple_SET_ITEM(products.p, j, output_targets(inst.products[j]));
    Object chain(inst.chain_id.empty() ? Py_NewRef(Py_None) : PyUnicode_FromString(inst.chain_id.c_str()));
    return Py_BuildValue("(sOOOsOOOOOO)", inst.name.c_str(), args.p, targets.p, source.p, inst.op_id.c_str(), outputs.p,
            records.p, control.p, guard.p, products.p, chain.p);
}
CircuitData* circuit_handle(PyObject* cap) {
    return static_cast<CircuitData*>(PyCapsule_GetPointer(cap, "magic-circuit-data-v1"));
}
void circuit_destructor(PyObject* cap) { delete circuit_handle(cap); }
PyObject* output_circuit(std::unique_ptr<CircuitData> circuit) {
    Object data(PyDict_New()), instructions(PyTuple_New(circuit->instructions.size()));
    for(size_t j = 0; j < circuit->instructions.size(); j++)
        PyTuple_SET_ITEM(instructions.p, j, output_instruction(circuit->instructions[j]));
    if(PyDict_SetItemString(data.p, "instructions", instructions.p) < 0) throw std::bad_alloc();
    phase_set(data.p, "num_records", PyLong_FromSize_t(circuit->num_records));
    phase_set(data.p, "num_qubits", PyLong_FromSize_t(circuit->num_qubits));
    Object affine(PyTuple_New(circuit->affine.size()));
    size_t pos = 0;
    for(const auto& [t, e] : circuit->affine) {
        Object target(output_target(t)), expr(output_expression(e));
        PyTuple_SET_ITEM(affine.p, pos++, PyTuple_Pack(2, target.p, expr.p));
    }
    if(PyDict_SetItemString(data.p, "affine", affine.p) < 0) throw std::bad_alloc();
    Object detectors(PyTuple_New(circuit->detectors.size())), required(PyTuple_New(circuit->required.size())),
            external(PyTuple_New(circuit->external.size()));
    for(size_t j = 0; j < circuit->detectors.size(); j++)
        PyTuple_SET_ITEM(detectors.p, j, output_expression(circuit->detectors[j]));
    for(size_t j = 0; j < circuit->required.size(); j++)
        PyTuple_SET_ITEM(required.p, j, output_expression(circuit->required[j]));
    for(size_t j = 0; j < circuit->external.size(); j++)
        PyTuple_SET_ITEM(external.p, j, PyLong_FromSize_t(circuit->external[j]));
    if(PyDict_SetItemString(data.p, "detectors", detectors.p) < 0 ||
            PyDict_SetItemString(data.p, "required", required.p) < 0 ||
            PyDict_SetItemString(data.p, "external", external.p) < 0)
        throw std::bad_alloc();
    phase_set(data.p, "text", PyUnicode_FromString(circuit->text.c_str()));
    phase_set(data.p, "source", PyUnicode_FromString(circuit->source.c_str()));
    phase_set(data.p, "format", PyUnicode_FromString(circuit->format_version().c_str()));
    Object cap(PyCapsule_New(circuit.release(), "magic-circuit-data-v1", circuit_destructor));
    return PyTuple_Pack(2, cap.p, data.p);
}
PyObject* parse_file(PyObject*, PyObject* args) {
    try {
        PyObject *text, *source, *controllers, *limit, *qubits;
        if(!PyArg_ParseTuple(args, "OOOOO", &text, &source, &controllers, &limit, &qubits)) return nullptr;
        auto contents = file_text(text), filename = file_text(source);
        size_t max_instructions = natural(limit), max_qubits = natural(qubits);
        Seq names(controllers);
        std::set<std::string> registry;
        for(size_t j = 0; j < names.size(); j++) registry.insert(file_text(names[j]));
        auto circuit = std::make_unique<CircuitData>();
        try {
            *circuit = native_work(
                    [&]() { return parse_circuit(contents, filename, registry, max_instructions, max_qubits); });
        } catch(const CircuitSyntaxError& e) {
            Object location(output_source(e.source));
            return Py_BuildValue("(O{s:s,s:O})", Py_None, "error", e.what(), "source", location.p);
        }
        return output_circuit(std::move(circuit));
    } catch(...) { return error(); }
}
PyObject* file_expression(PyObject*, PyObject* args) {
    try {
        const char* operation;
        PyObject *a, *b = Py_None, *c = Py_None;
        if(!PyArg_ParseTuple(args, "sO|OO", &operation, &a, &b, &c)) return nullptr;
        std::string op(operation);
        if(op == "expression") {
            auto* data = circuit_handle(a);
            if(!data) return nullptr;
            return output_expression(data->expression(read_target(b)));
        }
        if(op == "resolve") {
            CircuitData data;
            Seq affine(b), detectors(c);
            for(size_t j = 0; j < affine.size(); j++) {
                Seq row(affine[j]);
                if(row.size() != 2) throw std::invalid_argument("affine map shape");
                data.affine[read_target(row[0])] = read_expression(row[1]);
            }
            for(size_t j = 0; j < detectors.size(); j++) data.detectors.push_back(read_expression(detectors[j]));
            return output_expression(data.expression(read_target(a)));
        }
        if(op == "format") {
            auto* data = circuit_handle(a);
            if(!data) return nullptr;
            return PyUnicode_FromString(data->format_version().c_str());
        }
        throw std::invalid_argument("unknown file expression operation");
    } catch(...) { return error(); }
}
