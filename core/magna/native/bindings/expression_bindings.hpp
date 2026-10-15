// Immutable declaration handles and explicit value conversion.
int read_report_value(PyObject* raw) { return PyBool_Check(raw) ? int(raw == Py_True) : index_int(raw); }
Expression* expression_handle(PyObject* cap) {
    return static_cast<Expression*>(PyCapsule_GetPointer(cap, "magic-expression-v1"));
}
void expression_destructor(PyObject* cap) { delete expression_handle(cap); }
Expression read_declared_expression(PyObject* cap) {
    auto* expression = expression_handle(cap);
    if(!expression) throw std::invalid_argument("expression handle required");
    return *expression;
}
PyObject* make_expression(PyObject*, PyObject* args) {
    try {
        const char* kind;
        PyObject* raw;
        if(!PyArg_ParseTuple(args, "sO", &kind, &raw)) return nullptr;
        Seq children(raw);
        static const std::map<std::string, int> codes{
                {"const", 0}, {"record", 1}, {"not", 2}, {"xor", 3}, {"and", 4}, {"or", 5}, {"add", 6}, {"mul", 7}};
        auto found = codes.find(kind);
        if(found == codes.end()) throw std::invalid_argument("invalid control expression");
        int code = found->second;
        if(children.size() != size_t(code <= 2 ? 1 : 2)) throw std::invalid_argument("invalid control expression");
        Expression out;
        if(code == 0) {
            auto value = big(children[0]);
            int reduced = value.zero() ? 0 : int(value.d[0] & 7);
            out = constant_expression(value.neg ? -reduced : reduced);
        } else if(code == 1) out = record_expression(file_text(children[0]));
        else {
            std::vector<Expression> parts;
            for(size_t j = 0; j < children.size(); j++) parts.push_back(read_declared_expression(children[j]));
            out = composed_expression(code, std::move(parts));
        }
        auto owned = std::make_unique<Expression>(out);
        Object cap(PyCapsule_New(owned.release(), "magic-expression-v1", expression_destructor));
        return Py_BuildValue("(Oi)", cap.p, out->value);
    } catch(...) { return error(); }
}
PyObject* expression_operation(PyObject*, PyObject* args) {
    try {
        const char* operation;
        PyObject *cap, *raw = Py_None;
        if(!PyArg_ParseTuple(args, "sO|O", &operation, &cap, &raw)) return nullptr;
        auto e = read_declared_expression(cap);
        std::string op(operation);
        if(op == "records") {
            auto records = e->records();
            Object out(PyTuple_New(records.size()));
            size_t j = 0;
            for(const auto& r : records) PyTuple_SET_ITEM(out.p, j++, PyUnicode_FromString(r.c_str()));
            return Py_NewRef(out.p);
        }
        if(op == "evaluate") {
            Seq rows(raw);
            std::map<std::string, int> values;
            for(size_t j = 0; j < rows.size(); j++) {
                Seq row(rows[j]);
                if(row.size() != 2) throw std::invalid_argument("record value shape");
                values[file_text(row[0])] = read_report_value(row[1]);
            }
            return PyLong_FromLong(e->evaluate(values));
        }
        throw std::invalid_argument("unknown expression operation");
    } catch(...) { return error(); }
}
std::vector<DeclaredGate> read_declared_gates(PyObject* raw) {
    Seq rows(raw);
    std::vector<DeclaredGate> gates;
    for(size_t j = 0; j < rows.size(); j++) {
        Seq row(rows[j]);
        if(row.size() != 4) throw std::invalid_argument("declared gate shape");
        DeclaredGate gate;
        gate.kind = file_text(row[0]);
        gate.qubits = intvec(row[1]);
        if(row[2] != Py_None) gate.exponent = read_declared_expression(row[2]);
        gate.guard = read_declared_expression(row[3]);
        gates.push_back(std::move(gate));
    }
    return gates;
}
Operation read_operation(PyObject* raw) {
    Operation op;
    op.id = file_text(field(raw, "id"));
    op.kind = file_text(field(raw, "kind"));
    if(auto* qs = PyDict_GetItemString(raw, "qubits")) op.qubits = read_sizes(qs);
    if(auto* record = PyDict_GetItemString(raw, "record")) op.record = file_text(record);
    if(auto* sign = PyDict_GetItemString(raw, "sign")) op.sign = index_int(sign);
    if(auto* paulis = PyDict_GetItemString(raw, "paulis")) op.paulis = file_text(paulis);
    if(auto* pauli = PyDict_GetItemString(raw, "pauli")) op.pauli = file_text(pauli);
    if(auto* exponent = PyDict_GetItemString(raw, "exponent")) op.exponent = index_int(exponent);
    return op;
}
PyObject* defer_operations(PyObject*, PyObject* args) {
    try {
        PyObject *inputs, *outputs, *operations, *controls, *flips, *expose, *preserved;
        if(!PyArg_ParseTuple(args, "OOOOOOO", &inputs, &outputs, &operations, &controls, &flips, &expose, &preserved))
            return nullptr;
        auto in = read_sizes(inputs), out = read_sizes(outputs);
        Seq raw(operations), rules(controls), fs(flips), ps(preserved);
        std::vector<Operation> ops;
        for(size_t j = 0; j < raw.size(); j++) ops.push_back(read_operation(raw[j]));
        std::map<std::string, std::optional<std::vector<DeclaredGate>>> declarations;
        for(size_t j = 0; j < rules.size(); j++) {
            Seq row(rules[j]);
            if(row.size() != 2) throw std::invalid_argument("control rule shape");
            auto name = file_text(row[0]);
            if(row[1] == Py_None) declarations[name] = {};
            else declarations[name] = read_declared_gates(row[1]);
        }
        std::vector<std::string> flip_names;
        for(size_t j = 0; j < fs.size(); j++) flip_names.push_back(file_text(fs[j]));
        std::set<std::string> preserve_names;
        for(size_t j = 0; j < ps.size(); j++) preserve_names.insert(file_text(ps[j]));
        int expose_faults = PyObject_IsTrue(expose);
        if(expose_faults < 0) throw std::invalid_argument("report exposure flag");
        auto result = native_work([&]() {
            return defer_instrument(in, out, ops, declarations, flip_names, expose_faults, preserve_names);
        });
        Object result_ops(PyTuple_New(result.operations.size())), records(PyTuple_New(result.records.size())),
                anchors(PyTuple_New(result.report_anchors.size()));
        for(size_t j = 0; j < result.operations.size(); j++) {
            const auto& op = result.operations[j];
            PyTuple_SET_ITEM(
                    result_ops.p, j, op.original ? PyLong_FromSize_t(*op.original) : output_operation(op.value));
        }
        for(size_t j = 0; j < result.records.size(); j++) {
            const auto& [name, wire] = result.records[j];
            PyTuple_SET_ITEM(records.p, j, Py_BuildValue("(sn)", name.c_str(), Py_ssize_t(wire)));
        }
        for(size_t j = 0; j < result.report_anchors.size(); j++) {
            const auto& [name, id, wire] = result.report_anchors[j];
            PyTuple_SET_ITEM(anchors.p, j, Py_BuildValue("(ssn)", name.c_str(), id.c_str(), Py_ssize_t(wire)));
        }
        return PyTuple_Pack(3, result_ops.p, records.p, anchors.p);
    } catch(const UnsupportedFeedback& e) {
        PyErr_SetString(PyExc_NotImplementedError, e.what());
        return nullptr;
    } catch(...) { return error(); }
}
PyObject* declared_rule_operation(PyObject*, PyObject* args) {
    try {
        const char* operation;
        PyObject *raw, *values = Py_None;
        if(!PyArg_ParseTuple(args, "sO|O", &operation, &raw, &values)) return nullptr;
        auto gates = read_declared_gates(raw);
        std::string op(operation);
        if(op == "validate") {
            for(const auto& gate : gates) validate_declared_gate(gate);
            Py_RETURN_NONE;
        }
        if(op == "records") {
            auto names = declared_records(gates);
            Object out(PyTuple_New(names.size()));
            size_t j = 0;
            for(const auto& name : names) PyTuple_SET_ITEM(out.p, j++, PyUnicode_FromString(name.c_str()));
            return Py_NewRef(out.p);
        }
        Seq rows(values);
        if(op == "evaluate") {
            std::map<std::string, int> reports;
            for(size_t j = 0; j < rows.size(); j++) {
                Seq row(rows[j]);
                if(row.size() != 2) throw std::invalid_argument("record value shape");
                reports[file_text(row[0])] = read_report_value(row[1]);
            }
            auto selected = evaluate_declared_rule(gates, reports);
            Object out(PyList_New(selected.size()));
            for(size_t j = 0; j < selected.size(); j++) {
                const auto& gate = selected[j];
                Object qs(PyList_New(gate.qubits.size()));
                for(size_t k = 0; k < gate.qubits.size(); k++)
                    PyList_SET_ITEM(qs.p, k, PyLong_FromLong(gate.qubits[k]));
                Object item(Py_BuildValue("{s:s,s:O}", "kind", gate.kind.c_str(), "qubits", qs.p));
                if(gate.exponent) phase_set(item.p, "exponent", PyLong_FromLong(gate.exponent->value));
                PyList_SET_ITEM(out.p, j, Py_NewRef(item.p));
            }
            return Py_NewRef(out.p);
        }
        throw std::invalid_argument("unknown declared rule operation");
    } catch(const MissingRecord& e) {
        PyErr_SetString(PyExc_KeyError, e.what());
        return nullptr;
    } catch(const UnsupportedFeedback& e) {
        PyErr_SetString(PyExc_NotImplementedError, e.what());
        return nullptr;
    } catch(...) { return error(); }
}
