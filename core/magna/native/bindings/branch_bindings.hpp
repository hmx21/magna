// Callback objects and arbitrary annotation snapshots stay at the language boundary.
struct BranchScheduleBinding {
    std::unique_ptr<BranchSchedule> schedule;
    PyObject* callbacks = nullptr;
    PyObject* copy = nullptr;
    ~BranchScheduleBinding() {
        Py_XDECREF(callbacks);
        Py_XDECREF(copy);
    }
};
BranchScheduleBinding* branch_schedule(PyObject* cap) {
    return static_cast<BranchScheduleBinding*>(PyCapsule_GetPointer(cap, "magic-branch-schedule-v1"));
}
void branch_schedule_destructor(PyObject* cap) { delete branch_schedule(cap); }
struct CollectedBranchBinding {
    CollectedBranch branch;
    PyObject* owner = nullptr;
    std::vector<PyObject*> gates;
    ~CollectedBranchBinding() {
        Py_XDECREF(owner);
        for(auto* value : gates) Py_XDECREF(value);
    }
};
CollectedBranchBinding* collected_branch(PyObject* cap) {
    return static_cast<CollectedBranchBinding*>(PyCapsule_GetPointer(cap, "magic-collected-branch-v1"));
}
void collected_branch_destructor(PyObject* cap) { delete collected_branch(cap); }
PyObject* make_branch_schedule(PyObject*, PyObject* args) {
    try {
        PyObject *cap, *raw, *callbacks;
        if(!PyArg_ParseTuple(args, "OOO", &cap, &raw, &callbacks)) return nullptr;
        auto* layout = instrument_layout(cap);
        if(!layout) return nullptr;
        Seq rows(raw);
        std::map<std::string, std::optional<std::vector<DeclaredGate>>> rules;
        for(size_t j = 0; j < rows.size(); j++) {
            Seq row(rows[j]);
            if(row.size() != 2) throw std::invalid_argument("branch declaration shape");
            rules[file_text(row[0])] =
                    row[1] == Py_None ? std::nullopt : std::make_optional(read_declared_gates(row[1]));
        }
        auto owner = std::make_unique<BranchScheduleBinding>();
        owner->schedule = std::make_unique<BranchSchedule>(*layout, std::move(rules));
        owner->callbacks = Py_NewRef(callbacks);
        Object copy_module(PyImport_ImportModule("copy"));
        owner->copy = PyObject_GetAttrString(copy_module.p, "deepcopy");
        if(!owner->copy) throw std::invalid_argument("copy adapter unavailable");
        return PyCapsule_New(owner.release(), "magic-branch-schedule-v1", branch_schedule_destructor);
    } catch(...) { return execution_error(); }
}
PyObject* output_named_bits(const std::vector<std::pair<std::string, int>>& values) {
    Object out(PyDict_New());
    for(const auto& [name, bit] : values) phase_set(out.p, name.c_str(), PyLong_FromLong(bit));
    return Py_NewRef(out.p);
}
PyObject* output_branch_gate(const BranchGate& gate) {
    const auto& op = gate.operation.value;
    Object qs(PyList_New(op.qubits.size()));
    for(size_t j = 0; j < op.qubits.size(); j++) PyList_SET_ITEM(qs.p, j, PyLong_FromLong(op.qubits[j]));
    Object out(Py_BuildValue("{s:s,s:O}", "kind", op.kind.c_str(), "qubits", qs.p));
    if(gate.explicit_exponent) phase_set(out.p, "exponent", PyLong_FromLong(op.exponent));
    if(!op.pauli.empty()) phase_set(out.p, "pauli", PyUnicode_FromString(op.pauli.c_str()));
    return Py_NewRef(out.p);
}
std::vector<BranchGate> callback_gate_identities(PyObject* raw) {
    Seq rows(raw);
    std::vector<BranchGate> values;
    for(size_t j = 0; j < rows.size(); j++) {
        auto op = branch_gate(file_text(field(rows[j], "kind")));
        if(PyDict_GetItemString(rows[j], "record")) op.operation.record = "present";
        values.push_back(std::move(op));
    }
    return values;
}
PyObject* output_collected_data(CollectedBranchBinding& result, BranchScheduleBinding& owner) {
    Object decisions(PyTuple_New(result.branch.decisions.size()));
    for(size_t j = 0; j < result.branch.decisions.size(); j++) {
        const auto& decision = result.branch.decisions[j];
        if(!result.gates[j]) {
            Object gates(PyTuple_New(decision.gates.size()));
            for(size_t k = 0; k < decision.gates.size(); k++)
                PyTuple_SET_ITEM(gates.p, k, output_branch_gate(decision.gates[k]));
            result.gates[j] = Py_NewRef(gates.p);
        }
        Object parents(output_named_bits(decision.parents));
        PyTuple_SET_ITEM(decisions.p, j,
                Py_BuildValue("{s:s,s:s,s:O,s:O}", "id", decision.id.c_str(), "version", decision.version.c_str(),
                        "parents", parents.p, "gates", result.gates[j]));
    }
    std::vector<std::pair<std::string, int>> measured, reported;
    for(size_t j = 0; j < owner.schedule->layout.record_names.size(); j++) {
        const auto& name = owner.schedule->layout.record_names[j];
        measured.push_back({name, int(result.branch.true_word.bit(j))});
        reported.push_back({name, int(result.branch.reported_word.bit(j))});
    }
    Object actual(output_named_bits(measured)), reports(output_named_bits(reported)),
            word(output_bits(result.branch.true_word)), report_word(output_bits(result.branch.reported_word)),
            ordered(output_sizes(result.branch.ordered_faults));
    Object data(Py_BuildValue("{s:O,s:O,s:O,s:O,s:O,s:O}", "measured", actual.p, "reported", reports.p, "true_word",
            word.p, "reported_word", report_word.p, "decisions", decisions.p, "fault_order", ordered.p));
    return Py_NewRef(data.p);
}
PyObject* collect_branch(PyObject*, PyObject* args) {
    try {
        PyObject *cap, *truth, *flips, *faults;
        if(!PyArg_ParseTuple(args, "OOOO", &cap, &truth, &flips, &faults)) return nullptr;
        auto* owner = branch_schedule(cap);
        if(!owner) return nullptr;
        auto original = read_original_faults(faults);
        Seq true_rows(truth), flip_rows(flips);
        std::vector<std::pair<std::string, int>> true_values;
        std::vector<std::string> flip_names;
        for(size_t j = 0; j < true_rows.size(); j++) {
            Seq row(true_rows[j]);
            if(row.size() != 2) throw std::invalid_argument("true record shape");
            true_values.push_back({file_text(row[0]), index_int(row[1])});
        }
        for(size_t j = 0; j < flip_rows.size(); j++) flip_names.push_back(file_text(flip_rows[j]));
        auto result = std::make_unique<CollectedBranchBinding>();
        result->owner = Py_NewRef(cap);
        result->gates.resize(owner->schedule->controls.size(), nullptr);
        auto callback = [&](size_t index, const std::vector<std::pair<std::string, int>>& parents) {
            CallbackGIL gil;
            const auto& id = owner->schedule->layout.operations[owner->schedule->controls[index]].id;
            Object values(output_named_bits(parents)), permitted(PyDictProxy_New(values.p));
            Object raw(PyObject_CallOneArg(field(owner->callbacks, id.c_str()), permitted.p));
            Object sequence(PySequence_Tuple(raw.p)), owned(PyObject_CallOneArg(owner->copy, sequence.p));
            auto gates = callback_gate_identities(owned.p);
            result->gates[index] = Py_NewRef(owned.p);
            return gates;
        };
        result->branch = native_work([&]() {
            auto assignment = owner->schedule->layout.validate(true_values, flip_names, original);
            return owner->schedule->collect(assignment, original, callback);
        });
        Object data(output_collected_data(*result, *owner));
        Object handle(PyCapsule_New(result.release(), "magic-collected-branch-v1", collected_branch_destructor));
        return PyTuple_Pack(2, handle.p, data.p);
    } catch(...) { return execution_error(); }
}
PyObject* materialize_collected_branch(PyObject*, PyObject* args) {
    try {
        PyObject *cap, *raw;
        if(!PyArg_ParseTuple(args, "OO", &cap, &raw)) return nullptr;
        auto* branch = collected_branch(cap);
        if(!branch) return nullptr;
        auto* owner = branch_schedule(branch->owner);
        if(!owner) return nullptr;
        Seq original(raw);
        Object ops(PyTuple_New(branch->branch.steps.size()));
        for(size_t j = 0; j < branch->branch.steps.size(); j++) {
            const auto& step = branch->branch.steps[j];
            PyObject* source = nullptr;
            if(step.replacement) source = output_branch_gate(*step.replacement);
            else if(step.kind == 0) source = PyObject_CallOneArg(owner->copy, original[step.index]);
            else {
                Seq gates(branch->gates.at(step.index));
                source = PyObject_CallOneArg(owner->copy, gates[step.subindex]);
            }
            Object value(source);
            phase_set(value.p, "id", PyUnicode_FromString(step.id.c_str()));
            if(step.measured) phase_set(value.p, "true", PyLong_FromLong(*step.measured));
            if(step.reported) phase_set(value.p, "reported", PyLong_FromLong(*step.reported));
            PyTuple_SET_ITEM(ops.p, j, Py_NewRef(value.p));
        }
        owner->schedule->statistics.materializations++;
        return Py_NewRef(ops.p);
    } catch(...) { return execution_error(); }
}
