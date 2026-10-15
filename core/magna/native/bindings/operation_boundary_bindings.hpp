struct OperationBoundaryBinding {
    std::unique_ptr<OperationBoundarySampler> sampler;
    PyObject* callbacks = nullptr;
    PyObject* copy = nullptr;
    ~OperationBoundaryBinding() {
        Py_XDECREF(callbacks);
        Py_XDECREF(copy);
    }
};
OperationBoundaryBinding* operation_boundary(PyObject* cap) {
    return static_cast<OperationBoundaryBinding*>(PyCapsule_GetPointer(cap, "magic-operation-boundary-v1"));
}
void operation_boundary_destructor(PyObject* cap) { delete operation_boundary(cap); }
PyObject* operation_boundary_error() {
    try {
        throw;
    } catch(const UnsupportedFeedback& e) {
        PyErr_SetString(PyExc_NotImplementedError, e.what());
        return nullptr;
    } catch(const ControlMismatch& e) {
        // Raw calls: this handler must not throw. Fall back to RuntimeError if the type is unavailable.
        PyObject* module = PyImport_ImportModule("magna.feedback");
        PyObject* type = module ? PyObject_GetAttrString(module, "FeedbackMismatch") : nullptr;
        PyErr_Clear();
        PyErr_SetString(type ? type : PyExc_RuntimeError, e.what());
        Py_XDECREF(type);
        Py_XDECREF(module);
        return nullptr;
    } catch(...) { return execution_error(); }
}
NativeControl read_feedback_family(PyObject* raw) {
    Seq family(raw);
    if(family.size() != 3) throw std::invalid_argument("feedback family shape");
    auto axes = [](PyObject* raw) {
        Seq rows(raw);
        std::vector<std::vector<int>> out;
        for(size_t j = 0; j < rows.size(); j++) out.push_back(intvec(rows[j]));
        return out;
    };
    return feedback_family(intvec(family[0]), axes(family[1]), axes(family[2]));
}
PyObject* validate_phase_feedback(PyObject*, PyObject* family) {
    try {
        read_feedback_family(family);
        Py_RETURN_NONE;
    } catch(...) { return error(); }
}
Operation read_boundary_operation(PyObject* raw, bool original) {
    Object data(PyDict_Copy(raw));
    if(!original) phase_set(data.p, "id", PyUnicode_FromString(""));
    // Numerical phase values are represented modulo eight, just as in compiler input.
    auto kind = file_text(field(data.p, "kind"));
    if(kind == "T" || kind == "S" || kind == "PHASE") {
        auto* value = PyDict_GetItemString(data.p, "actual_exponent");
        if(!value) value = PyDict_GetItemString(data.p, "exponent");
        if(value) phase_set(data.p, "exponent", PyLong_FromLong(phase_exponent(value)));
    }
    if(!original && PyDict_GetItemString(data.p, "record"))
        throw std::invalid_argument("control expansion must be measurement-free");
    return read_operation(data.p);
}
PyObject* make_operation_boundary(PyObject*, PyObject* args) {
    try {
        PyObject *cap, *raw_initial, *raw_inputs, *raw_outputs, *raw_operations, *raw_controls, *raw_task, *raw_limits,
                *raw_database, *callbacks;
        if(!PyArg_ParseTuple(args, "OOOOOOOOOO", &cap, &raw_initial, &raw_inputs, &raw_outputs, &raw_operations,
                   &raw_controls, &raw_task, &raw_limits, &raw_database, &callbacks))
            return nullptr;
        auto* layout = instrument_layout(cap);
        if(!layout) return nullptr;
        auto initial = ket(raw_initial);
        auto inputs = read_sizes(raw_inputs), outputs = read_sizes(raw_outputs);
        Seq ops(raw_operations), controls(raw_controls);
        std::vector<Operation> operations;
        for(size_t j = 0; j < ops.size(); j++) operations.push_back(read_boundary_operation(ops[j], true));
        std::map<std::string, std::optional<std::vector<DeclaredGate>>> declarations;
        std::map<std::string, std::optional<NativeControl>> families;
        for(size_t j = 0; j < controls.size(); j++) {
            Seq row(controls[j]);
            if(row.size() != 3) throw std::invalid_argument("control boundary declaration shape");
            auto id = file_text(row[0]);
            declarations[id] = row[1] == Py_None ? std::nullopt : std::make_optional(read_declared_gates(row[1]));
            families[id] = row[2] == Py_None ? std::nullopt : std::make_optional(read_feedback_family(row[2]));
        }
        Seq task_rows(raw_task);
        if(task_rows.size() != 3) throw std::invalid_argument("boundary task shape");
        OperationBoundaryTask task;
        task.selection = read_named_rows(task_rows[0]);
        task.outcomes = intvec(task_rows[1]);
        Seq observables(task_rows[2]);
        for(size_t j = 0; j < observables.size(); j++) {
            Seq row(observables[j]);
            if(row.size() != 2) throw std::invalid_argument("observable shape");
            task.observables.push_back({bits(row[0]), bits(row[1])});
        }
        BoundaryLimits limits;
        limits.phase = phase_limits(raw_limits);
        if(auto* value = PyDict_GetItemString(raw_limits, "max_path_bits")) limits.paths = natural(value);
        if(auto* value = PyDict_GetItemString(raw_limits, "max_doubled_path_bits")) limits.doubled = natural(value);
        limits.plans = natural(field(raw_limits, "max_plans"));
        auto database = phase_database(raw_database);
        auto owner = std::make_unique<OperationBoundaryBinding>();
        owner->callbacks = Py_NewRef(callbacks);
        Object copy(PyImport_ImportModule("copy"));
        owner->copy = PyObject_GetAttrString(copy.p, "deepcopy");
        if(!owner->copy) throw std::invalid_argument("copy adapter unavailable");
        owner->sampler = native_work([&]() {
            return std::make_unique<OperationBoundarySampler>(
                    *layout, initial, inputs, outputs, operations, declarations, families, task, limits, database);
        });
        const auto& sampler = *owner->sampler;
        Object metadata(PyDict_New());
        phase_set(metadata.p, "direct", output_names(sampler.direct));
        phase_set(metadata.p, "complex", output_names(sampler.complex));
        phase_set(metadata.p, "required", output_names(sampler.required));
        phase_set(metadata.p, "layout", output_boundary_coordinates(sampler.layout.record_names, *sampler.selection));
        Object handle(PyCapsule_New(owner.release(), "magic-operation-boundary-v1", operation_boundary_destructor));
        return PyTuple_Pack(2, handle.p, metadata.p);
    } catch(...) { return operation_boundary_error(); }
}
PyObject* sample_operation_boundary(PyObject*, PyObject* args) {
    try {
        PyObject *cap, *raw_faults, *raw_flips, *rng;
        if(!PyArg_ParseTuple(args, "OOOO", &cap, &raw_faults, &raw_flips, &rng)) return nullptr;
        auto* owner = operation_boundary(cap);
        if(!owner) return nullptr;
        auto faults = read_original_faults(raw_faults);
        auto flips = input_names(raw_flips);
        auto callback = [&](const std::string& id, const std::vector<std::pair<std::string, int>>& parents) {
            CallbackGIL gil;
            Object values(output_named_bits(parents)), permitted(PyDictProxy_New(values.p)),
                    raw(PyObject_CallOneArg(field(owner->callbacks, id.c_str()), permitted.p));
            Object tuple(PySequence_Tuple(raw.p)), owned(PyObject_CallOneArg(owner->copy, tuple.p));
            Seq rows(owned.p);
            std::vector<Operation> out;
            for(size_t j = 0; j < rows.size(); j++) out.push_back(read_boundary_operation(rows[j], false));
            return out;
        };
        auto draw = native_work([&]() {
            return owner->sampler->sample(
                    faults, flips,
                    [&](size_t count) {
                        CallbackGIL gil;
                        return random_bits(rng)(count);
                    },
                    callback);
        });
        Object result(PyDict_New()), moments(PyTuple_New(draw.moments.size())), stages(PyList_New(draw.stages.size())),
                resources(PyList_New(draw.resources.size()));
        phase_set(result.p, "mass", output_scalar(draw.mass));
        phase_set(result.p, "acceptance", output_scalar(draw.acceptance));
        phase_set(result.p, "error", draw.error ? output_scalar(*draw.error) : Py_NewRef(Py_None));
        for(size_t j = 0; j < draw.moments.size(); j++) PyTuple_SET_ITEM(moments.p, j, output_scalar(draw.moments[j]));
        phase_set(result.p, "moments", Py_NewRef(moments.p));
        phase_set(result.p, "reports", output_named_bits(draw.reports));
        phase_set(result.p, "true", output_named_bits(draw.truth));
        for(size_t j = 0; j < draw.stages.size(); j++) {
            const auto& stage = draw.stages[j];
            Object info(output_boundary_sample_info(stage.sample));
            phase_set(info.p, "control", PyUnicode_FromString(stage.id.c_str()));
            phase_set(info.p, "records", output_names(stage.records));
            phase_set(info.p, "completion", PyUnicode_FromString("whole-instrument unitary future; no postselection"));
            PyList_SET_ITEM(stages.p, j, Py_NewRef(info.p));
        }
        for(size_t j = 0; j < draw.resources.size(); j++)
            PyList_SET_ITEM(resources.p, j, output_query_measurement(draw.resources[j]));
        Object diagnostic(Py_BuildValue("{s:s,s:O,s:O,s:n,s:n,s:n}", "route",
                owner->sampler->required.empty() ? "direct" : "complex-feedback", "stages", stages.p, "final_resources",
                resources.p, "controls_executed", Py_ssize_t(draw.controls), "sampled_records",
                Py_ssize_t(draw.reports.size()), "unsampled_records",
                Py_ssize_t(owner->sampler->layout.record_names.size() - draw.reports.size())));
        phase_set(result.p, "diagnostics", Py_NewRef(diagnostic.p));
        return Py_NewRef(result.p);
    } catch(...) { return operation_boundary_error(); }
}
PyObject* operation_boundary_plan_count(PyObject*, PyObject* cap) {
    try {
        auto* owner = operation_boundary(cap);
        if(!owner) return nullptr;
        std::lock_guard<std::mutex> lock(owner->sampler->mutex);
        return PyLong_FromSize_t(owner->sampler->plans.size());
    } catch(...) { return error(); }
}
