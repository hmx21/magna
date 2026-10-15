// Only Python conversion, callback invocation and annotation ownership live here.
struct FullRecordBinding {
    std::unique_ptr<FullRecordExecution> execution;
    PyObject* schedule = nullptr;
    ~FullRecordBinding() { Py_XDECREF(schedule); }
};
FullRecordBinding* full_record(PyObject* cap) {
    return static_cast<FullRecordBinding*>(PyCapsule_GetPointer(cap, "magic-full-record-v1"));
}
void full_record_destructor(PyObject* cap) { delete full_record(cap); }
PyObject* full_record_info(const FullRecordExecution& e) {
    Object info(e.global ? global_info(*e.global) : PyDict_New());
    phase_set(info.p, "literal_compilations", PyLong_FromSize_t(e.literal_compilations.load()));
    phase_set(info.p, "packed_fault_groups", PyLong_FromSize_t(e.packed_fault_groups.load()));
    if(!e.literal_reason.empty())
        phase_set(info.p, "literal_compilation_reason", PyUnicode_FromString(e.literal_reason.c_str()));
    return Py_NewRef(info.p);
}
PyObject* make_full_record(PyObject*, PyObject* args) {
    try {
        PyObject *schedule_cap, *initial, *inputs, *outputs, *operations, *families, *database, *limits;
        int reuse;
        if(!PyArg_ParseTuple(args, "OOOOOOOOp", &schedule_cap, &initial, &inputs, &outputs, &operations, &families,
                   &database, &limits, &reuse))
            return nullptr;
        auto* schedule = branch_schedule(schedule_cap);
        if(!schedule) return nullptr;
        auto k = ket(initial);
        auto in = intvec(inputs), out = intvec(outputs), ops_outcomes = std::vector<int>();
        auto ops = read_program_operations(operations);
        Seq original(operations), rules(families);
        for(size_t j = 0; j < original.size(); j++) {
            int value = 0;
            if(ops[j].value.kind.rfind("M", 0) == 0) {
                auto* p = PyDict_GetItemString(original[j], "true");
                if(!p) p = PyDict_GetItemString(original[j], "outcome");
                if(p) value = index_int(p);
            }
            ops_outcomes.push_back(value);
        }
        std::map<std::string, std::optional<NativeControl>> declarations;
        for(size_t j = 0; j < rules.size(); j++) {
            Seq row(rules[j]);
            if(row.size() != 2) throw std::invalid_argument("feedback family shape");
            declarations[file_text(row[0])] =
                    row[1] == Py_None ? std::nullopt : std::make_optional(read_feedback_family(row[1]));
        }
        auto db = phase_database(database);
        auto bound = phase_limits(limits);
        size_t paths = natural(field(limits, "max_path_bits")), bytes = natural(field(limits, "max_structure_bytes"));
        auto owner = std::make_unique<FullRecordBinding>();
        owner->schedule = Py_NewRef(schedule_cap);
        owner->execution = native_work([&]() {
            return std::make_unique<FullRecordExecution>(k, in, out, ops, ops_outcomes, *schedule->schedule,
                    declarations, db, bound, paths, bytes, database != Py_None, reuse);
        });
        auto& e = *owner->execution;
        Object info(full_record_info(e)), metadata(e.global ? output_compiler_metadata(e.input) : PyDict_New());
        PyObject* global = Py_NewRef(Py_None);
        if(e.global) {
            auto h = std::make_unique<GlobalHandle>(e.global);
            Py_DECREF(global);
            global = PyCapsule_New(h.release(), "magic-global-plan-v1", global_destructor);
        }
        Object global_owner(global);
        Object cap(PyCapsule_New(owner.release(), "magic-full-record-v1", full_record_destructor));
        return PyTuple_Pack(4, cap.p, global_owner.p, info.p, metadata.p);
    } catch(...) { return operation_boundary_error(); }
}
PyObject* output_full_certificate(const FullRecordResult& result) {
    Object out(PyDict_New()), resource(phase_output_resource(result.certificate.resource));
    phase_set(out.p, "normal", phase_output_tensor(result.normal));
    phase_set(out.p, "absorbed", phase_output_tensor(result.certificate.tensor));
    phase_set(resource.p, "resource_scope",
            PyUnicode_FromString("fixed-input residual after global radical and boundary contraction"));
    phase_set(resource.p, "absorption_performed", PyBool_FromLong(1));
    phase_set(resource.p, "selected_order", PyUnicode_FromString("single-sided-trace"));
    phase_set(resource.p, "constant_mod8", PyLong_FromLong(result.certificate.tensor.constant));
    phase_set(out.p, "resource", Py_NewRef(resource.p));
    phase_set(out.p, "states", phase_output_states(result.certificate.states));
    phase_set(out.p, "probability", output_scalar(result.probability));
    if(result.literal_input) {
        phase_set(out.p, "metadata", output_compiler_metadata(*result.literal_input));
        phase_set(out.p, "snapshot", output_compiler(*result.literal_compiler));
    }
    return Py_NewRef(out.p);
}
PyObject* evaluate_full_records(PyObject*, PyObject* args) {
    try {
        PyObject *cap, *raw;
        int certificate;
        if(!PyArg_ParseTuple(args, "OOp", &cap, &raw, &certificate)) return nullptr;
        auto* owner = full_record(cap);
        if(!owner) return nullptr;
        auto* schedule = branch_schedule(owner->schedule);
        auto& e = *owner->execution;
        FullRecordBatch batch(e, certificate);
        Object iterator(PyObject_GetIter(raw));
        std::vector<std::unique_ptr<CollectedBranchBinding>> branches;
        std::vector<std::unique_ptr<Object>> snapshots;
        std::unique_ptr<Object> previous_identity;
        auto tick = std::chrono::steady_clock::now();
        while(auto* next = PyIter_Next(iterator.p)) {
            Object item(next);
            Seq row(next);
            if(row.size() != 6) throw std::invalid_argument("full-record input shape");
            FullRecordInput input;
            input.faults = read_original_faults(row[2]);
            bool compact = PyObject_IsTrue(row[3]);
            std::vector<std::pair<std::string, int>> truth;
            std::vector<std::string> flips;
            if(compact) {
                auto word = bits(row[0]), report = bits(row[1]);
                if(word.size() > e.schedule.layout.record_names.size() ||
                        report.size() > e.schedule.layout.record_names.size())
                    throw std::invalid_argument("record word outside plan");
                for(size_t j = 0; j < e.schedule.layout.record_names.size(); j++) {
                    truth.push_back({e.schedule.layout.record_names[j], int(word.bit(j))});
                    if(report.bit(j)) flips.push_back(e.schedule.layout.record_names[j]);
                }
                input.immutable_fault_identity = reinterpret_cast<uintptr_t>(row[4]);
                previous_identity = std::make_unique<Object>(Py_NewRef(row[4]));
            } else {
                Seq values(row[0]);
                for(size_t j = 0; j < values.size(); j++) {
                    Seq pair(values[j]);
                    truth.push_back({file_text(pair[0]), index_int(pair[1])});
                }
                flips = input_names(row[1]);
            }
            input.values = e.schedule.layout.validate(truth, flips, input.faults);
            auto branch = std::make_unique<CollectedBranchBinding>();
            branch->owner = Py_NewRef(owner->schedule);
            branch->gates.resize(e.schedule.controls.size(), nullptr);
            auto callback = [&](size_t index, const std::vector<std::pair<std::string, int>>& parents) {
                CallbackGIL gil;
                const auto& id = e.schedule.layout.operations[e.schedule.controls[index]].id;
                Object values(output_named_bits(parents)), permitted(PyDictProxy_New(values.p)),
                        raw(PyObject_CallOneArg(field(schedule->callbacks, id.c_str()), permitted.p));
                Object sequence(PySequence_Tuple(raw.p)), owned(PyObject_CallOneArg(schedule->copy, sequence.p));
                auto ops = read_program_operations(owned.p, false);
                std::vector<BranchGate> gates;
                for(auto& op : ops) gates.push_back({std::move(op), false});
                branch->gates[index] = Py_NewRef(owned.p);
                return gates;
            };
            snapshots.push_back(std::make_unique<Object>(Py_NewRef(row[5])));
            native_work([&]() {
                batch.add(std::move(input), callback);
                return 0;
            });
            branches.push_back(std::move(branch));
        }
        if(PyErr_Occurred()) throw std::invalid_argument("assignment iterator failed");
        double packing = std::chrono::duration<double>(std::chrono::steady_clock::now() - tick).count();
        tick = std::chrono::steady_clock::now();
        auto results = native_work([&]() { return batch.finish(); });
        double kernel = std::chrono::duration<double>(std::chrono::steady_clock::now() - tick).count();
        Object output(PyList_New(results.size()));
        for(size_t j = 0; j < results.size(); j++) {
            if(!certificate) {
                PyList_SET_ITEM(output.p, j, output_scalar(results[j].probability));
                continue;
            }
            branches[j]->branch = std::move(*results[j].branch);
            Object data(output_collected_data(*branches[j], *schedule));
            Object handle(
                    PyCapsule_New(branches[j].release(), "magic-collected-branch-v1", collected_branch_destructor));
            Object value(output_full_certificate(results[j]));
            phase_set(value.p, "branch", PyTuple_Pack(3, handle.p, data.p, snapshots[j]->p));
            PyList_SET_ITEM(output.p, j, Py_NewRef(value.p));
        }
        Object timings(Py_BuildValue("{s:d,s:d,s:d,s:d}", "packing_validate_seconds", packing, "kernel_seconds", kernel,
                "native_control_seconds", batch.control_seconds, "feedback_fallback_seconds", batch.fallback_seconds));
        Object info(full_record_info(e));
        phase_set(info.p, "reusable_workspace_peak_capacity_bytes", PyLong_FromSize_t(batch.workspace_peak));
        return PyTuple_Pack(3, output.p, timings.p, info.p);
    } catch(...) { return operation_boundary_error(); }
}

PyObject* normalize_literal_program(PyObject*, PyObject* args) {
    try {
        PyObject *initial, *inputs, *outputs, *operations, *paths;
        if(!PyArg_ParseTuple(args, "OOOOO", &initial, &inputs, &outputs, &operations, &paths)) return nullptr;
        auto k = ket(initial);
        auto in = intvec(inputs), out = intvec(outputs);
        auto ops = read_program_operations(operations);
        Seq rows(operations);
        std::vector<int> outcomes;
        for(size_t j = 0; j < rows.size(); j++) {
            int value = 0;
            if(ops[j].value.kind.rfind("M", 0) == 0) {
                auto* p = PyDict_GetItemString(rows[j], "true");
                if(!p) p = PyDict_GetItemString(rows[j], "outcome");
                if(p) value = index_int(p);
            }
            outcomes.push_back(value);
        }
        auto limit = natural(paths);
        auto tensor = native_work([&]() {
            auto [input, assignment] = compile_literal_input(k, in, out, ops, outcomes, limit);
            GlobalPlan plan;
            plan.ir = input.ir;
            prepare_global(plan, std::numeric_limits<size_t>::max());
            auto value = plan.actual(assignment);
            value.environment.clear();
            for(auto index : plan.compiled.environment) value.environment.push_back(ops.at(size_t(index)).id);
            return value;
        });
        return phase_output_tensor(tensor);
    } catch(...) { return error(); }
}

struct FullRecordCursorBinding {
    RecordStreamPolicy policy;
    PyObject* owner;
    PyObject* iterator;
    FullRecordCursorBinding(bool compact, size_t size, PyObject* cap, PyObject* source)
        : policy(compact, size), owner(Py_NewRef(cap)), iterator(Py_NewRef(source)) {}
    ~FullRecordCursorBinding() {
        Py_DECREF(owner);
        Py_DECREF(iterator);
    }
};
FullRecordCursorBinding* full_record_cursor(PyObject* cap) {
    return static_cast<FullRecordCursorBinding*>(PyCapsule_GetPointer(cap, "magic-full-record-cursor-v1"));
}
void full_record_cursor_destructor(PyObject* cap) { delete full_record_cursor(cap); }
PyObject* make_full_record_cursor(PyObject*, PyObject* args) {
    try {
        PyObject *cap, *raw, *size;
        int compact;
        if(!PyArg_ParseTuple(args, "OOOp", &cap, &raw, &size, &compact)) return nullptr;
        if(!full_record(cap)) return nullptr;
        Object iterator(PyObject_GetIter(raw));
        auto cursor = std::make_unique<FullRecordCursorBinding>(compact, natural(size), cap, iterator.p);
        return PyCapsule_New(cursor.release(), "magic-full-record-cursor-v1", full_record_cursor_destructor);
    } catch(...) { return error(); }
}
PyObject* next_full_record_cursor(PyObject*, PyObject* cap) {
    try {
        auto* cursor = full_record_cursor(cap);
        if(!cursor) return nullptr;
        Object rows(PyList_New(0));
        for(size_t j = 0; j < cursor->policy.block_size; j++) {
            auto* next = PyIter_Next(cursor->iterator);
            if(!next) {
                if(PyErr_Occurred()) throw std::invalid_argument("assignment iterator failed");
                break;
            }
            Object value(next);
            if(PyList_Append(rows.p, value.p) < 0) throw std::bad_alloc();
        }
        if(!PyList_GET_SIZE(rows.p)) Py_RETURN_NONE;
        Object args(Py_BuildValue("(OOi)", cursor->owner, rows.p, 0));
        return evaluate_full_records(nullptr, args.p);
    } catch(...) { return operation_boundary_error(); }
}
