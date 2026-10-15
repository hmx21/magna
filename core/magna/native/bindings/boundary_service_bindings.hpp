struct BoundaryServiceBinding {
    std::unique_ptr<BoundaryService> service;
    PyObject* owner = nullptr;
    ~BoundaryServiceBinding() { Py_XDECREF(owner); }
};
BoundaryServiceBinding* boundary_service(PyObject* cap) {
    return static_cast<BoundaryServiceBinding*>(PyCapsule_GetPointer(cap, "magic-boundary-service-v1"));
}
void boundary_service_destructor(PyObject* cap) { delete boundary_service(cap); }
PyObject* make_boundary_service(PyObject*, PyObject* args) {
    try {
        PyObject *cap, *raw_locations, *raw_records, *raw_anchors;
        if(!PyArg_ParseTuple(args, "OOOO", &cap, &raw_locations, &raw_records, &raw_anchors)) return nullptr;
        auto* plan = boundary_handle(cap);
        if(!plan) return nullptr;
        auto owner = std::make_unique<BoundaryServiceBinding>();
        owner->owner = Py_NewRef(cap);
        owner->service = std::make_unique<BoundaryService>(plan);
        auto pairs = [](PyObject* raw) {
            Seq rows(raw);
            std::map<std::string, size_t> values;
            for(size_t j = 0; j < rows.size(); j++) {
                Seq row(rows[j]);
                if(row.size() != 2) throw std::invalid_argument("named index shape");
                values[file_text(row[0])] = natural(row[1]);
            }
            return values;
        };
        owner->service->locations = pairs(raw_locations);
        owner->service->records = pairs(raw_records);
        Seq anchors(raw_anchors);
        for(size_t j = 0; j < anchors.size(); j++) {
            Seq row(anchors[j]);
            if(row.size() != 3) throw std::invalid_argument("report anchor shape");
            owner->service->report_anchors[file_text(row[0])] = {file_text(row[1]), index_int(row[2])};
        }
        return PyCapsule_New(owner.release(), "magic-boundary-service-v1", boundary_service_destructor);
    } catch(...) { return error(); }
}
PyObject* output_names(const std::vector<std::string>& names) {
    Object out(PyTuple_New(names.size()));
    for(size_t j = 0; j < names.size(); j++)
        PyTuple_SET_ITEM(out.p, j, PyUnicode_FromStringAndSize(names[j].data(), names[j].size()));
    return Py_NewRef(out.p);
}
PyObject* output_boundary_coordinates(const std::vector<std::string>& names, const BoundaryCoordinates& layout) {
    Object out(PyDict_New());
    auto masks = [&](const char* key, size_t begin, size_t end) {
        std::vector<Bits> values(layout.transform.begin() + begin, layout.transform.begin() + end);
        phase_set(out.p, key, phase_output_bits(values));
    };
    phase_set(out.p, "record_order", output_names(names));
    masks("transform", 0, layout.transform.size());
    masks("postselection_rows", 0, layout.selection);
    masks("feedback_rows", layout.selection, layout.selection + layout.feedback);
    masks("trace_rows", layout.selection + layout.feedback, layout.transform.size());
    phase_set(out.p, "postselection_values", output_int_tuple(layout.values));
    phase_set(out.p, "postselection_rank", PyLong_FromSize_t(layout.selection));
    phase_set(out.p, "feedback_rank", PyLong_FromSize_t(layout.feedback));
    phase_set(out.p, "trace_rank", PyLong_FromSize_t(names.size() - layout.selection - layout.feedback));
    phase_set(out.p, "compatible", PyBool_FromLong(layout.compatible));
    phase_set(out.p, "classification", PyUnicode_FromString("task dependencies, not outcome randomness"));
    return Py_NewRef(out.p);
}
PyObject* boundary_service_rows(PyObject*, PyObject* args) {
    try {
        PyObject *cap, *raw;
        if(!PyArg_ParseTuple(args, "OO", &cap, &raw)) return nullptr;
        auto* owner = boundary_service(cap);
        if(!owner) return nullptr;
        auto names = read_named_rows(raw);
        auto rows = native_work([&]() { return owner->service->rows(names); });
        return phase_output_bits(rows);
    } catch(...) { return error(); }
}
PyObject* output_packed_assignment(const Assignment& input) {
    Object faults(PyTuple_New(input.faults.size())), word(output_bits(input.record));
    for(size_t j = 0; j < input.faults.size(); j++) {
        const auto& fault = input.faults[j];
        PyTuple_SET_ITEM(faults.p, j, Py_BuildValue("(nii)", Py_ssize_t(fault.location), fault.wire, fault.pauli));
    }
    return PyTuple_Pack(2, word.p, faults.p);
}
PyObject* boundary_service_pack(PyObject*, PyObject* args) {
    try {
        PyObject *cap, *raw, *flips;
        if(!PyArg_ParseTuple(args, "OOO", &cap, &raw, &flips)) return nullptr;
        auto* owner = boundary_service(cap);
        if(!owner) return nullptr;
        auto faults = read_original_faults(raw);
        auto names = input_names(flips);
        return output_packed_assignment(owner->service->pack(faults, names));
    } catch(...) { return error(); }
}
PyObject* boundary_service_totals(PyObject*, PyObject* args) {
    try {
        PyObject *cap, *raw, *queries;
        if(!PyArg_ParseTuple(args, "OOO", &cap, &raw, &queries)) return nullptr;
        auto* owner = boundary_service(cap);
        if(!owner) return nullptr;
        Seq input(raw), rows(queries);
        if(input.size() > 65536) throw std::length_error("scalar query batch limit");
        std::vector<Assignment> assignments;
        for(size_t j = 0; j < input.size(); j++)
            assignments.push_back(assignment(input[j], owner->service->plan->assignment_transport()));
        std::vector<BoundaryTotalsData::Query> specifications;
        std::vector<std::vector<int>> outcomes;
        for(size_t j = 0; j < rows.size(); j++) {
            Seq row(rows[j]);
            if(row.size() != 4) throw std::invalid_argument("scalar query shape");
            specifications.push_back({bitvec(row[0]), intvec(row[1]), bits(row[2]), bits(row[3])});
            outcomes.push_back(specifications.back().outcomes);
        }
        auto start = std::chrono::steady_clock::now();
        auto result = native_work([&]() {
            auto totals = owner->service->totals(specifications);
            return evaluate_totals(*totals, assignments, outcomes);
        });
        double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        Object values(PyList_New(result.values.size())), info(PyList_New(result.last.size()));
        for(size_t j = 0; j < result.values.size(); j++) {
            Object row(PyTuple_New(result.values[j].size()));
            for(size_t k = 0; k < result.values[j].size(); k++)
                PyTuple_SET_ITEM(row.p, k, output_scalar(result.values[j][k]));
            PyList_SET_ITEM(values.p, j, Py_NewRef(row.p));
        }
        for(size_t j = 0; j < result.last.size(); j++)
            PyList_SET_ITEM(info.p, j, boundary_result_info(result.last[j], 0, result.last_seconds[j]));
        return Py_BuildValue("(OOd)", values.p, info.p, elapsed);
    } catch(...) { return error(); }
}
PyObject* output_query_measurement(const QueryMeasurement& query) {
    return query_measurement_info(query.diagnostics, query.path_bits, query.rank, query.terms, query.affine, query.zero,
            query.retained, query.seconds);
}
PyObject* output_boundary_sample_info(const BoundarySampleRows& result) {
    Object info(Py_BuildValue("{s:O,s:n,s:n,s:s}", "common_axis_certificate", result.common ? Py_True : Py_False,
            "commuting_parities", Py_ssize_t(result.commuting), "sampled_affine_parities",
            Py_ssize_t(result.sampled_affine), "method", result.method.c_str()));
    auto report = [](const BoundarySampleRows::Query& query) { return output_query_measurement(query); };
    if(result.affine) phase_set(info.p, "affine_query_resource", report(*result.affine));
    Object final(report(result.final)), resources(PyList_New(1));
    PyList_SET_ITEM(resources.p, 0, Py_NewRef(final.p));
    phase_set(info.p, "resources", Py_NewRef(resources.p));
    return Py_NewRef(info.p);
}
PyObject* boundary_service_sample(PyObject*, PyObject* args) {
    try {
        PyObject *cap, *packed, *raw_rows, *raw_events, *raw_outcomes, *rng;
        if(!PyArg_ParseTuple(args, "OOOOOO", &cap, &packed, &raw_rows, &raw_events, &raw_outcomes, &rng))
            return nullptr;
        auto* owner = boundary_service(cap);
        if(!owner) return nullptr;
        auto& plan = *owner->service->plan;
        auto input = assignment(packed, plan.assignment_transport());
        auto rows = bitvec(raw_rows), events = bitvec(raw_events);
        auto outcomes = intvec(raw_outcomes);
        auto result = native_work([&]() {
            return sample_boundary_rows(plan, input, rows, events, outcomes, [&](size_t count) {
                CallbackGIL gil;
                return random_bits(rng)(count);
            });
        });
        Object word(output_bits(result.word)), mass(output_scalar(result.mass)),
                info(output_boundary_sample_info(result));
        double seconds = result.final.seconds;
        size_t count = 1;
        if(result.affine) {
            seconds += result.affine->seconds;
            count++;
        }
        return Py_BuildValue("(OOOnd)", word.p, mass.p, info.p, Py_ssize_t(count), seconds);
    } catch(...) { return execution_error(); }
}
