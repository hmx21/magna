BoundaryTracePlan* boundary_handle(PyObject* cap) {
    return static_cast<BoundaryTracePlan*>(PyCapsule_GetPointer(cap, "magic-boundary-trace-v1"));
}
void boundary_destructor(PyObject* cap) { delete boundary_handle(cap); }
BoundarySpec read_boundary_spec(PyObject* raw) {
    Seq row(raw);
    if(row.size() != 4) throw std::invalid_argument("boundary specification shape");
    return {bitvec(row[0]), bitvec(row[1]), bits(row[2]), bits(row[3])};
}
PyObject* compiled_network_info(const CompiledNetwork& network, bool maps = false) {
    Object info(Py_BuildValue("{s:n,s:n,s:n,s:O,s:d,s:d}", "retained_labels", Py_ssize_t(network.boundary.keep.size()),
            "reference_phase_coordinates", Py_ssize_t(network.reference_rank), "reference_terms",
            Py_ssize_t(network.reference_terms), "reference_zero", network.reference_zero ? Py_True : Py_False,
            "symbolic_seconds", network.reduction_seconds, "resource_seconds", network.resource_seconds));
    phase_set(info.p, "shared_symbolic_maps", PyBool_FromLong(network.shared_maps));
    phase_set(info.p, "reference_outcomes", output_int_tuple(network.reference_outcomes));
    phase_set(info.p, "uniform_term_bound",
            network.inherited ? PyLong_FromSize_t(network.uniform_term_bound()) : Py_NewRef(Py_None));
    phase_set(info.p, "direct_coverage",
            PyUnicode_FromString(
                    network.joint ? "matching reduced coordinate domain and phase trilinear form" : "unavailable"));
    phase_set(info.p, "inherited_coverage",
            PyUnicode_FromString(network.inherited
                            ? "all compatible Pauli offsets and event RHSs of the compiled input family"
                            : "unavailable"));
    phase_set(info.p, "direct_term_bound",
            network.joint ? PyLong_FromSize_t(network.joint->term_bound()) : Py_NewRef(Py_None));
    phase_set(info.p, "stable_pairing_term_bound",
            network.stable_pairing ? PyLong_FromSize_t(network.stable_pairing->term_bound()) : Py_NewRef(Py_None));
    phase_set(info.p, "stable_pairing_phase_coordinates",
            network.stable_pairing ? PyLong_FromSize_t(network.stable_pairing->reference.quotient.axes.size())
                                   : Py_NewRef(Py_None));
    phase_set(info.p, "inherited_phase_coordinates",
            network.inherited ? PyLong_FromSize_t(2 * network.inherited->reference.quotient.axes.size())
                              : Py_NewRef(Py_None));
    phase_set(info.p, "direct_limit", PyUnicode_FromString(network.joint_limit.c_str()));
    phase_set(info.p, "inherited_limit", PyUnicode_FromString(network.inherited_limit.c_str()));
    auto decomposition = [](const auto& certificate) {
        if(!certificate) return Py_NewRef(Py_None);
        const auto& d = certificate->reference;
        return Py_BuildValue("{s:n,s:n,s:n,s:n,s:O,s:O,s:O}", "original_terms", Py_ssize_t(d.original_terms),
                "t_rotations", Py_ssize_t(d.injection_rotations), "candidate_terms",
                Py_ssize_t(d.injection_candidate_terms), "selected_terms", Py_ssize_t(resource_entries(d).size()),
                "constructed", d.injection_constructed ? Py_True : Py_False, "selected",
                d.injection_selected ? Py_True : Py_False, "budget_skipped",
                d.injection_budget_skipped ? Py_True : Py_False);
    };
    phase_set(info.p, "direct_decomposition", decomposition(network.joint));
    phase_set(info.p, "stable_pairing_decomposition", decomposition(network.stable_pairing));
    phase_set(info.p, "inherited_decomposition", decomposition(network.inherited));
    if(maps) {
        phase_set(info.p, "input_dimension", PyLong_FromSize_t(network.input_dimension));
        phase_set(info.p, "input_forms", phase_output_bits(network.maps ? network.maps->forms : std::vector<Bits>{}));
        phase_set(info.p, "event_support_direction",
                phase_output_bits(network.maps ? network.maps->events->factor.kernel : std::vector<Bits>{}));
        auto certificate = [](const auto& certificate) {
            if(!certificate) return Py_NewRef(Py_None);
            Object out(PyDict_New());
            const auto& resource = certificate->reference;
            phase_set(out.p, "dimension", PyLong_FromSize_t(certificate->dimension));
            phase_set(out.p, "axes", phase_output_bits(resource.quotient.axes));
            phase_set(out.p, "lifts", phase_output_bits(resource.quotient.lifts));
            phase_set(out.p, "linear", output_int_tuple(resource.quotient.linear));
            phase_set(out.p, "upper", phase_output_bits(resource.quotient.upper));
            phase_set(out.p, "core", phase_output_terms(resource.quotient.core));
            phase_set(out.p, "constant", PyLong_FromLong(resource.constant_mod8));
            return Py_NewRef(out.p);
        };
        phase_set(info.p, "direct_certificate", certificate(network.joint));
        phase_set(info.p, "inherited_certificate", certificate(network.inherited));
    }
    return Py_NewRef(info.p);
}
PyObject* compile_boundary_queries(PyObject*, PyObject* args) {
    try {
        PyObject *cap, *raw;
        int maps = 0;
        if(!PyArg_ParseTuple(args, "OO|p", &cap, &raw, &maps)) return nullptr;
        auto* plan = boundary_handle(cap);
        if(!plan) return nullptr;
        Seq rows(raw);
        if(rows.size() > 4096) throw std::length_error("compiled query family budget");
        std::vector<BoundarySpec> specifications;
        for(size_t j = 0; j < rows.size(); j++) specifications.push_back(read_boundary_spec(rows[j]));
        auto networks = native_work([&]() {
            std::vector<std::shared_ptr<const CompiledNetwork>> out;
            for(const auto& spec : specifications) out.push_back(plan->compile(spec));
            plan->finish_preparation();
            return out;
        });
        Object out(PyTuple_New(networks.size()));
        for(size_t j = 0; j < networks.size(); j++)
            PyTuple_SET_ITEM(out.p, j, compiled_network_info(*networks[j], maps));
        return Py_NewRef(out.p);
    } catch(...) { return error(); }
}
PyObject* make_boundary_trace(PyObject*, PyObject* args) {
    try {
        PyObject *raw, *limits, *database = Py_None;
        unsigned long long max_doubled;
        if(!PyArg_ParseTuple(args, "OOK|O", &raw, &limits, &max_doubled, &database)) return nullptr;
        auto plan = std::make_unique<BoundaryTracePlan>();
        plan->ir = parse_compiler(raw);
        plan->limits = phase_limits(limits);
        plan->max_doubled = size_t(max_doubled);
        plan->database = phase_database(database);
        std::exception_ptr failure;
        auto tick = std::chrono::steady_clock::now();
        PyThreadState* state = PyEval_SaveThread();
        try {
            plan->prepare();
        } catch(...) { failure = std::current_exception(); }
        PyEval_RestoreThread(state);
        if(failure) std::rethrow_exception(failure);
        Object info(Py_BuildValue("{s:d,s:K,s:K,s:s,s:i,s:i}", "prepare_seconds",
                std::chrono::duration<double>(std::chrono::steady_clock::now() - tick).count(), "amplitude_path_bits",
                (unsigned long long)plan->amplitude_path_bits, "boundary_width",
                (unsigned long long)plan->compiled.plan.physical, "reduction", "single", "single_compilations", 1,
                "reverse_compilations", 0));
        Object cap(PyCapsule_New(plan.release(), "magic-boundary-trace-v1", boundary_destructor));
        return Py_BuildValue("(OO)", cap.p, info.p);
    } catch(...) { return error(); }
}
PyObject* query_measurement_info(const TraceDiagnostics& d, size_t path_bits, size_t rank, size_t terms, bool affine,
        bool zero, size_t retained, double elapsed) {
    Object info(Py_BuildValue("{s:d,s:K,s:K,s:K,s:O,s:O}", "query_seconds", elapsed, "trace_path_bits",
            (unsigned long long)path_bits, "quotient_rank", (unsigned long long)rank, "constructive_terms",
            (unsigned long long)terms, "affine", affine ? Py_True : Py_False, "zero", zero ? Py_True : Py_False));
    auto integer = [&](const char* key, size_t v) {
        Object value(PyLong_FromSize_t(v));
        if(PyDict_SetItemString(info.p, key, value.p) < 0) throw std::bad_alloc();
    };
    auto seconds = [&](const char* key, double v) {
        Object value(PyFloat_FromDouble(v));
        if(PyDict_SetItemString(info.p, key, value.p) < 0) throw std::bad_alloc();
    };
    Object mode(PyUnicode_FromString(d.wide ? "wide" : "single"));
    PyDict_SetItemString(info.p, "reduction", mode.p);
    PyDict_SetItemString(info.p, "fallback_used", Py_False);
    integer("single_input_bits", d.input_bits);
    integer("event_rows", d.event_rows);
    integer("event_rank", d.event_rank);
    integer("retained_labels", retained);
    integer("single_support_bits", d.support_bits);
    integer("single_phase_axes", d.phase_axes);
    integer("single_clifford_bits", d.clifford_bits);
    integer("difference_bits", d.difference_bits);
    integer("pairing_bits", d.pairing_bits);
    integer("support_factorizations", d.support_factorizations);
    integer("pairing_factorizations", d.pairing_factorizations);
    integer("single_side_cache_hits", d.side_hit);
    integer("support_cache_hits", d.support_hit);
    integer("pairing_cache_hits", d.pairing_hit);
    integer("expanded_entries", d.expanded_entries);
    integer("inherited_side_entries", d.inherited_side_entries);
    phase_set(info.p, "output_trace_reused", PyBool_FromLong(d.output_trace_reused));
    phase_set(info.p, "prepared_query", PyBool_FromLong(d.prepared_query));
    phase_set(info.p, "input_certificate_covered", PyBool_FromLong(d.input_certificate_covered));
    phase_set(info.p, "reference_entries_reused", PyBool_FromLong(d.reference_entries_reused));
    integer("resource_constructions", d.resource_constructions);
    integer("phase_coordinate_constructions", d.phase_coordinate_constructions);
    Object expansion(PyUnicode_FromString(d.inherited_product ? "inherited-product" : "joint"));
    PyDict_SetItemString(info.p, "resource_expansion", expansion.p);
    seconds("actual_update_seconds", d.actual_seconds);
    seconds("single_support_seconds", d.support_seconds);
    seconds("single_reduce_seconds", d.single_seconds);
    seconds("pairing_seconds", d.pairing_seconds);
    seconds("residual_reduce_seconds", d.reduce_seconds);
    seconds("resource_expand_seconds", d.expand_seconds);
    Py_INCREF(info.p);
    return info.p;
}
PyObject* boundary_result_info(const BoundaryResult& result, size_t retained, double elapsed) {
    return query_measurement_info(result.diagnostics, result.tensor.n, phase_quotient(result.tensor).axes.size(),
            result.terms.size(), result.affine, result.tensor.zero, retained, elapsed);
}
PyObject* query_boundary_trace_impl(PyObject* args) {
    try {
        PyObject *cap, *raw, *kept, *events, *outcomes, *px, *pz;
        int affine_only;
        if(!PyArg_ParseTuple(args, "OOOOOOOp", &cap, &raw, &kept, &events, &outcomes, &px, &pz, &affine_only))
            return nullptr;
        auto* p = boundary_handle(cap);
        if(!p) return nullptr;
        auto a = assignment(raw, p->assignment_transport());
        auto keep = bitvec(kept), rows = bitvec(events);
        auto values = intvec(outcomes);
        auto x = bits(px), z = bits(pz);
        BoundaryResult result;
        std::exception_ptr failure;
        auto tick = std::chrono::steady_clock::now();
        PyThreadState* state = PyEval_SaveThread();
        try {
            Workspace ws;
            result = p->query(a, keep, rows, values, x, z, affine_only, ws);
        } catch(...) { failure = std::current_exception(); }
        PyEval_RestoreThread(state);
        if(failure) std::rethrow_exception(failure);
        Object terms(phase_output_states(result.terms));
        Object info(boundary_result_info(
                result, keep.size(), std::chrono::duration<double>(std::chrono::steady_clock::now() - tick).count()));
        return Py_BuildValue("(OO)", terms.p, info.p);
    } catch(...) { return error(); }
}
PyObject* query_boundary_trace(PyObject*, PyObject* args) { return query_boundary_trace_impl(args); }
PyObject* sum_boundary_terms(PyObject*, PyObject* args) {
    try {
        PyObject *raw, *events, *outcomes;
        if(!PyArg_ParseTuple(args, "OOO", &raw, &events, &outcomes)) return nullptr;
        Seq items(raw);
        std::vector<std::pair<Scalar, Ket>> terms;
        for(size_t j = 0; j < items.size(); j++) {
            Seq row(items[j]);
            if(row.size() != 2) throw std::invalid_argument("probability term shape");
            terms.push_back({scalar(row[0]), ket(row[1])});
        }
        auto rows = bitvec(events);
        auto values = intvec(outcomes);
        Workspace ws;
        return output_scalar(boundary_sum(terms, rows, values, ws));
    } catch(...) { return error(); }
}

// Exact certificate probability/moment uses the identical production kernel.
PyObject* trace_phase_tensor(PyObject*, PyObject* args) {
    try {
        PyObject *raw, *lim, *db, *px, *pz;
        unsigned long long paths;
        if(!PyArg_ParseTuple(args, "OOOOOK", &raw, &lim, &db, &px, &pz, &paths)) return nullptr;
        auto t = phase_tensor(raw);
        auto limits = paired_phase_limits(phase_limits(lim));
        auto database = phase_database(db);
        auto x = bits(px), z = bits(pz);
        TraceFactorCache support, pairing;
        Workspace ws;
        auto result = single_trace_query(std::move(t), {}, {}, {}, x, z, false, ws, support, pairing,
                PathBudget(2 * size_t(paths), "2*max_path_bits"), limits, db == Py_None ? nullptr : &database);
        return output_scalar(boundary_sum(result.terms, {}, {}, ws));
    } catch(...) { return error(); }
}
