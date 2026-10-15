// Include inside bindings.cpp's anonymous namespace, after its conversion helpers.
using namespace magic_native;
PhaseTerms phase_parse_terms(PyObject* p) {
    Seq s(p);
    PhaseTerms out;
    for(size_t j = 0; j < s.size(); j++) {
        Seq row(s[j]);
        if(row.size() != 2) throw std::invalid_argument("phase term shape");
        out.push_back({bits(row[0]), index_int(row[1])});
    }
    return out;
}
PyObject* phase_output_bits(const std::vector<Bits>& v) {
    Object out(PyTuple_New(v.size()));
    for(size_t j = 0; j < v.size(); j++) PyTuple_SET_ITEM(out.p, j, output_bits(v[j]));
    Py_INCREF(out.p);
    return out.p;
}
PyObject* phase_output_ints(const std::vector<int>& v) {
    Object out(PyTuple_New(v.size()));
    for(size_t j = 0; j < v.size(); j++) PyTuple_SET_ITEM(out.p, j, PyLong_FromLong(v[j]));
    Py_INCREF(out.p);
    return out.p;
}
PyObject* phase_output_terms(const PhaseTerms& v) {
    Object out(PyTuple_New(v.size()));
    for(size_t j = 0; j < v.size(); j++) {
        Object mask(output_bits(v[j].first));
        PyTuple_SET_ITEM(out.p, j, Py_BuildValue("(Oi)", mask.p, v[j].second));
    }
    Py_INCREF(out.p);
    return out.p;
}
PhaseLimits phase_limits(PyObject* p) {
    PhaseLimits l;
    if(p == Py_None) return l;
    if(!PyDict_Check(p)) throw std::invalid_argument("phase limits dictionary required");
    if(auto* v = PyDict_GetItemString(p, "max_quotient_rank")) l.max_quotient_rank = natural(v);
    if(auto* v = PyDict_GetItemString(p, "max_terms")) l.max_terms = natural(v);
    if(auto* v = PyDict_GetItemString(p, "max_pair_contractions")) l.max_pairs = natural(v);
    if(auto* v = PyDict_GetItemString(p, "max_table_bytes")) l.max_table_bytes = natural(v);
    return l;
}
PhaseDatabase phase_database(PyObject* p) {
    PhaseDatabase db;
    if(p == Py_None) return db;
    if(!PyDict_Check(p)) throw std::invalid_argument("raw phase database dictionary required");
    if(integer(field(p, "schema_version")) != 1) throw std::invalid_argument("unsupported resource convention version");
    Seq entries(field(p, "entries"));
    for(size_t j = 0; j < entries.size(); j++) {
        PyObject* item = entries[j];
        PhaseDatabaseEntry e;
        e.dimension = natural(field(item, "dimension"));
        e.core = intvec(field(item, "core_phase_mod8"));
        Seq terms(field(item, "terms"));
        for(size_t k = 0; k < terms.size(); k++) {
            PhaseEntryTerm t;
            Seq c(field(terms[k], "coefficient"));
            if(c.size() != 4) throw std::invalid_argument("four database coefficients required");
            for(size_t i = 0; i < 4; i++) t.coefficient.c[i] = big(c[i]);
            t.coefficient.normalize();
            t.constraints = phase_parse_terms(field(terms[k], "constraints"));
            t.parities = phase_parse_terms(field(terms[k], "clifford_parities"));
            e.terms.push_back(std::move(t));
        }
        db.add(std::move(e));
    }
    return db;
}
PhaseTensor phase_tensor(PyObject* p) {
    if(!PyDict_Check(p)) throw std::invalid_argument("phase tensor dictionary required");
    PhaseTensor t;
    t.n = natural(field(p, "n"));
    t.linear = intvec(field(p, "linear"));
    t.upper = bitvec(field(p, "upper"));
    t.terms = phase_parse_terms(field(p, "terms"));
    t.constant = mod8(index_int(field(p, "constant")));
    t.power = integer(field(p, "power"));
    t.boundary = phase_parse_terms(field(p, "boundary"));
    t.output_count = natural(field(p, "output_count"));
    if(auto* v = PyDict_GetItemString(p, "zero")) t.zero = PyObject_IsTrue(v);
    if(auto* v = PyDict_GetItemString(p, "environment")) {
        Seq env(v);
        for(size_t j = 0; j < env.size(); j++) {
            if(!PyUnicode_Check(env[j])) throw std::invalid_argument("environment labels must be strings");
            Py_ssize_t length = 0;
            const char* label = PyUnicode_AsUTF8AndSize(env[j], &length);
            if(!label) throw std::invalid_argument("environment label UTF-8 conversion failed");
            t.environment.emplace_back(label, size_t(length));
        }
    }
    if(t.n > 1000000 || t.linear.size() != t.n || t.upper.size() != t.n ||
            t.output_count + t.environment.size() != t.boundary.size())
        throw std::invalid_argument("phase tensor dimensions");
    if(t.power < -1000000 || t.power > 1000000) throw std::invalid_argument("phase power budget");
    masks(t.upper, t.n);
    for(size_t j = 0; j < t.n; j++)
        if(t.linear[j] < 0 || t.linear[j] > 3 || t.upper[j] != t.upper[j].above(j))
            throw std::invalid_argument("phase quadratic convention");
    for(auto& [a, c] : t.terms)
        if(a.size() > t.n) throw std::invalid_argument("phase term dimension");
    for(auto& [a, b] : t.boundary)
        if(a.size() > t.n || (b != 0 && b != 1)) throw std::invalid_argument("phase boundary convention");
    return t;
}
void phase_set(PyObject* d, const char* key, PyObject* v) {
    Object own(v);
    if(PyDict_SetItemString(d, key, v) < 0) throw std::invalid_argument("phase dictionary output");
}
PyObject* phase_output_certificate(const PhaseCertificate& c) {
    Object out(PyDict_New());
    phase_set(out.p, "stage", PyUnicode_FromString(c.stage.c_str()));
    if(c.stage == "boundary_coordinate_change") {
        phase_set(out.p, "rows", phase_output_bits(c.rows));
        phase_set(out.p, "inverse_columns", phase_output_bits(c.inverse_columns));
    } else {
        phase_set(out.p, "input_variables", PyLong_FromSize_t(c.input_variables));
        phase_set(out.p, "axes", phase_output_bits(c.quotient.axes));
        phase_set(out.p, "separated_quadratic", PyBool_FromLong(1));
        if(c.stage == "global_radical_contraction") {
            phase_set(out.p, "signature_rank", PyLong_FromSize_t(c.quotient.axes.size()));
            phase_set(out.p, "section", phase_output_bits(c.quotient.lifts));
            phase_set(out.p, "core_parities", phase_output_terms(c.quotient.core));
            phase_set(out.p, "clifford_linear", phase_output_ints(c.quotient.linear));
            phase_set(out.p, "clifford_upper", phase_output_bits(c.quotient.upper));
            phase_set(out.p, "contracted_tensor", output_ket(c.contracted));
        }
    }
    Py_INCREF(out.p);
    return out.p;
}
PyObject* phase_output_tensor(const PhaseTensor& t) {
    Object out(PyDict_New());
    phase_set(out.p, "n", PyLong_FromSize_t(t.n));
    phase_set(out.p, "linear", phase_output_ints(t.linear));
    phase_set(out.p, "upper", phase_output_bits(t.upper));
    phase_set(out.p, "terms", phase_output_terms(t.terms));
    phase_set(out.p, "constant", PyLong_FromLong(t.constant));
    phase_set(out.p, "power", PyLong_FromLongLong(t.power));
    phase_set(out.p, "boundary", phase_output_terms(t.boundary));
    phase_set(out.p, "output_count", PyLong_FromSize_t(t.output_count));
    phase_set(out.p, "zero", PyBool_FromLong(t.zero));
    Object env(PyTuple_New(t.environment.size()));
    for(size_t j = 0; j < t.environment.size(); j++)
        PyTuple_SET_ITEM(
                env.p, j, PyUnicode_FromStringAndSize(t.environment[j].data(), Py_ssize_t(t.environment[j].size())));
    if(PyDict_SetItemString(out.p, "environment", env.p) < 0) throw std::invalid_argument("environment output");
    Object certs(PyTuple_New(t.certificates.size()));
    for(size_t j = 0; j < t.certificates.size(); j++)
        PyTuple_SET_ITEM(certs.p, j, phase_output_certificate(t.certificates[j]));
    if(PyDict_SetItemString(out.p, "certificates", certs.p) < 0) throw std::invalid_argument("certificate output");
    Py_INCREF(out.p);
    return out.p;
}
PyObject* phase_output_states(const std::vector<std::pair<Scalar, Ket>>& states) {
    Object out(PyTuple_New(states.size()));
    for(size_t j = 0; j < states.size(); j++) {
        Object c(output_scalar(states[j].first)), k(output_ket(states[j].second));
        PyTuple_SET_ITEM(out.p, j, Py_BuildValue("(OO)", c.p, k.p));
    }
    Py_INCREF(out.p);
    return out.p;
}
PyObject* phase_output_resource(const PhaseResource& d, bool public_fields = false) {
    Object out(PyDict_New());
    phase_set(out.p, "axes", phase_output_bits(d.quotient.axes));
    phase_set(out.p, "quotient_rank", PyLong_FromSize_t(d.quotient.axes.size()));
    phase_set(out.p, "section_lifts", phase_output_bits(d.quotient.lifts));
    phase_set(out.p, "core_phase_mod8", phase_output_ints(d.core_phase_mod8));
    phase_set(out.p, "clifford_linear_mod4", phase_output_ints(d.quotient.linear));
    phase_set(out.p, "clifford_quadratic_upper", phase_output_bits(d.quotient.upper));
    phase_set(out.p, "database_hit", PyBool_FromLong(d.database_hit));
    phase_set(out.p, "database_product", PyBool_FromLong(d.database_product));
    phase_set(out.p, "amplitude_denominator_power", PyLong_FromLong(d.amplitude_denominator_power));
    phase_set(out.p, "constant_mod8", PyLong_FromLong(d.constant_mod8));
    phase_set(out.p, "exact_signature_identity", PyBool_FromLong(1));
    phase_set(out.p, "constructive_terms", PyLong_FromSize_t(d.constructive_terms));
    phase_set(out.p, "constructive_stabilizer_operator_terms", PyLong_FromSize_t(resource_entries(d).size()));
    Object entries(PyTuple_New(resource_entries(d).size()));
    for(size_t j = 0; j < resource_entries(d).size(); j++) {
        auto& t = resource_entries(d)[j];
        Object c(output_scalar(t.coefficient)), s(phase_output_terms(t.constraints)), p(phase_output_terms(t.parities));
        PyTuple_SET_ITEM(entries.p, j, Py_BuildValue("(OOO)", c.p, s.p, p.p));
    }
    if(PyDict_SetItemString(out.p, "exact_entries", entries.p) < 0) throw std::invalid_argument("entry output");
    if(public_fields) {
        auto public_terms = public_resource_terms(d);
        Object adapted(PyTuple_New(public_terms.size()));
        for(size_t j = 0; j < public_terms.size(); j++) {
            const auto& term = public_terms[j];
            Object c(PyTuple_New(4));
            for(size_t k = 0; k < 4; k++) PyTuple_SET_ITEM(c.p, k, output_big(term.coefficient[k]));
            if(d.database_hit) {
                Object constraints(phase_output_terms(term.constraints)), parities(phase_output_terms(term.parities));
                PyTuple_SET_ITEM(adapted.p, j, PyTuple_Pack(3, c.p, constraints.p, parities.p));
            } else {
                Object axis(output_bits(term.axis));
                PyTuple_SET_ITEM(adapted.p, j, PyTuple_Pack(2, c.p, axis.p));
            }
        }
        phase_set(out.p, d.database_hit ? "stabilizer_terms" : "amplitude_terms", Py_NewRef(adapted.p));
    }
    phase_set(out.p, "resource_scope",
            PyUnicode_FromString("fixed-input residual after global radical and boundary contraction"));
    Py_INCREF(Py_None);
    phase_set(out.p, "minimum_stabilizer_rank", Py_None);
    Py_INCREF(out.p);
    return out.p;
}
PyObject* phase_absorb_test(PyObject*, PyObject* arg) {
    try {
        Workspace ws;
        return phase_output_tensor(absorb_phase(phase_tensor(arg), ws));
    } catch(...) { return error(); }
}
PyObject* phase_decompose_test(PyObject*, PyObject* args) {
    try {
        PyObject *raw, *lim, *database;
        int absorb = 1;
        if(!PyArg_ParseTuple(args, "OOO|p", &raw, &lim, &database, &absorb)) return nullptr;
        auto t = phase_tensor(raw);
        auto limits = phase_limits(lim);
        auto db = phase_database(database);
        Workspace ws;
        if(absorb) t = absorb_phase(std::move(t), ws);
        auto result = decompose_phase(std::move(t), limits, database == Py_None ? nullptr : &db, ws);
        Object tensor(phase_output_tensor(result.tensor)), resource(phase_output_resource(result.resource, true)),
                states(phase_output_states(result.states)), value(output_scalar(result.probability));
        return Py_BuildValue("(OOOO)", tensor.p, resource.p, states.p, value.p);
    } catch(...) { return error(); }
}

PyObject* phase_value_operation(PyObject*, PyObject* args) {
    try {
        const char* operation;
        PyObject* raw;
        if(!PyArg_ParseTuple(args, "sO", &operation, &raw)) return nullptr;
        auto t = phase_tensor(raw);
        std::string op(operation);
        if(op == "parities") return phase_output_terms(phase_as_parities(t));
        if(op == "scalar") return output_scalar(phase_scalar_value(t));
        if(op == "affine") {
            auto value = phase_affine_distribution(t);
            if(!value) Py_RETURN_NONE;
            Object total(output_scalar(value->first)), support(output_ket(value->second));
            return PyTuple_Pack(2, total.p, support.p);
        }
        throw std::invalid_argument("unknown phase value operation");
    } catch(...) { return error(); }
}
