StabilizerTerms read_stabilizer_terms(PyObject* raw) {
    Seq rows(raw);
    StabilizerTerms terms;
    for(size_t j = 0; j < rows.size(); j++) {
        Seq row(rows[j]);
        if(row.size() != 2) throw std::invalid_argument("term shape");
        terms.push_back({scalar(row[0]), ket(row[1])});
    }
    return terms;
}
RandomBits random_bits(PyObject* rng) {
    return [rng](size_t count) {
        Object value(PyObject_CallMethod(rng, "getrandbits", "n", Py_ssize_t(count)));
        return bits(value.p);
    };
}
// Only pass native values into this region. Python RNG/callback adapters stay outside it.
template <class Function> auto native_work(Function work) -> decltype(work()) {
    auto* state = PyEval_SaveThread();
    try {
        auto value = work();
        PyEval_RestoreThread(state);
        return value;
    } catch(...) {
        PyEval_RestoreThread(state);
        throw;
    }
}
PyObject* marginal_operation(PyObject*, PyObject* args) {
    try {
        const char* operation;
        PyObject *raw, *w, *a = Py_None, *b = Py_None;
        if(!PyArg_ParseTuple(args, "sOO|OO", &operation, &raw, &w, &a, &b)) return nullptr;
        ProbabilityMarginal marginal{read_stabilizer_terms(raw), natural(w)};
        for(const auto& [c, k] : marginal.terms)
            if(k.width != marginal.width) throw std::invalid_argument("marginal width mismatch");
        std::string op(operation);
        if(op == "total") {
            auto rows = a == Py_None ? std::vector<Bits>() : bitvec(a);
            auto outcomes = b == Py_None ? std::vector<int>() : intvec(b);
            return output_scalar(native_work([&]() { return marginal.total(rows, outcomes); }));
        }
        if(op == "value") {
            auto word = bits(a);
            return output_scalar(native_work([&]() { return marginal.value(word); }));
        }
        if(op == "sample") {
            auto result = marginal.sample(random_bits(a));
            Object word(output_bits(result.first));
            return Py_BuildValue("(Os)", word.p, result.second.c_str());
        }
        throw std::invalid_argument("unknown marginal operation");
    } catch(...) { return error(); }
}
void marginals_destructor(PyObject* cap) {
    delete static_cast<RecordMarginals*>(PyCapsule_GetPointer(cap, "magic-record-marginals-v1"));
}
PyObject* record_marginals_operation(PyObject*, PyObject* args) {
    try {
        const char* operation;
        PyObject *raw, *a = Py_None, *b = Py_None, *c = Py_None;
        if(!PyArg_ParseTuple(args, "sO|OOO", &operation, &raw, &a, &b, &c)) return nullptr;
        std::string op(operation);
        if(op == "make") {
            auto plan = std::make_unique<RecordMarginals>();
            plan->rows = bitvec(raw);
            plan->events = bitvec(a);
            plan->outcomes = intvec(b);
            plan->max_plans = natural(c);
            if(!plan->max_plans || plan->events.size() != plan->outcomes.size())
                throw std::invalid_argument("record plan shape/limit");
            return PyCapsule_New(plan.release(), "magic-record-marginals-v1", marginals_destructor);
        }
        auto* plan = static_cast<RecordMarginals*>(PyCapsule_GetPointer(raw, "magic-record-marginals-v1"));
        if(!plan) return nullptr;
        if(op == "evaluate") {
            auto terms = read_stabilizer_terms(a);
            auto x = bits(b), z = bits(c);
            return phase_output_states(native_work([&]() { return plan->evaluate(terms, x, z); }).terms);
        }
        throw std::invalid_argument("unknown record marginal operation");
    } catch(...) { return error(); }
}
PyObject* expansion_operation(PyObject*, PyObject* args) {
    try {
        const char* operation;
        PyObject *raw, *a = Py_None, *b = Py_None;
        if(!PyArg_ParseTuple(args, "sO|OO", &operation, &raw, &a, &b)) return nullptr;
        std::string op(operation);
        Workspace ws;
        if(op == "contract_target") return output_ket(contract_target(ket(raw), ket(a), natural(b), ws));
        auto terms = read_stabilizer_terms(raw);
        if(op == "norm") return output_scalar(phase_norm(terms, ws));
        if(op == "pauli") return output_scalar(expansion_pauli(terms, bits(a), bits(b)));
        if(op == "transform") {
            auto x = bits(a), z = bits(b);
            for(auto& [c, k] : terms) k = pauli_ket(std::move(k), x, z);
            return phase_output_states(terms);
        }
        if(op == "target") return output_scalar(target_weight(terms, read_stabilizer_terms(a), natural(b)));
        throw std::invalid_argument("unknown expansion operation");
    } catch(...) { return error(); }
}
