DecimalValue input_decimal_text(PyObject* raw) {
    auto text = file_text(raw);
    std::smatch match;
    static const std::regex pattern(R"(^([+-]?)([0-9]+)(?:\.([0-9]*))?(?:[eE]([+-]?[0-9]+))?$)");
    if(!std::regex_match(text, match, pattern)) throw std::invalid_argument("finite decimal required");
    int64_t exponent = match[4].matched ? std::stoll(match[4]) : 0;
    exponent -= int64_t(match[3].length());
    if(exponent < -1000000 || exponent > 1000000) throw std::length_error("decimal exponent budget");
    auto coefficient = integer_decimal(match[1].str() + match[2].str() + match[3].str());
    return {coefficient, exponent};
}
PyObject* scalar_decimal_value(PyObject*, PyObject* args) {
    try {
        PyObject *raw, *ratio = Py_None;
        if(!PyArg_ParseTuple(args, "O|O", &raw, &ratio)) return nullptr;
        auto probability = scalar(raw);
        std::optional<Rational> weight;
        if(ratio != Py_None) weight = input_rational(ratio);
        auto value = native_work(
                [&]() { return weight ? sample_weight_decimal(probability, *weight) : scalar_decimal(probability); });
        return PyUnicode_FromString(value.text().c_str());
    } catch(...) { return error(); }
}
ProposalEstimate* proposal_estimate(PyObject* cap) {
    return static_cast<ProposalEstimate*>(PyCapsule_GetPointer(cap, "magic-proposal-estimate-v1"));
}
void proposal_estimate_destructor(PyObject* cap) { delete proposal_estimate(cap); }
PyObject* make_proposal_estimate(PyObject*, PyObject*) {
    try {
        return PyCapsule_New(new ProposalEstimate(), "magic-proposal-estimate-v1", proposal_estimate_destructor);
    } catch(...) { return error(); }
}
PyObject* update_proposal_estimate(PyObject*, PyObject* args) {
    try {
        PyObject *cap, *raw = Py_None;
        if(!PyArg_ParseTuple(args, "O|O", &cap, &raw)) return nullptr;
        auto* estimate = proposal_estimate(cap);
        if(!estimate) return nullptr;
        if(raw == Py_None) {
            estimate->failure();
            Py_RETURN_NONE;
        }
        Seq row(raw);
        if(row.size() != 3) throw std::invalid_argument("estimate input shape");
        auto w = input_decimal_text(row[0]), a = input_decimal_text(row[1]), d = input_decimal_text(row[2]);
        native_work([&]() {
            estimate->add(w, a, d);
            return 0;
        });
        Py_RETURN_NONE;
    } catch(...) { return error(); }
}
PyObject* finish_proposal_estimate(PyObject*, PyObject* cap) {
    try {
        auto* estimate = proposal_estimate(cap);
        if(!estimate) return nullptr;
        auto result = native_work([&]() { return estimate->result(); });
        Object values(PyDict_New());
        for(const auto& [name, value] : result)
            phase_set(values.p, name.c_str(), value ? PyUnicode_FromString(value->text().c_str()) : Py_NewRef(Py_None));
        return Py_BuildValue("(nnOO)", Py_ssize_t(estimate->n), Py_ssize_t(estimate->zeros),
                estimate->sw.zero() ? Py_False : Py_True, values.p);
    } catch(...) { return error(); }
}

PyObject* sample_weight_value(PyObject*, PyObject* args) {
    try {
        PyObject *raw, *noise, *proposal;
        int logarithm = 0;
        if(!PyArg_ParseTuple(args, "OOO|p", &raw, &noise, &proposal, &logarithm)) return nullptr;
        auto probability = scalar(raw);
        auto weight = input_rational(noise) / input_rational(proposal);
        if(logarithm) return PyFloat_FromDouble(sample_weight_log2(probability, weight));
        auto result = sample_weight_decimal(probability, weight);
        return PyUnicode_FromString(result.text().c_str());
    } catch(...) { return error(); }
}
