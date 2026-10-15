// Value conversion only; statistical state and decisions live in C++.
void efficiency_field(PyObject* out, const char* name, std::optional<ScientificValue> value) {
    auto ranged = ranged_value(value);
    phase_set(out, name, ranged.ordinary ? PyFloat_FromDouble(*ranged.ordinary) : Py_NewRef(Py_None));
    phase_set(out, (std::string(name) + "_scientific").c_str(),
            value ? Py_BuildValue("(dL)", value->first, (long long)value->second) : Py_NewRef(Py_None));
}
PyObject* output_efficiency_summary(const HistoryEfficiency& value) {
    auto s = value.summary();
    Object out(Py_BuildValue("{s:s,s:n,s:n}", "status", s.status, "completed", Py_ssize_t(value.count), "zero_weights",
            Py_ssize_t(value.zeros)));
    efficiency_field(out.p, "kappa", s.kappa);
    efficiency_field(out.p, "variance_coefficient", s.variance);
    efficiency_field(out.p, "exact_marginal_variance_coefficient", s.reference_variance);
    efficiency_field(out.p, "ess", s.ess);
    efficiency_field(out.p, "rho", s.rho);
    efficiency_field(out.p, "mean_weight", s.mean_weight);
    phase_set(out.p, "comparison",
            PyUnicode_FromString(
                    "same noise proposal and terminal estimator; exact_marginal draws histories from their "
                    "true conditionals"));
    return Py_NewRef(out.p);
}
PyObject* decide_file_sampler(PyObject*, PyObject* args) {
    try {
        double kappa, capped, born;
        PyObject* raw_count;
        if(!PyArg_ParseTuple(args, "dddO", &kappa, &capped, &born, &raw_count)) return nullptr;
        auto result = sampler_decision(kappa, capped, born, natural(raw_count));
        return Py_BuildValue("{s:d,s:d,s:d,s:s}", "gamma", result.gamma, "affine_support_seconds_per_draw",
                result.affine_support_seconds_per_draw, "exact_marginal_seconds_per_draw",
                result.exact_marginal_seconds_per_draw, "recommended_sampler", result.recommended);
    } catch(...) { return error(); }
}
