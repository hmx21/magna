// Optional low-weight analysis. No work is added to ordinary query preparation.

PyObject* output_low_weight_aggregation(
        const ExactAggregation& aggregation, const Integer& denominator, const std::vector<bool>& missing) {
    const auto& sums = aggregation.sums;
    const auto& covered = aggregation.covered;
    auto omitted = denominator - covered;
    auto d = file_integer_scalar(denominator);
    Object exact(PyTuple_New(sums.size())), science(PyTuple_New(sums.size()));
    for(size_t i = 0; i < sums.size(); i++) {
        if(missing[i]) {
            Py_INCREF(Py_None);
            PyTuple_SET_ITEM(exact.p, i, Py_None);
            Py_INCREF(Py_None);
            PyTuple_SET_ITEM(science.p, i, Py_None);
        } else {
            PyTuple_SET_ITEM(exact.p, i, output_scalar(sums[i]));
            auto v = scaled_real_ratio(sums[i], i == 2 ? d * aggregation.error_normalization : d);
            PyTuple_SET_ITEM(science.p, i, Py_BuildValue("(dL)", v.first, (long long)v.second));
        }
    }
    Object bounds(PyDict_New());
    for(const auto& entry : aggregation.bounds) {
        const auto& a = entry.second.first;
        const auto& b = entry.second.second;
        phase_set(bounds.p, entry.first.c_str(),
                Py_BuildValue("((dL)(dL))", a.first, (long long)a.second, b.first, (long long)b.second));
    }
    if(aggregation.conditional_bound_missing) {
        Py_INCREF(Py_None);
        phase_set(bounds.p, "conditional_error", Py_None);
    }
    PyObject* conditional_raw;
    if(aggregation.has_conditional) {
        const auto& v = aggregation.conditional;
        conditional_raw = Py_BuildValue("(dL)", v.first, (long long)v.second);
    } else {
        Py_INCREF(Py_None);
        conditional_raw = Py_None;
    }
    Object conditional(conditional_raw);
    Object covered_raw(output_big(covered)), omitted_raw(output_big(omitted)), d_raw(output_big(denominator));
    Object out(Py_BuildValue("{s:O,s:O,s:O,s:O,s:O,s:O,s:O}", "numerators", exact.p, "scientific", science.p,
            "denominator", d_raw.p, "covered_numerator", covered_raw.p, "omitted_numerator", omitted_raw.p,
            "bounds_scientific", bounds.p, "conditional_scientific", conditional.p));
    phase_set(out.p, "error_normalization", output_scalar(aggregation.error_normalization));
    return Py_NewRef(out.p);
}
