PyObject* output_scientific(std::optional<ScientificValue> value) {
    return value ? Py_BuildValue("(dL)", value->first, (long long)value->second) : Py_NewRef(Py_None);
}
PyObject* output_ordinary(std::optional<double> value) {
    return value ? PyFloat_FromDouble(*value) : Py_NewRef(Py_None);
}
PyObject* output_history_contribution(const HistoryContribution& value) {
    Object ordinary(PyList_New(value.values.size())), scientific(PyList_New(value.values.size())),
            underflow(PyList_New(0)), overflow(PyList_New(0));
    for(size_t j = 0; j < value.values.size(); j++) {
        PyList_SET_ITEM(ordinary.p, j, output_ordinary(value.values[j].ordinary));
        PyList_SET_ITEM(scientific.p, j, output_scientific(value.values[j].scientific));
        Object index(PyLong_FromSize_t(j));
        if(value.values[j].underflow && PyList_Append(underflow.p, index.p) < 0) throw std::bad_alloc();
        if(value.values[j].overflow && PyList_Append(overflow.p, index.p) < 0) throw std::bad_alloc();
    }
    Object moments(PyList_GetSlice(ordinary.p, 2, value.values.size())), weight(output_scientific(value.weight));
    return Py_BuildValue("{s:O,s:O,s:O,s:O,s:O,s:O,s:O}", "acceptance", PyList_GET_ITEM(ordinary.p, 0), "error",
            PyList_GET_ITEM(ordinary.p, 1), "moments", moments.p, "scientific", scientific.p, "underflow_columns",
            underflow.p, "overflow_columns", overflow.p, "weight_scientific", weight.p);
}
PyObject* output_scientific_summary(const ScientificSummary& summary) {
    if(!summary.count) return Py_BuildValue("{s:O,s:O}", "estimates", Py_None, "standard_errors", Py_None);
    Object out(PyDict_New()), underflow(PyList_New(0)), overflow(PyList_New(0));
    for(const auto& entry : std::vector<std::pair<std::string, const std::vector<RangedValue>*>>{
                {"estimates", &summary.estimates}, {"standard_errors", &summary.standard_errors}}) {
        Object ordinary(PyDict_New()), science(PyDict_New()), moments(PyList_New(entry.second->size() - 3)),
                scientific_moments(PyList_New(entry.second->size() - 3));
        for(size_t j = 0; j < entry.second->size(); j++) {
            const auto& value = entry.second->at(j);
            std::string name = j < 3 ? std::vector<std::string>{"acceptance", "error", "conditional_error"}[j]
                                     : "moments." + std::to_string(j - 3);
            Object number(output_ordinary(value.ordinary)),
                    scientific(value.scientific ? Py_BuildValue("{s:d,s:L}", "mantissa", value.scientific->first,
                                                          "exponent2", (long long)value.scientific->second)
                                                : Py_NewRef(Py_None));
            if(j < 3) {
                phase_set(ordinary.p, name.c_str(), Py_NewRef(number.p));
                phase_set(science.p, name.c_str(), Py_NewRef(scientific.p));
            } else {
                PyList_SET_ITEM(moments.p, j - 3, Py_NewRef(number.p));
                PyList_SET_ITEM(scientific_moments.p, j - 3, Py_NewRef(scientific.p));
            }
            Object label(PyUnicode_FromString((entry.first + "." + name).c_str()));
            if(value.underflow && PyList_Append(underflow.p, label.p) < 0) throw std::bad_alloc();
            if(value.overflow && PyList_Append(overflow.p, label.p) < 0) throw std::bad_alloc();
        }
        phase_set(ordinary.p, "moments", Py_NewRef(moments.p));
        phase_set(science.p, "moments", Py_NewRef(scientific_moments.p));
        phase_set(out.p, entry.first.c_str(), Py_NewRef(ordinary.p));
        phase_set(out.p, (entry.first + "_scientific").c_str(), Py_NewRef(science.p));
    }
    phase_set(out.p, "numerics",
            Py_BuildValue("{s:s,s:O,s:O}", "representation", "mantissa * 2**exponent2", "underflow_fields", underflow.p,
                    "overflow_fields", overflow.p));
    return Py_NewRef(out.p);
}
