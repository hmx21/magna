struct FileEnumerationBinding {
    FileEnumeration enumeration;
    PyObject* owner;
    FileEnumerationBinding(std::shared_ptr<CompiledExperiment> run, size_t weight, size_t patterns, size_t branches,
            size_t bits, PyObject* owner)
        : enumeration(std::move(run), weight, patterns, branches, bits), owner(Py_NewRef(owner)) {}
    ~FileEnumerationBinding() { Py_DECREF(owner); }
};
FileEnumerationBinding* file_enumeration_handle(PyObject* cap) {
    return static_cast<FileEnumerationBinding*>(PyCapsule_GetPointer(cap, "magic-file-enumeration-v1"));
}
void file_enumeration_destructor(PyObject* cap) { delete file_enumeration_handle(cap); }
PyObject* make_file_enumeration(PyObject*, PyObject* args) {
    try {
        PyObject *cap, *weight, *patterns, *branches, *bits;
        if(!PyArg_ParseTuple(args, "OOOOO", &cap, &weight, &patterns, &branches, &bits)) return nullptr;
        auto* owner = execution_handle(cap);
        if(!owner) return nullptr;
        auto requested = big(weight);
        if(requested.neg) throw std::invalid_argument("negative enumeration weight");
        CompiledExperiment::ExclusiveUse use(*owner->run);
        auto categories = owner->run->noise.categories.size();
        size_t bounded = Integer::cmp(requested, Integer(int64_t(categories))) >= 0
                ? categories
                : (requested.zero() ? 0 : requested.d[0]);
        auto result = std::make_unique<FileEnumerationBinding>(
                owner->run, bounded, natural(patterns), natural(branches), natural(bits), cap);
        return PyCapsule_New(result.release(), "magic-file-enumeration-v1", file_enumeration_destructor);
    } catch(...) { return execution_error(); }
}
PyObject* next_file_enumeration(PyObject*, PyObject* cap) {
    try {
        auto* state = file_enumeration_handle(cap);
        if(!state) return nullptr;
        auto* owner = execution_handle(state->owner);
        if(!owner) return nullptr;
        auto hooks = execution_hooks(*owner, Py_None);
        CompiledExperiment::ExclusiveUse use(*state->enumeration.run);
        auto result = native_work([&]() { return state->enumeration.next(hooks); });
        if(!result) Py_RETURN_NONE;
        Object selected(PyTuple_New(result->pattern.selected.size())),
                values(PyTuple_New(result->response.values.size())), probability(output_big(result->pattern.numerator));
        for(size_t j = 0; j < result->pattern.selected.size(); j++)
            PyTuple_SET_ITEM(selected.p, j,
                    Py_BuildValue("(nn)", Py_ssize_t(result->pattern.selected[j].first),
                            Py_ssize_t(result->pattern.selected[j].second)));
        for(size_t j = 0; j < result->response.values.size(); j++)
            PyTuple_SET_ITEM(values.p, j,
                    result->response.values[j] ? output_scalar(*result->response.values[j]) : Py_NewRef(Py_None));
        return Py_BuildValue("{s:O,s:O,s:O,s:n,s:n}", "selected", selected.p, "response", values.p, "probability",
                probability.p, "weight", Py_ssize_t(result->pattern.weight), "branches",
                Py_ssize_t(result->response.branches));
    } catch(...) { return execution_error(); }
}
PyObject* file_enumeration_state(PyObject*, PyObject* cap) {
    try {
        auto* state = file_enumeration_handle(cap);
        if(!state) return nullptr;
        const auto& enumeration = state->enumeration;
        Object out(Py_BuildValue("{s:n,s:n,s:n}", "required_patterns", Py_ssize_t(enumeration.patterns.size()),
                "completed_patterns", Py_ssize_t(enumeration.next_index), "feedback_branches",
                Py_ssize_t(enumeration.branches)));
        phase_set(out.p, "denominator", output_big(enumeration.denominator));
        return Py_NewRef(out.p);
    } catch(...) { return execution_error(); }
}
PyObject* file_enumeration_totals(PyObject*, PyObject* cap) {
    try {
        auto* state = file_enumeration_handle(cap);
        if(!state) return nullptr;
        const auto& enumeration = state->enumeration;
        CompiledExperiment::ExclusiveUse use(*enumeration.run);
        auto aggregate = native_work([&]() { return enumeration.aggregate(); });
        Object totals(output_low_weight_aggregation(aggregate, enumeration.denominator, enumeration.missing)),
                sectors(PyList_New(enumeration.sectors.size()));
        size_t j = 0;
        for(const auto& entry : enumeration.sectors) {
            auto weight = entry.first;
            const auto& rows = entry.second;
            auto values = native_work([&]() {
                return aggregate_exact(
                        rows, enumeration.denominator, enumeration.missing, enumeration.run->target_normalization);
            });
            Object summary(output_low_weight_aggregation(values, enumeration.denominator, enumeration.missing));
            PyList_SET_ITEM(
                    sectors.p, j++, Py_BuildValue("(nnO)", Py_ssize_t(weight), Py_ssize_t(rows.size()), summary.p));
        }
        return PyTuple_Pack(2, totals.p, sectors.p);
    } catch(...) { return execution_error(); }
}
