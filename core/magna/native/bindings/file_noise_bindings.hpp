// CPython conversion and ownership for noise.hpp.
PyObject* output_file_noise_action(const FileNoiseAction& a) {
    Object anchor(PyUnicode_FromString(a.anchor.c_str()));
    if(a.anchor == "report") return Py_BuildValue("(Os)", anchor.p, a.report.c_str());
    Object gates(PyTuple_New(a.gates.size()));
    for(size_t j = 0; j < a.gates.size(); j++) {
        char pauli[2] = {a.gates[j].first, 0};
        PyTuple_SET_ITEM(gates.p, j, Py_BuildValue("(sn)", pauli, Py_ssize_t(a.gates[j].second)));
    }
    return Py_BuildValue("(OO)", anchor.p, gates.p);
}
using FileNoiseOwner = std::shared_ptr<const FileNoisePlan>;
FileNoiseOwner* file_noise_owner(PyObject* cap) {
    return static_cast<FileNoiseOwner*>(PyCapsule_GetPointer(cap, "magic-file-noise-v1"));
}
const FileNoisePlan* file_noise_handle(PyObject* cap) {
    auto* owner = file_noise_owner(cap);
    return owner ? owner->get() : nullptr;
}
void file_noise_destructor(PyObject* cap) { delete file_noise_owner(cap); }
PyObject* own_file_noise(std::unique_ptr<FileNoisePlan> plan) {
    auto owner = std::make_unique<FileNoiseOwner>(std::move(plan));
    auto* cap = PyCapsule_New(owner.get(), "magic-file-noise-v1", file_noise_destructor);
    if(cap) owner.release();
    return cap;
}
PyObject* compile_file_noise(PyObject*, PyObject* args) {
    try {
        PyObject *cap, *raw = Py_None;
        if(!PyArg_ParseTuple(args, "O|O", &cap, &raw)) return nullptr;
        auto* circuit = circuit_handle(cap);
        if(!circuit) return nullptr;
        std::optional<Rational> scale;
        if(raw != Py_None) {
            Seq p(raw);
            if(p.size() != 2) throw std::invalid_argument("proposal scale shape");
            scale = Rational(big(p[0]), big(p[1]));
        }
        auto plan = std::make_unique<FileNoisePlan>(native_work([&]() { return prepare_file_noise(*circuit, scale); }));
        Object metadata(Py_BuildValue("{s:n,s:n,s:n,s:n,s:n}", "categories", Py_ssize_t(plan->categories.size()),
                "actions", Py_ssize_t(plan->actions.size()), "tilted", Py_ssize_t(plan->tilted), "guarded",
                Py_ssize_t(plan->guarded), "deterministic", Py_ssize_t(plan->deterministic)));
        Object out(own_file_noise(std::move(plan)));
        if(!out.p) return nullptr;
        return PyTuple_Pack(2, out.p, metadata.p);
    } catch(...) { return error(); }
}
PyObject* file_noise_probabilities(PyObject*, PyObject* cap) {
    try {
        auto* plan = file_noise_handle(cap);
        if(!plan) return nullptr;
        Object out(PyTuple_New(plan->categories.size()));
        for(size_t j = 0; j < plan->categories.size(); j++) {
            const auto& ps = plan->categories[j]->probabilities;
            Object row(PyTuple_New(ps.size()));
            for(size_t k = 0; k < ps.size(); k++) {
                Object n(output_big(ps[k].n)), d(output_big(ps[k].d));
                PyTuple_SET_ITEM(row.p, k, PyTuple_Pack(2, n.p, d.p));
            }
            PyTuple_SET_ITEM(out.p, j, Py_NewRef(row.p));
        }
        return Py_NewRef(out.p);
    } catch(...) { return error(); }
}

PyObject* file_noise_actions(PyObject*, PyObject* cap) {
    try {
        auto* p = file_noise_handle(cap);
        if(!p) return nullptr;
        if(!p->has_actions) throw std::invalid_argument("noise action table required");
        Object rows(PyTuple_New(p->categories.size()));
        for(size_t j = 0; j < p->categories.size(); j++) {
            Object row(PyTuple_New(p->categories[j]->size));
            for(size_t k = 0; k < p->categories[j]->size; k++)
                PyTuple_SET_ITEM(row.p, k, output_file_noise_action(p->actions[p->offsets[j] + k]));
            Py_INCREF(row.p);
            PyTuple_SET_ITEM(rows.p, j, row.p);
        }
        Py_INCREF(rows.p);
        return rows.p;
    } catch(...) { return error(); }
}
