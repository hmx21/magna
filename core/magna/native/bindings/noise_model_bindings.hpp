// Value/RNG adapters. Kernel choices and likelihood arithmetic live in noise_model.hpp.
Rational input_rational(PyObject* raw) {
    Seq p(raw);
    if(p.size() != 2) throw std::invalid_argument("rational shape");
    return {big(p[0]), big(p[1])};
}
std::vector<Rational> input_rationals(PyObject* raw) {
    Seq ps(raw);
    std::vector<Rational> out;
    for(size_t j = 0; j < ps.size(); j++) out.push_back(input_rational(ps[j]));
    return out;
}
NoiseSite* noise_site_handle(PyObject* cap) {
    return static_cast<NoiseSite*>(PyCapsule_GetPointer(cap, "magic-noise-site-v1"));
}
void noise_site_destructor(PyObject* cap) { delete noise_site_handle(cap); }
RandomIndex input_randrange(PyObject* rng) {
    return [rng](const Integer& bound) {
        Object n(output_big(bound)), raw(PyObject_CallMethod(rng, "randrange", "O", n.p));
        auto value = big(raw.p);
        if(value.neg || Integer::cmp(value, bound) >= 0)
            throw std::invalid_argument("RNG index outside requested range");
        return value;
    };
}
PyObject* make_noise_site(PyObject*, PyObject* args) {
    try {
        PyObject *anchor, *wires, *channels, *probability, *weights;
        if(!PyArg_ParseTuple(args, "OOOOO", &anchor, &wires, &channels, &probability, &weights)) return nullptr;
        auto site = std::make_unique<NoiseSite>();
        site->anchor = file_text(anchor);
        site->probability = input_rational(probability);
        Seq qs(wires);
        for(size_t j = 0; j < qs.size(); j++) site->qubits.push_back(natural(qs[j]));
        Seq rows(channels);
        for(size_t j = 0; j < rows.size(); j++) {
            Seq row(rows[j]);
            FileNoiseAction action;
            action.anchor = site->anchor;
            if(row.size() == 2 && PyUnicode_Check(row[0]) && file_text(row[0]) == "REPORT" && PyUnicode_Check(row[1]) &&
                    site->qubits.empty()) {
                action.anchor = "report";
                action.report = file_text(row[1]);
            } else {
                if(row.size() != site->qubits.size() || row.size() == 0)
                    throw std::invalid_argument("noise channel must match original wires or name one report");
                for(size_t k = 0; k < row.size(); k++) {
                    auto pauli = file_text(row[k]);
                    if(pauli.size() != 1 || std::string("IXYZ").find(pauli[0]) == std::string::npos)
                        throw std::invalid_argument("noise channel must match original wires or name one report");
                    if(pauli != "I") action.gates.push_back({pauli[0], site->qubits[k]});
                }
            }
            site->channels.push_back(std::move(action));
        }
        if(weights != Py_None) site->weights = input_rationals(weights);
        site->validate();
        return PyCapsule_New(site.release(), "magic-noise-site-v1", noise_site_destructor);
    } catch(...) { return error(); }
}
PyObject* sample_noise_model(PyObject*, PyObject* args) {
    try {
        PyObject *raw, *rng, *physical, *scale;
        if(!PyArg_ParseTuple(args, "OOOO", &raw, &rng, &physical, &scale)) return nullptr;
        Seq rows(raw);
        std::vector<const NoiseSite*> sites;
        for(size_t j = 0; j < rows.size(); j++) {
            auto* site = noise_site_handle(rows[j]);
            if(!site) return nullptr;
            sites.push_back(site);
        }
        std::optional<Rational> p, s;
        if(physical != Py_None) p = input_rational(physical);
        if(scale != Py_None) s = input_rational(scale);
        auto draw = draw_noise_model(sites, input_randrange(rng), p, s);
        Object faults(PyList_New(0)), flips(PyList_New(0));
        for(const auto& [report, action] : draw.actions) {
            if(report) {
                Object r(PyUnicode_FromString(action.report.c_str()));
                if(PyList_Append(flips.p, r.p) < 0) throw std::bad_alloc();
            } else
                for(const auto& [pauli, wire] : action.gates) {
                    char name[2] = {pauli, 0};
                    Object f(Py_BuildValue("(sns)", action.anchor.c_str(), Py_ssize_t(wire), name));
                    if(PyList_Append(faults.p, f.p) < 0) throw std::bad_alloc();
                }
        }
        Object n(output_big(draw.weight.n)), d(output_big(draw.weight.d));
        return Py_BuildValue("(OO(OO))", faults.p, flips.p, n.p, d.p);
    } catch(...) { return error(); }
}
