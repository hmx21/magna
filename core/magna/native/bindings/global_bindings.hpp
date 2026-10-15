// Included after compiler/phase/control conversion helpers, inside bindings ns.
struct GlobalHandle {
    std::shared_ptr<GlobalPlan> owner;
    GlobalPlan& value;
    explicit GlobalHandle(std::shared_ptr<GlobalPlan> source) : owner(std::move(source)), value(*owner) {}
};
GlobalHandle* global_handle(PyObject* cap) {
    return static_cast<GlobalHandle*>(PyCapsule_GetPointer(cap, "magic-global-plan-v1"));
}
void global_destructor(PyObject* cap) { delete global_handle(cap); }
PyObject* global_info(const GlobalPlan& g) {
    auto& c = g.compiled;
    auto& p = g.plan();
    Object out(PyDict_New());
    phase_set(out.p, "engine", PyUnicode_FromString("native"));
    phase_set(out.p, "probability_kernel", PyUnicode_FromString("single-sided-trace"));
    phase_set(out.p, "native_global_compile", PyBool_FromLong(1));
    phase_set(out.p, "resource_order", PyUnicode_FromString("single-sided-trace"));
    phase_set(out.p, "retained_data_bytes", PyLong_FromSize_t(g.payload_bytes));
    phase_set(out.p, "skeleton_path_variables", PyLong_FromSize_t(c.path_variables));
    phase_set(out.p, "skeleton_support_variables", PyLong_FromSize_t(p.skeleton.columns.size()));
    phase_set(out.p, "projected_variables", PyLong_FromSize_t(p.basis.size()));
    phase_set(out.p, "retained_phase_legs", PyLong_FromSize_t(p.magic.size()));
    phase_set(out.p, "record_legs", PyLong_FromSize_t(p.nrecords));
    phase_set(out.p, "physical_width", PyLong_FromSize_t(p.physical));
    phase_set(out.p, "output_count", PyLong_FromSize_t(c.output_count));
    phase_set(out.p, "clifford_gate_count", PyLong_FromSize_t(c.gate_count));
    phase_set(out.p, "reference_layer", PyLong_FromLong(c.reference_layer));
    phase_set(out.p, "layer_slots", compiler_intvec(c.layer_slots));
    phase_set(out.p, "rotation_layers", PyLong_FromSize_t(c.rotation_layers));
    phase_set(out.p, "reference_layer_slots", PyLong_FromSize_t(c.reference_layer_slots));
    phase_set(out.p, "phase_ancillas", PyLong_FromSize_t(c.ancilla_count));
    phase_set(out.p, "shared_phase_coordinates", PyLong_FromSize_t(c.phase_coordinates));
    size_t nodes = 0;
    for(auto& [q, v] : p.suffix) nodes += v.size();
    phase_set(out.p, "fault_map_nodes", PyLong_FromSize_t(nodes));
    phase_set(out.p, "physical_boundary_injective", PyBool_FromLong(c.injective_physical));
    phase_set(out.p, "magic", compiler_intvec(p.magic));
    phase_set(out.p, "phase_controls", compiler_intvec(c.phase_controls));
    phase_set(out.p, "environment", compiler_intvec(c.environment));
    phase_set(out.p, "native_rule_controls", PyLong_FromSize_t(g.controls.size()));
    phase_set(out.p, "database_entries", PyLong_FromSize_t(g.database.entries.size()));
    phase_set(out.p, "compiler_seconds", PyFloat_FromDouble(g.compiler_seconds));
    phase_set(out.p, "compiler_parts",
            Py_BuildValue("{s:d,s:d,s:d,s:d,s:d}", "dilation_seconds", c.dilation_seconds, "path_seconds",
                    c.path_seconds, "contraction_seconds", c.contraction_seconds, "projection_seconds",
                    c.projection_seconds, "fault_map_seconds", c.fault_map_seconds));
    Py_INCREF(out.p);
    return out.p;
}
PyObject* debug_global_compiler(PyObject*, PyObject* cap) {
    try {
        auto* h = global_handle(cap);
        if(!h) return nullptr;
        return output_compiler(h->value.compiled);
    } catch(...) { return error(); }
}
