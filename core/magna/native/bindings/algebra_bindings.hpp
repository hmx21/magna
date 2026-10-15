// Adapt public Python values to the shared native algebra.
PyObject* output_bit_tuple(const std::vector<Bits>& values) {
    Object out(PyTuple_New(values.size()));
    for(size_t j = 0; j < values.size(); j++) PyTuple_SET_ITEM(out.p, j, output_bits(values[j]));
    Py_INCREF(out.p);
    return out.p;
}
PyObject* output_int_tuple(const std::vector<int>& values) {
    Object out(PyTuple_New(values.size()));
    for(size_t j = 0; j < values.size(); j++) PyTuple_SET_ITEM(out.p, j, PyLong_FromLong(values[j]));
    Py_INCREF(out.p);
    return out.p;
}
PyObject* image_basis(PyObject*, PyObject* rows) {
    try {
        return output_bit_tuple(phase_image_basis(bitvec(rows)));
    } catch(...) { return error(); }
}
PyObject* algebra_scalar(PyObject*, PyObject* args) {
    try {
        const char* op;
        PyObject *a, *b = Py_None;
        if(!PyArg_ParseTuple(args, "sO|O", &op, &a, &b)) return nullptr;
        std::string name(op);
        auto value = scalar(a);
        if(name == "normalize") return output_scalar(value);
        if(name == "add") return output_scalar(value + scalar(b));
        if(name == "multiply") return output_scalar(value * scalar(b));
        if(name == "conjugate") return output_scalar(value.conjugate());
        if(name == "scale") {
            Scalar scale;
            scale.c[0] = big(b);
            scale.normalize();
            return output_scalar(value * scale);
        }
        if(name == "ratio") {
            auto ratio = scaled_real_ratio(value, scalar(b));
            if(ratio.second > std::numeric_limits<int>::max()) {
                PyErr_SetString(PyExc_OverflowError, "ratio overflow");
                return nullptr;
            }
            return PyFloat_FromDouble(
                    ratio.second < std::numeric_limits<int>::min() ? 0 : std::ldexp(ratio.first, int(ratio.second)));
        }
        if(name == "log2") return PyFloat_FromDouble(scalar_log2(value));
        if(name == "complex") {
            auto result = scalar_complex(value);
            return PyComplex_FromDoubles(result.real(), result.imag());
        }
        throw std::invalid_argument("unknown scalar operation");
    } catch(...) { return error(); }
}
PyObject* algebra_ket(PyObject*, PyObject* args) {
    try {
        const char* op;
        PyObject *raw, *a = Py_None, *b = Py_None;
        if(!PyArg_ParseTuple(args, "sO|OO", &op, &raw, &a, &b)) return nullptr;
        auto k = ket(raw);
        std::string name(op);
        if(name == "validate") {
            validate_ket(k);
            Py_RETURN_NONE;
        }
        if(name == "norm") return output_scalar(norm(k));
        if(name == "project") return output_ket(project_ket(k, bitvec(a), intvec(b)));
        if(name == "pauli") return output_ket(pauli_ket(std::move(k), bits(a), bits(b)));
        if(name == "sample") {
            auto values = sample_ket_rows(k, bitvec(a), [&]() {
                Object value(PyObject_CallMethod(b, "integers", "i", 2));
                return index_int(value.p);
            });
            return output_int_tuple(values);
        }
        throw std::invalid_argument("unknown ket operation");
    } catch(...) { return error(); }
}
