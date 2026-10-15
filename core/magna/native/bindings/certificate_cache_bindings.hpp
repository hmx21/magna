CertificateCache* certificate_cache(PyObject* cap) {
    return static_cast<CertificateCache*>(PyCapsule_GetPointer(cap, "magic-certificate-cache-v1"));
}
void certificate_cache_destructor(PyObject* cap) { delete certificate_cache(cap); }
PyObject* make_certificate_cache(PyObject*, PyObject* args) {
    try {
        PyObject *budget, *entries;
        if(!PyArg_ParseTuple(args, "OO", &budget, &entries)) return nullptr;
        auto cache = std::make_unique<CertificateCache>(natural(budget), natural(entries));
        return PyCapsule_New(cache.release(), "magic-certificate-cache-v1", certificate_cache_destructor);
    } catch(...) { return error(); }
}
PyObject* certificate_cache_operation(PyObject*, PyObject* args) {
    try {
        const char* op;
        PyObject *cap, *key = Py_None, *blob = Py_None, *probability = Py_None, *extra = Py_None;
        double cost = 0;
        if(!PyArg_ParseTuple(args, "sO|OOOOd", &op, &cap, &key, &blob, &probability, &extra, &cost)) return nullptr;
        auto* cache = certificate_cache(cap);
        if(!cache) return nullptr;
        std::string operation(op);
        if(operation == "get") {
            auto entry = cache->get(file_text(key));
            if(!entry) Py_RETURN_NONE;
            return PyBytes_FromStringAndSize(entry->blob.data(), Py_ssize_t(entry->blob.size()));
        }
        if(operation == "put") {
            char* data;
            Py_ssize_t length;
            if(PyBytes_AsStringAndSize(blob, &data, &length) < 0) return nullptr;
            std::optional<Scalar> value;
            if(probability != Py_None) value = scalar(probability);
            cache->put({file_text(key), std::string(data, size_t(length)), value, natural(extra), cost});
            Py_RETURN_NONE;
        }
        throw std::invalid_argument("unknown certificate cache operation");
    } catch(...) { return error(); }
}
