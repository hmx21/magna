// Fixed-resolution physical noise descriptors and the scalar Philox CPU control.
// Included after cuda_experiment_bindings.hpp; no CUDA dependency in this module.

struct CudaPhilox {
    uint64_t shot, seed, block = 0, saved = 0;
    bool has_saved = false;
    CudaPhilox(uint64_t s, uint64_t k) : shot(s), seed(k) {}
    uint64_t operator()() {
        if(has_saved) { has_saved = false; return saved; }
        uint32_t a = uint32_t(shot), b = uint32_t(shot >> 32);
        uint32_t c = uint32_t(block), d = uint32_t(block >> 32);
        uint32_t k0 = uint32_t(seed), k1 = uint32_t(seed >> 32);
        block++;
        for(unsigned round = 0; round < 10; round++) {
            uint64_t p0 = uint64_t(0xd2511f53U) * a, p1 = uint64_t(0xcd9e8d57U) * c;
            a = uint32_t(p1 >> 32) ^ b ^ k0;
            b = uint32_t(p1);
            c = uint32_t(p0 >> 32) ^ d ^ k1;
            d = uint32_t(p0);
            k0 += 0x9e3779b9U;
            k1 += 0xbb67ae85U;
        }
        saved = uint64_t(c) | (uint64_t(d) << 32);
        has_saved = true;
        return uint64_t(a) | (uint64_t(b) << 32);
    }
};

PyObject* cuda_noise_metadata(PyObject*, PyObject* cap) {
    try {
        auto* bridge = cuda_experiment_handle(cap);
        if(!bridge) return nullptr;
        CompiledExperiment::ExclusiveUse use(*bridge->run);
        cuda_validate_run(*bridge);
        const auto& noise = bridge->run->noise;
        if(noise.categories.size() > std::numeric_limits<uint32_t>::max())
            throw std::length_error("CUDA noise site count exceeds uint32");
        std::vector<uint64_t> thresholds, conditional_thresholds;
        std::vector<uint32_t> block_offsets{0}, sites, choice_offsets{0}, choices;
        std::vector<uint8_t> certain, uniform, conditional_certain;
        for(const auto& group : noise.occurrences) {
            for(size_t first = 0; first < group.sites.size(); first += 64) {
                thresholds.push_back(group.coin.threshold);
                certain.push_back(group.coin.certain);
                for(size_t j = first; j < std::min(first + 64, group.sites.size()); j++)
                    sites.push_back(uint32_t(group.sites[j]));
                block_offsets.push_back(uint32_t(sites.size()));
            }
        }
        for(size_t site = 0; site < noise.categories.size(); site++) {
            const auto& category = *noise.categories[site];
            uniform.push_back(category.uniform);
            for(size_t j = 0; j < category.nonzero.size(); j++) {
                choices.push_back(uint32_t(noise.offsets[site] + category.nonzero[j]));
                if(!category.uniform && j < category.conditional.size()) {
                    conditional_thresholds.push_back(category.conditional[j].threshold);
                    conditional_certain.push_back(category.conditional[j].certain);
                } else {
                    conditional_thresholds.push_back(0);
                    conditional_certain.push_back(0);
                }
            }
            choice_offsets.push_back(uint32_t(choices.size()));
        }
        Object out(Py_BuildValue("{s:n,s:n}", "sites", Py_ssize_t(noise.categories.size()),
                                "blocks", Py_ssize_t(thresholds.size())));
        phase_set(out.p, "thresholds", cuda_output_bytes(thresholds));
        phase_set(out.p, "certain", cuda_output_bytes(certain));
        phase_set(out.p, "block_offsets", cuda_output_bytes(block_offsets));
        phase_set(out.p, "block_sites", cuda_output_bytes(sites));
        phase_set(out.p, "choice_offsets", cuda_output_bytes(choice_offsets));
        phase_set(out.p, "choices", cuda_output_bytes(choices));
        phase_set(out.p, "uniform", cuda_output_bytes(uniform));
        phase_set(out.p, "conditional_thresholds", cuda_output_bytes(conditional_thresholds));
        phase_set(out.p, "conditional_certain", cuda_output_bytes(conditional_certain));
        phase_set(out.p, "has_importance_weights", PyBool_FromLong(bridge->has_importance_weights));
        phase_set(out.p, "weighted_sites", cuda_output_bytes(bridge->weighted_sites));
        phase_set(out.p, "likelihood_mantissas", cuda_output_bytes(bridge->likelihood_mantissas));
        phase_set(out.p, "likelihood_exponents", cuda_output_bytes(bridge->likelihood_exponents));
        phase_set(out.p, "likelihood_values", cuda_output_bytes(bridge->likelihood_values));
        Object exact_weights(PyTuple_New(bridge->likelihood_values.size()));
        for(size_t count = 0; count < bridge->likelihood_values.size(); count++) {
            if(bridge->has_importance_weights) {
                const auto& value = noise.count_weights[0][count];
                Object n(output_big(value.n)), d(output_big(value.d));
                PyTuple_SET_ITEM(exact_weights.p, count, PyTuple_Pack(2, n.p, d.p));
            } else PyTuple_SET_ITEM(exact_weights.p, count, Py_BuildValue("(ii)", 1, 1));
        }
        phase_set(out.p, "likelihood_exact", Py_NewRef(exact_weights.p));
        return Py_NewRef(out.p);
    } catch(...) { return execution_error(); }
}

PyObject* cuda_noise_validate(PyObject*, PyObject* cap) {
    try {
        auto* bridge = cuda_experiment_handle(cap);
        if(!bridge) return nullptr;
        CompiledExperiment::ExclusiveUse use(*bridge->run);
        cuda_validate_run(*bridge);
        Py_RETURN_NONE;
    } catch(...) { return execution_error(); }
}

uint64_t cuda_noise_uint64(PyObject* value) {
    if(!PyLong_CheckExact(value)) throw std::invalid_argument("CUDA noise counters must be uint64 integers");
    auto result = PyLong_AsUnsignedLongLong(value);
    if(PyErr_Occurred()) throw std::invalid_argument("CUDA noise counter outside uint64");
    return uint64_t(result);
}

PyObject* cuda_noise_cpu_sample(PyObject*, PyObject* args) {
    try {
        PyObject *cap, *raw_shots, *raw_seed, *raw_offset;
        int record = 0, weighted = 0;
        if(!PyArg_ParseTuple(args, "OOOO|pp", &cap, &raw_shots, &raw_seed, &raw_offset, &record, &weighted)) return nullptr;
        auto* bridge = cuda_experiment_handle(cap);
        if(!bridge) return nullptr;
        auto shots = cuda_noise_uint64(raw_shots), seed = cuda_noise_uint64(raw_seed);
        auto offset = cuda_noise_uint64(raw_offset);
        if(shots > cuda_experiment_max_shots || (shots && offset > UINT64_MAX - (shots - 1)))
            throw std::invalid_argument("CUDA noise shot count or offset limit");
        CompiledExperiment::ExclusiveUse use(*bridge->run);
        cuda_validate_run(*bridge);
        if(bridge->has_importance_weights && !weighted)
            throw std::invalid_argument("importance draws require the weighted noise interface");
        const auto& noise = bridge->run->noise;
        size_t stride = 2 * bridge->words;
        std::vector<uint64_t> frames(size_t(shots) * stride), offsets;
        std::vector<uint32_t> indices;
        std::vector<double> likelihoods;
        if(weighted) likelihoods.resize(size_t(shots));
        if(record) offsets.push_back(0);
        native_work([&]() {
            SignalCheck signals;
            for(uint64_t shot = 0; shot < shots; shot++) {
                CudaPhilox rng(offset + shot, seed);
                auto* frame = frames.data() + size_t(shot) * stride;
                std::copy(bridge->base.begin(), bridge->base.end(), frame);
                size_t first_index = indices.size();
                size_t weight_count = 0;
                auto select = [&](size_t site) {
                    const auto& category = *noise.categories[site];
                    size_t selected = category.nonzero.size() - 1;
                    if(category.uniform) selected = uniform_index(category.nonzero.size(), rng);
                    else for(size_t j = 0; j < category.conditional.size(); j++)
                        if(category.conditional[j].draw(rng)) { selected = j; break; }
                    size_t action = noise.offsets[site] + category.nonzero[selected];
                    const auto* column = bridge->columns.data() + action * stride;
                    for(size_t word = 0; word < stride; word++) frame[word] ^= column[word];
                    if(record) indices.push_back(uint32_t(action));
                    if(weighted) weight_count += bridge->weighted_sites[site];
                };
                for(const auto& group : noise.occurrences) {
                    for(size_t first = 0; first < group.sites.size(); first += 64) {
                        size_t count = std::min(size_t(64), group.sites.size() - first);
                        if(count < 8) {
                            for(size_t j = 0; j < count; j++)
                                if(group.coin.draw(rng)) select(group.sites[first + j]);
                        } else {
                            uint64_t live = count == 64 ? UINT64_MAX : (uint64_t(1) << count) - 1;
                            auto selected = group.coin.draw_bits(rng, live);
                            while(selected) {
                                size_t bit = unsigned(__builtin_ctzll(selected));
                                selected &= selected - 1;
                                select(group.sites[first + bit]);
                            }
                        }
                    }
                }
                if(record) {
                    std::sort(indices.begin() + first_index, indices.end());
                    offsets.push_back(indices.size());
                }
                if(weighted) likelihoods[size_t(shot)] = bridge->likelihood_values.at(weight_count);
                signals();
            }
            return 0;
        });
        Object output(cuda_output_bytes(frames));
        if(weighted) {
            Object weights(cuda_output_bytes(likelihoods));
            if(!record) return PyTuple_Pack(2, output.p, weights.p);
            Object raw_offsets(cuda_output_bytes(offsets)), raw_indices(cuda_output_bytes(indices));
            return PyTuple_Pack(4, output.p, weights.p, raw_offsets.p, raw_indices.p);
        }
        if(!record) return Py_NewRef(output.p);
        Object raw_offsets(cuda_output_bytes(offsets)), raw_indices(cuda_output_bytes(indices));
        return PyTuple_Pack(3, output.p, raw_offsets.p, raw_indices.p);
    } catch(...) { return execution_error(); }
}
