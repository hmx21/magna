// Optional CUDA plan bridge. Device kernels live in native/cuda and are loaded
// lazily through Python/CuPy. This binding exports exact integer maps and keeps
// the original native probability kernel available as a validation control.

struct CudaExperimentBinding {
    std::shared_ptr<CompiledExperiment> run;
    std::shared_ptr<FileBoundary> engine;
    std::shared_ptr<FileBoundary> target;
    size_t frame_bits = 0, words = 0;
    std::vector<uint64_t> columns, base;
    std::vector<size_t> action_sites;
    bool has_importance_weights = false;
    std::vector<uint8_t> weighted_sites;
    std::vector<double> likelihood_mantissas, likelihood_values;
    std::vector<int32_t> likelihood_exponents;
};
constexpr size_t cuda_experiment_max_shots = 1000000;
CudaExperimentBinding* cuda_experiment_handle(PyObject* cap) {
    return static_cast<CudaExperimentBinding*>(PyCapsule_GetPointer(cap, "magna-cuda-experiment-v1"));
}
void cuda_experiment_destructor(PyObject* cap) { delete cuda_experiment_handle(cap); }
template <class T> PyObject* cuda_output_bytes(const std::vector<T>& values) {
    if(values.size() > size_t(PY_SSIZE_T_MAX) / sizeof(T)) throw std::length_error("CUDA byte buffer too large");
    return PyBytes_FromStringAndSize(reinterpret_cast<const char*>(values.data()), Py_ssize_t(values.size() * sizeof(T)));
}
template <class T> std::vector<T> cuda_read_bytes(PyObject* raw) {
    if(!PyBytes_CheckExact(raw)) throw std::invalid_argument("CUDA bridge requires immutable bytes buffers");
    auto bytes = size_t(PyBytes_GET_SIZE(raw));
    if(bytes % sizeof(T)) throw std::invalid_argument("CUDA byte buffer has an incomplete integer");
    std::vector<T> out(bytes / sizeof(T));
    if(bytes) std::memcpy(out.data(), PyBytes_AS_STRING(raw), bytes);
    return out;
}
void cuda_validate_run(const CudaExperimentBinding& bridge) {
    const auto& run = *bridge.run;
    const auto stage = run.source.stage("recovery");
    auto found = run.plans.prepared.find(stage);
    if(!run.complete || found == run.plans.prepared.end() || found->second != bridge.engine)
        throw std::invalid_argument("compiled circuit was reconfigured; recreate the CUDA experiment");
    if(bridge.target) {
        auto target = run.plans.prepared.find("target");
        if(target == run.plans.prepared.end() || target->second != bridge.target)
            throw std::invalid_argument("target was reconfigured; recreate the CUDA experiment");
    }
}
void cuda_pack_frame(const Frame& frame, size_t words, uint64_t* out) {
    if(frame.x.w.size() > words || frame.z.w.size() > words) throw std::logic_error("CUDA frame width mismatch");
    std::fill(out, out + 2 * words, uint64_t(0));
    std::copy(frame.x.w.begin(), frame.x.w.end(), out);
    std::copy(frame.z.w.begin(), frame.z.w.end(), out + words);
}

// A single native count table is an exact sufficient statistic for these
// homogeneous, independent importance proposals. Mixed factor groups and
// channels requiring draw-time arbitrary precision remain unsupported.
void cuda_prepare_likelihood(CudaExperimentBinding& bridge) {
    const auto& noise = bridge.run->noise;
    bridge.weighted_sites.assign(noise.categories.size(), 0);
    if(noise.factor_powers.empty()) {
        bridge.likelihood_mantissas = {.5};
        bridge.likelihood_exponents = {1};
        bridge.likelihood_values = {1.};
        return;
    }
    auto unsupported = []() {
        throw std::invalid_argument("CUDA importance sampling requires one complete native homogeneous count table");
    };
    if(noise.count_weights.size() != 1 || noise.count_groups.size() != noise.categories.size() ||
            noise.factor_groups.size() != noise.categories.size()) unsupported();
    size_t counted = 0;
    for(size_t site = 0; site < noise.categories.size(); site++) {
        if(noise.count_groups[site] == 0) {
            if(noise.categories[site]->occurrence.certain || !noise.categories[site]->occurrence.threshold)
                throw std::invalid_argument("CUDA importance proposal must have a nontrivial uint64 occurrence probability");
            bridge.weighted_sites[site] = 1;
            counted++;
        } else if(noise.count_groups[site] != -1 || noise.factor_groups[site] != std::array<int64_t, 2>{-1, -1})
            unsupported();
    }
    if(counted < 2 || counted > 512 || noise.count_weights[0].size() != counted + 1 ||
            noise.count_scientific.size() != counted + 1) unsupported();
    for(const auto& [mantissa, exponent] : noise.count_scientific) {
        if(exponent < std::numeric_limits<int32_t>::min() || exponent > std::numeric_limits<int32_t>::max())
            throw std::invalid_argument("CUDA likelihood exponent exceeds int32 range");
        double value = std::ldexp(mantissa, int(exponent));
        if(!std::isfinite(value) || value <= 0)
            throw std::invalid_argument("CUDA importance likelihood table exceeds positive float64 range");
        bridge.likelihood_mantissas.push_back(mantissa);
        bridge.likelihood_exponents.push_back(int32_t(exponent));
        bridge.likelihood_values.push_back(value);
    }
    bridge.has_importance_weights = true;
}
PyObject* make_cuda_experiment(PyObject*, PyObject* cap) {
    try {
        const uint16_t endian = 1;
        if(*reinterpret_cast<const uint8_t*>(&endian) != 1)
            throw std::invalid_argument("CUDA experiment buffers require a little-endian host");
        auto* owner = execution_handle(cap);
        if(!owner) return nullptr;
        auto bridge = std::make_unique<CudaExperimentBinding>();
        bridge->run = owner->run;
        auto& run = *bridge->run;
        CompiledExperiment::ExclusiveUse use(run);
        native_work([&]() {
            if(run.needs_history() || run.nonlinear() || !run.source.guarded_ids.empty() || run.noise.guarded)
                throw std::invalid_argument("CUDA experiment requires a fixed circuit without sampled histories, callbacks or guarded noise");
            cuda_prepare_likelihood(*bridge);
            if(!run.task.boundaries.observables.empty())
                throw std::invalid_argument("CUDA experiment currently exports acceptance and target error only");
            if(!run.complete) run.prepare();
            bridge->engine = run.plans.get(run.source.stage("recovery"));
            if(run.source.target) {
                bridge->target = run.plans.get("target");
                if(!bridge->target->plan->compiled.plan.continuation)
                    throw std::invalid_argument("CUDA experiment requires target frame continuation");
            }
            const auto& plan = bridge->engine->plan->compiled.plan;
            bridge->frame_bits = plan.physical + plan.magic.size();
            bridge->words = std::max(size_t(1), (bridge->frame_bits + 63) / 64);
            size_t stride = 2 * bridge->words, actions = run.noise.actions.size();
            if(actions > std::numeric_limits<uint32_t>::max() || actions > bridge->columns.max_size() / stride)
                throw std::length_error("CUDA action table exceeds index capacity");
            bridge->columns.resize(actions * stride);
            bridge->base.resize(stride);
            bridge->action_sites.resize(actions);
            cuda_pack_frame(bridge->engine->feedback_constant, bridge->words, bridge->base.data());
            for(size_t site = 0; site < run.noise.categories.size(); site++)
                for(size_t choice = 0; choice < run.noise.categories[site]->size; choice++) {
                    auto index = run.noise.offsets[site] + choice;
                    bridge->action_sites.at(index) = site;
                    FileNoiseDraw selected;
                    selected.selected.push_back(int64_t(index));
                    auto noise = noise_actions(run.noise, selected);
                    auto packed = bridge->engine->pack(run.source.original_faults(noise), noise.flips);
                    auto frame = plan.fault_frame(std::move(packed.faults), false);
                    cuda_pack_frame(frame, bridge->words, bridge->columns.data() + index * stride);
                }
            return 0;
        });
        Object metadata(Py_BuildValue("{s:n,s:n,s:n,s:n,s:n}", "frame_bits", Py_ssize_t(bridge->frame_bits),
                "words", Py_ssize_t(bridge->words), "actions", Py_ssize_t(run.noise.actions.size()),
                "num_sites", Py_ssize_t(run.noise.categories.size()), "max_shots", Py_ssize_t(cuda_experiment_max_shots)));
        phase_set(metadata.p, "columns", cuda_output_bytes(bridge->columns));
        phase_set(metadata.p, "base_frame", cuda_output_bytes(bridge->base));
        phase_set(metadata.p, "target_normalization", output_scalar(run.target_normalization));
        phase_set(metadata.p, "has_importance_weights", PyBool_FromLong(bridge->has_importance_weights));
        Object offsets(PyTuple_New(run.noise.offsets.size()));
        for(size_t j = 0; j < run.noise.offsets.size(); j++)
            PyTuple_SET_ITEM(offsets.p, j, PyLong_FromSize_t(run.noise.offsets[j]));
        phase_set(metadata.p, "offsets", Py_NewRef(offsets.p));
        Object handle(PyCapsule_New(bridge.get(), "magna-cuda-experiment-v1", cuda_experiment_destructor));
        bridge.release();
        return PyTuple_Pack(2, handle.p, metadata.p);
    } catch(...) { return execution_error(); }
}
struct CudaNoiseBatch {
    std::vector<uint64_t> offsets;
    std::vector<uint32_t> indices;
    size_t shots() const { return offsets.size() - 1; }
};
CudaNoiseBatch cuda_read_batch(const CudaExperimentBinding& bridge, PyObject* raw_offsets, PyObject* raw_indices) {
    CudaNoiseBatch batch{cuda_read_bytes<uint64_t>(raw_offsets), cuda_read_bytes<uint32_t>(raw_indices)};
    if(batch.offsets.empty() || batch.shots() > cuda_experiment_max_shots || batch.offsets.front() != 0 ||
            batch.offsets.back() != batch.indices.size())
        throw std::invalid_argument("invalid CUDA CSR offsets or shot limit");
    for(size_t shot = 0; shot < batch.shots(); shot++) {
        auto first = batch.offsets[shot], end = batch.offsets[shot + 1];
        if(first > end || end > batch.indices.size()) throw std::invalid_argument("CUDA CSR offsets are not monotone");
        size_t previous_site = 0;
        for(auto k = first; k < end; k++) {
            auto index = batch.indices[size_t(k)];
            if(index >= bridge.action_sites.size()) throw std::invalid_argument("CUDA action index outside plan");
            auto site = bridge.action_sites[index];
            if(k != first && site <= previous_site)
                throw std::invalid_argument("CUDA actions must be ordered and select at most one outcome per noise site");
            previous_site = site;
        }
    }
    return batch;
}
PyObject* cuda_draw_noise(PyObject*, PyObject* args) {
    try {
        PyObject *cap, *raw_seeds;
        if(!PyArg_ParseTuple(args, "OO", &cap, &raw_seeds)) return nullptr;
        auto* bridge = cuda_experiment_handle(cap);
        if(!bridge) return nullptr;
        Seq input(raw_seeds);
        if(input.size() > cuda_experiment_max_shots) throw std::invalid_argument("CUDA noise batch exceeds shot limit");
        std::vector<uint64_t> seeds;
        seeds.reserve(input.size());
        for(size_t j = 0; j < input.size(); j++) {
            if(!PyLong_CheckExact(input[j])) throw std::invalid_argument("CUDA seeds must be uint64 integers");
            auto seed = PyLong_AsUnsignedLongLong(input[j]);
            if(PyErr_Occurred()) throw std::invalid_argument("CUDA seed outside uint64");
            seeds.push_back(uint64_t(seed));
        }
        CompiledExperiment::ExclusiveUse use(*bridge->run);
        cuda_validate_run(*bridge);
        CudaNoiseBatch batch;
        batch.offsets.push_back(0);
        native_work([&]() {
            SignalCheck signals;
            for(auto seed : seeds) {
                auto draw = draw_noise(bridge->run->noise, seed, true);
                for(auto index : draw.selected) batch.indices.push_back(uint32_t(index));
                batch.offsets.push_back(uint64_t(batch.indices.size()));
                signals();
            }
            return 0;
        });
        Object offsets(cuda_output_bytes(batch.offsets)), indices(cuda_output_bytes(batch.indices));
        return PyTuple_Pack(2, offsets.p, indices.p);
    } catch(...) { return execution_error(); }
}
PyObject* cuda_map_frames(PyObject*, PyObject* args) {
    try {
        PyObject *cap, *raw_offsets, *raw_indices;
        if(!PyArg_ParseTuple(args, "OOO", &cap, &raw_offsets, &raw_indices)) return nullptr;
        auto* bridge = cuda_experiment_handle(cap);
        if(!bridge) return nullptr;
        auto batch = cuda_read_batch(*bridge, raw_offsets, raw_indices);
        CompiledExperiment::ExclusiveUse use(*bridge->run);
        cuda_validate_run(*bridge);
        size_t stride = 2 * bridge->words;
        std::vector<uint64_t> output;
        if(batch.shots() > output.max_size() / stride) throw std::length_error("CUDA frame output too large");
        output.resize(batch.shots() * stride);
        native_work([&]() {
            SignalCheck signals;
            for(size_t shot = 0; shot < batch.shots(); shot++) {
                auto* row = output.data() + shot * stride;
                std::copy(bridge->base.begin(), bridge->base.end(), row);
                for(auto k = batch.offsets[shot]; k < batch.offsets[shot + 1]; k++) {
                    const auto* column = bridge->columns.data() + size_t(batch.indices[size_t(k)]) * stride;
                    for(size_t w = 0; w < stride; w++) row[w] ^= column[w];
                }
                signals();
            }
            return 0;
        });
        return cuda_output_bytes(output);
    } catch(...) { return execution_error(); }
}
PyObject* cuda_output_masses(const std::vector<HistoryMasses>& results) {
    Object output(PyTuple_New(results.size()));
    for(size_t j = 0; j < results.size(); j++) {
        const auto& masses = results[j];
        Object history(output_scalar(masses.history)), acceptance(output_scalar(masses.acceptance));
        Object error(masses.error ? output_scalar(*masses.error) : Py_NewRef(Py_None));
        auto* row = PyTuple_Pack(3, history.p, acceptance.p, error.p);
        if(!row) throw std::bad_alloc();
        PyTuple_SET_ITEM(output.p, j, row);
    }
    return Py_NewRef(output.p);
}
PyObject* cuda_evaluate_frames(PyObject*, PyObject* args) {
    try {
        PyObject *cap, *raw_frames;
        if(!PyArg_ParseTuple(args, "OO", &cap, &raw_frames)) return nullptr;
        auto* bridge = cuda_experiment_handle(cap);
        if(!bridge) return nullptr;
        auto frames = cuda_read_bytes<uint64_t>(raw_frames);
        auto stride = 2 * bridge->words;
        if(frames.size() % stride || frames.size() / stride > cuda_experiment_max_shots)
            throw std::invalid_argument("CUDA frame buffer shape or shot limit");
        if(bridge->frame_bits % 64 || !bridge->frame_bits) {
            auto valid = bridge->frame_bits ? (uint64_t(1) << (bridge->frame_bits % 64)) - 1 : uint64_t(0);
            for(size_t j = 0; j < frames.size(); j += bridge->words)
                if(frames[j + bridge->words - 1] & ~valid)
                    throw std::invalid_argument("CUDA frame contains nonzero padding bits");
        }
        CompiledExperiment::ExclusiveUse use(*bridge->run);
        cuda_validate_run(*bridge);
        std::vector<HistoryMasses> results;
        results.reserve(frames.size() / stride);
        native_work([&]() {
            SignalCheck signals;
            for(size_t first = 0; first < frames.size(); first += stride) {
                Frame frame;
                frame.x.w.assign(frames.begin() + first, frames.begin() + first + bridge->words);
                frame.z.w.assign(frames.begin() + first + bridge->words, frames.begin() + first + stride);
                frame.x.trim();
                frame.z.trim();
                results.push_back(bridge->run->evaluate({}, {}, {}, {}, {}, {}, nullptr, nullptr, nullptr, &frame));
                signals();
            }
            return 0;
        });
        return cuda_output_masses(results);
    } catch(...) { return execution_error(); }
}
PyObject* cuda_evaluate_noise(PyObject*, PyObject* args) {
    try {
        PyObject *cap, *raw_offsets, *raw_indices;
        if(!PyArg_ParseTuple(args, "OOO", &cap, &raw_offsets, &raw_indices)) return nullptr;
        auto* bridge = cuda_experiment_handle(cap);
        if(!bridge) return nullptr;
        auto batch = cuda_read_batch(*bridge, raw_offsets, raw_indices);
        CompiledExperiment::ExclusiveUse use(*bridge->run);
        cuda_validate_run(*bridge);
        std::vector<HistoryMasses> results;
        results.reserve(batch.shots());
        native_work([&]() {
            ExecutionHooks hooks;
            hooks.seed = []() -> uint64_t { throw std::logic_error("unexpected random draw in fixed-frame evaluation"); };
            SignalCheck signals;
            for(size_t shot = 0; shot < batch.shots(); shot++) {
                FileNoiseDraw selected;
                for(auto k = batch.offsets[shot]; k < batch.offsets[shot + 1]; k++)
                    selected.selected.push_back(batch.indices[size_t(k)]);
                auto noise = noise_actions(bridge->run->noise, selected);
                results.push_back(bridge->run->sample(hooks, std::move(noise)).masses);
                signals();
            }
            return 0;
        });
        return cuda_output_masses(results);
    } catch(...) { return execution_error(); }
}
