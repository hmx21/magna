// Static exact probability plans for the experimental CUDA sampler.
#pragma once

struct CudaQueryAffine {
    Bits row;
    int constant = 0;
};
inline CudaQueryAffine cuda_query_xor(CudaQueryAffine a, const CudaQueryAffine& b) {
    a.row ^= b.row;
    a.constant ^= b.constant;
    return a;
}
struct CudaQueryPlan {
    size_t n = 0, frame_words = 0, input_bits = 0;
    int mode = 0; // 0 zero; 1 dynamic exact Gauss; 2 static Gauss; 3 exact phase histogram.
    int constant = 0, power_sqrt2 = 0, output_power = 0;
    size_t phase_rank = 0;
    std::vector<CudaQueryAffine> features;
    std::map<std::pair<Bits, int>, uint32_t, bool(*)(const std::pair<Bits,int>&, const std::pair<Bits,int>&)> feature_index;
    std::vector<uint32_t> support, guards, linear_shift;
    std::vector<int32_t> constant_linear, constant_pairs, magic;
    std::vector<uint64_t> magic_masks, upper;
    std::vector<uint8_t> linear, phase_table;
    std::vector<int32_t> steps;
    std::vector<uint32_t> step_offsets, step_indices;
    std::vector<uint8_t> step_tags;
    CudaQueryPlan() : feature_index([](const auto& a, const auto& b) {
        if(a.first != b.first) return BitsLess{}(a.first, b.first);
        return a.second < b.second;
    }) {}
    uint32_t feature(CudaQueryAffine value) {
        auto key = std::make_pair(value.row, value.constant);
        auto found = feature_index.find(key);
        if(found != feature_index.end()) return found->second;
        if(features.size() >= 1000000) throw std::length_error("CUDA query feature budget exceeded");
        auto index = uint32_t(features.size());
        feature_index.emplace(std::move(key), index);
        features.push_back(std::move(value));
        return index;
    }
};
struct CudaQueriesBinding {
    std::shared_ptr<CompiledExperiment> run;
    std::shared_ptr<FileBoundary> engine, target;
    size_t words = 0, frame_bits = 0;
    std::vector<CudaQueryPlan> queries;
};

inline Bits cuda_query_frame_bits(const Frame& frame, size_t words) {
    return frame.x ^ (frame.z << (64 * words));
}

// Compose every query-coordinate parity with the exact prefix continuation.
// This removes a separate target-frame transfer/kernel without changing its map.
inline std::pair<std::vector<Bits>, std::vector<Bits>> cuda_query_frame_rows(
        const CudaExperimentBinding& bridge, const FileBoundary& engine, bool target) {
    const auto& plan = engine.plan->compiled.plan;
    size_t bits = plan.physical + plan.magic.size();
    std::vector<Bits> x(bits), z(bits);
    for(size_t axis = 0; axis < 2; axis++)
        for(size_t bit = 0; bit < bridge.frame_bits; bit++) {
            Frame image;
            (axis ? image.z : image.x).set(bit);
            if(target) image = plan.continuation->apply(image);
            const size_t input = axis * 64 * bridge.words + bit;
            image.x.each([&](size_t j) { x.at(j).set(input); });
            image.z.each([&](size_t j) { z.at(j).set(input); });
        }
    return {std::move(x), std::move(z)};
}

inline std::vector<uint8_t> cuda_query_feature_values(const CudaQueryPlan& plan, const uint64_t* frame) {
    std::vector<uint8_t> values;
    values.reserve(plan.features.size());
    for(const auto& feature : plan.features) {
        uint64_t combined = 0;
        for(size_t w = 0; w < feature.row.w.size(); w++) combined ^= feature.row.w[w] & frame[w];
        values.push_back(uint8_t(feature.constant ^ (__builtin_popcountll(combined) & 1)));
    }
    return values;
}

// Build the same actual Clifford data as CompiledNetwork::evaluate_fixed.
// Non-Clifford odd phases stay in phase_table for the exact histogram route.
inline int cuda_query_actual(const CudaQueryPlan& p, const std::vector<uint8_t>& f,
        std::vector<int>& l, std::vector<Bits>& upper, bool make_upper = true) {
    l.assign(p.linear.begin(), p.linear.end());
    size_t words = (p.n + 63) / 64;
    if(make_upper) {
        upper.resize(p.n);
        for(size_t j = 0; j < p.n; j++) {
            upper[j].w.assign(p.upper.begin() + j * words, p.upper.begin() + (j + 1) * words);
            upper[j].trim();
        }
    }
    int phase = p.constant;
    for(size_t j = 0; j < p.constant_linear.size(); j += 2)
        phase = mod8(phase + p.constant_linear[j + 1] * f[size_t(p.constant_linear[j])]);
    for(size_t j = 0; j < p.constant_pairs.size(); j += 2)
        phase = mod8(phase + 4 * f[size_t(p.constant_pairs[j])] * f[size_t(p.constant_pairs[j + 1])]);
    for(size_t j = 0; j < p.n; j++) l[j] = mod4(l[j] + 2 * f[p.linear_shift[j]]);
    for(size_t j = 0; j < p.magic.size() / 8; j++) {
        const auto* m = p.magic.data() + 8 * j;
        int actual = mod8((f[size_t(m[4])] ? -m[0] : m[0]) + 4 * f[size_t(m[5])]);
        int before[2] = {m[2], m[3]}, after[2] = {m[2] ^ f[size_t(m[6])], m[3] ^ f[size_t(m[7])]};
        for(size_t side = 0; side < 2; side++) {
            int sign = side ? 1 : -1;
            phase = mod8(phase + sign * (actual * after[side] - m[1] * before[side]));
            int delta = mod8(sign * (actual * (after[side] ? -1 : 1) - m[1] * (before[side] ? -1 : 1)));
            if(delta & 1) throw std::logic_error("CUDA query update is not Clifford");
            int c = delta / 2;
            if(!c) continue;
            Bits mask;
            auto first = p.magic_masks.begin() + (2 * j + side) * words;
            mask.w.assign(first, first + words);
            mask.trim();
            mask.each([&](size_t k) {
                l[k] = mod4(l[k] + c);
                if(make_upper && (c & 1)) upper[k] ^= mask.above(k);
            });
        }
    }
    return mod8(phase);
}

inline CudaQueryPlan cuda_prepare_query(const CudaExperimentBinding& bridge,
        const FileBoundary& engine, const char* purpose, bool target) {
    CudaQueryPlan out;
    out.frame_words = 2 * bridge.words;
    out.input_bits = bridge.frame_bits;
    const auto& parent = *engine.reference_networks.at(purpose);
    const auto& network = *parent.reference;
    if(!network.boundary.keep.empty() || !parent.closed.empty() || parent.capped_history)
        throw std::invalid_argument("CUDA exact queries require scalar fixed-history boundaries");
    if(network.traced.clifford.zero) return out;
    const auto* certificate = network.fixed_certificate();
    if(!certificate) throw std::invalid_argument("CUDA exact query requires a fixed phase certificate");
    const auto& resource = certificate->reference;
    out.n = network.restricted.clifford.n;
    out.phase_rank = resource.quotient.axes.size();
    if(out.n > 1024) throw std::invalid_argument("CUDA exact query dimension exceeds 1024");
    if(out.phase_rank && out.n > 12)
        throw std::invalid_argument("CUDA non-Clifford histogram dimension exceeds 12; no CPU fallback");
    if(resource.quotient.linear.size() != out.n)
        throw std::logic_error("CUDA fixed certificate does not use the scalar query domain");
    if(!out.phase_rank) {
        const auto& entries = resource_entries(resource);
        if(entries.size() != 1 || !entries[0].constraints.empty() || !entries[0].parities.empty() ||
                entries[0].coefficient.c != Scalar(1).c || entries[0].coefficient.power != 0)
            throw std::invalid_argument("CUDA Gauss query requires the certified unit entry");
    }
    out.mode = out.phase_rank ? 3 : (out.n <= 64 ? 1 : 2);
    const auto& base = out.phase_rank ? network.restricted.clifford.linear : resource.quotient.linear;
    const auto& base_upper = out.phase_rank ? network.restricted.clifford.upper : resource.quotient.upper;
    out.linear.assign(base.begin(), base.end());
    size_t words = (out.n + 63) / 64;
    out.upper.assign(out.n * words, 0);
    for(size_t j = 0; j < out.n; j++)
        for(size_t w = 0; w < base_upper[j].w.size(); w++) out.upper[j * words + w] = base_upper[j].w[w];
    out.constant = mod8(network.restricted.clifford.constant + network.phase_constant);
    auto power = network.restricted.clifford.power;
    if(power < -1000000 || power > 1000000) throw std::invalid_argument("CUDA query scale exceeds int32 budget");
    out.power_sqrt2 = int(power);
    auto target_power = target ? bridge.run->source.target->traced.size() : 0;
    if(target_power > 1000000) throw std::invalid_argument("CUDA target trace scale exceeds int32 budget");
    out.output_power = int(target_power);
    const auto& plan = engine.plan->compiled.plan;
    auto [xrows, zrows] = cuda_query_frame_rows(bridge, engine, target);
    std::vector<CudaQueryAffine> rhs;
    for(size_t j = 0; j < network.boundary.events.size(); j++)
        rhs.push_back({xors(xrows, network.boundary.events[j]),
                parent.selection[j] ^ network.traced.clifford.boundary[j].second});
    auto combine = [](const std::vector<CudaQueryAffine>& values, const Bits& mask) {
        CudaQueryAffine result;
        mask.each([&](size_t j) { result = cuda_query_xor(std::move(result), values.at(j)); });
        return result;
    };
    const auto& factor = network.event_support->factor;
    for(const auto& row : factor.zeros) out.support.push_back(out.feature(combine(rhs, row)));
    std::vector<CudaQueryAffine> shift(network.traced.clifford.n);
    for(const auto& [j, pair] : factor.pivots)
        shift[j] = cuda_query_xor(combine(shift, pair.first), combine(rhs, pair.second));
    const auto& traced = network.traced.clifford;
    for(size_t j = 0; j < shift.size(); j++) {
        auto a = out.feature(shift[j]);
        if(traced.linear[j]) out.constant_linear.insert(out.constant_linear.end(), {int32_t(a), 2 * traced.linear[j]});
        traced.upper[j].each([&](size_t k) {
            out.constant_pairs.insert(out.constant_pairs.end(), {int32_t(a), int32_t(out.feature(shift[k]))});
        });
    }
    auto observation = CudaQueryAffine{xors(zrows, network.boundary.x) ^ xors(xrows, network.boundary.z), 0};
    out.constant_linear.insert(out.constant_linear.end(), {int32_t(out.feature(observation)), 4});
    for(const auto& row : network.maps->event_polar) out.linear_shift.push_back(out.feature(combine(shift, row)));
    std::map<std::pair<size_t, size_t>, CudaQueryAffine> structural;
    for(size_t j = 0; j < network.magic.size(); j++) {
        CudaQueryAffine xm{xrows.at(plan.physical + j), 0}, zm{zrows.at(plan.physical + j), 0};
        auto bm = combine(shift, network.maps->bra_magic[j]), km = combine(shift, network.maps->ket_magic[j]);
        out.magic.insert(out.magic.end(), {mod8(plan.magic[j]), mod8(network.reference_exponents[j]),
                network.bra_offsets[j], network.ket_offsets[j], int32_t(out.feature(xm)), int32_t(out.feature(zm)),
                int32_t(out.feature(bm)), int32_t(out.feature(km))});
        const Bits* masks[2] = {&network.maps->bra_reduced[j], &network.maps->ket_reduced[j]};
        CudaQueryAffine changes[2] = {cuda_query_xor(xm, bm), cuda_query_xor(xm, km)};
        for(size_t side = 0; side < 2; side++) {
            auto start = out.magic_masks.size();
            out.magic_masks.resize(start + words);
            std::copy(masks[side]->w.begin(), masks[side]->w.end(), out.magic_masks.begin() + start);
            if(out.mode == 2 && (plan.magic[j] & 1)) {
                auto indices = masks[side]->indices();
                for(size_t a = 0; a < indices.size(); a++)
                    for(size_t b = a; b < indices.size(); b++) {
                        auto key = std::make_pair(indices[a], indices[b]);
                        structural[key] = cuda_query_xor(std::move(structural[key]), changes[side]);
                    }
            }
        }
    }
    if(out.mode == 3) {
        out.phase_table.resize(size_t(1) << out.n);
        for(size_t value = 0; value < out.phase_table.size(); value++) {
            int phase = 0;
            for(const auto& [mask, coefficient] : network.fixed_phase)
                phase = mod8(phase + coefficient * mask.parity_and(Bits(value)));
            out.phase_table[value] = uint8_t(mod8(phase));
        }
    }
    if(out.mode == 2) {
        std::vector<Bits> structural_rows;
        for(const auto& [position, value] : structural)
            if(!value.row.empty()) structural_rows.push_back(value.row);
        auto independent = phase_image_basis(structural_rows);
        Bits allowed_base;
        allowed_base.w = bridge.base;
        allowed_base.trim();
        for(const auto& row : independent) {
            for(size_t a = 0; a < bridge.action_sites.size(); a++) {
                uint64_t combined = 0;
                for(size_t w = 0; w < row.w.size(); w++) combined ^= row.w[w] & bridge.columns[a * out.frame_words + w];
                if(__builtin_popcountll(combined) & 1)
                    throw std::invalid_argument("CUDA large query has varying quadratic geometry on the noise span");
            }
            out.guards.push_back(out.feature({row, row.parity_and(allowed_base)}));
        }
        auto features = cuda_query_feature_values(out, bridge.base.data());
        std::vector<int> l;
        std::vector<Bits> upper;
        cuda_query_actual(out, features, l, upper);
        CliffordContraction contraction(std::vector<Bits>(out.n), Quadratic{0, l, upper});
        if(!contraction.keep.empty()) throw std::logic_error("CUDA scalar contraction retained outputs");
        out.step_offsets.push_back(0);
        for(const auto& step : contraction.steps) {
            out.steps.insert(out.steps.end(), {int32_t(step.h), int32_t(step.p), int32_t(step.odd)});
            const auto& entries = step.odd ? step.neighbors : step.support | step.neighbors;
            entries.each([&](size_t k) {
                out.step_indices.push_back(uint32_t(k));
                out.step_tags.push_back(uint8_t(step.support.bit(k) | (2 * step.neighbors.bit(k))));
            });
            out.step_offsets.push_back(uint32_t(out.step_indices.size()));
        }
    }
    return out;
}

inline PyObject* cuda_query_plan_output(const CudaQueryPlan& p) {
    Object out(Py_BuildValue("{s:n,s:n,s:i,s:i,s:i,s:i,s:n,s:n}", "n", Py_ssize_t(p.n),
            "frame_words", Py_ssize_t(p.frame_words), "mode", p.mode, "constant", p.constant,
            "power_sqrt2", p.power_sqrt2, "output_power", p.output_power,
            "phase_rank", Py_ssize_t(p.phase_rank), "feature_count", Py_ssize_t(p.features.size())));
    std::vector<uint32_t> offsets{0}, words;
    std::vector<uint64_t> masks;
    std::vector<uint8_t> constants;
    for(const auto& feature : p.features) {
        for(size_t w = 0; w < feature.row.w.size(); w++)
            if(feature.row.w[w]) { words.push_back(uint32_t(w)); masks.push_back(feature.row.w[w]); }
        offsets.push_back(uint32_t(words.size()));
        constants.push_back(uint8_t(feature.constant));
    }
    phase_set(out.p, "feature_offsets", cuda_output_bytes(offsets));
    phase_set(out.p, "feature_words", cuda_output_bytes(words));
    phase_set(out.p, "feature_masks", cuda_output_bytes(masks));
    phase_set(out.p, "feature_constants", cuda_output_bytes(constants));
    phase_set(out.p, "support", cuda_output_bytes(p.support));
    phase_set(out.p, "guards", cuda_output_bytes(p.guards));
    phase_set(out.p, "linear_shift", cuda_output_bytes(p.linear_shift));
    phase_set(out.p, "constant_linear", cuda_output_bytes(p.constant_linear));
    phase_set(out.p, "constant_pairs", cuda_output_bytes(p.constant_pairs));
    phase_set(out.p, "magic", cuda_output_bytes(p.magic));
    phase_set(out.p, "magic_masks", cuda_output_bytes(p.magic_masks));
    phase_set(out.p, "upper", cuda_output_bytes(p.upper));
    phase_set(out.p, "linear", cuda_output_bytes(p.linear));
    phase_set(out.p, "phase_table", cuda_output_bytes(p.phase_table));
    phase_set(out.p, "steps", cuda_output_bytes(p.steps));
    phase_set(out.p, "step_offsets", cuda_output_bytes(p.step_offsets));
    phase_set(out.p, "step_indices", cuda_output_bytes(p.step_indices));
    phase_set(out.p, "step_tags", cuda_output_bytes(p.step_tags));
    return Py_NewRef(out.p);
}

inline CudaQueriesBinding* cuda_queries_handle(PyObject* cap) {
    return static_cast<CudaQueriesBinding*>(PyCapsule_GetPointer(cap, "magna-cuda-queries-v1"));
}
inline void cuda_queries_destructor(PyObject* cap) { delete cuda_queries_handle(cap); }

PyObject* make_cuda_queries(PyObject*, PyObject* cap) {
    try {
        auto* bridge = cuda_experiment_handle(cap);
        if(!bridge) return nullptr;
        CompiledExperiment::ExclusiveUse use(*bridge->run);
        cuda_validate_run(*bridge);
        auto owner = std::make_unique<CudaQueriesBinding>();
        owner->run = bridge->run;
        owner->engine = bridge->engine;
        owner->target = bridge->target;
        owner->words = bridge->words;
        owner->frame_bits = bridge->frame_bits;
        native_work([&]() {
            owner->queries.push_back(cuda_prepare_query(*bridge, *bridge->engine, "acceptance", false));
            if(bridge->target) owner->queries.push_back(cuda_prepare_query(*bridge, *bridge->target, "fidelity", true));
            return 0;
        });
        Object metadata(PyTuple_New(owner->queries.size()));
        for(size_t j = 0; j < owner->queries.size(); j++)
            PyTuple_SET_ITEM(metadata.p, j, cuda_query_plan_output(owner->queries[j]));
        Object handle(PyCapsule_New(owner.get(), "magna-cuda-queries-v1", cuda_queries_destructor));
        owner.release();
        return PyTuple_Pack(2, handle.p, metadata.p);
    } catch(...) { return execution_error(); }
}

inline Scalar cuda_query_cpu_value(const CudaQueryPlan& p, const uint64_t* frame) {
    if(!p.mode) return {};
    auto f = cuda_query_feature_values(p, frame);
    for(auto guard : p.guards)
        if(f[guard]) throw std::invalid_argument("frame is outside the certified large-query quadratic geometry");
    for(auto event : p.support)
        if(f[event]) return {};
    std::vector<int> l;
    std::vector<Bits> upper;
    int phase = cuda_query_actual(p, f, l, upper, p.mode != 2);
    Scalar value;
    if(p.mode == 3) {
        std::array<int64_t, 8> counts{};
        for(size_t u = 0; u < p.phase_table.size(); u++)
            counts[mod8(phase + 2 * qvalue(Bits(u), l, upper) + p.phase_table[u])]++;
        for(size_t k = 0; k < 8; k++) value = value + Scalar(counts[k]) * omega(int(k), p.power_sqrt2);
    } else if(p.mode == 1) {
        CliffordContraction contraction(std::vector<Bits>(p.n), Quadratic{0, l, upper});
        auto ket = contraction.evaluate(0, {}, std::move(l), p.power_sqrt2, phase);
        if(!ket.zero) value = omega(ket.phase, ket.power);
    } else {
        int power = p.power_sqrt2;
        for(size_t s = 0; s < p.steps.size() / 3; s++) {
            auto h = p.steps[3 * s], pivot = p.steps[3 * s + 1], odd = p.steps[3 * s + 2];
            int lh = l[size_t(h)];
            if(odd) {
                int sign = lh == 1 ? -1 : 1;
                for(auto k = p.step_offsets[s]; k < p.step_offsets[s + 1]; k++)
                    l[p.step_indices[k]] = mod4(l[p.step_indices[k]] + sign);
                power++;
                phase = mod8(phase + (lh == 1 ? 1 : -1));
            } else {
                power += 2;
                int rhs = lh / 2;
                if(size_t(pivot) == p.n) { if(rhs) return {}; continue; }
                int lp = l[size_t(pivot)];
                phase = mod8(phase + 2 * rhs * lp);
                for(auto k = p.step_offsets[s]; k < p.step_offsets[s + 1]; k++) {
                    int sj = p.step_tags[k] & 1, nj = (p.step_tags[k] >> 1) & 1;
                    auto j = p.step_indices[k];
                    l[j] = mod4(l[j] + lp * sj + 2 * sj * nj + 2 * rhs * (nj ^ ((lp & 1) * sj)));
                }
            }
        }
        value = omega(phase, power);
    }
    return value * Scalar(1, p.output_power);
}

PyObject* cuda_query_cpu_evaluate(PyObject*, PyObject* args) {
    try {
        PyObject *cap, *raw;
        if(!PyArg_ParseTuple(args, "OO", &cap, &raw)) return nullptr;
        auto* owner = cuda_queries_handle(cap);
        if(!owner) return nullptr;
        auto frames = cuda_read_bytes<uint64_t>(raw);
        size_t stride = 2 * owner->words, shots = frames.size() / stride, queries = owner->queries.size();
        if(frames.size() % stride || shots > cuda_experiment_max_shots) throw std::invalid_argument("CUDA query frame shape/shot limit");
        if(owner->frame_bits % 64 || !owner->frame_bits) {
            uint64_t valid = owner->frame_bits ? (uint64_t(1) << (owner->frame_bits % 64)) - 1 : 0;
            for(size_t first = 0; first < frames.size(); first += owner->words)
                if(frames[first + owner->words - 1] & ~valid) throw std::invalid_argument("CUDA query frame padding is nonzero");
        }
        CompiledExperiment::ExclusiveUse use(*owner->run);
        auto found = owner->run->plans.prepared.find(owner->run->source.stage("recovery"));
        if(!owner->run->complete || found == owner->run->plans.prepared.end() || found->second != owner->engine)
            throw std::invalid_argument("compiled circuit was reconfigured; recreate CUDA queries");
        if(owner->target) {
            auto target = owner->run->plans.prepared.find("target");
            if(target == owner->run->plans.prepared.end() || target->second != owner->target)
                throw std::invalid_argument("target was reconfigured; recreate CUDA queries");
        }
        std::vector<int64_t> coefficients(shots * queries * 4);
        std::vector<int32_t> powers(shots * queries), status(shots * queries);
        native_work([&]() {
            SignalCheck signals;
            for(size_t shot = 0; shot < shots; shot++) {
                for(size_t query = 0; query < queries; query++) {
                    auto value = cuda_query_cpu_value(owner->queries[query], frames.data() + shot * stride);
                    size_t index = shot * queries + query;
                    powers[index] = int32_t(value.power);
                    for(size_t k = 0; k < 4; k++) {
                        const auto& c = value.c[k];
                        if(c.d.size() > 2 || (c.d.size() == 2 && c.d[1] >= (uint32_t(1) << 31)))
                            throw std::overflow_error("CUDA exact query coefficient exceeds int64 proof bound");
                        uint64_t v = c.d.empty() ? 0 : c.d[0];
                        if(c.d.size() == 2) v |= uint64_t(c.d[1]) << 32;
                        coefficients[4 * index + k] = c.neg ? -int64_t(v) : int64_t(v);
                    }
                }
                signals();
            }
            return 0;
        });
        Object c(cuda_output_bytes(coefficients)), p(cuda_output_bytes(powers)), s(cuda_output_bytes(status));
        return PyTuple_Pack(3, c.p, p.p, s.p);
    } catch(...) { return execution_error(); }
}
