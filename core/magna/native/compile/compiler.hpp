// Whole-instrument Clifford dilation and global affine preparation.
// All numerical construction owns C++ values; no Python callbacks or states.
#pragma once
#include "plan.hpp"
#include "../foundation/limits.hpp"
#include <chrono>
#include <set>

namespace magic_native {
struct CompilerOp {
    std::string kind;
    std::vector<int> qubits;
    int exponent = 0, sign = 0, record = -1, location = -1, feedback_slot = -1;
    std::string pauli, paulis;
};
struct CompilerIR {
    Ket initial;
    std::vector<int> inputs, outputs;
    std::vector<CompilerOp> operations;
    size_t records = 0, max_path_bits = 2048;
};
struct CompilerPreparation {
    Plan plan;
    std::vector<int> linear;
    std::vector<Bits> upper;
    std::vector<std::pair<Bits, int>> magic_terms;
    std::vector<int> phase_controls, environment;
    size_t path_variables = 0, output_count = 0, gate_count = 0, shared_path_gates = 0;
    size_t rotation_layers = 0, reference_layer_slots = 0, ancilla_count = 0, phase_coordinates = 0;
    int reference_layer = -1;
    std::vector<int> layer_slots;
    std::vector<Frame> rotation_axes;
    bool injective_physical = false;
    double dilation_seconds = 0, path_seconds = 0, contraction_seconds = 0, projection_seconds = 0,
           fault_map_seconds = 0;
};

// Ordering is the nonnegative packed-integer order used by Python's canonical
// parity map, including dimensions beyond one machine word.

struct CompilerForm {
    Bits mask;
    int offset = 0;
};
// Preparation-local expression of the whole fixed-input amplitude. There is
// no contraction, resource expansion or output boundary in this shared prefix.
struct CompilerPathPrefix {
    Ket initial, expression;
    std::vector<int> inputs;
    std::vector<CompilerOp> gates, source_gates;
    int reference_layer = -1;
    std::vector<int> endpoints, frame_positions;
    size_t frame_physical = 0, frame_magic = 0;
    std::map<int, CompilerForm> wires;
    std::vector<CompilerForm> captures;
    bool track_global_phase = true;
    bool matches(const CompilerIR& ir, const std::vector<CompilerOp>& next, bool phase, bool source = false) const {
        const auto& k = ir.initial;
        const auto& previous = source ? source_gates : gates;
        if(previous.empty() || phase != track_global_phase || inputs != ir.inputs || previous.size() > next.size() ||
                initial.width != k.width || initial.base != k.base || initial.columns != k.columns ||
                initial.linear != k.linear || initial.upper != k.upper || initial.power != k.power ||
                initial.phase != k.phase || initial.zero != k.zero)
            return false;
        for(size_t j = 0; j < previous.size(); j++) {
            const auto& a = previous[j];
            const auto& b = next[j];
            if(a.kind != b.kind || a.qubits != b.qubits || a.exponent != b.exponent || a.pauli != b.pauli ||
                    a.feedback_slot != b.feedback_slot)
                return false;
        }
        return true;
    }
};
inline CompilerForm compiler_xor(CompilerForm a, const CompilerForm& b) {
    a.mask ^= b.mask;
    a.offset ^= b.offset;
    return a;
}

// Construct only an affine path tensor. Summation is deferred until the entire
// dilation, including every phase, record and retired environment leg, exists.
inline Ket compiler_path_tensor(const CompilerIR& ir, const std::vector<CompilerOp>& gates,
        const std::vector<int>& outputs, size_t& path_variables, bool track_global_phase = true,
        CompilerPathPrefix* prefix = nullptr, size_t* shared_gates = nullptr,
        std::vector<CompilerForm>* captures_out = nullptr) {
    Ket k = ir.initial;
    if(!track_global_phase) k.phase = 0;
    if(k.width != ir.inputs.size() || k.power != -int64_t(k.columns.size()) || k.zero)
        throw std::invalid_argument("normalized input stabilizer on declared inputs required");
    for(auto& v : column_change(k.columns))
        if(xors(k.columns, v).empty()) throw std::invalid_argument("independent input stabilizer support required");
    size_t n = k.columns.size();
    auto path_budget = [&](size_t needed) {
        return ResourceLimitError("compiled network exceeds the path-variable budget", "max_path_bits",
                double(ir.max_path_bits), double(needed));
    };
    if(n > ir.max_path_bits) throw path_budget(n);
    std::map<int, CompilerForm> wires;
    auto rows = transpose(k.columns, k.width);
    for(size_t j = 0; j < ir.inputs.size(); j++) wires[ir.inputs[j]] = {rows[j], int(k.base.bit(j))};
    std::vector<CompilerForm> captures;
    size_t start = 0;
    if(prefix && prefix->matches(ir, gates, track_global_phase)) {
        k = prefix->expression;
        wires = prefix->wires;
        captures = prefix->captures;
        n = k.linear.size();
        start = prefix->gates.size();
        if(n > ir.max_path_bits) throw path_budget(n);
    }
    if(shared_gates) *shared_gates = start;
    auto fresh = [&]() {
        if(n >= ir.max_path_bits) throw path_budget(n + 1);
        k.linear.push_back(0);
        k.upper.emplace_back();
        return CompilerForm{Bits::unit(n++), 0};
    };
    auto phase = [&](const CompilerForm& f, int exponent) {
        if(exponent & 1) throw std::invalid_argument("non-Clifford phase in Clifford dilation");
        if(track_global_phase) k.phase = mod8(k.phase + exponent * f.offset);
        int c = mod4((f.offset ? -exponent : exponent) / 2);
        for(auto j : f.mask.indices()) {
            k.linear[j] = mod4(k.linear[j] + c);
            if(c & 1) k.upper[j] ^= f.mask.above(j);
        }
    };
    auto product = [&](const CompilerForm& a, const CompilerForm& b) {
        // The CZ contribution is (-1)^(a b), including affine offsets.
        if(track_global_phase) k.phase = mod8(k.phase + 4 * a.offset * b.offset);
        Bits support = a.mask | b.mask;
        for(auto j : support.indices()) {
            bool aj = a.mask.bit(j), bj = b.mask.bit(j);
            k.linear[j] = mod4(k.linear[j] + 2 * ((aj && bj) ^ (a.offset && bj) ^ (b.offset && aj)));
            Bits row = (aj ? b.mask : Bits()) ^ (bj ? a.mask : Bits());
            k.upper[j] ^= row.above(j);
        }
    };
    for(size_t index = start; index < gates.size(); index++) {
        const auto& g = gates[index];
        auto& qs = g.qubits;
        auto& kind = g.kind;
        if(kind == "PREP_ZERO" || kind == "PREP_PLUS") {
            if(qs.size() != 1 || wires.count(qs[0])) throw std::invalid_argument("invalid dilation preparation");
            bool plus = kind == "PREP_PLUS";
            wires[qs[0]] = plus ? fresh() : CompilerForm{};
            k.power -= plus;
            continue;
        }
        for(auto q : qs)
            if(!wires.count(q)) throw std::invalid_argument("dilation operation outside lifetime");
        if(kind == "PHASE_CAPTURE") {
            CompilerForm f;
            for(auto q : qs) f = compiler_xor(f, wires.at(q));
            captures.push_back(std::move(f));
        } else if(kind == "H") {
            auto f = fresh();
            product(wires.at(qs.at(0)), f);
            wires[qs[0]] = f;
            k.power--;
        } else if(kind == "CX") wires[qs.at(1)] = compiler_xor(wires.at(qs[1]), wires.at(qs[0]));
        else if(kind == "CZ") product(wires.at(qs.at(0)), wires.at(qs.at(1)));
        else if(kind == "PHASE") {
            CompilerForm f;
            for(auto q : qs) f = compiler_xor(f, wires.at(q));
            phase(f, g.exponent);
        } else if(kind == "PAULI" || kind == "X" || kind == "Y" || kind == "Z") {
            auto p = kind == "PAULI" ? g.pauli : kind;
            auto& f = wires.at(qs.at(0));
            if(p == "Y" || p == "Z") phase(f, 4);
            if(p == "Y" && track_global_phase) k.phase = mod8(k.phase + 2);
            if(p == "X" || p == "Y") f.offset ^= 1;
        } else if(kind != "I") throw std::invalid_argument("unsupported Clifford dilation gate");
    }
    if(prefix) {
        prefix->initial = ir.initial;
        prefix->inputs = ir.inputs;
        prefix->gates = gates;
        prefix->expression = k;
        prefix->wires = wires;
        prefix->captures = captures;
        prefix->track_global_phase = track_global_phase;
    }
    if(wires.size() != outputs.size()) throw std::invalid_argument("incomplete dilation boundary");
    std::vector<Bits> forms;
    k.base = Bits();
    for(size_t j = 0; j < outputs.size(); j++) {
        auto f = wires.at(outputs[j]);
        forms.push_back(f.mask);
        if(f.offset) k.base.set(j);
    }
    k.width = outputs.size();
    k.columns = transpose(forms, n);
    k.phase = mod8(k.phase);
    if(captures_out) *captures_out = std::move(captures);
    path_variables = n;
    return k;
}

inline Frame compiler_scaled(Frame f, int p) {
    f.p = mod4(f.p + p);
    return f;
}
inline void compiler_fault_map(Plan& plan, const std::vector<CompilerOp>& gates, const std::vector<int>& outputs,
        const std::vector<int>& positions = {}, size_t physical = 0, std::vector<Frame>* rotation_axes_only = nullptr,
        size_t begin = 0) {
    std::map<int, std::pair<Frame, Frame>> images;
    if(gates.size() > size_t(std::numeric_limits<int>::max())) throw std::length_error("gate index capacity");
    int end = int(gates.size());
    for(size_t j = 0; j < outputs.size(); j++) {
        int at = positions.empty() ? int(j) : positions[j];
        auto x = at < 0 ? Frame{} : Frame{Bits::unit(size_t(at)), Bits(), 0};
        auto z = at < 0 ? Frame{} : Frame{Bits(), Bits::unit(size_t(at)), 0};
        images[outputs[j]] = {x, z};
        if(!rotation_axes_only) plan.suffix[outputs[j]].push_back({end, x, z});
    }
    for(int i = end - 1; i >= int(begin); i--) {
        auto& g = gates[size_t(i)];
        auto& qs = g.qubits;
        auto& kind = g.kind;
        if(rotation_axes_only && kind == "PHASE_SLOT") {
            Frame axis;
            for(auto q : qs) axis = multiply(axis, images.at(q).second);
            rotation_axes_only->at(size_t(g.feedback_slot)) = std::move(axis);
        }
        if(kind == "PREP_ZERO" || kind == "PREP_PLUS" || kind == "I" || kind == "PHASE_SLOT") continue;
        std::map<int, std::pair<Frame, Frame>> old;
        for(auto q : qs) old[q] = images.at(q);
        if(kind == "PHASE_CAPTURE") {
            for(auto q : qs) images[q].first.x.flip(physical + size_t(g.feedback_slot));
        } else if(kind == "H") {
            auto q = qs.at(0);
            images[q] = {old[q].second, old[q].first};
        } else if(kind == "CX" || kind == "CZ") {
            auto c = qs.at(0), t = qs.at(1);
            auto [xc, zc] = old[c];
            auto [xt, zt] = old[t];
            if(kind == "CX") {
                images[c] = {multiply(xc, xt), zc};
                images[t] = {xt, multiply(zc, zt)};
            } else {
                images[c] = {multiply(xc, zt), zc};
                images[t] = {multiply(zc, xt), zt};
            }
        } else if(kind == "PHASE") {
            int e = g.exponent / 2;
            Frame all;
            for(auto q : qs) all = multiply(all, old[q].second);
            for(auto q : qs) {
                auto [x, z] = old[q];
                images[q] = {compiler_scaled((e & 1) ? multiply(x, all) : x, e), z};
            }
        } else if(kind == "PAULI" || kind == "X" || kind == "Y" || kind == "Z") {
            auto q = qs.at(0);
            auto p = kind == "PAULI" ? g.pauli : kind;
            auto [x, z] = old[q];
            images[q] = {compiler_scaled(x, 2 * int(p == "Y" || p == "Z")),
                    compiler_scaled(z, 2 * int(p == "X" || p == "Y"))};
        } else throw std::invalid_argument("unsupported fault-map gate");
        if(!rotation_axes_only)
            for(auto q : qs) plan.suffix[q].push_back({i, images[q].first, images[q].second});
    }
    for(auto& [q, ss] : plan.suffix) std::reverse(ss.begin(), ss.end());
}

// Lower the original operations into a Clifford dilation whose rotations are
// PHASE_SLOT placeholders. Endpoints are the outputs, the retired environment
// wires, then the measurement records.
struct CompilerDilation {
    std::vector<CompilerOp> gates;
    std::vector<int> endpoints;
};
inline CompilerDilation compiler_dilation(const CompilerIR& ir, CompilerPreparation& result) {
    auto& plan = result.plan;
    std::set<int> unique_inputs(ir.inputs.begin(), ir.inputs.end()),
            unique_outputs(ir.outputs.begin(), ir.outputs.end());
    if(unique_inputs.size() != ir.inputs.size() || unique_outputs.size() != ir.outputs.size())
        throw std::invalid_argument("duplicate input/output wire");
    if(ir.operations.size() > size_t(std::numeric_limits<int>::max()))
        throw std::length_error("operation index capacity");
    std::map<int, int> live;
    int count = 0;
    for(auto q : ir.inputs) {
        live[q] = count;
        plan.histories[q].push_back({-1, count++});
    }
    std::vector<CompilerOp> gates;
    std::vector<int> record_wires(ir.records, -1), environment_wires;
    auto emit = [&](std::string kind, std::vector<int> qubits, int exponent = 0, std::string pauli = "") {
        CompilerOp g;
        g.kind = std::move(kind);
        g.qubits = std::move(qubits);
        g.exponent = exponent;
        g.pauli = std::move(pauli);
        gates.push_back(std::move(g));
    };
    auto fresh = [&](bool plus = false) {
        if(count == std::numeric_limits<int>::max()) throw std::length_error("dilation wire capacity");
        int q = count++;
        emit(plus ? "PREP_PLUS" : "PREP_ZERO", {q});
        return q;
    };
    auto diagonal = [&](const std::vector<int>& qs, int exponent, int feedback_slot = -1) {
        exponent = mod8(exponent);
        if(!exponent && feedback_slot < 0) return;
        if(!(exponent & 1) && feedback_slot < 0) {
            emit("PHASE", qs, exponent);
            return;
        }
        if(count == std::numeric_limits<int>::max()) throw std::length_error("dilation wire capacity");
        emit("PHASE_SLOT", qs);
        gates.back().feedback_slot = int(plan.magic.size());
        // Reserve an ID, without creating a wire. A continuation then keeps
        // every nonreference ancillary ID even if it adds physical lifetimes.
        gates.back().sign = count++;
        plan.magic.push_back(exponent);
        result.phase_controls.push_back(feedback_slot);
    };
    auto measure = [&](const std::string& kind, const std::vector<int>& qs, int record, int sign = 0) {
        if(record < 0 || size_t(record) >= ir.records || record_wires[size_t(record)] >= 0)
            throw std::invalid_argument("invalid measurement record");
        bool destructive = kind == "MZ_DESTRUCT" || kind == "MX_DESTRUCT";
        int r;
        if(destructive && qs.size() != 1) throw std::invalid_argument("destructive measurement arity");
        if(kind == "MZ" || kind == "MZ_DESTRUCT") {
            r = destructive ? qs[0] : fresh();
            if(!destructive)
                for(auto q : qs) emit("CX", {q, r});
        } else if(kind == "MX" || kind == "MXX" || kind == "MX_DESTRUCT") {
            r = destructive ? qs[0] : fresh(true);
            if(!destructive)
                for(auto q : qs) emit("CX", {r, q});
            emit("H", {r});
        } else throw std::invalid_argument("unsupported measurement kind");
        if(sign) emit("X", {r});
        record_wires[size_t(record)] = r;
    };
    std::set<int> seen_locations;
    for(size_t index = 0; index < ir.operations.size(); index++) {
        auto& op = ir.operations[index];
        auto& physical = op.qubits;
        auto& kind = op.kind;
        if(std::set<int>(physical.begin(), physical.end()).size() != physical.size())
            throw std::invalid_argument("repeated qubit within operation");
        if(kind == "PREP_ZERO" || kind == "PREP_PLUS" || kind == "RESET_ZERO" || kind == "RESET_PLUS" ||
                kind == "DISCARD") {
            if(physical.size() != 1) throw std::invalid_argument("preparation/reset/discard arity");
            int q = physical[0];
            bool retired = kind == "RESET_ZERO" || kind == "RESET_PLUS" || kind == "DISCARD";
            if(retired) {
                if(!live.count(q)) throw std::invalid_argument("reset/discard outside wire lifetime");
                environment_wires.push_back(live[q]);
                result.environment.push_back(op.location);
                live.erase(q);
            } else if(live.count(q)) throw std::invalid_argument("double preparation");
            if(kind != "DISCARD") live[q] = fresh(kind == "PREP_PLUS" || kind == "RESET_PLUS");
            plan.histories[q].push_back({int(index), live.count(q) ? live[q] : -1});
        } else {
            std::vector<int> qs;
            for(auto q : physical) {
                if(!live.count(q)) throw std::invalid_argument("operation outside lifetime");
                qs.push_back(live[q]);
            }
            if(kind == "T" || kind == "S" || kind == "PHASE") diagonal(qs, op.exponent, op.feedback_slot);
            else if(kind == "CCZ") {
                if(qs.size() != 3) throw std::invalid_argument("CCZ arity");
                // Match the original parity expansion's stable layer order.
                for(size_t a = 0; a < 3; a++) diagonal({qs[a]}, 1);
                for(size_t a = 0; a < 3; a++)
                    for(size_t b = a + 1; b < 3; b++) diagonal({qs[a], qs[b]}, -1);
                diagonal(qs, 1);
            } else if(kind == "MP") {
                if(op.paulis.size() != qs.size() || op.paulis.find_first_not_of("XYZ") != std::string::npos)
                    throw std::invalid_argument("Pauli measurement body");
                if(op.sign != 0 && op.sign != 1) throw std::invalid_argument("Pauli measurement sign");
                std::vector<CompilerOp> before;
                for(size_t j = 0; j < qs.size(); j++) {
                    if(op.paulis[j] == 'Y') {
                        CompilerOp g;
                        g.kind = "PHASE";
                        g.qubits = {qs[j]};
                        g.exponent = -2;
                        before.push_back(g);
                    }
                    if(op.paulis[j] == 'X' || op.paulis[j] == 'Y') {
                        CompilerOp g;
                        g.kind = "H";
                        g.qubits = {qs[j]};
                        before.push_back(g);
                    }
                }
                for(auto& g : before) emit(g.kind, g.qubits, g.exponent);
                measure("MZ", qs, op.record, op.sign);
                for(auto it = before.rbegin(); it != before.rend(); ++it) emit(it->kind, it->qubits, -it->exponent);
            } else if(kind == "MZ" || kind == "MX" || kind == "MXX" || kind == "MZ_DESTRUCT" || kind == "MX_DESTRUCT") {
                measure(kind, qs, op.record);
                if(kind == "MZ_DESTRUCT" || kind == "MX_DESTRUCT") {
                    live.erase(physical[0]);
                    plan.histories[physical[0]].push_back({int(index), -1});
                }
            } else if(kind == "H" || kind == "CX" || kind == "CZ" || kind == "PAULI" || kind == "X" || kind == "Y" ||
                    kind == "Z" || kind == "I") {
                size_t arity = kind == "CX" || kind == "CZ" ? 2 : 1;
                if(kind != "I" && qs.size() != arity) throw std::invalid_argument("Clifford gate arity");
                if(kind == "PAULI" && op.pauli != "I" && op.pauli != "X" && op.pauli != "Y" && op.pauli != "Z")
                    throw std::invalid_argument("Pauli name");
                emit(kind, qs, 0, op.pauli);
            } else throw std::invalid_argument("unsupported original operation: " + kind);
        }
        if(op.location >= 0) {
            if(!seen_locations.insert(op.location).second) throw std::invalid_argument("duplicate original location");
            if(size_t(op.location) != plan.locations.size())
                throw std::invalid_argument("nonsequential original locations");
            if(gates.size() > size_t(std::numeric_limits<int>::max())) throw std::length_error("gate index capacity");
            plan.locations.push_back({int(index), int(gates.size())});
        }
    }
    std::set<int> live_set;
    for(auto& [q, v] : live) live_set.insert(q);
    if(live_set != unique_outputs) throw std::invalid_argument("outputs must account for every live wire");
    std::vector<int> outputs;
    for(auto q : ir.outputs) outputs.push_back(live.at(q));
    outputs.insert(outputs.end(), environment_wires.begin(), environment_wires.end());
    result.output_count = ir.outputs.size();
    plan.physical = outputs.size();
    plan.retained = plan.physical + plan.magic.size();
    for(auto q : record_wires)
        if(q < 0) throw std::invalid_argument("missing measurement record");
    outputs.insert(outputs.end(), record_wires.begin(), record_wires.end());
    plan.nrecords = ir.records;
    return {std::move(gates), std::move(outputs)};
}
// Group the ordered rotations into layers of mutually commuting axes in a
// common Clifford endpoint frame and choose the layer with the most slots.
struct CompilerRotationLayers {
    std::vector<size_t> layers;
    size_t count = 0, best_layer = 0, best_slots = 0;
};
inline CompilerRotationLayers compiler_rotation_layers(const std::vector<CompilerOp>& gates,
        const std::vector<int>& outputs, size_t magic_count, std::vector<Frame>& rotation_axes) {
    CompilerRotationLayers out;
    auto& layers = out.layers;
    auto& best_layer = out.best_layer;
    auto& best_slots = out.best_slots;
    size_t layer = 0, current_slots = 0;
    Plan axes_plan;
    std::vector<Frame> captured_rotation_axes(magic_count);
    if(magic_count) compiler_fault_map(axes_plan, gates, outputs, {}, 0, &captured_rotation_axes);
    std::vector<Frame> layer_basis;
    std::map<size_t, Bits> layer_pivots;
    for(size_t i = 0; i < gates.size(); i++)
        if(gates[i].kind == "PHASE_SLOT") {
            Frame axis = std::move(captured_rotation_axes.at(layers.size()));
            bool compatible = true;
            for(const auto& prior : layer_basis)
                if(axis.x.parity_and(prior.z) ^ axis.z.parity_and(prior.x)) {
                    compatible = false;
                    break;
                }
            if(!compatible) {
                layer++;
                current_slots = 0;
                layer_basis.clear();
                layer_pivots.clear();
            }
            auto residual = axis.x | (axis.z << outputs.size());
            while(!residual.empty()) {
                auto found = layer_pivots.find(residual.size() - 1);
                if(found == layer_pivots.end()) break;
                residual ^= found->second;
            }
            if(!residual.empty()) {
                layer_pivots[residual.size() - 1] = residual;
                layer_basis.push_back(axis);
            }
            layers.push_back(layer);
            rotation_axes.push_back(std::move(axis));
            if(++current_slots > best_slots) {
                best_slots = current_slots;
                best_layer = layer;
            }
        }
    out.count = layers.empty() ? 0 : layer + 1;
    return out;
}
// Retained phases are observations of the common path coordinates, not
// fresh quantum wires. Only rotations outside the reference layer receive
// copying gadgets; returns those gadgets' ancillary wires.
inline std::vector<int> compiler_capture_phases(
        std::vector<CompilerOp>& gates, const std::vector<size_t>& layers, size_t best_layer, Plan& plan) {
    std::vector<int> magic_wires;
    std::vector<CompilerOp> annotated;
    annotated.reserve(gates.size());
    std::vector<size_t> positions(gates.size() + 1);
    size_t slot = 0;
    for(size_t i = 0; i < gates.size(); i++) {
        positions[i] = annotated.size();
        auto g = gates[i];
        if(g.kind == "PHASE_SLOT") {
            if(layers[slot] != best_layer) {
                int q = g.sign;
                magic_wires.push_back(q);
                CompilerOp prep;
                prep.kind = "PREP_ZERO";
                prep.qubits = {q};
                annotated.push_back(std::move(prep));
                for(auto c : g.qubits) {
                    CompilerOp cx;
                    cx.kind = "CX";
                    cx.qubits = {c, q};
                    annotated.push_back(std::move(cx));
                }
                g.qubits = {q};
            }
            g.kind = "PHASE_CAPTURE";
            g.feedback_slot = int(slot++);
        }
        annotated.push_back(std::move(g));
    }
    positions.back() = annotated.size();
    for(auto& location : plan.locations) location.position = int(positions.at(size_t(location.position)));
    gates = std::move(annotated);
    return magic_wires;
}
// Compress the phase observations before Clifford contraction. This is an
// injective change of their shared coordinates, with no scalar change.
// Returns each capture's coordinates in the independent axes appended to path.
inline std::vector<Bits> compiler_compress_captures(
        const std::vector<CompilerForm>& captures, Ket& path, size_t endpoint_count, size_t& phase_coordinates) {
    std::vector<Bits> capture_coordinates;
    std::vector<Bits> capture_axes;
    std::map<size_t, std::pair<Bits, Bits>> capture_pivots;
    for(const auto& capture : captures) {
        Bits residual = capture.mask, coordinates;
        while(!residual.empty()) {
            auto found = capture_pivots.find(residual.size() - 1);
            if(found == capture_pivots.end()) break;
            residual ^= found->second.first;
            coordinates ^= found->second.second;
        }
        if(!residual.empty()) {
            auto unit = Bits::unit(capture_axes.size());
            capture_axes.push_back(capture.mask);
            capture_pivots[residual.size() - 1] = {residual, coordinates ^ unit};
            coordinates = unit;
        }
        capture_coordinates.push_back(std::move(coordinates));
    }
    phase_coordinates = capture_axes.size();
    auto path_rows = transpose(path.columns, path.width);
    path_rows.resize(endpoint_count);
    path_rows.insert(path_rows.end(), capture_axes.begin(), capture_axes.end());
    path.width = path_rows.size();
    path.base = path.base.low(endpoint_count);
    path.columns = transpose(path_rows, path.linear.size());
    return capture_coordinates;
}
// Project the dilation onto the measurement-record kernel for amplitude
// consumers: path basis, quadratic form, retained forms and magic terms.
inline void compiler_project_amplitude(CompilerPreparation& result) {
    auto& plan = result.plan;
    plan.basis = plan.records.kernel;
    auto q = restrict_q(plan.skeleton.linear, plan.skeleton.upper, Bits(), plan.basis);
    result.linear = std::move(q.linear);
    result.upper = std::move(q.upper);
    {
        std::vector<Bits> columns;
        for(auto& v : plan.basis) columns.push_back(xors(plan.skeleton.columns, v).low(plan.retained));
        plan.forms = transpose(columns, plan.retained);
    }
    plan.polar = plan.skeleton.upper;
    for(size_t j = 0; j < plan.skeleton.columns.size(); j++) {
        if(plan.skeleton.linear[j] & 1) plan.polar[j].set(j);
        for(auto h : plan.skeleton.upper[j].indices()) plan.polar[h].set(j);
    }
    {
        std::map<Bits, int, BitsLess> terms;
        for(size_t j = 0; j < plan.magic.size(); j++) terms[plan.forms[plan.physical + j]] += plan.magic[j];
        for(auto& [a, c] : terms)
            if(!a.empty() && mod8(c)) result.magic_terms.push_back({a, mod8(c)});
    }
    {
        std::vector<Bits> physical_forms(plan.forms.begin(), plan.forms.begin() + plan.physical);
        result.injective_physical = Factor(physical_forms, plan.basis.size()).pivots.size() == plan.basis.size();
    }
}
// Map the parent prefix's endpoint generators, captured at the continuation
// cut, to this plan's boundary frame.
inline FrameContinuation compiler_continuation(const Plan& plan, const std::vector<int>& prior_endpoints,
        const std::vector<int>& prior_positions, size_t prior_physical, size_t prior_magic, size_t continuation_cut) {
    FrameContinuation map;
    map.source_physical = prior_physical;
    map.source_magic = prior_magic;
    map.target_physical = plan.physical;
    size_t width = prior_physical + prior_magic;
    for(auto at : prior_positions)
        if(at >= 0) width = std::max(width, size_t(at) + 1);
    map.x.resize(width);
    map.z.resize(width);
    for(size_t j = 0; j < prior_endpoints.size(); j++)
        if(prior_positions[j] >= 0) {
            const auto& suffix = plan.suffix.at(prior_endpoints[j]);
            auto image = std::lower_bound(suffix.begin(), suffix.end(), int(continuation_cut),
                    [](const auto& a, int position) { return a.position < position; });
            if(image == suffix.end()) throw std::logic_error("missing compiler continuation endpoint");
            map.x[size_t(prior_positions[j])] = image->x;
            map.z[size_t(prior_positions[j])] = image->z;
        }
    return map;
}
enum class CompilerPurpose { Amplitude, Trace };
inline CompilerPreparation prepare_compiler(const CompilerIR& ir, bool track_global_phase = true,
        CompilerPathPrefix* prefix = nullptr, bool retain_fault_map = true,
        CompilerPurpose purpose = CompilerPurpose::Amplitude, bool branch_transport_only = false) {
    CompilerPreparation result;
    auto& plan = result.plan;
    auto tick = std::chrono::steady_clock::now();
    auto elapsed = [&]() {
        auto now = std::chrono::steady_clock::now();
        double dt = std::chrono::duration<double>(now - tick).count();
        tick = now;
        return dt;
    };
    auto dilation = compiler_dilation(ir, result);
    auto& gates = dilation.gates;
    auto& outputs = dilation.endpoints;

    // The path input uses the dilation's internal wire IDs, not physical labels.
    CompilerIR path_ir;
    path_ir.initial = ir.initial;
    path_ir.max_path_bits = ir.max_path_bits;
    for(size_t j = 0; j < ir.inputs.size(); j++) path_ir.inputs.push_back(int(j));
    // Inspect the ordered rotations in a common Clifford endpoint frame.
    // PREP/reset lifetimes have already been expanded into distinct wire IDs.
    // This caller needs only rotation axes, not a history for later fault queries.
    auto rotations = compiler_rotation_layers(gates, outputs, plan.magic.size(), result.rotation_axes);
    const auto& layers = rotations.layers;
    size_t best_layer = rotations.best_layer, best_slots = rotations.best_slots;
    // Continuations retain their parent's selected layer. Shared boundary/path
    // preparation takes priority over saving additional auxiliary gadgets.
    bool inherited =
            prefix && prefix->reference_layer >= 0 && prefix->matches(path_ir, gates, track_global_phase, true);
    if(inherited) {
        best_layer = size_t(prefix->reference_layer);
        best_slots = size_t(std::count(layers.begin(), layers.end(), best_layer));
    }
    auto source_gates = prefix ? gates : std::vector<CompilerOp>{};
    result.rotation_layers = rotations.count;
    result.reference_layer_slots = best_slots;
    result.reference_layer = layers.empty() ? -1 : int(best_layer);
    result.layer_slots.resize(result.rotation_layers);
    for(auto id : layers) result.layer_slots[id]++;
    auto magic_wires = compiler_capture_phases(gates, layers, best_layer, plan);
    result.ancilla_count = magic_wires.size();
    // Auxiliary endpoints are captured symbolically and are not physical trace legs.
    auto endpoint_count = outputs.size();
    outputs.insert(outputs.end(), magic_wires.begin(), magic_wires.end());
    const bool has_continuation = prefix && prefix->matches(path_ir, gates, track_global_phase);
    const size_t continuation_cut = has_continuation ? prefix->gates.size() : 0;
    auto prior_endpoints = has_continuation ? prefix->endpoints : std::vector<int>{};
    auto prior_positions = has_continuation ? prefix->frame_positions : std::vector<int>{};
    const size_t prior_physical = has_continuation ? prefix->frame_physical : 0;
    const size_t prior_magic = has_continuation ? prefix->frame_magic : 0;
    result.dilation_seconds = elapsed();
    {
        std::vector<CompilerForm> captures;
        auto path = compiler_path_tensor(path_ir, gates, outputs, result.path_variables, track_global_phase, prefix,
                &result.shared_path_gates, &captures);
        if(prefix) {
            prefix->source_gates = std::move(source_gates);
            prefix->reference_layer = result.reference_layer;
        }
        auto capture_coordinates = compiler_compress_captures(captures, path, endpoint_count, result.phase_coordinates);
        result.path_seconds = elapsed();
        {
            Workspace ws;
            plan.skeleton = contract(std::move(path), ws, track_global_phase);
        }
        result.contraction_seconds = elapsed();
        if(plan.skeleton.zero) throw std::runtime_error("Clifford isometry cannot vanish");
        // Expand only affine metadata for the established per-rotation assignment
        // interface. No ancillary gates or second Clifford contraction are created.
        auto shared_rows = transpose(plan.skeleton.columns, plan.skeleton.width);
        std::vector<Bits> rows(shared_rows.begin(), shared_rows.begin() + plan.physical);
        Bits expanded_base = plan.skeleton.base.low(plan.physical);
        std::vector<Bits> phase_rows(shared_rows.begin() + endpoint_count, shared_rows.end());
        for(size_t j = 0; j < captures.size(); j++) {
            rows.push_back(xors(phase_rows, capture_coordinates[j]));
            bool bit = captures[j].offset ^ ((plan.skeleton.base >> endpoint_count).parity_and(capture_coordinates[j]));
            if(bit) expanded_base.set(plan.physical + j);
        }
        for(size_t j = 0; j < ir.records; j++) {
            rows.push_back(shared_rows[plan.physical + j]);
            if(plan.skeleton.base.bit(plan.physical + j)) expanded_base.set(plan.retained + j);
        }
        plan.skeleton.width = rows.size();
        plan.skeleton.base = std::move(expanded_base);
        plan.skeleton.columns = transpose(rows, plan.skeleton.columns.size());
        if(purpose == CompilerPurpose::Amplitude)
            plan.records =
                    Factor(std::vector<Bits>(rows.begin() + plan.retained, rows.end()), plan.skeleton.columns.size());
    }
    // Trace consumers contract the dilation itself. The amplitude projection
    // below has no consumer there; retain the same path/prefix and fault maps.
    if(purpose == CompilerPurpose::Amplitude) compiler_project_amplitude(result);
    result.projection_seconds = elapsed();
    std::vector<int> frame_positions;
    for(size_t j = 0; j < outputs.size(); j++)
        frame_positions.push_back(j < plan.physical ? int(j)
                        : j < endpoint_count        ? int(plan.retained + j - plan.physical)
                                                    : -1);
    if(retain_fault_map)
        compiler_fault_map(plan, gates, outputs, frame_positions, plan.physical, nullptr,
                branch_transport_only && has_continuation ? continuation_cut : 0);
    if(has_continuation && retain_fault_map) {
        plan.continuation = compiler_continuation(
                plan, prior_endpoints, prior_positions, prior_physical, prior_magic, continuation_cut);
        if(branch_transport_only) {
            // This branch owns only the cut-to-boundary generator map.
            // Explicit standalone assignments can prepare their own map lazily.
            plan.suffix.clear();
            plan.histories.clear();
            plan.locations.clear();
        }
    }
    if(prefix) {
        prefix->endpoints = outputs;
        prefix->frame_positions = frame_positions;
        prefix->frame_physical = plan.physical;
        prefix->frame_magic = plan.magic.size();
    }
    result.fault_map_seconds = elapsed();
    result.gate_count = gates.size() - plan.magic.size();
    return result;
}
} // namespace magic_native
