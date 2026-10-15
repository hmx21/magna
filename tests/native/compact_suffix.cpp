// Lossless suffix storage and ordered manual faults, with set-valued Paulis.
#include "compile/plan.hpp"
#include <iostream>
#include <random>
#include <set>
using namespace magic_native;

size_t checks = 0;
void require(bool condition, const char* message) {
    checks++;
    if(!condition) throw std::runtime_error(message);
}
using Positions = std::set<size_t>;
struct Pauli {
    Positions x, z;
    int phase = 0;
};
Positions symmetric_difference(const Positions& a, const Positions& b) {
    Positions out;
    for(auto q : a)
        if(!b.count(q)) out.insert(q);
    for(auto q : b)
        if(!a.count(q)) out.insert(q);
    return out;
}
Pauli product(const Pauli& a, const Pauli& b) {
    int sign = 0;
    for(auto q : a.z) sign ^= bool(b.x.count(q));
    return {symmetric_difference(a.x, b.x), symmetric_difference(a.z, b.z), (a.phase + b.phase + 2 * sign) % 4};
}
Bits packed(const Positions& values) {
    Bits out;
    for(auto q : values) out.set(q);
    return out;
}
Frame packed(const Pauli& p) { return {packed(p.x), packed(p.z), p.phase}; }
Positions positions(const Bits& bits) {
    Positions out;
    for(size_t j = 0; j < bits.size(); j++)
        if(bits.bit(j)) out.insert(j);
    return out;
}
void same(const Frame& actual, const Pauli& expected) {
    require(positions(actual.x) == expected.x, "X positions lost in suffix storage/transport");
    require(positions(actual.z) == expected.z, "Z positions lost in suffix storage/transport");
    require(actual.p == expected.phase, "Pauli scalar phase lost in suffix storage/transport");
}

void roundtrip(const Positions& values) {
    Bits original = packed(values);
    StoredBits saved(original);
    require(positions(saved.materialize()) == values, "stored bitset roundtrip");
    Positions base = {1, 63, 65, 100003};
    Bits target = packed(base);
    saved.xor_into(target);
    require(positions(target) == symmetric_difference(base, values), "stored XOR preserves bits outside its span");
    saved.xor_into(target);
    require(positions(target) == base, "stored XOR is involutive");
    target = original;
    saved.xor_into(target);
    require(target.empty(), "stored XOR trims complete cancellation");
    auto copy = saved;
    StoredBits moved(std::move(copy));
    require(positions(moved.materialize()) == values, "stored data own their copy/move lifetime");
    original = Bits();
    require(positions(saved.materialize()) == values, "stored bits do not alias input");
}

void check_storage() {
    roundtrip({});
    for(auto bit : {size_t(0), size_t(63), size_t(64), size_t(65), size_t(127), size_t(128), size_t(129), size_t(65535),
                size_t(65536), size_t(100000)}) {
        roundtrip({bit});
        roundtrip({0, bit});
        roundtrip({bit, bit + 1, bit + 64});
    }
    for(auto first : {size_t(0), size_t(61), size_t(64), size_t(99328)}) {
        Positions dense;
        for(size_t q = first; q < first + 257; q++) dense.insert(q);
        roundtrip(dense);
        Positions spaced;
        for(size_t q = first; q < first + 257; q += 64) spaced.insert(q);
        roundtrip(spaced);
    }
    // Separate large endpoint from populated word count: one high bit has a
    // short dense payload; two remote bits must not retain the intervening zeros.
    StoredBits distant(packed(Positions{0, 100000})), high(packed(Positions{100000}));
    require(distant.sparse(), "distant populated words select sparse storage");
    require(!high.sparse(), "one offset word uses contiguous storage");
    require(distant.payload_bytes() < 128 && high.payload_bytes() < 128,
            "high bit index does not allocate its zero prefix");
    std::mt19937_64 random(2026100501);
    for(size_t trial = 0; trial < 64; trial++) {
        Positions sparse, dense;
        for(size_t j = 0; j < 1 + trial % 19; j++) sparse.insert(random() % 100001);
        for(size_t j = 0; j < 1 + trial * 4; j++)
            if(random() & 1) dense.insert(63 + j);
        roundtrip(sparse);
        roundtrip(dense);
        for(int phase = 0; phase < 4; phase++) {
            Pauli value{sparse, dense, phase};
            StoredFrame stored(packed(value));
            same(stored.materialize(), value);
            Pauli initial{{0, 65, 100000}, {63, 129}, 3};
            Frame target = packed(initial);
            stored.xor_into(target);
            same(target,
                    {symmetric_difference(initial.x, sparse), symmetric_difference(initial.z, dense), initial.phase});
        }
    }
}

void check_range_capacity() {
    // Private storage can only come from an allocated dense source. Check
    // exact endpoint arithmetic and canonicalization across packed boundaries;
    // no forged invalid private header or impossible giant allocation is used.
    for(size_t high : {size_t(63), size_t(64), size_t(65), size_t(127), size_t(128), size_t(65535), size_t(65536),
                size_t(100000)}) {
        for(bool distant : {false, true}) {
            Positions source = {high};
            if(distant) source.insert(0);
            Bits bits = packed(source);
            StoredBits stored(bits);
            require(stored.word_limit() == high / 64 + 1, "stored endpoint must not truncate or wrap its word index");
            require(stored.word_limit() <= bits.w.size(), "stored range is bounded by its allocated source");
            bits.w.resize(bits.w.size() + 65, 0);
            StoredBits padded(bits);
            require(padded.word_limit() == high / 64 + 1, "trailing empty words do not change the stored range");
            require(positions(padded.materialize()) == source, "trailing empty words retain the same bitset");
        }
    }
    Bits zero;
    zero.w.resize(100, 0);
    StoredBits saved_zero(zero);
    require(saved_zero.word_limit() == 0 && saved_zero.materialize().empty(),
            "all-zero allocated source has an empty range");
}

struct Row {
    int position;
    Pauli x, z;
};
void add(Plan& plan, std::map<int, std::vector<Row>>& reference, int wire, int at, Pauli x, Pauli z) {
    plan.suffix[wire].push_back({at, packed(x), packed(z)});
    reference[wire].push_back({at, std::move(x), std::move(z)});
}
Pauli expected_faults(const Plan& plan, const std::map<int, std::vector<Row>>& rows, const std::vector<Fault>& input) {
    // Iterate original locations, then original input order: independent of
    // Plan's stable-sort/lower-bound implementation and compact byte layout.
    Pauli out;
    for(size_t location = 0; location < plan.locations.size(); location++)
        for(const auto& fault : input) {
            if(fault.location != location || fault.pauli == 0) continue;
            const auto& loc = plan.locations.at(location);
            int internal = -1;
            for(const auto& lifetime : plan.histories.at(fault.wire))
                if(lifetime.first <= loc.index) internal = lifetime.second;
            if(internal < 0) throw std::invalid_argument("reference fault outside wire lifetime");
            const Row* row = nullptr;
            for(const auto& candidate : rows.at(internal))
                if(candidate.position >= loc.position) {
                    row = &candidate;
                    break;
                }
            if(!row) throw std::invalid_argument("reference suffix missing");
            Pauli image = fault.pauli == 1 ? row->x : row->z;
            if(fault.pauli == 3) {
                image = product(row->x, row->z);
                image.phase = (image.phase + 1) % 4;
            }
            out = product(image, out);
        }
    return out;
}
void check_manual_assignments() {
    Plan plan;
    std::map<int, std::vector<Row>> rows;
    plan.locations = {{0, 0}, {1, 2}, {2, 5}, {3, 6}, {4, 9}, {5, 12}};
    plan.histories[5] = {{-1, 10}, {3, 11}, {5, -1}};
    plan.histories[7] = {{-1, 20}};
    add(plan, rows, 10, 1, {{0, 100000}, {65}, 1}, {{63}, {0}, 2});
    add(plan, rows, 10, 4, {{65}, {0, 100000}, 2}, {{0}, {63, 100000}, 3});
    add(plan, rows, 10, 6, {{}, {}, 3}, {{100000}, {100000}, 1});
    add(plan, rows, 11, 8, {{129}, {65}, 0}, {{65}, {129}, 1});
    add(plan, rows, 11, 12, {{0, 64, 100000}, {63}, 2}, {{63}, {0, 64, 100000}, 3});
    for(int at : {1, 4, 8, 12})
        add(plan, rows, 20, at, {{size_t(at), 64}, {100000}, at % 4}, {{100000}, {size_t(at), 64}, (at + 1) % 4});
    for(const auto& [wire, values] : rows)
        for(size_t index = 0; index < values.size(); index++) {
            const auto& actual = plan.suffix.at(wire).at(index);
            const auto& expected = values[index];
            same(actual.x.materialize(), expected.x);
            same(actual.z.materialize(), expected.z);
        }
    std::vector<std::vector<Fault>> cases = {{{4, 5, 3}, {0, 5, 1}, {4, 5, 2}, {1, 7, 3}}, {{1, 5, 1}, {1, 5, 2}},
            {{1, 5, 2}, {1, 5, 1}}, {{0, 5, 3}, {0, 5, 3}}, {{5, 7, 3}, {2, 5, 2}, {3, 5, 1}, {4, 7, 2}}, {{0, 999, 0}},
            {}};
    std::mt19937_64 random(2026100502);
    for(size_t trial = 0; trial < 128; trial++) {
        std::vector<Fault> faults;
        for(size_t j = 0; j < trial % 11; j++) {
            auto loc = size_t(random() % 6);
            int wire = (loc < 5 && (random() & 1)) ? 5 : 7;
            faults.push_back({loc, wire, int(random() % 4)});
        }
        cases.push_back(std::move(faults));
    }
    for(const auto& faults : cases) {
        auto expected = expected_faults(plan, rows, faults);
        same(plan.fault_frame(faults), expected);
        expected.phase = 0;
        same(plan.fault_frame(faults, false), expected);
        Plan copied = plan;
        same(copied.fault_frame(faults), expected_faults(plan, rows, faults));
    }
    for(const auto& faults : std::vector<std::vector<Fault>>{{{5, 5, 1}}, {{0, 999, 1}}}) {
        bool rejected = false;
        try {
            plan.fault_frame(faults);
        } catch(const std::invalid_argument&) { rejected = true; }
        require(rejected, "original-location fault must respect wire lifetime");
    }
}

int main() {
    check_storage();
    check_range_capacity();
    check_manual_assignments();
    std::cout << "native compact suffix checks passed: " << checks << "\n";
}
