// Independent unpacked-bit oracles for shared GF(2) and exact quadratic kernels.
#include "foundation/stabilizer.hpp"
#include <iostream>
#include <random>
#include <set>
using namespace magic_native;
namespace {
using Row = std::vector<unsigned char>;
using Matrix = std::vector<Row>;
size_t checks = 0, factor_cases = 0, quadratic_cases = 0;
std::mt19937_64 random_bits(2026100409);
void require(bool ok, const char* why) {
    ++checks;
    if(!ok) throw std::runtime_error(why);
}
Bits pack(const Row& row) {
    Bits result;
    for(size_t j = 0; j < row.size(); j++)
        if(row[j]) result.set(j);
    return result;
}
Row unpack(const Bits& row, size_t n) {
    Row result(n);
    for(size_t j = 0; j < n; j++) result[j] = row.bit(j);
    return result;
}
Row random_row(size_t n, unsigned density = 2) {
    Row row(n);
    for(auto& b : row) b = random_bits() % density == 0;
    return row;
}
std::vector<Bits> pack(const Matrix& matrix) {
    std::vector<Bits> result;
    for(const auto& row : matrix) result.push_back(pack(row));
    return result;
}
unsigned dot(const Row& a, const Row& b) {
    unsigned result = 0;
    for(size_t j = 0; j < std::min(a.size(), b.size()); j++) result ^= a[j] * b[j];
    return result;
}
Row product(const Matrix& a, const Row& x) {
    Row result;
    for(const auto& row : a) result.push_back(dot(row, x));
    return result;
}
Row combination(const Matrix& a, const Row& mask, size_t n) {
    Row result(n);
    for(size_t i = 0; i < a.size(); i++)
        if(i < mask.size() && mask[i])
            for(size_t j = 0; j < n; j++) result[j] ^= a[i][j];
    return result;
}
size_t rank(Matrix rows, size_t n) {
    size_t active = 0;
    // Left-to-right byte Gaussian elimination, unlike production high-bit pivots.
    for(size_t column = 0; column < n && active < rows.size(); column++) {
        size_t pivot = active;
        while(pivot < rows.size() && !rows[pivot][column]) pivot++;
        if(pivot == rows.size()) continue;
        std::swap(rows[active], rows[pivot]);
        for(size_t i = 0; i < rows.size(); i++)
            if(i != active && rows[i][column])
                for(size_t j = column; j < n; j++) rows[i][j] ^= rows[active][j];
        active++;
    }
    return active;
}
bool consistent(const Matrix& a, const Row& rhs, size_t n) {
    auto augmented = a;
    for(size_t i = 0; i < a.size(); i++) augmented[i].push_back(rhs[i]);
    return rank(a, n) == rank(augmented, n + 1);
}
void check_factor(const Matrix& a, size_t n) {
    factor_cases++;
    auto rows = pack(a);
    Factor factor(rows, n);
    size_t r = rank(a, n);
    require(factor.pivots.size() == r, "Factor rank disagrees with independent byte elimination");
    require(factor.kernel.size() == n - r, "Factor complete right nullspace dimension");
    require(factor.zeros.size() == a.size() - r, "Factor complete left nullspace dimension");
    Matrix kernel, left;
    for(const auto& v : factor.kernel) {
        require(v.size() <= n, "right nullspace width");
        auto row = unpack(v, n);
        require(product(a, row) == Row(a.size()), "right nullspace annihilation");
        kernel.push_back(row);
    }
    for(const auto& v : factor.zeros) {
        require(v.size() <= a.size(), "left nullspace width");
        auto row = unpack(v, a.size());
        require(combination(a, row, n) == Row(n), "zero-row provenance");
        left.push_back(row);
    }
    require(rank(kernel, n) == n - r, "right nullspace independence");
    require(rank(left, a.size()) == a.size() - r, "left nullspace independence");
    for(const auto& [pivot, rt] : factor.pivots) {
        require(rt.first.size() == pivot + 1, "highest-bit pivot convention");
        require(combination(a, unpack(rt.second, a.size()), n) == unpack(rt.first, n), "pivot row exact provenance");
    }
    size_t trials = a.size() <= 5 ? size_t(1) << a.size() : 16;
    for(size_t trial = 0; trial < trials; trial++) {
        Row rhs = a.size() <= 5 ? unpack(Bits(trial), a.size()) : random_row(a.size());
        if(trial % 2 && a.size() > 5) rhs = product(a, random_row(n));
        bool expected = consistent(a, rhs, n);
        require(factor.consistent(pack(rhs)) == expected, "Factor consistency including contradictions");
        if(!expected) continue;
        auto solution = factor.solve(pack(rhs));
        require(solution.size() <= n, "particular solution width");
        require(product(a, unpack(solution, n)) == rhs, "particular solution");
        auto initial = random_row(n);
        auto seeded = factor.solve(pack(rhs), pack(initial));
        require(product(a, unpack(seeded, n)) == rhs, "solve with seeded free coordinates");
        for(size_t j = 0; j < n; j++)
            if(!factor.pivots.count(j)) require(seeded.bit(j) == bool(initial[j]), "free coordinate convention");
        if(n <= 5) {
            std::set<Row> expected_solutions, generated;
            for(size_t x = 0; x < (size_t(1) << n); x++) {
                auto row = unpack(Bits(x), n);
                if(product(a, row) == rhs) expected_solutions.insert(row);
            }
            for(size_t mask = 0; mask < (size_t(1) << kernel.size()); mask++) {
                auto row = combination(kernel, unpack(Bits(mask), kernel.size()), n);
                auto particular = unpack(solution, n);
                for(size_t j = 0; j < n; j++) row[j] ^= particular[j];
                generated.insert(row);
            }
            require(expected_solutions == generated,
                    "complete affine solution set by exhaustive independent enumeration");
        }
    }
}
void factor_tests() {
    for(size_t n = 0; n <= 5; n++)
        for(size_t m = 0; m <= 5; m++)
            for(size_t trial = 0; trial < 3; trial++) {
                Matrix a(m, Row(n));
                if(trial)
                    for(auto& row : a) row = random_row(n);
                if(trial == 2 && m > 1) a.back() = a.front();
                check_factor(a, n);
            }
    for(size_t n : {size_t(63), size_t(64), size_t(65), size_t(127), size_t(128), size_t(129), size_t(257)}) {
        for(size_t m : {size_t(0), size_t(1), size_t(7), size_t(70)}) {
            Matrix a(m, Row(n));
            for(auto& row : a) row = random_row(n);
            if(m > 2) {
                a.back() = a.front();
                a[m - 2] = Row(n);
            }
            check_factor(a, n);
        }
        Matrix identity(n, Row(n));
        for(size_t j = 0; j < n; j++) identity[j][j] = 1;
        check_factor(identity, n);
    }
}
void packed_tests() {
    require(Bits(1).low(std::numeric_limits<size_t>::max()) == Bits(1), "low enormous bound preserves finite prefix");
    require(Bits().low(std::numeric_limits<size_t>::max()).empty(), "low enormous bound of empty bits");
    for(size_t n : {size_t(0), size_t(1), size_t(63), size_t(64), size_t(65), size_t(127), size_t(128), size_t(129),
                size_t(257)})
        for(size_t m : {size_t(0), size_t(1), size_t(3), size_t(4), size_t(7), size_t(63), size_t(64), size_t(65),
                    size_t(137)})
            for(unsigned density : {2u, 17u}) {
                Matrix a(m, Row(n));
                for(auto& row : a) row = random_row(n, density);
                if(m > 2) {
                    a[0] = Row(n);
                    a[m - 1] = a[m - 2];
                }
                auto rows = pack(a);
                auto value = random_row(n + 3);
                auto expected = product(a, value);
                for(size_t limit : {size_t(0), size_t(1), size_t(63), size_t(64), size_t(65), size_t(128), n, n + 91}) {
                    auto low = value;
                    for(size_t j = limit; j < low.size(); j++) low[j] = 0;
                    require(pack(value).low(limit) == pack(low), "low prefix exact around word boundaries");
                }
                require(parities(rows, pack(value)) == pack(expected), "batched parity versus unpacked dot products");
                require(parities(rows, Bits()).empty(), "zero parity vector remains canonical");
                auto mask = random_row(m);
                require(xors(rows, pack(mask)) == pack(combination(a, mask, n)),
                        "selected-row XOR versus scalar columns");
                auto extended = mask;
                extended.resize(m + 67);
                extended.back() = 1;
                for(size_t limit : {size_t(0), size_t(1), size_t(63), size_t(64), size_t(65), m, m + 91}) {
                    auto initial = random_row(n + 7);
                    auto expected_xor = initial;
                    for(size_t i = 0; i < std::min(m, limit); i++)
                        if(extended[i])
                            for(size_t j = 0; j < n; j++) expected_xor[j] ^= a[i][j];
                    auto output = pack(initial);
                    xor_rows_prefix(output, rows, pack(extended), limit);
                    require(output == pack(expected_xor), "selected-row prefix clipping and destination preservation");
                }
                auto transposed = transpose(rows, n);
                require(transposed.size() == n, "transpose rectangular shape");
                for(size_t i = 0; i < n; i++)
                    for(size_t j = 0; j < m; j++)
                        require(transposed[i].bit(j) == bool(a[j][i]), "transpose independent entries");
            }
    bool rejected = false;
    try {
        xors({Bits(1)}, Bits::unit(64));
    } catch(const std::invalid_argument&) { rejected = true; }
    require(rejected, "xors preserves invalid selection rejection");
}
int polynomial(const Row& x, const std::vector<int>& linear, const Matrix& upper) {
    int value = 0;
    for(size_t j = 0; j < x.size(); j++)
        if(x[j]) {
            value += linear[j];
            for(size_t k = j + 1; k < x.size(); k++)
                if(x[k] && upper[j][k]) value += 2;
        }
    return value & 3;
}
void quadratic_tests() {
    const std::vector<std::pair<size_t, size_t>> shapes = {{0, 0}, {0, 5}, {1, 0}, {1, 1}, {3, 5}, {5, 3}, {7, 7},
            {63, 6}, {64, 7}, {65, 8}, {65, 65}, {129, 67}, {137, 129}};
    for(const auto& [n, k] : shapes)
        for(unsigned density : {2u, 11u}) {
            quadratic_cases++;
            std::vector<int> linear(n);
            Matrix upper(n, Row(n)), basis(k, Row(n));
            for(size_t j = 0; j < n; j++) {
                linear[j] = int(random_bits() % 4);
                for(size_t l = j + 1; l < n; l++) upper[j][l] = random_bits() % density == 0;
            }
            for(auto& row : basis) row = random_row(n, density);
            if(k > 2) {
                basis[0] = Row(n);
                basis[k - 1] = basis[k - 2];
            }
            auto shift = random_row(n);
            auto q = restrict_q(linear, pack(upper), pack(shift), pack(basis));
            require(q.constant == polynomial(shift, linear, upper), "quadratic affine constant modulo four");
            auto no_constant = restrict_q(linear, pack(upper), pack(shift), pack(basis), false);
            require(no_constant.constant == 0 && no_constant.linear == q.linear && no_constant.upper == q.upper,
                    "phase omission changes only global constant");
            Matrix result_upper;
            for(size_t j = 0; j < k; j++) {
                result_upper.push_back(unpack(q.upper[j], k));
                for(size_t l = 0; l <= j; l++) require(!q.upper[j].bit(l), "restricted upper triangle canonical");
            }
            size_t count = k <= 8 ? size_t(1) << k : 64;
            for(size_t sample = 0; sample < count; sample++) {
                auto y = k <= 8 ? unpack(Bits(sample), k) : random_row(k);
                auto x = combination(basis, y, n);
                for(size_t j = 0; j < n; j++) x[j] ^= shift[j];
                int expected = polynomial(x, linear, upper),
                    actual = (q.constant + polynomial(y, q.linear, result_upper)) & 3;
                require(actual == expected, "exact affine quadratic restriction modulo four");
                for(int phase = 0; phase < 8; phase++)
                    require(((phase + 2 * actual) & 7) == ((phase + 2 * expected) & 7),
                            "scalar phase modulo eight preserved");
            }
        }
}
}
int main() {
    try {
        packed_tests();
        factor_tests();
        quadratic_tests();
        std::cout << "{\"status\":\"passed\",\"checks\":" << checks << ",\"factor_cases\":" << factor_cases
                  << ",\"quadratic_cases\":" << quadratic_cases << ",\"seed\":2026100409}" << std::endl;
        return 0;
    } catch(const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }
}
