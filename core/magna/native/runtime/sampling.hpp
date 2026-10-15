// Python-independent native implementation.
#pragma once
#include "../evaluate/trace_query.hpp"
#include "../foundation/numeric.hpp"
#include "marginal.hpp"
#include <random>
#include <cmath>

namespace magic_native {
inline Scalar file_integer_scalar(const Integer& value) {
    Scalar s;
    s.c[0] = value;
    s.normalize();
    return s;
}

inline bool file_exact_bernoulli(const Scalar& numerator, const Scalar& denominator, std::mt19937_64& rng) {
    const Scalar minus_one(-1);
    if(real_sign(denominator) <= 0 || real_sign(numerator) < 0 || real_sign(denominator + numerator * minus_one) < 0)
        throw std::invalid_argument("invalid exact Bernoulli probability");
    if(numerator.zero()) return false;
    if((denominator + numerator * minus_one).zero()) return true;
    // Reveal a uniform dyadic interval lazily. Comparisons stay in Z[omega,1/2].
    // No conversion to double, finite-resolution CDF or biased denominator cap.
    Integer prefix;
    for(size_t depth = 1;; depth++) {
        prefix = prefix.shift(1) + Integer(int64_t(rng() & 1));
        Scalar lower = file_integer_scalar(prefix);
        lower.power -= int64_t(depth);
        Scalar upper = file_integer_scalar(prefix + Integer(1));
        upper.power -= int64_t(depth);
        if(real_sign(numerator + denominator * upper * minus_one) >= 0) return true;
        if(real_sign(numerator + denominator * lower * minus_one) <= 0) return false;
    }
}

inline Bits sample_boundary(const std::vector<std::pair<Scalar, Ket>>& terms, size_t width, uint64_t seed,
        Scalar* selected_mass = nullptr) {
    Bits word;
    Workspace ws;
    std::mt19937_64 rng(seed);
    Scalar total = boundary_sum(terms, {}, {}, ws);
    if(real_sign(total) <= 0) throw std::invalid_argument("cannot sample zero or negative marginal");
    // Most retained decoder information is an affine mixture. Recognizing
    // individually nonnegative constant-phase terms avoids one contraction
    // per visible coordinate. Signed/complex components use the exact route.
    bool mixture = true;
    std::vector<Scalar> weights;
    for(auto& [c, k] : terms) {
        if(std::any_of(k.linear.begin(), k.linear.end(), [](int x) { return x != 0; }) ||
                std::any_of(k.upper.begin(), k.upper.end(), [](const Bits& x) { return !x.empty(); })) {
            mixture = false;
            break;
        }
        auto w = c * omega(k.phase, k.power);
        w.power += int64_t(k.columns.size());
        if(!w.c[2].zero() || !(w.c[1] == -w.c[3]) || real_sign(w) < 0) {
            mixture = false;
            break;
        }
        weights.push_back(w);
    }
    if(mixture) {
        auto remaining = total;
        bool selected = false;
        for(size_t j = 0; j < terms.size(); j++) {
            if(file_exact_bernoulli(weights[j], remaining, rng)) {
                const auto& k = terms[j].second;
                word = k.base;
                for(auto& column : k.columns)
                    if(rng() & 1) word ^= column;
                selected = true;
                break;
            }
            remaining = remaining + weights[j] * Scalar(-1);
        }
        if(!selected) throw std::runtime_error("exact mixture failed to select a component");
        if(selected_mass) *selected_mass = ProbabilityMarginal{terms, width}.value(word);
    } else {
        std::vector<Bits> rows;
        std::vector<int> outcomes;
        auto current = total;
        for(size_t j = 0; j < width; j++) {
            rows.push_back(Bits::unit(j));
            outcomes.push_back(0);
            auto zero = boundary_sum(terms, rows, outcomes, ws);
            int bit = !file_exact_bernoulli(zero, current, rng);
            current = bit ? current + zero * Scalar(-1) : zero;
            outcomes.back() = bit;
            if(bit) word.set(j);
        }
        if(selected_mass) *selected_mass = current;
    }
    return word;
}
struct FileOnlineMoment {
    size_t count = 0;
    long double mean = 0, m2 = 0;
    void add(long double value) {
        count++;
        auto delta = value - mean;
        mean += delta / count;
        m2 += delta * (value - mean);
    }
    long double standard_error() const { return std::sqrt(std::max(0.L, m2) / (count * (long double)(count - 1))); }
};

struct FileContribution {
    double acceptance;
    bool has_error;
    double error;
    std::vector<double> moments;
};
struct FileStatistics {
    FileOnlineMoment acceptance, error;
    std::vector<FileOnlineMoment> moments;
    bool has_error = false;
    long double cross = 0;
    void add(const FileContribution& row) {
        if(!acceptance.count) {
            has_error = row.has_error;
            moments.resize(row.moments.size());
        } else if(row.has_error != has_error) throw std::invalid_argument("inconsistent error metric presence");
        if(row.moments.size() != moments.size()) throw std::invalid_argument("inconsistent moment dimensions");
        if(!std::isfinite(row.acceptance) || !std::isfinite(row.error))
            throw std::invalid_argument("finite draw value required");
        for(auto value : row.moments)
            if(!std::isfinite(value)) throw std::invalid_argument("finite draw value required");
        long double old_a = acceptance.mean;
        acceptance.add(row.acceptance);
        if(has_error) {
            error.add(row.error);
            cross += (row.acceptance - old_a) * (row.error - error.mean);
        }
        for(size_t k = 0; k < moments.size(); k++) moments[k].add(row.moments[k]);
    }
    bool has_ratio() const { return has_error && acceptance.mean != 0; }
    long double ratio() const { return has_ratio() ? error.mean / acceptance.mean : 0; }
    long double ratio_error() const {
        if(!has_ratio() || acceptance.count < 2) return 0;
        auto r = ratio();
        auto residual = error.m2 - 2 * r * cross + r * r * acceptance.m2;
        return std::sqrt(std::max(0.L, residual) / (acceptance.count * (long double)(acceptance.count - 1))) /
                std::abs(acceptance.mean);
    }
};
struct WeightStatistics {
    double ess = 0, mantissa = 0;
    int64_t exponent = 0;
    size_t completed = 0;
};
// Streaming importance-weight statistics, accumulated at the largest positive
// weight exponent seen so far and rescaled exactly when it grows.
struct WeightAccumulator {
    std::optional<int64_t> top;
    long double sum = 0, squares = 0;
    size_t count = 0;
    static long double rescale(long double value, int64_t shift) {
        return shift < -32000 ? 0 : std::ldexp(value, int(shift));
    }
    static void validate(const std::pair<double, int64_t>& v) {
        double m = v.first;
        auto e = v.second;
        if(!std::isfinite(m) || m < 0 || m >= 1 || (m > 0 && m < .5) || e < -1000000000 || e > 1000000000)
            throw std::invalid_argument("normalized nonnegative scientific weight required");
    }
    void add(const std::pair<double, int64_t>& v) {
        validate(v);
        double m = v.first;
        auto e = v.second;
        if(m > 0 && (!top || e > *top)) {
            if(top) {
                sum = rescale(sum, *top - e);
                squares = rescale(squares, 2 * (*top - e));
            }
            top = e;
        }
        auto delta = e - top.value_or(0);
        long double x = delta < -32000 ? 0 : std::ldexp((long double)m, int(delta));
        sum += x;
        squares += x * x;
        count++;
    }
    WeightStatistics summary() const {
        int exponent = 0;
        double mantissa = count ? double(std::frexp(sum / count, &exponent)) : 0;
        if(mantissa == 1) {
            mantissa = .5;
            exponent++;
        }
        return {squares ? double(sum * sum / squares) : 0., mantissa, mantissa ? top.value_or(0) + exponent : 0, count};
    }
};
inline WeightStatistics summarize_weights(const std::vector<std::pair<double, int64_t>>& values) {
    WeightAccumulator accumulator;
    for(const auto& v : values) accumulator.add(v);
    return accumulator.summary();
}

} // namespace magic_native
