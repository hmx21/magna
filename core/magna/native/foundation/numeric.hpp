// Range-safe conversion of exact real scalars, independent of Python and tasks.
#pragma once
#include "scalar.hpp"
#include <cfenv>
#include <cmath>
#include <complex>

namespace magic_native {
inline size_t integer_bits(const Integer& value) {
    return value.d.empty() ? 0 : 32 * (value.d.size() - 1) + 32 - __builtin_clz(value.d.back());
}

namespace numeric_detail {
inline uint64_t word(const Integer& value) {
    return value.zero() ? 0 : uint64_t(value.d[0]) | (value.d.size() == 2 ? uint64_t(value.d[1]) << 32 : 0);
}

// Integer likelihoods need a single quotient estimate and exact remainder,
// not the algebraic sign searches used for irrational amplitudes.
inline std::pair<double, int64_t> integer_ratio(const Integer& n, const Integer& d) {
    auto nb = integer_bits(n), db = integer_bits(d);
    int64_t exponent = int64_t(nb) - int64_t(db);
    int comparison =
            exponent >= 0 ? Integer::cmp(n, d.shift(size_t(exponent))) : Integer::cmp(n.shift(size_t(-exponent)), d);
    if(comparison >= 0) exponent++;
    auto top = [](const Integer& v) {
        size_t k = v.d.size() - 1;
        double x = v.d[k];
        int bits = 32 - __builtin_clz(v.d[k]);
        if(k) {
            x = std::ldexp(x, 32) + v.d[k - 1];
            bits += 32;
        }
        if(k > 1) {
            x = std::ldexp(x, 32) + v.d[k - 2];
            bits += 32;
        }
        return std::ldexp(x, -bits);
    };
    uint64_t q = uint64_t(std::ldexp(top(n) / top(d), int(int64_t(nb) - int64_t(db) + 53 - exponent)));
    int64_t shift = 53 - exponent;
    Integer a = shift >= 0 ? n.shift(size_t(shift)) : n;
    Integer b = shift < 0 ? d.shift(size_t(-shift)) : d;
    Integer remainder = a - b * Integer(int64_t(q));
    while(remainder.neg) {
        q--;
        remainder = remainder + b;
    }
    while(Integer::cmp(remainder, b) >= 0) {
        q++;
        remainder = remainder - b;
    }
    int half = Integer::cmp(remainder.shift(1), b);
    if(half > 0 || (half == 0 && (q & 1))) q++;
    if(q == (uint64_t(1) << 53)) {
        q >>= 1;
        exponent++;
    }
    return {std::ldexp(double(q), -53), exponent};
}

// Positive real ratio; called only after validation and exact sign handling.
inline std::pair<double, int64_t> exact_ratio(const Scalar& numerator, const Scalar& denominator) {
    // For (a+b sqrt(2)), the nonzero integral norm a^2-2b^2 bounds
    // cancellation. Locate the binary exponent without converting coefficients.
    size_t nb = std::max(integer_bits(numerator.c[0]), integer_bits(numerator.c[1]));
    size_t db = std::max(integer_bits(denominator.c[0]), integer_bits(denominator.c[1]));
    if(nb > size_t(std::numeric_limits<int64_t>::max() / 4) || db > size_t(std::numeric_limits<int64_t>::max() / 4))
        throw std::length_error("scalar ratio coefficient size budget");
    int64_t difference = numerator.power - denominator.power;
    int64_t bound = int64_t(nb) + int64_t(db) + 6;
    int64_t lower = difference - bound, upper = difference + bound;
    auto compare_power = [&](int64_t exponent) {
        auto scaled = denominator;
        scaled.power += exponent;
        return real_sign(numerator + scaled * Scalar(-1));
    };
    while(upper - lower > 1) {
        int64_t middle = lower + (upper - lower) / 2;
        if(compare_power(middle) >= 0) lower = middle;
        else upper = middle;
    }
    int64_t exponent = lower + 1;
    auto scaled = numerator;
    scaled.power += 53 - exponent;
    uint64_t lo = uint64_t(1) << 52, hi = uint64_t(1) << 53;
    while(hi - lo > 1) {
        uint64_t middle = lo + (hi - lo) / 2;
        if(real_sign(scaled + denominator * Scalar(-int64_t(middle))) >= 0) lo = middle;
        else hi = middle;
    }
    int half = real_sign(scaled * Scalar(2) + denominator * Scalar(-int64_t(2 * lo + 1)));
    if(half > 0 || (half == 0 && (lo & 1))) lo++;
    if(lo == (uint64_t(1) << 53)) {
        lo >>= 1;
        exponent++;
    }
    return {std::ldexp(double(lo), -53), exponent};
}
} // namespace numeric_detail

inline std::pair<double, int64_t> scaled_real_ratio(Scalar numerator, const Scalar& denominator) {
    int sign = real_sign(numerator);
    if(real_sign(denominator) <= 0) throw std::invalid_argument("scalar ratio denominator must be positive");
    if(!sign) return {0., 0};
    if(sign < 0) numerator = numerator * Scalar(-1);
    int64_t difference = numerator.power - denominator.power;
    // Identical coefficient vectors cancel exactly, including irrational ones.
    if(numerator.c == denominator.c) return {sign * .5, difference + 1};
    // Exactly representable integer operands, a normal-range quotient and IEEE
    // nearest-even division certify this route. Keep powers separate so even
    // extreme dyadic exponents never underflow or overflow during conversion.
    if(numerator.c[1].zero() && denominator.c[1].zero() && integer_bits(numerator.c[0]) <= 53 &&
            integer_bits(denominator.c[0]) <= 53 && std::numeric_limits<double>::is_iec559 &&
            std::numeric_limits<double>::digits == 53 && std::fegetround() == FE_TONEAREST) {
        double ratio = double(numeric_detail::word(numerator.c[0])) / double(numeric_detail::word(denominator.c[0]));
        int exponent;
        double mantissa = std::frexp(ratio, &exponent);
        return {sign * mantissa, difference + exponent};
    }
    auto result = numerator.c[1].zero() && denominator.c[1].zero()
            ? numeric_detail::integer_ratio(numerator.c[0], denominator.c[0])
            : numeric_detail::exact_ratio(numerator, denominator);
    if(numerator.c[1].zero() && denominator.c[1].zero()) result.second += difference;
    result.first *= sign;
    return result;
}
inline double scalar_log2(const Scalar& value) {
    auto sign = real_sign(value);
    if(sign < 0) throw std::invalid_argument("negative probability");
    if(!sign) return -std::numeric_limits<double>::infinity();
    auto ratio = scaled_real_ratio(value, Scalar(1));
    return std::log2(ratio.first) + ratio.second;
}
inline std::complex<double> scalar_complex(const Scalar& value) {
    std::complex<double> result;
    for(size_t j = 0; j < 4; j++) {
        double coefficient = 0;
        for(size_t k = value.c[j].d.size(); k-- > 0;) coefficient = std::ldexp(coefficient, 32) + value.c[j].d[k];
        if(!std::isfinite(coefficient)) throw std::overflow_error("complex coefficient exceeds floating range");
        if(value.c[j].neg) coefficient = -coefficient;
        result += coefficient * std::polar(1., double(j) * std::acos(-1.) / 4);
    }
    if(value.power > std::numeric_limits<int>::max()) throw std::overflow_error("complex scale exceeds floating range");
    double scale = value.power < std::numeric_limits<int>::min() ? 0. : std::ldexp(1., int(value.power));
    if(!std::isfinite(scale)) throw std::overflow_error("complex scale exceeds floating range");
    return result * scale;
}
inline std::array<Integer, 4> integral_coefficients(Scalar value) {
    if(value.power < 0) throw std::invalid_argument("nonintegral coefficient");
    for(auto& coefficient : value.c) coefficient = coefficient.shift(size_t(value.power));
    return value.c;
}
} // namespace magic_native
