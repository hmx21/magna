// Finite nearest-even decimal arithmetic for the existing precision-80 estimator.
#pragma once
#include "rational.hpp"
namespace magic_native {
inline Integer integer_power10(size_t exponent) {
    Integer out(1), base(10);
    while(exponent) {
        if(exponent & 1) out = out * base;
        exponent >>= 1;
        if(exponent) base = base * base;
    }
    return out;
}
inline size_t decimal_digits(const Integer& value) {
    if(value.zero()) return 1;
    size_t digits = size_t((integer_bits(value) - 1) * .3010299956639812) + 1;
    auto bound = integer_power10(digits);
    if(Integer::cmp(value, bound) >= 0) return digits + 1;
    if(digits > 1 && Integer::cmp(value, integer_power10(digits - 1)) < 0) return digits - 1;
    return digits;
}
inline std::string integer_decimal_text(Integer value) {
    if(value.zero()) return "0";
    bool negative = value.neg;
    value.neg = false;
    std::vector<uint32_t> groups;
    while(!value.zero()) {
        auto divided = integer_divmod(value, Integer(1000000000));
        groups.push_back(divided.second.zero() ? 0 : divided.second.d[0]);
        value = std::move(divided.first);
    }
    std::string text = negative ? "-" : "";
    text += std::to_string(groups.back());
    for(size_t j = groups.size() - 1; j-- > 0;) {
        auto part = std::to_string(groups[j]);
        text += std::string(9 - part.size(), '0') + part;
    }
    return text;
}
inline int64_t rational_decimal_order(const Integer& numerator, const Integer& denominator) {
    int64_t exponent = int64_t(decimal_digits(numerator)) - int64_t(decimal_digits(denominator));
    int comparison = exponent >= 0 ? Integer::cmp(numerator, denominator * integer_power10(size_t(exponent)))
                                   : Integer::cmp(numerator * integer_power10(size_t(-exponent)), denominator);
    return comparison < 0 ? exponent - 1 : exponent;
}
inline Integer integer_sqrt(const Integer& value) {
    if(value.zero()) return {};
    auto x = Integer(1).shift((integer_bits(value) + 1) / 2);
    while(true) {
        auto y = x + integer_divmod(value, x).first;
        y.half();
        if(Integer::cmp(y, x) >= 0) return x;
        x = std::move(y);
    }
}
struct DecimalValue {
    Integer coefficient;
    int64_t exponent = 0;
    DecimalValue(Integer coefficient = Integer(), int64_t exponent = 0)
        : coefficient(std::move(coefficient)), exponent(exponent) {}
    bool zero() const { return coefficient.zero(); }
    Rational rational() const {
        if(exponent >= 0) return {coefficient * integer_power10(size_t(exponent))};
        return {coefficient, integer_power10(size_t(-exponent))};
    }
    std::string text() const { return integer_decimal_text(coefficient) + "E" + std::to_string(exponent); }
    static DecimalValue rounded(const Rational& value, size_t precision = 80) {
        if(value.n.zero()) return {};
        auto n = value.n;
        n.neg = false;
        int64_t exponent = rational_decimal_order(n, value.d) - int64_t(precision) + 1;
        auto d = value.d;
        if(exponent >= 0) d = d * integer_power10(size_t(exponent));
        else n = n * integer_power10(size_t(-exponent));
        auto [q, r] = integer_divmod(n, d);
        int compare = Integer::cmp(r.shift(1), d);
        if(compare > 0 || (compare == 0 && !q.even())) q = q + Integer(1);
        q.neg = value.n.neg && !q.zero();
        return {q, exponent};
    }
    static DecimalValue add(const DecimalValue& a, const DecimalValue& b, size_t precision = 80) {
        return rounded(a.rational() + b.rational(), precision);
    }
    static DecimalValue sub(const DecimalValue& a, const DecimalValue& b, size_t precision = 80) {
        return rounded(a.rational() - b.rational(), precision);
    }
    static DecimalValue mul(const DecimalValue& a, const DecimalValue& b, size_t precision = 80) {
        return rounded(a.rational() * b.rational(), precision);
    }
    static DecimalValue div(const DecimalValue& a, const DecimalValue& b, size_t precision = 80) {
        return rounded(a.rational() / b.rational(), precision);
    }
    static DecimalValue sqrt(const DecimalValue& value, size_t precision = 80) {
        if(value.zero()) return {};
        if(value.coefficient.neg) throw std::invalid_argument("negative square root");
        auto r = value.rational();
        auto order = rational_decimal_order(r.n, r.d);
        int64_t half = order >= 0 ? order / 2 : -((-order + 1) / 2);
        int64_t exponent = half - int64_t(precision) + 1;
        if(exponent >= 0) r.d = r.d * integer_power10(size_t(2 * exponent));
        else r.n = r.n * integer_power10(size_t(-2 * exponent));
        auto root = integer_sqrt(integer_divmod(r.n, r.d).first);
        auto midpoint = root * Integer(2) + Integer(1);
        int compare = Integer::cmp(r.n.shift(2), r.d * midpoint * midpoint);
        if(compare > 0 || (compare == 0 && !root.even())) root = root + Integer(1);
        return {root, exponent};
    }
};
inline DecimalValue scalar_decimal(const Scalar& value) {
    if(!value.c[2].zero() || !(value.c[1] == -value.c[3])) throw std::runtime_error("expected an exactly real scalar");
    size_t width = 0;
    for(const auto& c : value.c) width = std::max(width, integer_bits(c));
    size_t precision = std::max(size_t(80), size_t(width * .302) + 40);
    auto root = DecimalValue::sqrt(DecimalValue(Integer(2)), precision);
    auto real = DecimalValue::add(
            DecimalValue(value.c[0]), DecimalValue::mul(DecimalValue(value.c[1]), root, precision), precision);
    Rational scale = value.power >= 0 ? Rational(Integer(1).shift(size_t(value.power)))
                                      : Rational(Integer(1), Integer(1).shift(size_t(-value.power)));
    return DecimalValue::mul(real, DecimalValue::rounded(scale, precision), precision);
}
inline DecimalValue sample_weight_decimal(const Scalar& probability, const Rational& ratio) {
    return DecimalValue::div(
            DecimalValue::mul(scalar_decimal(probability), DecimalValue(ratio.n)), DecimalValue(ratio.d));
}
} // namespace magic_native
