// Fixed-resolution noise probabilities prepared from the circuit's rational input.
#pragma once
#include "scalar.hpp"

namespace magic_native {
struct RationalCoin {
    uint64_t threshold = 0;
    bool certain = false;
    RationalCoin(Integer numerator, Integer denominator) {
        if(numerator.neg || denominator.neg || denominator.zero() || Integer::cmp(numerator, denominator) > 0)
            throw std::invalid_argument("invalid rational Bernoulli probability");
        certain = numerator == denominator;
        if(numerator.zero() || certain) return;
        // Keep floor(p*2^64). The discarded remainder is below one random
        // word's resolution; there is no arbitrary-precision sampling tail.
        if(denominator.d.size() <= 2) {
            auto word = [](const Integer& v) {
                return v.zero() ? uint64_t(0) : uint64_t(v.d[0]) | (v.d.size() == 2 ? uint64_t(v.d[1]) << 32 : 0);
            };
            auto small_d = word(denominator), small_r = word(numerator);
            for(size_t bit = 0; bit < 64; bit++) {
                threshold <<= 1;
                auto complement = small_d - small_r;
                if(small_r >= complement) {
                    threshold |= 1;
                    small_r -= complement;
                } else small_r += small_r;
            }
        } else {
            auto remainder = std::move(numerator);
            for(size_t bit = 0; bit < 64; bit++) {
                threshold <<= 1;
                remainder = remainder.shift(1);
                if(Integer::cmp(remainder, denominator) >= 0) {
                    threshold |= 1;
                    remainder = remainder - denominator;
                }
            }
        }
    }
    template <class Rng> bool draw(Rng& rng) const { return certain || (threshold && rng() < threshold); }
    // Compare up to 64 independent random words with the same threshold,
    // transposed into random bit planes. Resolved lanes need no further bits.
    // This is exactly the scalar fixed-resolution law, including its endpoints.
    template <class Rng> uint64_t draw_bits(Rng& rng, uint64_t live) const {
        if(certain) return live;
        uint64_t result = 0, pending = live, remaining = threshold;
        while(pending && remaining) {
            auto bits = rng();
            if(remaining & (uint64_t(1) << 63)) {
                result |= pending & ~bits;
                pending &= bits;
            } else pending &= ~bits;
            remaining <<= 1;
        }
        return result;
    }
};

template <class Rng> size_t uniform_index(size_t count, Rng& rng) {
    if(!count) throw std::invalid_argument("empty categorical distribution");
    if(count == 1) return 0;
    uint64_t n = count, reject = (uint64_t(0) - n) % n, value;
    do { value = rng(); } while(value < reject);
    return size_t(value % n);
}
} // namespace magic_native
