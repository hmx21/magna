// Check finite noise thresholds and shared scalar conversion independently of Python.
#include "foundation/rational_random.hpp"
#include "foundation/numeric.hpp"
#include <cassert>
#include <initializer_list>
#include <random>

using namespace magic_native;
struct ScriptedRng {
    std::vector<uint64_t> values;
    size_t position = 0;
    ScriptedRng(std::initializer_list<uint64_t> v) : values(v) {}
    uint64_t operator()() {
        assert(position < values.size());
        return values[position++];
    }
};
int main() {
    RationalCoin third(Integer(1), Integer(3));
    assert(third.threshold == 0x5555555555555555ULL);
    ScriptedRng below{third.threshold - 1}, above{third.threshold + 1};
    assert(third.draw(below));
    assert(!third.draw(above));
    ScriptedRng boundary{third.threshold};
    assert(!third.draw(boundary));
    assert(boundary.position == 1);
    RationalCoin half(Integer(1), Integer(2));
    ScriptedRng at_half{uint64_t(1) << 63};
    assert(!half.draw(at_half));
    ScriptedRng no_bits{};
    assert(!RationalCoin(Integer(0), Integer(1)).draw(no_bits));
    assert(RationalCoin(Integer(1), Integer(1)).draw(no_bits));
    auto resolution = Integer(1).shift(64);
    RationalCoin one_bucket(Integer(1), resolution), sub_bucket(Integer(1), resolution.shift(1));
    assert(one_bucket.threshold == 1);
    assert(sub_bucket.threshold == 0);
    ScriptedRng first_bucket{0}, second_bucket{1};
    assert(one_bucket.draw(first_bucket));
    assert(!one_bucket.draw(second_bucket));
    assert(!sub_bucket.draw(no_bits));
    RationalCoin near_one(resolution - Integer(1), resolution);
    ScriptedRng last_bucket{~uint64_t(0)};
    assert(!near_one.draw(last_bucket));
    assert(last_bucket.position == 1);
    Integer maximum;
    maximum.d = {0xffffffff, 0xffffffff};
    RationalCoin wide(maximum - Integer(1), maximum);
    assert(wide.threshold == ~uint64_t(0) - 1);
    ScriptedRng wide_yes{wide.threshold - 1}, wide_no{wide.threshold};
    assert(wide.draw(wide_yes));
    assert(!wide.draw(wide_no));
    // Ordinary fault-tolerance rates: threshold = (2^64 - (2^64 mod d))/d.
    for(uint64_t d : {1000ULL, 10000ULL}) {
        RationalCoin physical(Integer(1), Integer(int64_t(d)));
        assert(physical.threshold == UINT64_MAX / d);
        ScriptedRng just_below{physical.threshold - 1}, at_cutoff{physical.threshold};
        assert(physical.draw(just_below));
        assert(!physical.draw(at_cutoff));
    }
    ScriptedRng uniform{0, 1};
    assert(uniform_index(15, uniform) == 1);
    assert(uniform.position == 2); // Reject the incomplete modulo bucket.
    assert(uniform_index(1, no_bits) == 0);

    // Hardware-division route agrees bit for bit with the general exact route.
    std::mt19937_64 rng(20260927);
    // Independent oracle: supply 64 ordinary words as transposed bit planes,
    // then compare each original word with the threshold using scalar '<'.
    for(const auto& coin : {RationalCoin(Integer(0), Integer(1)), half, third, one_bucket, sub_bucket, near_one,
                RationalCoin(Integer(1), Integer(1))}) {
        for(size_t trial = 0; trial < 80; trial++) {
            std::array<uint64_t, 64> words;
            for(auto& word : words) word = rng();
            words[0] = coin.threshold;
            words[1] = coin.threshold - 1;
            words[2] = 0;
            words[3] = UINT64_MAX;
            ScriptedRng planes{};
            planes.values.assign(64, 0);
            for(size_t lane = 0; lane < 64; lane++)
                for(size_t bit = 0; bit < 64; bit++) planes.values[bit] |= ((words[lane] >> (63 - bit)) & 1) << lane;
            uint64_t live = trial == 0 ? UINT64_MAX : rng(), expected = 0;
            for(size_t lane = 0; lane < 64; lane++)
                if(coin.certain || words[lane] < coin.threshold) expected |= uint64_t(1) << lane;
            assert(coin.draw_bits(planes, live) == (expected & live));
        }
    }
    for(size_t i = 0; i < 512; i++) {
        Scalar n(int64_t((rng() >> 11) | 1), int64_t(i) - 1000);
        Scalar d(int64_t((rng() >> 11) | 1), 1000 - int64_t(i));
        assert(scaled_real_ratio(n, d) == numeric_detail::exact_ratio(n, d));
    }
    // A non-nearest process rounding mode must not enter hardware division.
    int saved = std::fegetround();
    assert(std::fesetround(FE_UPWARD) == 0);
    assert(scaled_real_ratio(Scalar(1), Scalar(3)) == numeric_detail::exact_ratio(Scalar(1), Scalar(3)));
    assert(std::fesetround(saved) == 0);
}
