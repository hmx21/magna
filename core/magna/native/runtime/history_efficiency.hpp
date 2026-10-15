// Constant-storage diagnostics for completed importance draws.
#pragma once
#include "file_statistics.hpp"
#include <cfloat>
namespace magic_native {
inline ScientificValue efficiency_scaled(long double x, int64_t exponent = 0) {
    if(!x) return {0, 0};
    int shift = 0;
    auto mantissa = std::frexp(x, &shift);
    double m = double(mantissa);
    if(m == 1) {
        m = .5;
        shift++;
    }
    return {m, exponent + shift};
}
inline ScientificValue efficiency_product(ScientificValue a, ScientificValue b) {
    return efficiency_scaled((long double)a.first * b.first, a.second + b.second);
}
inline ScientificValue efficiency_quotient(ScientificValue a, ScientificValue b) {
    return efficiency_scaled((long double)a.first / b.first, a.second - b.second);
}
inline long double efficiency_shift(long double x, int64_t exponent) {
    return exponent < LDBL_MIN_EXP - LDBL_MANT_DIG ? 0 : std::ldexp(x, int(exponent));
}
struct EfficiencyMoments {
    long double mass = 0, mean = 0, m2 = 0;
    int64_t exponent = 0;
    void add(ScientificValue weight, long double value) {
        if(!weight.first) return;
        if(!mass) exponent = weight.second;
        if(weight.second > exponent) {
            auto factor = efficiency_shift(1, exponent - weight.second);
            mass *= factor;
            m2 *= factor;
            exponent = weight.second;
        }
        auto w = efficiency_shift(weight.first, weight.second - exponent);
        if(!w) return;
        auto total = mass + w, delta = value - mean;
        m2 += delta * delta * (mass / total) * w;
        mean += delta * (w / total);
        mass = total;
    }
    ScientificValue sum() const { return efficiency_scaled(mass, exponent); }
    ScientificValue residual(long double center) const {
        auto delta = mean - center;
        return efficiency_scaled(m2 + mass * delta * delta, exponent);
    }
};
struct HistoryEfficiency {
    size_t count = 0, zeros = 0;
    bool missing_error = false, range_lost = false;
    EfficiencyMoments acceptance, proposal, born, weights, weight_squares;
    void add(ScientificValue a, std::optional<ScientificValue> error, ScientificValue w, ScientificValue noise) {
        count++;
        zeros += w.first == 0;
        weights.add(w, 0);
        weight_squares.add(efficiency_product(w, w), 0);
        if(!error) {
            missing_error = true;
            return;
        }
        if(!a.first) return;
        if(!w.first || !noise.first) throw std::invalid_argument("positive contribution requires positive likelihoods");
        auto ratio = efficiency_quotient(*error, a);
        if(ratio.first && (ratio.second < LDBL_MIN_EXP / 2 || ratio.second > 1)) {
            range_lost = true;
            return;
        }
        long double value = ratio.first ? std::ldexp((long double)ratio.first, int(ratio.second)) : 0;
        acceptance.add(a, value);
        auto square = efficiency_product(a, a);
        proposal.add(square, value);
        born.add(efficiency_product(square, efficiency_quotient(noise, w)), value);
    }
    const char* status() const {
        if(missing_error) return "no_error_metric";
        if(range_lost) return "diagnostic_range_exceeded";
        if(count < 2) return "insufficient_samples";
        if(!acceptance.mass) return "zero_acceptance";
        if(!born.residual(acceptance.mean).first) return "zero_residual_variance";
        return "available";
    }
    std::optional<ScientificValue> kappa() const {
        if(std::string(status()) != "available") return {};
        return efficiency_quotient(proposal.residual(acceptance.mean), born.residual(acceptance.mean));
    }
    std::optional<ScientificValue> variance(bool reference) const {
        if(std::string(status()) != "available") return {};
        auto residual = (reference ? born : proposal).residual(acceptance.mean);
        return efficiency_product(efficiency_quotient(residual, efficiency_product(acceptance.sum(), acceptance.sum())),
                efficiency_scaled(count));
    }
    // Everything a progress report or final summary shows, in one place.
    struct Summary {
        const char* status;
        std::optional<ScientificValue> kappa, variance, reference_variance, ess, rho, mean_weight;
    };
    Summary summary() const {
        Summary out{status(), kappa(), variance(false), variance(true), {}, {}, {}};
        // A ratio that exists but leaves the ordinary floating-point range cannot guide a choice.
        if(out.kappa && !ranged_value(out.kappa).ordinary) out.status = "diagnostic_range_exceeded";
        if(count) out.mean_weight = efficiency_quotient(weights.sum(), efficiency_scaled(count));
        if(weights.mass) {
            out.ess = efficiency_quotient(efficiency_product(weights.sum(), weights.sum()), weight_squares.sum());
            out.rho = efficiency_quotient(efficiency_scaled(count), *out.ess);
        } else out.ess = ScientificValue{0, 0};
        return out;
    }
};
struct SamplerDecision {
    double affine_support_seconds_per_draw, exact_marginal_seconds_per_draw, gamma;
    const char* recommended;
};
inline SamplerDecision sampler_decision(double kappa, double capped_seconds, double born_seconds, size_t count) {
    if(!count || !std::isfinite(kappa) || kappa < 0 || !std::isfinite(capped_seconds) || capped_seconds <= 0 ||
            !std::isfinite(born_seconds) || born_seconds <= 0)
        throw std::invalid_argument("finite positive pilot times and count required");
    auto gamma = kappa * (capped_seconds / born_seconds);
    if(!std::isfinite(gamma)) throw std::invalid_argument("pilot Gamma exceeds floating-point range");
    return {capped_seconds / count, born_seconds / count, gamma, gamma > 1 ? "exact_marginal" : "affine_support"};
}
} // namespace magic_native
