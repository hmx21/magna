// Configured resource budgets. An exceeded budget names its setting, the
// configured value and, when known, a lower bound on the amount required.
#pragma once
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <optional>
#include <stdexcept>
#include <string>

namespace magic_native {
inline std::string limit_number(double value) {
    char text[32];
    if(value == std::floor(value) && std::fabs(value) < 9007199254740992.)
        std::snprintf(text, sizeof text, "%.0f", value);
    else std::snprintf(text, sizeof text, "%g", value);
    return text;
}
struct ResourceLimitError : std::runtime_error {
    std::string option;
    double limit;
    std::optional<double> needed;
    ResourceLimitError(
            const std::string& context, const std::string& option, double limit, std::optional<double> needed = {})
        : std::runtime_error(context + " (" + option + " = " + limit_number(limit) +
                  (needed ? ", needs at least " + limit_number(*needed) : "") + ")"),
          option(option), limit(limit), needed(needed) {}
    // Adds context while keeping the setting, its value and the requirement.
    ResourceLimitError(const ResourceLimitError& cause, const std::string& context)
        : std::runtime_error(cause.what() + context), option(cause.option), limit(cause.limit), needed(cause.needed) {}
};
// Decomposition, table and path budgets. Internal callers may catch these
// to keep an existing construction or try another one within the same budget.
struct PhaseBudgetError : ResourceLimitError {
    using ResourceLimitError::ResourceLimitError;
};
// The doubled-network path budget and the setting it comes from.
struct PathBudget {
    std::size_t bits;
    const char* option;
    PathBudget(std::size_t bits, const char* option = "max_doubled_path_bits") : bits(bits), option(option) {}
};
} // namespace magic_native
