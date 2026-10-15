// Python-independent native implementation.
#pragma once
#include "../foundation/stabilizer.hpp"

namespace magic_native {
// Exact record marginal of a completed stabilizer pair. No physical-time state.
struct RecordTracePlan {
    size_t width = 0, na = 0, nb = 0;
    std::vector<Bits> ac, bc, rows, events, visible;
    std::vector<int> outcomes;
    Factor factor;
    RecordTracePlan(const Ket& a, const Ket& b, std::vector<Bits> r, std::vector<Bits> e, std::vector<int> o)
        : width(a.width), na(a.columns.size()), nb(b.columns.size()), ac(a.columns), bc(b.columns), rows(std::move(r)),
          events(std::move(e)), outcomes(std::move(o)) {
        if(a.width != b.width || events.size() != outcomes.size())
            throw std::invalid_argument("record trace dimensions");
        validate_masks(rows, width);
        validate_masks(events, width);
        for(int bit : outcomes)
            if(bit != 0 && bit != 1) throw std::invalid_argument("nonbinary event");
        auto cols = ac;
        cols.insert(cols.end(), bc.begin(), bc.end());
        auto equations = transpose(cols, width);
        auto aforms = transpose(ac, width);
        for(auto& event : events) equations.push_back(xors(aforms, event));
        factor = Factor(equations, na + nb);
        visible.resize(na + nb);
        for(size_t j = 0; j < na; j++)
            for(size_t k = 0; k < rows.size(); k++)
                if((rows[k] & ac[j]).parity()) visible[j].set(k);
    }
    Ket evaluate(const Ket& a, const Ket& b, Workspace& ws) const {
        if(a.width != width || b.width != width || a.columns != ac || b.columns != bc)
            throw std::invalid_argument("record trace structure changed");
        Bits rhs = a.base ^ b.base;
        for(size_t j = 0; j < events.size(); j++)
            if(outcomes[j] ^ (events[j] & a.base).parity()) rhs.set(width + j);
        Ket out;
        out.width = rows.size();
        if(!factor.consistent(rhs)) {
            out.zero = true;
            return out;
        }
        auto shift = factor.solve(rhs);
        auto linear = a.linear;
        for(auto& v : linear) v = mod4(-v);
        linear.insert(linear.end(), b.linear.begin(), b.linear.end());
        auto upper = a.upper;
        for(auto& v : b.upper) upper.push_back(v << na);
        auto q = restrict_q(linear, upper, shift, factor.kernel);
        for(size_t j = 0; j < rows.size(); j++)
            if((rows[j] & a.base).parity()) out.base.set(j);
        out.base ^= xors(visible, shift);
        for(auto& v : factor.kernel) out.columns.push_back(xors(visible, v));
        out.linear = std::move(q.linear);
        out.upper = std::move(q.upper);
        out.power = a.power + b.power;
        out.phase = b.phase - a.phase + 2 * q.constant;
        return contract(std::move(out), ws);
    }
};

} // namespace magic_native
