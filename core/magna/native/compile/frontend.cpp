#include "frontend.hpp"
#include <sstream>
#include <cctype>
namespace magic_native {
namespace {
const std::map<std::string, std::string> aliases{{"CNOT", "CX"}, {"ZCX", "CX"}, {"ZCY", "CY"}, {"ZCZ", "CZ"},
        {"MZ", "M"}, {"MRZ", "MR"}, {"RZ", "R"}, {"E", "CORRELATED_ERROR"}, {"H_XZ", "H"}};
const std::set<std::string> singles{"I", "X", "Y", "Z", "H", "S", "S_DAG", "T", "T_DAG", "SQRT_X", "SQRT_X_DAG",
        "SQRT_Y", "SQRT_Y_DAG", "H_XY", "H_YZ", "C_XYZ", "C_ZYX"};
const std::set<std::string> doubles{"CX", "CY", "CZ", "SWAP", "ISWAP", "ISWAP_DAG", "CS", "CS_DAG", "CH"};
const std::set<std::string> triples{"CCZ", "CCX"};
const std::set<std::string> measurements{"M", "MX", "MY", "MR", "MRX", "MRY", "MXX", "MYY", "MZZ", "MPP"};
const std::set<std::string> resets{"R", "RX", "RY"};
const std::map<std::string, size_t> noise_args{{"X_ERROR", 1}, {"Y_ERROR", 1}, {"Z_ERROR", 1}, {"DEPOLARIZE1", 1},
        {"DEPOLARIZE2", 1}, {"PAULI_CHANNEL_1", 3}, {"PAULI_CHANNEL_2", 15}, {"CORRELATED_ERROR", 1},
        {"ELSE_CORRELATED_ERROR", 1}};
const std::set<std::string> annotations{"DETECTOR", "OBSERVABLE_INCLUDE", "QUBIT_COORDS", "SHIFT_COORDS", "TICK"};
const std::set<std::string> controlled{"CX", "CY", "CZ", "CS", "CS_DAG", "CH", "CCZ", "CCX"};
bool builtin(const std::string& name) { return name == "XOR" || name == "NOT"; }
bool classical(const FileTarget& t) { return t.kind == "rec" || t.kind == "ctrl"; }
std::vector<std::string> words(const std::string& s) {
    std::istringstream stream(s);
    std::string word;
    std::vector<std::string> out;
    while(stream >> word) out.push_back(word);
    return out;
}
std::vector<std::string> split(const std::string& text, char separator) {
    std::vector<std::string> out;
    size_t first = 0;
    for(size_t j = 0; j <= text.size(); j++)
        if(j == text.size() || text[j] == separator) {
            out.push_back(text.substr(first, j - first));
            first = j + 1;
        }
    return out;
}
struct Block {
    enum Kind { instruction, pending, repeat } kind;
    std::string text;
    size_t count = 0;
    SourceLocation source;
    std::vector<Block> children;
};
std::vector<Block> blocks(const std::string& text, const std::string& source, size_t limit) {
    std::vector<Block> root;
    std::vector<std::vector<Block>*> stack{&root};
    std::vector<SourceLocation> headers;
    std::istringstream lines(text);
    std::string original;
    size_t line = 0;
    static const std::regex repeat_start(R"(^REPEAT\b)", std::regex::icase),
            repeat_header(R"(^REPEAT\s+([0-9]+)$)", std::regex::icase);
    while(std::getline(lines, original)) {
        line++;
        auto body = original.substr(0, original.find('#'));
        for(size_t begin = 0; begin < body.size();) {
            size_t end = begin + 1;
            if(body[begin] != '{' && body[begin] != '}')
                while(end < body.size() && body[end] != '{' && body[end] != '}') end++;
            auto raw = body.substr(begin, end - begin), token = trim(raw);
            size_t column = begin + raw.find_first_not_of(" \t\r\n\f\v") + 1;
            begin = end;
            if(token.empty()) continue;
            SourceLocation loc{source, line, column, {}};
            auto& current = *stack.back();
            if(token == "{") {
                if(current.empty() || current.back().kind != Block::pending)
                    throw CircuitSyntaxError("unexpected '{'", loc);
                current.back().kind = Block::repeat;
                headers.push_back(current.back().source);
                stack.push_back(&current.back().children);
            } else if(token == "}") {
                if(stack.size() == 1) throw CircuitSyntaxError("unmatched '}'", loc);
                if(!current.empty() && current.back().kind == Block::pending)
                    throw CircuitSyntaxError("REPEAT requires '{'", loc);
                stack.pop_back();
                headers.pop_back();
            } else {
                if(!current.empty() && current.back().kind == Block::pending)
                    throw CircuitSyntaxError("REPEAT requires '{'", loc);
                if(std::regex_search(token, repeat_start)) {
                    std::smatch m;
                    if(!std::regex_match(token, m, repeat_header) || integer_decimal(m[1]).zero())
                        throw CircuitSyntaxError("REPEAT needs a positive integer count", loc);
                    // Counts above the expansion cap need no machine-width conversion.
                    auto number = integer_decimal(m[1]);
                    size_t count = limit + 1;
                    if(integer_bits(number) < std::numeric_limits<size_t>::digits)
                        count = size_t(numeric_detail::word(number));
                    current.push_back({Block::pending, {}, count, loc, {}});
                } else current.push_back({Block::instruction, token, 0, loc, {}});
            }
        }
    }
    if(stack.size() != 1) throw CircuitSyntaxError("unclosed REPEAT block", headers.back());
    if(!root.empty() && root.back().kind == Block::pending)
        throw CircuitSyntaxError("REPEAT requires '{'", root.back().source);
    return root;
}
ResourceLimitError instruction_budget(const SourceLocation& where, size_t limit, double needed) {
    return ResourceLimitError(
            where.filename + ":" + std::to_string(where.line) + ": flattened circuit exceeds the instruction budget",
            "max_instructions", double(limit), needed);
}
size_t expanded_size(const std::vector<Block>& block, size_t limit) {
    size_t total = 0;
    for(const auto& item : block) {
        size_t count = 1;
        if(item.kind == Block::repeat) {
            auto size = expanded_size(item.children, limit);
            if(size && item.count > limit / size)
                throw instruction_budget(item.source, limit, double(total) + double(size) * double(item.count));
            count = size * item.count;
        }
        if(count > limit - total) throw instruction_budget(item.source, limit, double(total) + double(count));
        total += count;
    }
    return total;
}
struct Parser {
    CircuitData data;
    const std::set<std::string>& controllers;
    size_t limit, max_qubits;
    std::map<size_t, int64_t> versions;
    FileTarget target(const std::string& token, const SourceLocation& loc, bool output = false) {
        static const std::regex cls(R"(^(!?)(rec|ctrl|det)\[(-?[0-9]+)\]$)"), quantum(R"(^(!?)([XYZ]?)([0-9]+)$)");
        std::smatch m;
        if(std::regex_match(token, m, cls)) {
            bool inverted = m[1].length() != 0;
            std::string kind = m[2];
            int64_t value;
            try {
                value = std::stoll(m[3]);
            } catch(const std::exception&) { throw CircuitSyntaxError("classical index outside native range", loc); }
            if(output) {
                if(kind != "ctrl" || value < 0 || inverted)
                    throw CircuitSyntaxError("CONTROL outputs must be nonnegative ctrl registers", loc);
                return {"ctrl", size_t(value)};
            }
            if(kind == "det") {
                int64_t index = value < 0 ? int64_t(data.detectors.size()) + value : value;
                if(index < 0 || uint64_t(index) >= data.detectors.size())
                    throw CircuitSyntaxError("detector reference " + token + " does not name an earlier detector", loc);
                return {kind, size_t(index), -1, inverted};
            }
            if(kind == "rec") {
                if(value >= 0 || int64_t(data.num_records) + value < 0)
                    throw CircuitSyntaxError("record lookback " + token + " does not name an earlier measurement", loc);
                return {kind, size_t(int64_t(data.num_records) + value), -1, inverted};
            }
            if(value < 0 || !versions.count(size_t(value)))
                throw CircuitSyntaxError("uninitialized control register " + token, loc);
            return {kind, size_t(value), versions.at(size_t(value)), inverted};
        }
        if(output) throw CircuitSyntaxError("CONTROL outputs must be ctrl registers", loc);
        if(std::regex_match(token, m, quantum)) {
            size_t q;
            try {
                q = std::stoull(m[3]);
            } catch(const std::exception&) { throw CircuitSyntaxError("qubit index outside native range", loc); }
            if(q >= max_qubits)
                throw ResourceLimitError(loc.filename + ":" + std::to_string(loc.line) + ": qubit index " +
                                std::to_string(q) + " exceeds the qubit budget",
                        "max_qubits", double(max_qubits), double(q) + 1);
            data.num_qubits = std::max(data.num_qubits, q + 1);
            return {m[2].length() ? "pauli" : "qubit", q, -1, m[1].length() != 0, m[2]};
        }
        throw CircuitSyntaxError("invalid target '" + token + "'", loc);
    }
    void instruction(const std::string& line, const SourceLocation& loc) {
        static const std::regex syntax(R"(^([A-Za-z_][A-Za-z_0-9]*)(?:\[([^\]]+)\])?(?:\(([^)]*)\))?(?:\s+(.*))?$)");
        static const std::regex controller_name(R"(^[A-Za-z_][A-Za-z0-9_.-]*$)");
        std::smatch match;
        if(!std::regex_match(line, match, syntax)) throw CircuitSyntaxError("invalid instruction syntax", loc);
        std::string name = match[1];
        for(auto& c : name) c = char(std::toupper(static_cast<unsigned char>(c)));
        if(aliases.count(name)) name = aliases.at(name);
        if(!supported_instructions().count(name))
            throw CircuitSyntaxError("unsupported instruction '" + name + "'", loc);
        if((name == "CONTROL") != match[2].matched)
            throw CircuitSyntaxError("only CONTROL requires a [function] name", loc);
        FileInstruction inst;
        inst.name = name;
        inst.control_name = match[2];
        inst.source = loc;
        inst.op_id = "op" + std::to_string(data.instructions.size());
        if(match[3].matched) try {
                for(auto& text : split(match[3], ',')) inst.args.push_back(parse_rational(text));
            } catch(const std::exception&) {
                throw CircuitSyntaxError("arguments must be finite decimal or rational numbers", loc);
            }
        std::string rest = match[4];
        auto& targets = inst.targets;
        auto& args = inst.args;
        auto& outputs = inst.outputs;
        auto& products = inst.products;
        auto fail = [&](const std::string& message) { throw CircuitSyntaxError(message, loc); };
        auto unique = [](const std::vector<FileTarget>& group) {
            std::set<size_t> seen;
            for(const auto& t : group)
                if(!seen.insert(t.index).second) return false;
            return true;
        };
        if(name == "CONTROL") {
            if(!std::regex_match(inst.control_name, controller_name)) fail("invalid controller name");
            auto arrow = rest.find("->");
            if(arrow == std::string::npos || rest.find("->", arrow + 2) != std::string::npos)
                fail("CONTROL requires inputs -> ctrl outputs");
            for(const auto& word : words(rest.substr(0, arrow))) targets.push_back(target(word, loc));
            for(const auto& word : words(rest.substr(arrow + 2))) outputs.push_back(target(word, loc, true));
            if(outputs.empty() || !unique(outputs)) fail("CONTROL needs distinct nonempty output registers");
            for(const auto& t : targets)
                if(!classical(t) && t.kind != "det")
                    fail("CONTROL inputs must be records, detectors or control registers");
            AffineExpression expr;
            if(builtin(inst.control_name)) {
                if(inst.control_name == "NOT" && (!args.empty() || targets.size() != 1 || outputs.size() != 1))
                    fail("NOT takes exactly one input and one output, with no arguments");
                if(inst.control_name == "XOR" &&
                        (outputs.size() != 1 || args.size() > 1 ||
                                (!args.empty() && args[0] != Rational(0) && args[0] != Rational(1))))
                    fail("XOR takes optional constant 0/1 and exactly one output");
                expr.constant = inst.control_name == "NOT" ? 1 : !args.empty() && args[0] == Rational(1);
                for(const auto& t : targets) expr ^= data.expression(t);
            } else {
                if(!args.empty()) fail("external controller configuration belongs in its explicit registry binding");
                if(!controllers.count(inst.control_name))
                    fail("missing controller binding '" + inst.control_name +
                            "'; install and explicitly register its decoder");
                for(const auto& t : targets)
                    if(t.kind == "rec")
                        fail("external CONTROL inputs must be det[...] or ctrl[...]; declare DETECTOR rec[...] first and pass its det[...]");
                for(const auto& t : targets) {
                    auto value = data.expression(t);
                    if(!value.terms.empty() &&
                            std::find(data.required.begin(), data.required.end(), value) == data.required.end())
                        data.required.push_back(std::move(value));
                }
            }
            for(auto& t : outputs) {
                t.version = versions.count(t.index) ? versions[t.index] + 1 : 0;
                versions[t.index] = t.version;
                data.affine[t] = builtin(inst.control_name) ? expr : AffineExpression{0, {t}};
            }
        } else {
            if(rest.find("->") != std::string::npos) fail("only CONTROL writes registers");
            if(name == "MPP") {
                static const std::regex stars(R"(\s*\*\s*)");
                auto joined = std::regex_replace(rest, stars, "*");
                for(const auto& product : words(joined)) {
                    std::vector<FileTarget> group;
                    for(const auto& word : split(product, '*')) {
                        auto t = target(word, loc);
                        group.push_back(t);
                        targets.push_back(t);
                    }
                    products.push_back(std::move(group));
                }
            } else
                for(const auto& word : words(rest)) targets.push_back(target(word, loc));
            if(noise_args.count(name) && !targets.empty() && classical(targets.front())) {
                inst.guard = targets.front();
                targets.erase(targets.begin());
            }
            if(noise_args.count(name)) {
                bool valid = args.size() == noise_args.at(name);
                for(const auto& p : args) valid = valid && Rational(0) <= p && p <= Rational(1);
                if(!valid) fail(name + " requires " + std::to_string(noise_args.at(name)) + " probabilities in [0, 1]");
                if(name.rfind("PAULI_CHANNEL", 0) == 0) {
                    Rational sum;
                    for(const auto& p : args) sum = sum + p;
                    if(sum > Rational(1)) fail("Pauli channel probabilities sum to more than one");
                }
            } else if(measurements.count(name)) {
                if(args.size() > 1 || (!args.empty() && (!(Rational(0) <= args[0]) || !(args[0] <= Rational(1)))))
                    fail("measurement accepts one report-flip probability in [0, 1]");
            } else if(name == "OBSERVABLE_INCLUDE") {
                if(args.size() != 1 || args[0] < Rational(0) || !(args[0].d == Integer(1)))
                    fail("OBSERVABLE_INCLUDE needs one nonnegative integer index");
            } else if(name == "POSTSELECT") {
                if(!args.empty()) fail("POSTSELECT takes no arguments; write !det[n] or !ctrl[k] to require a one");
            } else if(name != "DETECTOR" && name != "QUBIT_COORDS" && name != "SHIFT_COORDS" && !args.empty())
                fail(name + " does not take arguments");
            if(name == "POSTSELECT") {
                if(targets.empty()) fail("POSTSELECT requires detector or control references");
                for(const auto& t : targets)
                    if(t.kind != "det" && t.kind != "ctrl")
                        fail("POSTSELECT targets must be det[n], ctrl[k] or their complements");
            } else if(name == "TICK" || name == "SHIFT_COORDS") {
                if(!targets.empty()) fail(name + " does not take targets");
            } else if(name == "DETECTOR") {
                for(const auto& t : targets)
                    if(t.kind != "rec" || t.inverted) fail("DETECTOR targets must be measurement records");
            } else if(name == "OBSERVABLE_INCLUDE") {
                for(const auto& t : targets)
                    if((t.kind != "rec" && t.kind != "ctrl") || t.inverted)
                        fail("OBSERVABLE_INCLUDE targets must be measurement records or control registers");
            } else if(name == "MPP" || name == "CORRELATED_ERROR" || name == "ELSE_CORRELATED_ERROR") {
                if(targets.empty()) fail(name + " requires Pauli targets");
                for(const auto& t : targets)
                    if(t.kind != "pauli") fail(name + " requires Pauli targets");
                if(name != "MPP") {
                    for(const auto& t : targets)
                        if(t.inverted) fail("Pauli error targets cannot be inverted");
                    products = {targets};
                }
                for(const auto& group : products)
                    if(!unique(group)) fail("Pauli product repeats a qubit");
            } else {
                if(targets.empty()) fail(name + " requires targets");
                size_t arity = triples.count(name) ? 3
                        : (doubles.count(name) || name == "MXX" || name == "MYY" || name == "MZZ" ||
                                  name == "DEPOLARIZE2" || name == "PAULI_CHANNEL_2")
                        ? 2
                        : 1;
                if(targets.size() % arity) fail(name + " needs target groups of " + std::to_string(arity));
                if(name == "CZ")
                    for(size_t j = 0; j < targets.size(); j += 2)
                        if(targets[j].kind == "qubit" && classical(targets[j + 1]))
                            std::swap(targets[j], targets[j + 1]);
                for(size_t j = 0; j < targets.size(); j += arity) {
                    bool control = controlled.count(name) && classical(targets[j]);
                    std::vector<FileTarget> group(targets.begin() + j + control, targets.begin() + j + arity);
                    for(const auto& t : group) {
                        if(t.kind != "qubit") fail("invalid quantum target for " + name);
                        if(t.inverted && !measurements.count(name)) fail("inverted target is not allowed for " + name);
                    }
                    if(!unique(group)) fail(name + " repeats a qubit within one operation");
                }
                if(name == "MXX" || name == "MYY" || name == "MZZ") {
                    for(size_t j = 0; j < targets.size(); j += 2) {
                        auto group = std::vector<FileTarget>(targets.begin() + j, targets.begin() + j + 2);
                        for(auto& t : group) {
                            t.kind = "pauli";
                            t.pauli = name.substr(1, 1);
                        }
                        products.push_back(std::move(group));
                    }
                } else if(measurements.count(name)) {
                    auto axis = name == "MX" || name == "MRX" ? "X" : name == "MY" || name == "MRY" ? "Y" : "Z";
                    for(auto t : targets) {
                        t.kind = "pauli";
                        t.pauli = axis;
                        products.push_back({t});
                    }
                }
            }
            if(name == "CORRELATED_ERROR" || name == "ELSE_CORRELATED_ERROR") {
                if(name == "CORRELATED_ERROR") inst.chain_id = inst.op_id;
                else if(data.instructions.empty() ||
                        (data.instructions.back().name != "CORRELATED_ERROR" &&
                                data.instructions.back().name != "ELSE_CORRELATED_ERROR"))
                    fail("ELSE_CORRELATED_ERROR must immediately follow a correlated error alternative");
                else if(data.instructions.back().guard != inst.guard)
                    fail("correlated alternatives must have the same control guard");
                else inst.chain_id = data.instructions.back().chain_id;
            }
        }
        size_t count = measurements.count(name)
                ? ((name == "MPP" || name == "MXX" || name == "MYY" || name == "MZZ") ? products.size()
                                                                                      : targets.size())
                : 0;
        for(size_t j = 0; j < count; j++) inst.records.push_back(data.num_records++);
        if(name == "DETECTOR") {
            AffineExpression expr;
            for(const auto& t : targets) expr ^= data.expression(t);
            data.detectors.push_back(std::move(expr));
        }
        if(name == "CONTROL" && !builtin(inst.control_name)) data.external.push_back(data.instructions.size());
        data.instructions.push_back(std::move(inst));
    }
    void expand(const std::vector<Block>& block, const std::vector<size_t>& address = {}) {
        for(const auto& item : block) {
            if(item.kind == Block::repeat) {
                if(!expanded_size(item.children, limit)) continue;
                for(size_t j = 0; j < item.count; j++) {
                    auto next = address;
                    next.push_back(j);
                    expand(item.children, next);
                }
            } else {
                if(data.instructions.size() >= limit)
                    throw instruction_budget(item.source, limit, double(data.instructions.size()) + 1);
                auto loc = item.source;
                loc.repeat = address;
                instruction(item.text, loc);
            }
        }
    }
};
} // namespace
std::vector<size_t> CircuitData::qubits() const {
    std::set<size_t> used;
    auto add = [&](const FileTarget& target) {
        if(target.kind == "qubit" || target.kind == "pauli") used.insert(target.index);
    };
    for(const auto& inst : instructions) {
        for(const auto& target : inst.targets) add(target);
        for(const auto& product : inst.products)
            for(const auto& target : product) add(target);
    }
    return {used.begin(), used.end()};
}
const std::set<std::string>& supported_instructions() {
    static const auto result = []() {
        std::set<std::string> out = singles;
        for(const auto* group : {&doubles, &triples, &measurements, &resets, &annotations})
            out.insert(group->begin(), group->end());
        for(const auto& [name, count] : noise_args) out.insert(name);
        out.insert("CONTROL");
        out.insert("POSTSELECT");
        return out;
    }();
    return result;
}
AffineExpression CircuitData::expression(const FileTarget& target) const {
    FileTarget plain{target.kind, target.index, target.version};
    AffineExpression out;
    if(target.kind == "rec") out.terms.insert(plain);
    else if(target.kind == "ctrl") out = affine.at(plain);
    else if(target.kind == "det") out = detectors.at(target.index);
    else throw std::invalid_argument("only a reported record, detector or control has a classical expression");
    out.constant ^= target.inverted;
    return out;
}
std::string CircuitData::format_version() const { return "magna-0.1"; }
CircuitData parse_circuit(const std::string& text, const std::string& source, const std::set<std::string>& controllers,
        size_t max_instructions, size_t max_qubits) {
    if(!max_instructions || max_instructions == std::numeric_limits<size_t>::max())
        throw std::invalid_argument("max_instructions must be a positive integer");
    if(!max_qubits || max_qubits == std::numeric_limits<size_t>::max())
        throw std::invalid_argument("max_qubits must be a positive integer");
    Parser parser{{}, controllers, max_instructions, max_qubits, {}};
    parser.data.source = source;
    parser.data.text = text;
    auto tree = blocks(text, source, max_instructions);
    expanded_size(tree, max_instructions);
    parser.expand(tree);
    return std::move(parser.data);
}
} // namespace magic_native
