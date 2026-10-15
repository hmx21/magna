#include "compile/compiler.hpp"
#include "evaluate/reference_network.hpp"
#include <iostream>
using namespace magic_native;
void require(bool ok, const char* text) {
    if(!ok) throw std::runtime_error(text);
}
int main() {
    size_t generators = 0;
    for(int variant = 0; variant < 8; variant++) {
        CompilerIR ir;
        ir.inputs = {0, 1, 2};
        ir.outputs = {2, 0, 1};
        ir.initial.width = 3;
        auto add = [&](std::string kind, std::vector<int> wires, int exponent = 0, int record = -1) {
            CompilerOp op;
            op.kind = kind;
            op.qubits = wires;
            op.exponent = exponent;
            op.record = record;
            op.location = ir.operations.size();
            ir.operations.push_back(op);
        };
        add("H", {0});
        add("H", {1});
        add("PHASE", {0}, variant & 1 ? 7 : 1);
        add("CX", {0, 1});
        add("PHASE", {1, 2}, 3);
        if(variant & 2) {
            add("H", {0});
            add("PHASE", {0}, 5);
        }
        if(variant & 4) {
            add("MZ", {1}, 0, 0);
            ir.records = 1;
            add("RESET_PLUS", {2});
        }
        CompilerPathPrefix prefix;
        auto source = prepare_compiler(ir, false, &prefix, true, CompilerPurpose::Trace);
        const auto source_ops = ir.operations.size();
        add("PREP_PLUS", {3});
        add("CX", {0, 3});
        add("PHASE", {3}, 7);
        add("H", {2});
        add("CZ", {2, 3});
        add("PHASE", {0, 2}, 3);
        ir.outputs = {3, 1, 0, 2};
        auto branch_prefix = prefix;
        auto branch = prepare_compiler(ir, false, &branch_prefix, true, CompilerPurpose::Trace, true);
        require(branch.plan.suffix.empty() && branch.plan.locations.empty() && branch.plan.histories.empty(),
                "branch retains duplicate protocol transport");
        auto target = prepare_compiler(ir, false, &prefix, true, CompilerPurpose::Trace);
        require(bool(target.plan.continuation), "compiled continuation missing");
        for(size_t location = 0; location < source_ops; location++)
            for(int wire = 0; wire < 3; wire++)
                for(int pauli = 1; pauli <= 3; pauli++) {
                    std::vector<Fault> fault{{location, wire, pauli}};
                    auto original = source.plan.fault_frame(fault, false);
                    auto expected = target.plan.fault_frame(fault, false);
                    auto actual = branch.plan.continuation->apply(original);
                    require(actual.x == expected.x && actual.z == expected.z, "continuation generator identity");
                    generators++;
                }
    }
    std::cout << "reference reuse checks passed: " << generators << " assignment generators\n";
}
