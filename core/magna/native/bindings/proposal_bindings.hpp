std::vector<std::string> input_names(PyObject* raw) {
    Seq rows(raw);
    std::vector<std::string> out;
    for(size_t j = 0; j < rows.size(); j++) out.push_back(file_text(rows[j]));
    return out;
}
PyObject* output_record_proposal(const std::vector<std::string>& names, const RecordProposalDraw& draw) {
    Object records(PyDict_New()), n(output_big(draw.probability.n)), d(output_big(draw.probability.d));
    for(size_t j = 0; j < names.size(); j++)
        phase_set(records.p, names[j].c_str(), PyLong_FromLong(draw.true_word.bit(j)));
    return Py_BuildValue("(O(OO))", records.p, n.p, d.p);
}
PyObject* draw_independent_proposal(PyObject*, PyObject* args) {
    try {
        PyObject *raw_names, *raw_probabilities, *rng;
        if(!PyArg_ParseTuple(args, "OOO", &raw_names, &raw_probabilities, &rng)) return nullptr;
        auto names = input_names(raw_names);
        Seq rows(raw_probabilities);
        std::map<std::string, Rational> probabilities;
        for(size_t j = 0; j < rows.size(); j++) {
            Seq row(rows[j]);
            if(row.size() != 2) throw std::invalid_argument("proposal entry shape");
            probabilities[file_text(row[0])] = input_rational(row[1]);
        }
        auto draw = independent_proposal(names, probabilities, input_randrange(rng));
        return output_record_proposal(names, draw);
    } catch(...) { return error(); }
}
AffineProposal* affine_proposal(PyObject* cap) {
    return static_cast<AffineProposal*>(PyCapsule_GetPointer(cap, "magic-affine-proposal-v1"));
}
void affine_proposal_destructor(PyObject* cap) { delete affine_proposal(cap); }
PyObject* make_affine_proposal(PyObject*, PyObject* args) {
    try {
        PyObject *names, *rows, *rhs;
        if(!PyArg_ParseTuple(args, "OOO", &names, &rows, &rhs)) return nullptr;
        auto proposal = std::make_unique<AffineProposal>(input_names(names), bitvec(rows), intvec(rhs));
        const auto& s = proposal->solution;
        Object base(output_bits(s.particular)), basis(phase_output_bits(s.kernel)), reduced(phase_output_bits(s.rows)),
                solution(PyTuple_Pack(3, base.p, basis.p, reduced.p));
        Object cap(PyCapsule_New(proposal.release(), "magic-affine-proposal-v1", affine_proposal_destructor));
        return PyTuple_Pack(2, cap.p, solution.p);
    } catch(...) { return error(); }
}
PyObject* draw_affine_proposal(PyObject*, PyObject* args) {
    try {
        PyObject *cap, *names, *rng;
        if(!PyArg_ParseTuple(args, "OOO", &cap, &names, &rng)) return nullptr;
        auto* proposal = affine_proposal(cap);
        if(!proposal) return nullptr;
        auto draw = proposal->draw(input_names(names), random_bits(rng));
        return output_record_proposal(proposal->records, draw);
    } catch(...) { return error(); }
}
