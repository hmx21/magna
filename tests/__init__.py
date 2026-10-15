"""Keep standard unittest discovery aware of the grouped regression sources."""

def load_tests(loader, tests, pattern):
    from scripts.project_checks import load_suite
    return load_suite(pattern=pattern or 'test_*.py')
