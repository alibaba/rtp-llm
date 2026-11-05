"""Assertions operate on declared values, never business type names."""


def compare(actual, expected):
    for operator, target in expected.items():
        if operator == 'eq':
            good = actual == target
        elif operator == 'count':
            good = len(actual) == target
        elif operator == 'gte':
            good = bool(actual) and all(v >= target for v in actual) if isinstance(actual, list) else actual >= target
        else:
            good = bool(actual) and all(v <= target for v in actual) if isinstance(actual, list) else actual <= target
        if not good:
            return False
    return True
