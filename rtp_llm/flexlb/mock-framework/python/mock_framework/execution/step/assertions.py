def matches_expectation(actual, expected):
    for operator, target in expected.items():
        if operator == 'eq':
            good = actual == target
        elif operator == 'count':
            good = len(actual) == target
        elif operator == 'gte':
            values = actual if isinstance(actual, list) else [actual]
            good = bool(values) and all(value >= target for value in values)
        else:
            values = actual if isinstance(actual, list) else [actual]
            good = bool(values) and all(value <= target for value in values)
        if not good:
            return False
    return True
