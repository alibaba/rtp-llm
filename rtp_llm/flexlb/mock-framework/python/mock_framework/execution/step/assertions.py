def matches_expectation(actual, expected):
    for operator, target in expected.items():
        if operator == 'eq':
            good = actual == target
        elif operator == 'count':
            good = len(actual) == target
        elif operator == 'gte':
            good = actual >= target
        else:
            good = actual <= target
        if not good:
            return False
    return True
