from typing import Optional


def integer_within_tolerance(
    expected: int, actual: Optional[int], tolerance: int
) -> bool:
    if tolerance < 0:
        raise ValueError("integer comparison tolerance must be non-negative")
    return actual is not None and abs(actual - expected) <= tolerance
