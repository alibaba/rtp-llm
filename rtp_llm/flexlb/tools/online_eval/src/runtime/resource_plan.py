"""Backend-independent integer and port interval admission tools."""


class ResourcePlanError(ValueError):
    """The selected instances cannot be executed within a valid port lease."""


def validate_integer(value: int, name: str, minimum: int) -> None:
    if type(value) is not int or value < minimum:
        raise ResourcePlanError(f"{name} must be an integer >= {minimum}")


def port_numbers(intervals) -> tuple[int, ...]:
    return tuple(port for _, lo, hi in intervals for port in range(lo, hi + 1))


def validate_disjoint_intervals(intervals) -> None:
    for i, (name, lo, hi) in enumerate(intervals):
        for other, start, end in intervals[:i]:
            if lo <= end and start <= hi:
                raise ResourcePlanError(f"{name} {lo}..{hi} overlaps {other} {start}..{end}")
