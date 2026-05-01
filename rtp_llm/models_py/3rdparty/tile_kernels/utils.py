def ceil_div(x: int, y: int) -> int:
    return (x + y - 1) // y


def align(x: int, y: int) -> int:
    return ceil_div(x, y) * y


def is_power_of_two(x: int) -> bool:
    return x > 0 and (x & (x - 1)) == 0
