"""Integer launch geometry without Triton's constexpr dispatch on the host."""


def ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def next_power_of_2(value: int) -> int:
    if isinstance(value, int):
        return 0 if value <= 0 else 1 << (value - 1).bit_length()
    # Preserve Triton's behavior for symbolic or NumPy scalar inputs.
    import triton

    return triton.next_power_of_2(value)
