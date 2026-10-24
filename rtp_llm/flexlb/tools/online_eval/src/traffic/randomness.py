"""Stable SplitMix64 event-index randomness; changing this changes experiment identity."""


def event_uniform(index, seed):
    mask = (1 << 64) - 1
    z = (index + seed + 0x9E3779B97F4A7C15) & mask
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & mask
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & mask
    z ^= z >> 31
    return (z >> 11) * 2.0**-53
