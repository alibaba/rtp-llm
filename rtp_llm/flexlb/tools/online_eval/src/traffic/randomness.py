"""Stable SplitMix64 event-index randomness; changing this changes experiment identity."""

def event_uniform(index, seed):
    mask = (1 << 64) - 1
    z = (index + seed + 0x9e3779b97f4a7c15) & mask
    z = ((z ^ (z >> 30)) * 0xbf58476d1ce4e5b9) & mask
    z = ((z ^ (z >> 27)) * 0x94d049bb133111eb) & mask
    z ^= z >> 31
    return (z >> 11) * 2.0 ** -53
