"""Released prefill kernel ABI; other geometry uses its validated backend.

The raw-key state ring has two compression groups. These constants are kernel
specializations, not deployment options or a promise of arbitrary geometry.
"""

PREFILL_COMPRESS_RATIO = 4
PREFILL_STATE_RING_ROWS = 2 * PREFILL_COMPRESS_RATIO
PREFILL_HEAD_DIM = 128
