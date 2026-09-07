"""balance cases in stable execution order."""

from ...registry import collect_cases
from .balance_concurrent_mix import balance_concurrent_mix
from .balance_decode_spread import balance_decode_spread
from .balance_len_mixed import balance_len_mixed
from .balance_overload_avoid_decode import balance_overload_avoid_decode
from .balance_overload_avoid_prefill import balance_overload_avoid_prefill
from .balance_uniform_serial import balance_uniform_serial

BALANCE_CASES = collect_cases(
    "balance",
    [
        balance_uniform_serial,
        balance_concurrent_mix,
        balance_overload_avoid_prefill,
        balance_overload_avoid_decode,
        balance_decode_spread,
        balance_len_mixed,
    ],
)
