"""master cases in stable execution order."""

from ...registry import collect_cases
from .direct_generate_error import inject_generate_error
from .failback_wraparound import failback_wraparound
from .fallback_direct import fallback_direct
from .fallback_negative_errorcode import fallback_negative_errorcode
from .master_coldstart_burst import coldstart_burst
from .master_freeze import master_freeze
from .master_ha_failover import master_ha_failover
from .master_kill import master_kill
from .master_quota_block import master_quota_block

MASTER_CASES = collect_cases(
    "master",
    [
        master_kill,
        master_quota_block,
        coldstart_burst,
        master_freeze,
        master_ha_failover,
        fallback_direct,
        fallback_negative_errorcode,
        failback_wraparound,
        inject_generate_error,
    ],
)
