"""Nine ordered category lists consumed by the functional-test runner.

Category packages explicitly collect their individual Python cases.
Elastic retains its module during migration. Shared components live in
flexlb_ft.support and do not register cases.
"""

from .admission import ADMISSION_CASES
from .balance import BALANCE_CASES
from .cancel import CANCEL_CASES
from .elastic import ELASTIC_CASES
from .engine_fault import ENGINE_FAULT_CASES
from .kv import KV_CASES
from .master import MASTER_CASES
from .priority import PRIORITY_CASES
from .status import STATUS_CASES

__all__ = [
    "ADMISSION_CASES",
    "BALANCE_CASES",
    "CANCEL_CASES",
    "ELASTIC_CASES",
    "ENGINE_FAULT_CASES",
    "KV_CASES",
    "MASTER_CASES",
    "PRIORITY_CASES",
    "STATUS_CASES",
]
