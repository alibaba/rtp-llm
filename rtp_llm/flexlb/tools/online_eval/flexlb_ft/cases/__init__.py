"""flexlb_ft case categories (task #85 reorg): nine scenario modules.

Each ``cases/<category>.py`` registers its cases into a CATEGORY_CASES
list (cancel -> CANCEL_CASES, status -> STATUS_CASES, kv -> KV_CASES,
balance -> BALANCE_CASES, elastic -> ELASTIC_CASES, engine_fault ->
ENGINE_FAULT_CASES, master -> MASTER_CASES, admission ->
ADMISSION_CASES, direct -> DIRECT_CASES); the runner imports the nine
lists and concatenates them into ALL_CASES.  The framework files
(harness / context / engine_ops / grade) stay in flexlb_ft/ — this
package holds only scenario definitions, one contract theme per module.
"""

from .admission import ADMISSION_CASES
from .balance import BALANCE_CASES
from .cancel import CANCEL_CASES
from .direct import DIRECT_CASES
from .elastic import ELASTIC_CASES
from .engine_fault import ENGINE_FAULT_CASES
from .kv import KV_CASES
from .master import MASTER_CASES
from .status import STATUS_CASES

__all__ = [
    "ADMISSION_CASES",
    "BALANCE_CASES",
    "CANCEL_CASES",
    "DIRECT_CASES",
    "ELASTIC_CASES",
    "ENGINE_FAULT_CASES",
    "KV_CASES",
    "MASTER_CASES",
    "STATUS_CASES",
]
