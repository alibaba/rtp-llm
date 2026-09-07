"""status cases in stable execution order."""

from ...registry import collect_cases
from .status_ack_empty_no_crash import status_ack_empty_no_crash
from .status_ack_multi_error import status_ack_multi_error
from .status_ack_partial_fail import status_ack_partial_fail
from .status_batch_async_partial_fail import status_batch_async_partial_fail
from .status_cursor_regress import status_cursor_regress
from .status_decode_before_prefill import status_decode_before_prefill
from .status_decode_running_before_prefill import status_decode_running_before_prefill
from .status_decode_suppress_finished import status_decode_suppress_finished
from .status_decode_waiting_before_prefill import status_decode_waiting_before_prefill
from .status_duplicate_finished import status_duplicate_finished
from .status_fetch_error import inject_fetch_error
from .status_finished_then_running import status_finished_then_running
from .status_foreign_batchid import status_foreign_batchid
from .status_inflight_ttl_cleanup import inflight_ttl_cleanup
from .status_prefill_suppress_all import status_prefill_suppress_all
from .status_prefill_suppress_finished import status_prefill_suppress_finished
from .status_special_ids import status_special_ids
from .status_status_no_respond import status_status_no_respond
from .status_unbatched_single_request import status_unbatched_single_request
from .status_unknown_batchid import status_unknown_batchid
from .status_unknown_rid_finished import status_unknown_rid_finished
from .status_unknown_rid_running import status_unknown_rid_running
from .status_version_regress import status_version_regress
from .status_zombie_completed_running import status_zombie_completed_running
from .status_zombie_fake_running import status_zombie_fake_running

STATUS_CASES = collect_cases(
    "status",
    [
        inflight_ttl_cleanup,
        status_ack_partial_fail,
        status_batch_async_partial_fail,
        status_ack_multi_error,
        status_ack_empty_no_crash,
        status_prefill_suppress_all,
        status_prefill_suppress_finished,
        status_status_no_respond,
        status_unknown_rid_finished,
        status_version_regress,
        status_decode_suppress_finished,
        status_decode_before_prefill,
        status_decode_running_before_prefill,
        status_decode_waiting_before_prefill,
        status_unknown_rid_running,
        status_unknown_batchid,
        status_special_ids,
        status_unbatched_single_request,
        status_foreign_batchid,
        status_duplicate_finished,
        status_cursor_regress,
        status_finished_then_running,
        status_zombie_completed_running,
        status_zombie_fake_running,
        inject_fetch_error,
    ],
)
