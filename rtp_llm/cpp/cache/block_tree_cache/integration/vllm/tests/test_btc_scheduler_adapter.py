# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast
from unittest.mock import Mock, call, patch

import pytest

from btc_vllm.cache_integration import BtcCacheBlocks, BtcCacheIntegration
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.outputs import KVConnectorOutput, ModelRunnerOutput

from btc_vllm.connector import BtcTpuConnector
from btc_vllm.scheduler_adapter import BtcSchedulerAdapter

if TYPE_CHECKING:
    from block_tree_cache import Plan


@dataclass
class _Request:
    request_id: str
    num_computed_tokens: int = 16
    num_in_flight_tokens: int = 8
    num_stale_output_tokens: int = 0

    def is_finished(self) -> bool:
        return False


def test_btc_output_publication_respects_committed_pages_and_stop_order():
    events = []
    requests = {
        request_id: _Request(request_id)
        for request_id in ("stream", "continuing", "stale", "failed", "invalid")
    }
    requests["stale"].num_stale_output_tokens = 4

    def prepare(output, protected):
        assert protected == {9}
        events.append("receipts")
        output.failed_recving.add("failed")
        return output

    def stop(request):
        events.append("stopped")
        request.num_computed_tokens = 0  # A streaming stop moves the frontier.
        return False

    def update(step, output):
        assert output.kv_connector_output.failed_recving == {"failed"}
        invalid = scheduler._handle_invalid_blocks(
            output.kv_connector_output.invalid_block_ids, step.num_scheduled_tokens
        )
        for request_id, tokens in step.num_scheduled_tokens.items():
            request = requests[request_id]
            request.num_in_flight_tokens -= tokens
            request.num_stale_output_tokens = max(
                0, request.num_stale_output_tokens - tokens
            )
            if request_id not in invalid and request_id == "stream":
                scheduler._handle_stopped_request(request)
        return {}

    scheduler = SimpleNamespace(
        requests=requests,
        update_from_output=update,
        _handle_stopped_request=stop,
        _handle_invalid_blocks=lambda *_: {"invalid"},
        _update_waiting_for_remote_kv=lambda _: None,
    )
    connector = SimpleNamespace(
        prepare_scheduler_output=prepare,
        publish_computed_gpu_blocks=lambda req, end: events.append(
            (req.request_id, end)
        ),
        balance_device=lambda protected: events.append(("balance", protected)),
    )
    BtcSchedulerAdapter(cast(Scheduler, scheduler), cast(BtcTpuConnector, connector))
    step = SchedulerOutput.make_empty()
    step.new_block_ids_to_zero = [9]
    step.num_scheduled_tokens = dict.fromkeys(requests, 4)
    scheduler.update_from_output(
        step,
        ModelRunnerOutput(
            [], {}, kv_connector_output=KVConnectorOutput(invalid_block_ids={1})
        ),
    )

    assert events == ["receipts", ("stream", 12), "stopped", ("continuing", 12), ("balance", {9})]
    assert requests["stale"].num_stale_output_tokens == 0


def test_btc_load_publication_uses_cached_frontier_before_replay_trim():
    publications = []
    request = _Request("load", num_in_flight_tokens=0)

    def finish_load(req):
        req.num_computed_tokens -= 1

    scheduler = SimpleNamespace(
        update_from_output=lambda *_: {},
        _handle_stopped_request=lambda _: False,
        _handle_invalid_blocks=lambda *_: set(),
        _update_waiting_for_remote_kv=finish_load,
        failed_recving_kv_req_ids=set(),
    )
    connector = SimpleNamespace(
        publish_computed_gpu_blocks=lambda _req, end: publications.append(end),
    )
    BtcSchedulerAdapter(cast(Scheduler, scheduler), cast(BtcTpuConnector, connector))
    scheduler._update_waiting_for_remote_kv(request)
    assert publications == [16]
    assert request.num_computed_tokens == 15

    scheduler.failed_recving_kv_req_ids.add("load")
    scheduler._update_waiting_for_remote_kv(request)
    assert publications == [16]


def test_stale_plan_returns_entire_provisional_request_row():
    request = _Request("retry", num_computed_tokens=0)
    pending = BtcCacheBlocks(((),), plan=cast("Plan", object()))
    released = []
    native_calls = []
    adapter = object.__new__(BtcCacheIntegration)
    adapter._allocation = None
    adapter._allocate = lambda *_args, **_kwargs: BtcCacheBlocks(((),))
    adapter.manager = SimpleNamespace(free=released.append)

    def reject_binding(req, plan):
        native_calls.append((req, plan))
        return False

    adapter.btc = SimpleNamespace(
        bind_allocated=reject_binding, abandon_admission=Mock(),
    )
    assert adapter.allocate_slots(request, 0, new_computed_blocks=pending) is None
    assert native_calls == [(request, pending.plan)]
    assert released == [request]
    assert adapter._allocation is None

    def fail_binding(_req, _plan):
        raise RuntimeError("native bind failed")

    adapter.btc.bind_allocated = fail_binding
    try:
        adapter.allocate_slots(request, 0, new_computed_blocks=pending)
    except RuntimeError as error:
        assert str(error) == "native bind failed"
    else:
        raise AssertionError("BTC must propagate a native binding error")
    assert released == [request, request]
    assert adapter.btc.abandon_admission.call_args_list == [
        call(request.request_id, pending.plan), call(request.request_id, pending.plan),
    ]
    assert adapter._allocation is None


def test_allocation_shortage_abandons_even_an_empty_claimed_plan():
    request = _Request("shortage", num_computed_tokens=0)
    blocks = BtcCacheBlocks(((),), plan=cast("Plan", object()))
    adapter = object.__new__(BtcCacheIntegration)
    adapter._allocation = None
    adapter._allocate = Mock(return_value=None)
    adapter.manager = SimpleNamespace(free=Mock())
    adapter.btc = Mock()

    assert adapter.allocate_slots(request, 1, new_computed_blocks=blocks) is None
    adapter.btc.abandon_admission.assert_called_once_with(request.request_id, blocks.plan)
    adapter.btc.bind_allocated.assert_not_called()
    adapter.manager.free.assert_not_called()
    assert adapter._allocation is None

    # vLLM can discard a cache hit before allocating a normal row.
    adapter.btc.reset_mock()
    assert adapter.allocate_slots(request, 1) is None
    adapter.btc.abandon_admission.assert_called_once_with(request.request_id)
    adapter.btc.bind_allocated.assert_not_called()


def _admission_scheduler():
    from btc_vllm.scheduler import BtcScheduler

    config = SimpleNamespace(
        kv_transfer_config=SimpleNamespace(kv_connector_extra_config={}),
        model_config=SimpleNamespace(model="fixture"),
        cache_config=SimpleNamespace(cache_dtype="auto"),
    )
    scheduler = BtcScheduler(config, None)
    scheduler._native = Mock()
    scheduler._remote = Mock()
    scheduler.step = 8
    scheduler.group_sizes = (8,)
    scheduler.manager = SimpleNamespace(
        empty_kv_cache_blocks=BtcCacheBlocks(((),)),
        prefix_cache_lookup_enabled=lambda _: True,
        get_blocks=lambda _: BtcCacheBlocks(((),)),
    )
    scheduler.pool = SimpleNamespace(
        enable_kv_cache_events=False,
        null_block=SimpleNamespace(is_null=True),
        blocks=[None, SimpleNamespace(block_id=1, is_null=False)],
    )
    return scheduler


def test_pending_admission_polls_until_claim_and_reuses_selected_pages():
    from block_tree_cache import Limits, Page, PageState, Position, Status

    scheduler = _admission_scheduler()
    hold = scheduler._holds[1] = Page(0, 1, 5, 5)
    native = scheduler.native
    plan = SimpleNamespace(
        ready_tokens=8, recoverable_tokens=16, close=Mock(),
        pages=(SimpleNamespace(position=Position(0, 8), state=PageState.READY, pages=(hold,)),
               SimpleNamespace(position=Position(0, 16), state=PageState.LOAD, pages=())),
    )
    native.match.return_value = None
    native.poll_match.side_effect = [None, plan]
    request = SimpleNamespace(
        request_id="one", block_hashes=[b"a", b"b"], num_tokens=17,
        kv_transfer_params={"max_load_tokens": 8},
    )

    for _ in range(2):
        assert scheduler.match(request, None) == (scheduler.manager.empty_kv_cache_blocks, 0)
        assert scheduler.external(request) == (None, False)
    blocks, ready = scheduler.match(request, None)
    assert ready == 8
    assert blocks.blocks[0][0] is scheduler.pool.blocks[1]
    assert scheduler._selected_device_slots() == {1}
    assert scheduler.external(request) == (8, True)
    assert scheduler.match(request, None)[0] is blocks
    native.match.assert_called_once_with(1, (b"a", b"b"), Limits(16, max_load_tokens=8))
    assert native.poll_match.call_args_list == [call(1), call(1)]
    scheduler.finish_request("one")
    native.cancel.assert_called_once_with(1)
    plan.close.assert_called_once()
    assert scheduler._selected_device_slots() == set()
    assert scheduler.external(request) == (0, False)


@pytest.mark.parametrize("retry_initial", [True, False])
def test_admission_retry_is_abandoned_and_changed_input_gets_a_new_id(retry_initial):
    from block_tree_cache import Error, Status

    scheduler = _admission_scheduler()
    native = scheduler.native
    native.match.side_effect = [Error(Status.RETRY, "stale") if retry_initial else None, None, None]
    native.poll_match.side_effect = Error(Status.RETRY, "stale")
    request = SimpleNamespace(
        request_id="retry", block_hashes=[b"a"], num_tokens=9, kv_transfer_params={},
    )
    empty = scheduler.manager.empty_kv_cache_blocks
    assert scheduler.match(request, None) == (empty, 0)
    if not retry_initial:
        assert scheduler.external(request) == (None, False)
        assert scheduler.match(request, None) == (empty, 0)
    assert scheduler.external(request) == (0, False)
    assert scheduler.match(request, None) == (empty, 0)
    request.block_hashes[0] = b"b"
    assert scheduler.match(request, None) == (empty, 0)
    assert [(entry.args[0], entry.args[1]) for entry in native.match.call_args_list] == [
        (1, (b"a",)), (2, (b"a",)), (3, (b"b",)),
    ]
    assert native.cancel.call_args_list == [call(1), call(2)]


def test_binding_a_local_admission_does_not_cancel_the_native_request():
    from block_tree_cache import Status

    scheduler = _admission_scheduler()
    native = scheduler.native
    plan = SimpleNamespace(ready_tokens=0, recoverable_tokens=0, pages=(), close=Mock())
    native.match.return_value = plan
    native.bind.return_value = Status.OK
    request = SimpleNamespace(
        request_id="local", block_hashes=[b"a"], num_tokens=9, kv_transfer_params={},
    )
    blocks, ready = scheduler.match(request, None)
    assert ready == 0
    assert scheduler.match(request, None)[0] is blocks
    assert scheduler.bind_allocated(request, blocks.plan)
    scheduler.allocated(request, blocks, 0)
    native.match.assert_called_once()
    native.poll_match.assert_not_called()
    native.bind.assert_called_once_with(plan, [])
    native.cancel.assert_not_called()
    plan.close.assert_called_once()


def test_ordinary_scheduler_uses_no_optional_btc_imports():
    from vllm_torchtpu.distributed.kv_transfer.btc.scheduler_adapter import (
        install_btc_scheduler_adapter,
    )

    def plain_init(self, config):
        self.config = config

    config = SimpleNamespace(
        kv_transfer_config=SimpleNamespace(kv_connector="TPURaidenConnector"),
    )
    with patch.object(Scheduler, "__init__", plain_init):
        install_btc_scheduler_adapter()
        with patch("builtins.__import__", side_effect=AssertionError("unexpected import")):
            assert Scheduler(config).config is config


def test_rank_receipt_keeps_failed_copy_live_until_safe():
    from block_tree_cache import Operation, Result, WorkKind
    from btc_vllm.connector import BtcRankReceipts
    from btc_vllm.scheduler import BtcScheduler
    from btc_vllm.torchtpu.rank_transfers import RankResult

    op = Operation(8, 1)
    work = SimpleNamespace(id=op, kind=WorkKind.HOST_TO_DEVICE, participants=(0, 1, 2), pages=())
    completed = []
    published = []
    scheduler = SimpleNamespace(
        _device_work={op: work},
        _device_receipts={op: {}},
        _finished=set(),
        _failed=set(),
        native=SimpleNamespace(complete=completed.extend),
        _advance=lambda: None,
        _publish_loaded=published.append,
    )
    early = BtcRankReceipts({op: {1: RankResult(False, False)}})
    late = BtcRankReceipts({op: {1: RankResult(True, True)}})
    assert early.aggregate(late).jobs == late.aggregate(early).jobs == {
        op: {1: RankResult(False, True)},
    }
    BtcScheduler.settle(scheduler, {op: {0: RankResult(True, True), 2: RankResult(True, True)}}, set())
    BtcScheduler.settle(scheduler, early.jobs, set())
    assert op in scheduler._device_work
    assert [(r.participant, r.result, r.safe) for r in completed] == [
        (0, Result.SUCCESS, True), (2, Result.SUCCESS, True), (1, Result.FAILED, False),
    ]
    BtcScheduler.settle(scheduler, late.jobs, set())
    assert (completed[-1].participant, completed[-1].result, completed[-1].safe) == (
        1, Result.SUCCESS, True,
    )
    assert op not in scheduler._device_work and not published


def test_device_balance_uses_physical_capacity_and_protected_holds():
    from block_tree_cache import Page
    from btc_vllm.scheduler import BtcScheduler
    from vllm.v1.core.kv_cache_utils import KVCacheBlockCopy

    native = Mock()
    native.device_balance_needed.return_value = True
    idle, active, watched, copying, zeroed, handed_off = (
        Page(0, slot, slot, slot) for slot in range(1, 7)
    )
    pool = SimpleNamespace(
        num_gpu_blocks=9, get_num_free_blocks=lambda: 1,
        blocks=[SimpleNamespace(ref_cnt=2 if slot == active.slot else 1) for slot in range(9)],
        _reuse_watchers={watched.slot: object()},
    )
    scheduler = BtcScheduler.__new__(BtcScheduler)
    scheduler._native = native
    scheduler._holds = {page.slot: page for page in (idle, active, watched, copying, zeroed, handed_off)}
    scheduler.pool = pool
    scheduler._admissions = {}
    scheduler._advance = lambda: ()
    step = SchedulerOutput.make_empty()
    step.kv_cache_block_copies = [KVCacheBlockCopy(copying.slot, 8)]
    step.new_block_ids_to_zero = [zeroed.slot]
    step.kv_connector_block_state = SimpleNamespace(boundary_state_offloads={"req": [(0, handed_off.slot, 8)]})
    assert scheduler.dispatch(step) == {}
    native.balance_device.assert_called_once_with(0, 8, 1, [idle])
    native.device_balance_needed.return_value = False
    native.balance_device.reset_mock()
    scheduler.balance_device(set())
    native.balance_device.assert_not_called()
    scheduler.reclaim(2, {copying.slot, zeroed.slot, handed_off.slot})
    native.reclaim.assert_called_once_with(0, 1, [idle])


def test_demote_rechecks_before_dispatch_and_delivers_real_inflight_receipts():
    from block_tree_cache import Operation, Page, Receipt, Result, WorkKind
    from btc_vllm.scheduler import BtcScheduler
    from btc_vllm.torchtpu.rank_transfers import RankResult

    page = Page(0, 1, 5, 5)
    op = Operation(8, 2)
    work = SimpleNamespace(id=op, kind=WorkKind.DEVICE_TO_HOST, request=0, participants=(0,),
                           pages=(SimpleNamespace(source=(page,)),))
    scheduler = BtcScheduler.__new__(BtcScheduler)
    scheduler._native = native = Mock()
    native.device_balance_needed.return_value = True
    scheduler.pool = SimpleNamespace(num_gpu_blocks=3, get_num_free_blocks=lambda: 0,
                                     blocks=[None, SimpleNamespace(ref_cnt=1)], _reuse_watchers={})
    scheduler._holds = {1: page}
    scheduler._device_work = {}
    scheduler._device_receipts = {}
    scheduler._deferred = set()
    scheduler._finished, scheduler._failed = set(), set()
    scheduler._admissions = {}
    scheduler._advance = lambda: (work,)
    step = SchedulerOutput.make_empty()
    step.new_block_ids_to_zero = [page.slot]
    assert scheduler.dispatch(step) == {}
    assert native.method_calls == [call.device_balance_needed(0, 2, 0), call.balance_device(0, 2, 0, []),
                                   call.complete([Receipt(op, 0, Result.REJECTED, True)])]
    native.reset_mock()
    step.new_block_ids_to_zero = None
    assert scheduler.dispatch(step) == {op: work}
    native.reset_mock()
    scheduler.pool.blocks[1].ref_cnt = 2
    scheduler.settle({op: {0: RankResult(True, True)}}, set())
    assert native.method_calls == [call.device_balance_needed(0, 2, 0), call.balance_device(0, 2, 0, []),
                                   call.complete([Receipt(op, 0, Result.SUCCESS, True)])]


def test_publish_loaded_requires_exact_native_target_installation():
    from block_tree_cache import Binding, Page, Position
    from btc_vllm.scheduler import BtcScheduler
    from vllm.v1.core.kv_cache_utils import make_block_hash_with_group_id

    good, wrong, recycled = (Page(0, slot, slot, slot) for slot in range(1, 4))
    keys = (b"prefix", b"longer")
    pairs = [(Position(0, 8), good), (Position(1, 16), wrong), (Position(0, 16), recycled)]
    work = SimpleNamespace(pages=tuple(SimpleNamespace(position=p, target=(page,)) for p, page in pairs))
    scheduler = BtcScheduler.__new__(BtcScheduler)
    scheduler._native = Mock()
    scheduler._native.installed_device_pages.return_value = (True, False)
    scheduler._load_keys = {good.hold: keys, wrong.hold: keys}
    scheduler._holds = {1: good, 2: wrong, 3: Page(0, 3, 8, 8)}
    scheduler.step = 8
    scheduler.pool = SimpleNamespace(blocks=[None, object(), object(), object()])
    scheduler.cache = Mock()
    scheduler._publish_loaded(work)
    scheduler.native.installed_device_pages.assert_called_once_with(
        keys, [Binding(position, (page,)) for position, page in pairs[:2]],
    )
    scheduler.cache.register_loaded_block.assert_called_once_with(
        make_block_hash_with_group_id(b"prefix", 0), scheduler.pool.blocks[1], 8,
    )
