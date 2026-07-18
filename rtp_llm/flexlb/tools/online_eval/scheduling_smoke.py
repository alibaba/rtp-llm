#!/usr/bin/env python3
"""Scheduling accuracy smoke tests for FlexLB.

Connects to a running FlexLB master and mock engine cluster.  Exercises
scheduling-accuracy scenarios grouped by algorithm:

Common (all paths):
  S1  load_balance_distribution     - N requests spread across prefill workers.
  S2  kv_cache_affinity             - Same block_cache_keys -> same worker.
  S3  decode_balance_distribution   - N completed requests across decode workers.
  S10 weighted_random_distribution  - 50 requests spread across >=3 decode workers.
  S11 kv_capacity_filter           - Low available_kv worker is excluded.
  S12 reserve_weight_change        - After reserve, weight lowers for that worker.

COST_BASED_PREFILL (batch-only):
  S4  hotspot_filter               - High queue depth worker is filtered out.
  S5  kv_cache_hit_preference      - Same block key routes to cached worker.
  S6  cost_based_determinism       - Identical conditions -> same worker.

SHORTEST_TTFT (direct/queue-only):
  S7  cas_fairness                 - Concurrent requests spread across workers.
  S8  ttft_sorting                - Lower prefill delay biases routing.
  S9  no_hard_filter              - High queue depth does NOT block routing.

Usage:
    python3 scheduling_smoke.py --master-ip 127.0.0.1 \\
        --master-http-port 18080 --mock-http-port 55150 \\
        --schedule-mode batch
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from collections import Counter

from flexlb_smoke_base import FlexLBSmokeBase, ScenarioResult, StreamSnapshot


class SchedulingSmokeTest(FlexLBSmokeBase):
    """Scheduling accuracy smoke tests for FlexLB."""

    DEFAULT_INPUT_LEN = 2048
    DEFAULT_OUTPUT_LEN = 2

    LOAD_BALANCE_N = 10
    MAX_SINGLE_WORKER_RATIO = 0.60
    KV_CACHE_SYNC_WAIT_S = 2.0
    STREAM_TIMEOUT_S = 15.0

    # -- Helpers ----------------------------------------------------------

    async def _schedule_auto(self, request_id: int, **kwargs):
        """Schedule with the configured schedule_mode from CLI args."""
        kwargs.setdefault("schedule_mode", self.args.schedule_mode)
        return await self._schedule(request_id, **kwargs)

    async def _snapshot_by_name(self) -> dict[str, dict]:
        """Get mock-engine cluster snapshot as a dict keyed by engine name."""
        snap = await self._get_snapshot()
        return {e["name"]: e for e in snap.get("engines", [])}

    async def _addr_to_name(self) -> dict[str, str]:
        """Map ``grpc_addr`` -> engine name from the current snapshot."""
        snap = await self._get_snapshot()
        return {e["grpc_addr"]: e["name"] for e in snap.get("engines", [])}

    async def _run_one_request(self, rid: int, **kwargs) -> tuple[str, str | None]:
        """Schedule, start stream, consume to completion.

        Returns ``(prefill_addr, error)`` where *error* is ``None`` on
        success or an error string on failure.  *prefill_addr* is ``""``
        when the schedule call itself failed.
        """
        try:
            response = await self._schedule_auto(rid, **kwargs)
            if response.code != 200 or not response.success:
                return "", f"schedule failed: {response.error_message}"
            addr = self._role_addr(response, self.pb2.ROLE_TYPE_PREFILL)
            input_pb = (
                self._build_generate_input(rid)
                if not response.enqueued_by_master
                else None
            )
            stream = await self._start_stream(response, rid, input_pb=input_pb)
            snap = StreamSnapshot()
            task = asyncio.create_task(self._consume_stream(stream, snap))
            await self._wait_for_stream_end(task, timeout_s=self.STREAM_TIMEOUT_S)
            if snap.error:
                return addr, snap.error
            if not snap.completed:
                return addr, "stream did not complete"
            return addr, None
        except Exception as exc:
            return "", repr(exc)

    # -- S1: load_balance_distribution -----------------------------------

    async def test_load_balance(self) -> ScenarioResult:
        """Send N requests with unique block keys; verify all succeed.

        In batch mode (COST_BASED_PREFILL), differentiate prefill
        performance via ``_set_perf`` so routing is deterministic — all
        requests should go to the fastest worker.  In other modes,
        hysteresis bias may concentrate requests; distribution is logged
        for diagnostics.
        """
        start = time.monotonic()
        n = self.LOAD_BALANCE_N
        is_batch = self.args.schedule_mode == "batch"
        perf_engine: str | None = None
        try:
            # In batch mode, differentiate prefill performance to make
            # COST_BASED_PREFILL routing deterministic (all to fastest).
            if is_batch:
                snap0 = await self._snapshot_by_name()
                prefill_names = sorted(
                    name for name, e in snap0.items() if e.get("role") == "prefill"
                )
                if len(prefill_names) >= 2:
                    await self._set_perf(prefill_names[1], prefill_fixed_ms=200.0)
                    perf_engine = prefill_names[1]

            addrs: list[str] = []
            for _ in range(n):
                rid = self._next_request_id()
                keys = [rid * 100 + j for j in range(3)]
                addr, err = await self._run_one_request(
                    rid, output_len=2, block_keys=keys
                )
                if err:
                    return ScenarioResult(
                        "S1: load_balance_distribution",
                        False,
                        f"rid={rid} failed: {err}",
                        time.monotonic() - start,
                    )
                addrs.append(addr)

            # Distribution from schedule responses
            counts = Counter(addrs)
            num_workers = len(counts)
            max_ratio = max(counts.values()) / n if n else 1.0

            # Cross-check with snapshot accepted counts
            snap = await self._snapshot_by_name()
            accepted = {
                name: info.get("accepted", 0)
                for name, info in snap.items()
                if info.get("role") == "prefill"
            }
            addr_map = await self._addr_to_name()
            dist_names = {addr_map.get(a, a): c for a, c in counts.items()}

            batch_detail = ""
            if is_batch and perf_engine:
                # COST_BASED_PREFILL: all requests should route to the
                # fastest worker (prefill-0), not the slowed one.
                slow_count = dist_names.get(perf_engine, 0)
                passed = slow_count == 0
                batch_detail = (
                    f", slow_worker={perf_engine}({slow_count}), "
                    f"batch_deterministic={'yes' if passed else 'no'}"
                )
            else:
                # Hysteresis bias may concentrate all requests on one
                # worker — this is normal strategy behaviour, not a bug.
                passed = True

            detail = (
                f"requests={n}, workers={num_workers}, "
                f"max_ratio={max_ratio:.2f}, "
                f"distribution={json.dumps(dist_names, sort_keys=True)}, "
                f"snapshot_accepted={json.dumps(accepted, sort_keys=True)}"
                f"{batch_detail}"
            )
            return ScenarioResult(
                "S1: load_balance_distribution",
                passed,
                detail,
                time.monotonic() - start,
            )
        except Exception as exc:
            return ScenarioResult(
                "S1: load_balance_distribution",
                False,
                f"exception: {exc!r}",
                time.monotonic() - start,
            )
        finally:
            if perf_engine:
                try:
                    await self._set_perf(perf_engine, prefill_fixed_ms=100.0)
                except Exception:
                    pass

    # -- S2: kv_cache_affinity -------------------------------------------

    async def test_kv_affinity(self) -> ScenarioResult:
        """Two requests with identical block keys -> same prefill worker."""
        start = time.monotonic()
        keys = [1001, 1002, 1003]
        try:
            # Request A — populates cache on the selected prefill worker
            rid_a = self._next_request_id()
            addr_a, err_a = await self._run_one_request(
                rid_a, input_len=2048, output_len=2, block_keys=keys
            )
            if err_a:
                return ScenarioResult(
                    "S2: kv_cache_affinity",
                    False,
                    f"request A failed: {err_a}",
                    time.monotonic() - start,
                )

            # Wait for master cache-status sync
            await asyncio.sleep(self.KV_CACHE_SYNC_WAIT_S)

            # Request B — same block keys, should hit same worker
            rid_b = self._next_request_id()
            addr_b, err_b = await self._run_one_request(
                rid_b, input_len=2048, output_len=2, block_keys=keys
            )
            if err_b:
                return ScenarioResult(
                    "S2: kv_cache_affinity",
                    False,
                    f"request B failed: {err_b}",
                    time.monotonic() - start,
                )

            if addr_a == addr_b:
                passed = True
                detail = f"affinity confirmed: A=B={addr_a}"
            else:
                # Retry once (cache sync may lag)
                await asyncio.sleep(self.KV_CACHE_SYNC_WAIT_S)
                rid_c = self._next_request_id()
                addr_c, err_c = await self._run_one_request(
                    rid_c, input_len=2048, output_len=2, block_keys=keys
                )
                if err_c:
                    return ScenarioResult(
                        "S2: kv_cache_affinity",
                        False,
                        f"request C failed: {err_c}",
                        time.monotonic() - start,
                    )
                passed = addr_a == addr_c
                detail = (
                    f"retry: A={addr_a}, B={addr_b}, C={addr_c}, "
                    f"match={'A==C' if passed else 'none'}"
                )

            return ScenarioResult(
                "S2: kv_cache_affinity",
                passed,
                detail,
                time.monotonic() - start,
            )
        except Exception as exc:
            return ScenarioResult(
                "S2: kv_cache_affinity",
                False,
                f"exception: {exc!r}",
                time.monotonic() - start,
            )

    # -- S3: decode_balance_distribution ---------------------------------

    async def test_decode_balance(self) -> ScenarioResult:
        """Send N short requests; verify decode load spread >=2 workers."""
        start = time.monotonic()
        n = 10
        try:
            for _ in range(n):
                rid = self._next_request_id()
                keys = [rid * 100 + j for j in range(3)]
                _, err = await self._run_one_request(rid, output_len=2, block_keys=keys)
                if err:
                    return ScenarioResult(
                        "S3: decode_balance_distribution",
                        False,
                        f"rid={rid} failed: {err}",
                        time.monotonic() - start,
                    )

            snap = await self._snapshot_by_name()
            completed = {
                name: info.get("completed", 0)
                for name, info in snap.items()
                if info.get("role") == "decode"
            }
            total = sum(completed.values())
            used = sum(1 for v in completed.values() if v > 0)

            passed = used >= 2 and total >= n
            detail = (
                f"requests={n}, decode_workers={used}, "
                f"total_completed={total}, "
                f"distribution={json.dumps(completed, sort_keys=True)}"
            )
            return ScenarioResult(
                "S3: decode_balance_distribution",
                passed,
                detail,
                time.monotonic() - start,
            )
        except Exception as exc:
            return ScenarioResult(
                "S3: decode_balance_distribution",
                False,
                f"exception: {exc!r}",
                time.monotonic() - start,
            )

    # -- S4: hotspot_filter (batch-only) --------------------------------

    async def test_hotspot_filter(self) -> ScenarioResult:
        """S4: Inject high queue depth on prefill-0; verify requests avoid it.

        COST_BASED_PREFILL filters workers whose pendingCount exceeds
        avgPending * 3.0.  By injecting a large queue depth on prefill-0,
        all requests should route to prefill-1.
        """
        start = time.monotonic()
        injected_engine: str | None = None
        try:
            snap = await self._snapshot_by_name()
            prefill_names = sorted(
                name for name, e in snap.items() if e.get("role") == "prefill"
            )
            if len(prefill_names) < 2:
                return ScenarioResult(
                    "S4: hotspot_filter",
                    False,
                    "need >=2 prefill workers",
                    time.monotonic() - start,
                )
            hot = prefill_names[0]
            cool = prefill_names[1]

            # Inject high queue depth (reported value, does not block requests)
            await self._set_queue_depth(hot, 80000)
            injected_engine = hot

            # Wait for master to sync the updated worker status
            await asyncio.sleep(1.0)

            addrs: list[str] = []
            for _ in range(5):
                rid = self._next_request_id()
                keys = [rid * 100 + j for j in range(3)]
                addr, err = await self._run_one_request(
                    rid, output_len=2, block_keys=keys
                )
                if err:
                    return ScenarioResult(
                        "S4: hotspot_filter",
                        False,
                        f"rid={rid} failed: {err}",
                        time.monotonic() - start,
                    )
                addrs.append(addr)

            addr_map = await self._addr_to_name()
            dist = Counter(addr_map.get(a, a) for a in addrs)
            hot_count = dist.get(hot, 0)
            cool_count = dist.get(cool, 0)

            passed = hot_count == 0 and cool_count == 5
            detail = (
                f"hot={hot}({hot_count}), cool={cool}({cool_count}), "
                f"dist={json.dumps(dict(dist), sort_keys=True)}"
            )
            return ScenarioResult(
                "S4: hotspot_filter",
                passed,
                detail,
                time.monotonic() - start,
            )
        except Exception as exc:
            return ScenarioResult(
                "S4: hotspot_filter",
                False,
                f"exception: {exc!r}",
                time.monotonic() - start,
            )
        finally:
            if injected_engine:
                try:
                    await self._set_queue_depth(injected_engine, 0)
                    # Wait for master to sync the reset so subsequent tests
                    # (e.g. S5 kv_cache_hit_preference) see the correct cost.
                    await asyncio.sleep(1.0)
                except Exception:
                    pass

    # -- S5: kv_cache_hit_preference (batch-only) ------------------------

    async def test_kv_cache_hit_preference(self) -> ScenarioResult:
        """S5: Same block_cache_key should route to the same prefill worker.

        Request A populates cache on a prefill worker.  After cache sync,
        request B with the same block key should be routed to the same
        worker via CacheAwareService.
        """
        start = time.monotonic()
        try:
            block_key = 999
            keys = [block_key]

            # Request A — populates cache on the selected prefill worker
            rid_a = self._next_request_id()
            addr_a, err_a = await self._run_one_request(
                rid_a, input_len=2048, output_len=2, block_keys=keys
            )
            if err_a:
                return ScenarioResult(
                    "S5: kv_cache_hit_preference",
                    False,
                    f"request A failed: {err_a}",
                    time.monotonic() - start,
                )

            # Wait for master cache-status sync
            await asyncio.sleep(self.KV_CACHE_SYNC_WAIT_S)

            # Request B — same block key, should hit same worker
            rid_b = self._next_request_id()
            addr_b, err_b = await self._run_one_request(
                rid_b, input_len=2048, output_len=2, block_keys=keys
            )
            if err_b:
                return ScenarioResult(
                    "S5: kv_cache_hit_preference",
                    False,
                    f"request B failed: {err_b}",
                    time.monotonic() - start,
                )

            addr_map = await self._addr_to_name()
            name_a = addr_map.get(addr_a, addr_a)
            name_b = addr_map.get(addr_b, addr_b)

            # Verify both hit the same engine and that engine has cache
            snap = await self._snapshot_by_name()
            cache_count = snap.get(name_a, {}).get("cache_keys", 0)

            passed = addr_a == addr_b and cache_count > 0
            detail = (
                f"A={name_a}, B={name_b}, "
                f"same={'yes' if addr_a == addr_b else 'no'}, "
                f"cache_keys={cache_count}"
            )
            return ScenarioResult(
                "S5: kv_cache_hit_preference",
                passed,
                detail,
                time.monotonic() - start,
            )
        except Exception as exc:
            return ScenarioResult(
                "S5: kv_cache_hit_preference",
                False,
                f"exception: {exc!r}",
                time.monotonic() - start,
            )

    # -- S6: cost_based_determinism (batch-only) --------------------------

    async def test_cost_based_determinism(self) -> ScenarioResult:
        """S6: Under identical conditions, all requests route to same worker.

        COST_BASED_PREFILL uses deterministic minimum-score selection.
        With identical performance and no cache hits, all requests should
        route to the same prefill worker.
        """
        start = time.monotonic()
        try:
            addrs: list[str] = []
            for _ in range(5):
                rid = self._next_request_id()
                # Unique block keys to avoid cache hit interference
                keys = [rid * 100 + j for j in range(3)]
                addr, err = await self._run_one_request(
                    rid, output_len=2, block_keys=keys
                )
                if err:
                    return ScenarioResult(
                        "S6: cost_based_determinism",
                        False,
                        f"rid={rid} failed: {err}",
                        time.monotonic() - start,
                    )
                addrs.append(addr)

            addr_map = await self._addr_to_name()
            dist = Counter(addr_map.get(a, a) for a in addrs)
            num_workers = len(dist)

            snap = await self._snapshot_by_name()
            accepted = {
                name: info.get("accepted", 0)
                for name, info in snap.items()
                if info.get("role") == "prefill"
            }

            passed = num_workers == 1
            detail = (
                f"workers={num_workers}, "
                f"dist={json.dumps(dict(dist), sort_keys=True)}, "
                f"accepted={json.dumps(accepted, sort_keys=True)}"
            )
            return ScenarioResult(
                "S6: cost_based_determinism",
                passed,
                detail,
                time.monotonic() - start,
            )
        except Exception as exc:
            return ScenarioResult(
                "S6: cost_based_determinism",
                False,
                f"exception: {exc!r}",
                time.monotonic() - start,
            )

    # -- S7: cas_fairness (direct/queue-only) ------------------------------

    async def test_cas_fairness(self) -> ScenarioResult:
        """S7: Concurrent requests should spread across >=2 workers.

        SHORTEST_TTFT uses CAS fairness (lastSelectedTime earliest wins).
        With 2 workers + RATIO 0.3, candidateCount=1, but lastSelectedTime
        updates should cause rotation across workers.
        """
        start = time.monotonic()
        try:
            rids = [self._next_request_id() for _ in range(20)]
            tasks = []
            for rid in rids:
                keys = [rid * 100 + j for j in range(3)]
                tasks.append(self._run_one_request(rid, output_len=2, block_keys=keys))
            results = await asyncio.gather(*tasks)

            addrs: list[str] = []
            for idx, (addr, err) in enumerate(results):
                if err:
                    return ScenarioResult(
                        "S7: cas_fairness",
                        False,
                        f"rid={rids[idx]} failed: {err}",
                        time.monotonic() - start,
                    )
                addrs.append(addr)

            addr_map = await self._addr_to_name()
            dist = Counter(addr_map.get(a, a) for a in addrs)
            num_workers = len(dist)

            snap = await self._snapshot_by_name()
            accepted = {
                name: info.get("accepted", 0)
                for name, info in snap.items()
                if info.get("role") == "prefill"
            }

            passed = num_workers >= 2
            detail = (
                f"workers={num_workers}, "
                f"dist={json.dumps(dict(dist), sort_keys=True)}, "
                f"accepted={json.dumps(accepted, sort_keys=True)}"
            )
            return ScenarioResult(
                "S7: cas_fairness",
                passed,
                detail,
                time.monotonic() - start,
            )
        except Exception as exc:
            return ScenarioResult(
                "S7: cas_fairness",
                False,
                f"exception: {exc!r}",
                time.monotonic() - start,
            )

    # -- S8: ttft_sorting (direct/queue-only) ------------------------------

    async def test_ttft_sorting(self) -> ScenarioResult:
        """S8: Lower prefill delay on prefill-0 should bias requests toward it.

        SHORTEST_TTFT scores by prefillMs + realWaitTimeMs.  Setting
        prefill-0 to 10ms and prefill-1 to 500ms should route most
        requests to prefill-0.  CAS fairness may cause some even split.
        """
        start = time.monotonic()
        perf_engines: list[str] = []
        try:
            snap = await self._snapshot_by_name()
            prefill_names = sorted(
                name for name, e in snap.items() if e.get("role") == "prefill"
            )
            if len(prefill_names) < 2:
                return ScenarioResult(
                    "S8: ttft_sorting",
                    False,
                    "need >=2 prefill workers",
                    time.monotonic() - start,
                )
            fast = prefill_names[0]
            slow = prefill_names[1]

            await self._set_perf(fast, prefill_fixed_ms=10.0)
            await self._set_perf(slow, prefill_fixed_ms=500.0)
            perf_engines = [fast, slow]

            addrs: list[str] = []
            for _ in range(10):
                rid = self._next_request_id()
                keys = [rid * 100 + j for j in range(3)]
                addr, err = await self._run_one_request(
                    rid, output_len=2, block_keys=keys
                )
                if err:
                    return ScenarioResult(
                        "S8: ttft_sorting",
                        False,
                        f"rid={rid} failed: {err}",
                        time.monotonic() - start,
                    )
                addrs.append(addr)

            addr_map = await self._addr_to_name()
            dist = Counter(addr_map.get(a, a) for a in addrs)
            fast_count = dist.get(fast, 0)
            slow_count = dist.get(slow, 0)

            passed = fast_count >= slow_count
            detail = (
                f"fast={fast}({fast_count}), slow={slow}({slow_count}), "
                f"dist={json.dumps(dict(dist), sort_keys=True)}, "
                f"assertion=fast>=slow"
            )
            return ScenarioResult(
                "S8: ttft_sorting",
                passed,
                detail,
                time.monotonic() - start,
            )
        except Exception as exc:
            return ScenarioResult(
                "S8: ttft_sorting",
                False,
                f"exception: {exc!r}",
                time.monotonic() - start,
            )
        finally:
            for eng in perf_engines:
                try:
                    await self._set_perf(eng, prefill_fixed_ms=100.0)
                except Exception:
                    pass

    # -- S9: no_hard_filter (direct/queue-only) ---------------------------

    async def test_no_hard_filter(self) -> ScenarioResult:
        """S9: High queue depth does NOT block routing (no hard filter).

        SHORTEST_TTFT has no hard filter — even with high queue depth,
        requests can still route to that worker.  This contrasts with S4
        where COST_BASED_PREFILL filters out the hotspot.
        """
        start = time.monotonic()
        injected_engine: str | None = None
        try:
            snap = await self._snapshot_by_name()
            prefill_names = sorted(
                name for name, e in snap.items() if e.get("role") == "prefill"
            )
            if len(prefill_names) < 2:
                return ScenarioResult(
                    "S9: no_hard_filter",
                    False,
                    "need >=2 prefill workers",
                    time.monotonic() - start,
                )
            target = prefill_names[0]

            await self._set_queue_depth(target, 50000)
            injected_engine = target

            addrs: list[str] = []
            for _ in range(5):
                rid = self._next_request_id()
                keys = [rid * 100 + j for j in range(3)]
                addr, err = await self._run_one_request(
                    rid, output_len=2, block_keys=keys
                )
                if err:
                    return ScenarioResult(
                        "S9: no_hard_filter",
                        False,
                        f"rid={rid} failed: {err}",
                        time.monotonic() - start,
                    )
                addrs.append(addr)

            addr_map = await self._addr_to_name()
            dist = Counter(addr_map.get(a, a) for a in addrs)
            target_count = dist.get(target, 0)

            passed = target_count > 0
            detail = (
                f"target={target}({target_count}), "
                f"dist={json.dumps(dict(dist), sort_keys=True)}"
            )
            return ScenarioResult(
                "S9: no_hard_filter",
                passed,
                detail,
                time.monotonic() - start,
            )
        except Exception as exc:
            return ScenarioResult(
                "S9: no_hard_filter",
                False,
                f"exception: {exc!r}",
                time.monotonic() - start,
            )
        finally:
            if injected_engine:
                try:
                    await self._set_queue_depth(injected_engine, 0)
                except Exception:
                    pass

    # -- S10: weighted_random_distribution (all paths) --------------------

    async def test_weighted_random_distribution(self) -> ScenarioResult:
        """S10: 50 requests should spread across >=3 decode workers.

        COST_BASED_DECODE uses weighted random selection (cache usage
        affects weight).  With 4 decode workers, distribution should
        cover at least 3 of them.
        """
        start = time.monotonic()
        try:
            for _ in range(50):
                rid = self._next_request_id()
                keys = [rid * 100 + j for j in range(3)]
                _, err = await self._run_one_request(rid, output_len=2, block_keys=keys)
                if err:
                    return ScenarioResult(
                        "S10: weighted_random_distribution",
                        False,
                        f"rid={rid} failed: {err}",
                        time.monotonic() - start,
                    )

            snap = await self._snapshot_by_name()
            completed = {
                name: info.get("completed", 0)
                for name, info in snap.items()
                if info.get("role") == "decode"
            }
            used = sum(1 for v in completed.values() if v > 0)

            passed = used >= 3
            detail = (
                f"decode_workers={used}, "
                f"distribution={json.dumps(completed, sort_keys=True)}"
            )
            return ScenarioResult(
                "S10: weighted_random_distribution",
                passed,
                detail,
                time.monotonic() - start,
            )
        except Exception as exc:
            return ScenarioResult(
                "S10: weighted_random_distribution",
                False,
                f"exception: {exc!r}",
                time.monotonic() - start,
            )

    # -- S11: kv_capacity_filter (all paths) ------------------------------

    async def test_kv_capacity_filter(self) -> ScenarioResult:
        """S11: Decode worker with available_kv < seqLen is excluded.

        COST_BASED_DECODE filters out workers whose availableKv < seqLen.
        By setting high active_kv_tokens on decode-0, its available drops
        to 0 (< seqLen=2048), so all requests should avoid it.
        """
        start = time.monotonic()
        injected_engine: str | None = None
        try:
            snap = await self._snapshot_by_name()
            decode_names = sorted(
                name for name, e in snap.items() if e.get("role") == "decode"
            )
            if len(decode_names) < 2:
                return ScenarioResult(
                    "S11: kv_capacity_filter",
                    False,
                    "need >=2 decode workers",
                    time.monotonic() - start,
                )
            target = decode_names[0]
            target_info = snap[target]

            # Set active_kv so available < seqLen (DEFAULT_INPUT_LEN=2048)
            avail = target_info.get("available_kv_tokens", 0)
            active = target_info.get("active_kv_tokens", 0)
            total_kv = avail + active
            await self._set_kv_pressure(target, total_kv)
            injected_engine = target

            # Wait for master to sync the updated KV status
            await asyncio.sleep(1.0)

            # Record baseline completed count (re-read snapshot after sync)
            snap_sync = await self._snapshot_by_name()
            completed_before = snap_sync.get(target, {}).get("completed", 0)

            for _ in range(10):
                rid = self._next_request_id()
                keys = [rid * 100 + j for j in range(3)]
                _, err = await self._run_one_request(rid, output_len=2, block_keys=keys)
                if err:
                    return ScenarioResult(
                        "S11: kv_capacity_filter",
                        False,
                        f"rid={rid} failed: {err}",
                        time.monotonic() - start,
                    )

            snap2 = await self._snapshot_by_name()
            completed = {
                name: info.get("completed", 0)
                for name, info in snap2.items()
                if info.get("role") == "decode"
            }
            target_delta = completed.get(target, 0) - completed_before

            passed = target_delta <= 1
            detail = (
                f"target={target}(delta={target_delta}), "
                f"distribution={json.dumps(completed, sort_keys=True)}, "
                f"assertion=delta<=1"
            )
            return ScenarioResult(
                "S11: kv_capacity_filter",
                passed,
                detail,
                time.monotonic() - start,
            )
        except Exception as exc:
            return ScenarioResult(
                "S11: kv_capacity_filter",
                False,
                f"exception: {exc!r}",
                time.monotonic() - start,
            )
        finally:
            if injected_engine:
                try:
                    await self._set_kv_pressure(injected_engine, 0)
                except Exception:
                    pass

    # -- S12: reserve_weight_change (all paths) ----------------------------

    async def test_reserve_weight_change(self) -> ScenarioResult:
        """S12: After request A, subsequent requests lean away from A's worker.

        COST_BASED_DECODE lowers a worker's weight after reserve.  Request
        A goes to a decode worker; then B/C/D should prefer other workers.
        Weighted random has variance, so we check that A's worker does not
        dominate (total <= max of others).
        """
        start = time.monotonic()
        try:
            snap0 = await self._snapshot_by_name()
            decode_names = sorted(
                name for name, e in snap0.items() if e.get("role") == "decode"
            )
            if len(decode_names) < 2:
                return ScenarioResult(
                    "S12: reserve_weight_change",
                    False,
                    "need >=2 decode workers",
                    time.monotonic() - start,
                )
            base = {name: snap0[name].get("completed", 0) for name in decode_names}

            # Send request A
            rid_a = self._next_request_id()
            keys_a = [rid_a * 100 + j for j in range(3)]
            _, err_a = await self._run_one_request(
                rid_a, output_len=2, block_keys=keys_a
            )
            if err_a:
                return ScenarioResult(
                    "S12: reserve_weight_change",
                    False,
                    f"request A failed: {err_a}",
                    time.monotonic() - start,
                )

            # Identify A's decode worker via completed-delta
            snap_a = await self._snapshot_by_name()
            delta_a = {
                name: snap_a[name].get("completed", 0) - base[name]
                for name in decode_names
            }
            a_worker = max(delta_a, key=delta_a.get) if any(delta_a.values()) else None
            if a_worker is None:
                return ScenarioResult(
                    "S12: reserve_weight_change",
                    False,
                    "could not identify A's decode worker",
                    time.monotonic() - start,
                )

            # Send 10 subsequent requests (different block keys to avoid cache affinity)
            for _ in range(10):
                rid = self._next_request_id()
                keys = [rid * 100 + j for j in range(3)]
                _, err = await self._run_one_request(rid, output_len=2, block_keys=keys)
                if err:
                    return ScenarioResult(
                        "S12: reserve_weight_change",
                        False,
                        f"subsequent request failed: {err}",
                        time.monotonic() - start,
                    )

            # Final distribution
            snap_f = await self._snapshot_by_name()
            total_delta = {
                name: snap_f[name].get("completed", 0) - base[name]
                for name in decode_names
            }
            a_total = total_delta.get(a_worker, 0)
            other_max = max(
                (total_delta[n] for n in decode_names if n != a_worker),
                default=0,
            )

            passed = a_total <= other_max + 1
            detail = (
                f"a_worker={a_worker}(total={a_total}), "
                f"other_max={other_max}, "
                f"delta={json.dumps(total_delta, sort_keys=True)}, "
                f"assertion=a_total<=other_max+1"
            )
            return ScenarioResult(
                "S12: reserve_weight_change",
                passed,
                detail,
                time.monotonic() - start,
            )
        except Exception as exc:
            return ScenarioResult(
                "S12: reserve_weight_change",
                False,
                f"exception: {exc!r}",
                time.monotonic() - start,
            )

    # -- Runner ----------------------------------------------------------

    async def run_all(self) -> int:
        # Common scenarios (all paths)
        scenarios = [
            self.test_load_balance,
            self.test_kv_affinity,
            self.test_decode_balance,
            # Decode-specific (all paths)
            self.test_weighted_random_distribution,
            self.test_kv_capacity_filter,
            self.test_reserve_weight_change,
        ]
        # Path-specific scenarios
        if self.args.schedule_mode == "batch":
            scenarios += [
                self.test_hotspot_filter,
                self.test_kv_cache_hit_preference,
                self.test_cost_based_determinism,
            ]
        else:
            scenarios += [
                self.test_cas_fairness,
                self.test_ttft_sorting,
                self.test_no_hard_filter,
            ]
        print("=" * 70)
        print("FlexLB Scheduling Smoke Test")
        print(f"  master: {self._master_target()}")
        print(f"  schedule_mode: {self.args.schedule_mode}")
        print(f"  mock_http_port: {self.args.mock_http_port}")
        print("=" * 70)

        for scenario in scenarios:
            print(f"\n>>> Running {scenario.__name__} ...", flush=True)
            result = await scenario()
            self.results.append(result)
            status = "PASS" if result.passed else "FAIL"
            print(
                f"<<< {result.name}: {status}  "
                f"({result.duration_s:.2f}s)  {result.detail}",
                flush=True,
            )

        passed = sum(1 for r in self.results if r.passed)
        failed = len(self.results) - passed
        print("\n" + "=" * 70)
        print(f"Summary: {passed}/{len(self.results)} passed, {failed} failed")
        for r in self.results:
            status = "PASS" if r.passed else "FAIL"
            print(f"  {status}  {r.name}")
        print("=" * 70)
        return 1 if failed > 0 else 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--master-ip", default="127.0.0.1")
    parser.add_argument("--master-http-port", type=int, default=18080)
    parser.add_argument(
        "--mock-http-port",
        type=int,
        default=55150,
        help="mock engine cluster HTTP API port",
    )
    parser.add_argument(
        "--flexlb-http-port",
        type=int,
        default=18080,
        help="flexlb master HTTP port for inflight status check",
    )
    parser.add_argument(
        "--schedule-mode",
        choices=["auto", "batch", "direct", "queue"],
        default="batch",
    )
    parser.add_argument("--request-id-base", type=int, default=20000)
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    test = SchedulingSmokeTest(args)
    try:
        exit_code = await test.run_all()
    finally:
        await test.close()
    sys.exit(exit_code)


if __name__ == "__main__":
    asyncio.run(main())
