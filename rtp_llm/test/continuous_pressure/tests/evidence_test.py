import copy
import dataclasses
import json
import tempfile
import tracemalloc
import unittest
from pathlib import Path
from types import SimpleNamespace as NS

from rtp_llm.test.continuous_pressure import results
from rtp_llm.test.continuous_pressure.config import resolve_pressure_config
from rtp_llm.test.continuous_pressure.diagnostics import (
    GroupSample,
    LocalStoreSample,
    MockCacheSample,
    MockMemorySample,
    MockRemoteWriteSample,
    PoolSample,
    PressureCoverage,
    RankSample,
    decode_pressure_response,
    group_valid,
    raw_pressure_record,
    select_targets,
    validate_inventory,
)
from rtp_llm.test.continuous_pressure.ports import JsonlSink
from rtp_llm.test.continuous_pressure.results import (
    PressureEvidence,
    PressureWriter,
    RequestWindow,
)


def settings(**changes):
    values = dict(
        seed_timeout_s=30,
        fill_timeout_s=120,
        fill_max_requests=1000,
        fill_no_progress_s=30,
        fill_input_tokens=4096,
        gpu_tolerance_blocks=0,
        activity_timeout_s=30,
        drain_timeout_s=60,
        capture_timeout_s=120,
    )
    values.update(changes)
    return values


def configured(**changes):
    return resolve_pressure_config(settings(**changes))


def rank(rank_id=0, groups=None, pools=None, instance="instance"):
    return RankSample(
        rank_id,
        instance,
        1,
        1.0,
        1.0,
        1.1,
        tuple(
            pools
            or (
                PoolSample("gpu", 100, 90, 0, 0.8, 0.9),
                PoolSample("host", 100, 30, 0, 0.8, 0.9, 25),
            )
        ),
        tuple(
            groups
            if groups is not None
            else (GroupSample(0, "FULL", ("gpu",), "host", 1, 1),)
        ),
    )


class PB(NS):
    def HasField(self, name):
        return name in self.__dict__ and self.__dict__[name] is not None

    def SerializeToString(self):
        return b"complete raw wire evidence including steps/activity"


class RankPB(PB):
    DESCRIPTOR = NS(
        fields_by_name={
            "mock_memory",
            "mock_cache",
            "mock_remote_write",
            "remote_runtime",
        }
    )


def wire():
    gpu = PB(
        pool_id="gpu",
        tier=1,
        total_blocks=100,
        free_blocks=0,
        available_blocks=0,
        used_blocks=90,
        pending_release_blocks=0,
        published_host_blocks=0,
    )
    host = PB(
        pool_id="host",
        tier=2,
        total_blocks=100,
        free_blocks=0,
        available_blocks=0,
        used_blocks=30,
        pending_release_blocks=0,
        published_host_blocks=25,
    )
    group = PB(
        group_id=0,
        group_type=1,
        device_pool_ids=["gpu"],
        host_pool_id=PB(value="host"),
        device_watermark=PB(low_ratio=0.8, high_ratio=0.9),
        host_watermark=PB(low_ratio=0.8, high_ratio=0.9),
        watermark_trigger_count=1,
    )
    transfer = dict(
        group_id=0, operation=2, source_tier=1, target_tier=2, published_blocks=7
    )
    return PB(
        schema_version=1,
        requested_ranks=[0],
        ranks=[
            RankPB(
                world_rank=0,
                service_instance="instance",
                sequence=1,
                sampled_monotonic_ns=1000000000,
                collect_duration_ns=0,
                error="",
                pools=[gpu, host],
                groups=[group],
                transfers=[
                    PB(**transfer, pool_id="gpu", pool_tier=1),
                    PB(**transfer, pool_id="host", pool_tier=2),
                ],
            )
        ],
    )


def shared_wire(t=1, baseline=False):
    message = wire()
    message.requested_ranks = [0, 1]
    message.ranks.append(copy.deepcopy(message.ranks[0]))
    for i, item in enumerate(message.ranks):
        item.world_rank = i
        item.service_instance = f"instance-{i}"
        item.sequence = t + 2
        item.sampled_monotonic_ns = int((t + i * 100) * 1e9)
        item.ownership = PB(
            mode=2,
            owner_world_rank=PB(value=0),
            participant_world_ranks=[0, 1],
            publication_scope=2,
        )
        item.groups[0].group_tags = ["gpu-tag"]
        item.groups[0].block_tokens = 512
        item.groups[0].payload_bytes = item.groups[0].storage_bytes = 4096
        item.groups[0].crc_enabled = False
        for pool in item.pools:
            pool.pool_name = "gpu-tag" if pool.tier == 1 else "host-tag"
            pool.block_size_bytes = 4096
        if baseline or i:
            item.groups[0].watermark_trigger_count = 0
            for transfer in item.transfers:
                transfer.published_blocks = 0
        if i:
            for pool in item.pools:
                pool.used_blocks = pool.published_host_blocks = 0
    return message


class SharedOwnershipTest(unittest.TestCase):
    def test_first_sparse_transfer_endpoint_is_progress_not_reset(self):
        from rtp_llm.test.continuous_pressure.diagnostics import PressureCoverage

        initial = shared_wire(0, True)
        initial.ranks[0].transfers = []
        initial.ranks[1].transfers = []
        baseline = decode_pressure_response(initial, 0, 0, (0, 1)).samples
        evaluator = PressureCoverage(baseline, configured())
        for t in range(1, 4):
            current = shared_wire(t)
            current.ranks[1].transfers = []
            evaluator.add(decode_pressure_response(current, t, t, (0, 1)).samples, t)
        self.assertTrue(evaluator.assess(1, 3).valid)
        vanished = shared_wire(4)
        vanished.ranks[0].transfers = []
        vanished.ranks[1].transfers = []
        evaluator.add(decode_pressure_response(vanished, 4, 4, (0, 1)).samples, 4)
        self.assertFalse(evaluator.epoch_valid)
        self.assertIn("counter_reset", evaluator.reasons)

    def test_rejects_missing_scope_owner_and_incompatible_layout(self):
        mutations = [
            lambda m: setattr(m.ranks[0].ownership, "owner_world_rank", None),
            lambda m: setattr(m.ranks[0].ownership, "owner_world_rank", PB(value=9)),
            lambda m: setattr(m.ranks[0].ownership, "publication_scope", 1),
            lambda m: setattr(m.ranks[0].ownership, "participant_world_ranks", [0]),
            lambda m: setattr(m.ranks[1], "ownership", None),
            lambda m: setattr(m.ranks[1].ownership, "mode", 0),
            lambda m: setattr(m.ranks[1].groups[0], "group_tags", ["other"]),
            lambda m: setattr(m.ranks[1].groups[0], "group_type", 2),
            lambda m: setattr(m.ranks[1].groups[0], "block_tokens", 256),
            lambda m: setattr(m.ranks[1].groups[0], "payload_bytes", 2048),
            lambda m: setattr(m.ranks[1].groups[0], "storage_bytes", 8192),
            lambda m: setattr(m.ranks[1].groups[0], "crc_enabled", True),
            lambda m: setattr(m.ranks[1].pools[0], "pool_name", "other"),
            lambda m: setattr(m.ranks[1].pools[0], "block_size_bytes", 8192),
            lambda m: setattr(m.ranks[1].pools[0], "total_blocks", 101),
            lambda m: m.ranks.pop(),
            lambda m: m.ranks.pop(0),
        ]
        for mutate in mutations:
            message = shared_wire()
            mutate(message)
            with self.subTest(mutate=mutate), self.assertRaises(ValueError):
                decode_pressure_response(message, 1, 1, (0, 1))

    def test_partial_page_does_not_hide_observed_restart_layout_or_counter_reset(self):
        from rtp_llm.test.continuous_pressure.diagnostics import PressureCoverage

        baseline = decode_pressure_response(shared_wire(0, True), 0, 0, (0, 1)).samples
        current = decode_pressure_response(shared_wire(1), 1, 1, (0, 1)).samples
        mutations = [
            lambda s: dataclasses.replace(s, instance="restarted"),
            lambda s: dataclasses.replace(
                s, pools=(dataclasses.replace(s.pools[0], capacity=101), s.pools[1])
            ),
            lambda s: dataclasses.replace(
                s, groups=(dataclasses.replace(s.groups[0], triggers=0),)
            ),
            lambda s: dataclasses.replace(s, ownership=None),
        ]
        for mutate in mutations:
            evaluator = PressureCoverage(baseline, configured())
            evaluator.add(current, 1)
            observed = mutate(
                dataclasses.replace(
                    current[0],
                    sequence=4,
                    server_s=2,
                    client_send_s=2,
                    client_receive_s=2,
                )
            )
            evaluator.add((observed,), 2)
            self.assertFalse(evaluator.epoch_valid)

    def test_incomplete_page_preserves_counter_reference_without_coverage(self):
        from rtp_llm.test.continuous_pressure.diagnostics import PressureCoverage

        baseline = decode_pressure_response(shared_wire(0, True), 0, 0, (0, 1)).samples
        current = decode_pressure_response(shared_wire(1), 1, 1, (0, 1)).samples
        evaluator = PressureCoverage(baseline, configured())
        evaluator.add(current, 1, complete=False)
        self.assertTrue(evaluator.epoch_valid)
        self.assertEqual(evaluator.last_seen[0], current[0])
        later = dataclasses.replace(
            current[0],
            sequence=current[0].sequence + 1,
            groups=(dataclasses.replace(current[0].groups[0], triggers=0),),
        )
        evaluator.add((later,), 2, complete=False)
        self.assertFalse(evaluator.epoch_valid)
        self.assertIn("counter_reset", evaluator.reasons)

    def test_owner_clock_boundary_and_missing_stale_peer_never_gain_coverage(self):
        from rtp_llm.test.continuous_pressure.diagnostics import PressureCoverage

        baseline = decode_pressure_response(shared_wire(0, True), 0, 0, (0, 1)).samples
        for failure in ("missing", "stale", "none"):
            evaluator = PressureCoverage(baseline, configured())
            for t in range(1, 5):
                samples = list(
                    decode_pressure_response(shared_wire(t), t, t, (0, 1)).samples
                )
                samples[0] = dataclasses.replace(samples[0], offset_bounds=(-0.2, 0.2))
                if t >= 3 and failure == "missing":
                    samples.pop()
                if t >= 3 and failure == "stale":
                    samples[1] = dataclasses.replace(
                        samples[1], offset_bounds=(-110, -100)
                    )
                evaluator.add(samples, t)
            if failure == "none":
                self.assertEqual(evaluator.intervals(1, 4), [(1.2, 3.8)])
            else:
                self.assertEqual(
                    evaluator.intervals(1, 4),
                    [(1.2, 1.8 if failure == "missing" else 2.0)],
                )


class PressurePredicateTest(unittest.TestCase):
    def test_full_gate_needs_publication_and_effective_occupancy(self):
        cfg = configured()
        pools = {
            "gpu": PoolSample("gpu", 100, 90, 15, 0.8, 0.9),
            "host": PoolSample("host", 100, 30, 0, 0.8, 0.9, 25),
        }
        base = GroupSample(0, "FULL", ("gpu",), "host", 0, 0)
        current = dataclasses.replace(base, triggers=1, gpu_host_published=1)
        self.assertFalse(group_valid(current, pools, base, cfg))
        pools["gpu"] = dataclasses.replace(pools["gpu"], pending_release=0)
        self.assertTrue(group_valid(current, pools, base, cfg))
        self.assertFalse(group_valid(base, pools, base, cfg))
        self.assertEqual((cfg.steady_s, cfg.measure_s, cfg.profile_steps), (3, 60, 20))

    def test_rounding_and_any_device(self):
        pools = {
            "a": PoolSample("a", 101, 79, 0, 0.8, 0.9),
            "b": PoolSample("b", 101, 80, 0, 0.8, 0.9),
            "host": PoolSample("host", 101, 26, 0, 0.8, 0.9, 26),
        }
        base = GroupSample(0, "FULL", ("a", "b"), "host", 0, 0)
        group = dataclasses.replace(base, triggers=1, gpu_host_published=1)
        self.assertTrue(group_valid(group, pools, base, configured()))
        pools["host"] = dataclasses.replace(pools["host"], host_resident=25)
        self.assertFalse(group_valid(group, pools, base, configured()))

    def test_config_validation_and_frozen(self):
        cfg = configured()
        self.assertEqual(
            (
                cfg.host_ratio,
                cfg.sample_period_s,
                cfg.max_sample_age_s,
                cfg.max_sample_gap_s,
                cfg.min_coverage,
                cfg.max_invalid_s,
                cfg.steady_timeout_s,
                cfg.support_s,
            ),
            (0.25, 1, 2, 2, 0.95, 2, 60, 60),
        )
        with self.assertRaises(dataclasses.FrozenInstanceError):
            cfg.measure_s = 1
        for key in settings():
            values = settings()
            del values[key]
            with self.subTest(missing=key), self.assertRaises(ValueError):
                resolve_pressure_config(values)
        for key, value in [
            ("fill_timeout_s", 0),
            ("fill_input_tokens", True),
            ("fill_max_requests", 1.5),
            ("fill_no_progress_s", float("nan")),
            ("activity_timeout_s", float("inf")),
            ("host_ratio", 1.1),
            ("gpu_tolerance_blocks", -1),
            ("sample_period_s", 3),
            ("nsys_duration_s", 0),
        ]:
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                configured(**{key: value})

    def test_targets_are_pure_all_ranks_and_full_only(self):
        first = rank(
            groups=[
                GroupSample(0, "FULL", ("gpu",), "host", 1, 1),
                GroupSample(1, "SWA", ("gpu",), "host", 0, 0),
            ]
        )
        targets = select_targets((first, rank(1)), (0, 1))
        self.assertEqual(len(targets), 2)
        self.assertEqual(
            select_targets((rank(instance="new"),), (0,))[0].instance, "new"
        )
        with self.assertRaises(ValueError):
            validate_inventory(select_targets((rank(),), (0,)), targets)
        with self.assertRaises(ValueError):
            select_targets((rank(),), (0, 1))
        with self.assertRaises(ValueError):
            select_targets((rank(groups=[]),), (0,))
        with self.assertRaises(ValueError):
            select_targets(
                (rank(pools=[PoolSample("gpu", 100, 90, 0, 0.8, 0.9)]),), (0,)
            )

    def test_invalid_pools_resets_and_mapping(self):
        sample = rank()
        group = sample.groups[0]
        pools = {p.pool_id: p for p in sample.pools}
        for bad in [
            dataclasses.replace(group, triggers=0),
            dataclasses.replace(group, gpu_host_published=0),
            dataclasses.replace(group, host_pool_id="missing"),
        ]:
            with self.assertRaises(ValueError):
                group_valid(bad, pools, group, configured())
        with self.assertRaises(ValueError):
            group_valid(group, pools, group, configured(gpu_tolerance_blocks=81))
        for bad in [
            dataclasses.replace(pools["gpu"], pending_release=91),
            dataclasses.replace(pools["gpu"], low_ratio=0.9),
        ]:
            with self.assertRaises(ValueError):
                select_targets(
                    (dataclasses.replace(sample, pools=(bad, pools["host"])),), (0,)
                )
        with self.assertRaises(ValueError):
            select_targets(
                (
                    dataclasses.replace(
                        sample,
                        pools=sample.pools
                        + (dataclasses.replace(pools["gpu"], used=99),),
                    ),
                ),
                (0,),
            )

    def test_live_free_and_available_blocks_are_projected(self):
        message = wire()
        message.ranks[0].pools[0].total_blocks = 128
        message.ranks[0].pools[0].free_blocks = 17
        message.ranks[0].pools[0].available_blocks = 23
        sample = decode_pressure_response(message, 1.0, 1.1, (0,)).samples[0]
        self.assertEqual(sample.pools[0].capacity, 128)
        self.assertEqual(sample.pools[0].free, 17)
        self.assertEqual(sample.pools[0].available, 23)
        before = select_targets((sample,), (0,))
        message.ranks[0].pools[0].free_blocks = 19
        message.ranks[0].pools[0].available_blocks = 24
        after = decode_pressure_response(message, 1.0, 1.1, (0,)).samples
        validate_inventory(select_targets(after, (0,)), before)

    def test_live_free_and_available_cannot_exceed_capacity(self):
        for field in ("free_blocks", "available_blocks"):
            with self.subTest(field=field):
                message = wire()
                setattr(message.ranks[0].pools[0], field, 101)
                with self.assertRaisesRegex(ValueError, "invalid physical occupancy"):
                    decode_pressure_response(message, 1.0, 1.1, (0,))

    def test_real_wire_endpoint_projection_and_raw_evidence(self):
        message = wire()
        decoded = decode_pressure_response(message, 1.0, 1.1, (0,))
        self.assertEqual(decoded.raw_wire, message.SerializeToString())
        self.assertEqual(decoded.samples[0].groups[0].gpu_host_published, 7)
        self.assertEqual(decoded.samples[0].server_s, 1)
        self.assertEqual(decoded.samples[0].pools[1].host_resident, 25)
        message.ranks[0].pools.append(message.ranks[0].pools[0])
        self.assertEqual(
            len(decode_pressure_response(message, 1.0, 1.1, (0,)).samples[0].pools), 2
        )

    def test_store_publication_qualifies_without_double_counting_source(self):
        message = wire()
        # Stable low-watermark occupancy cannot establish the prior DEVICE trigger.
        message.ranks[0].pools[0].used_blocks = 80
        for transfer in message.ranks[0].transfers:
            transfer.operation = 3
        sample = decode_pressure_response(message, 1.0, 1.1, (0,)).samples[0]
        group = sample.groups[0]
        self.assertEqual(group.gpu_host_published, 7)
        baseline = dataclasses.replace(group, triggers=0, gpu_host_published=0)
        self.assertTrue(
            group_valid(
                group, {p.pool_id: p for p in sample.pools}, baseline, configured()
            )
        )
        self.assertFalse(
            group_valid(
                dataclasses.replace(group, triggers=0),
                {p.pool_id: p for p in sample.pools},
                baseline,
                configured(),
            )
        )
        message.ranks[0].groups[0].watermark_trigger_count = 0
        decoded = decode_pressure_response(message, 1.0, 1.1, (0,)).samples[0]
        self.assertEqual(decoded.groups[0].triggers, 0)
        self.assertFalse(
            group_valid(
                decoded.groups[0],
                {p.pool_id: p for p in decoded.pools},
                baseline,
                configured(),
            )
        )

    def test_low_swa_occupancy_does_not_gate_full(self):
        sample = rank()
        full = sample.groups[0]
        sample = dataclasses.replace(
            sample,
            pools=sample.pools + (PoolSample("cold_swa", 100, 0, 0, 0.8, 0.9),),
            groups=sample.groups + (GroupSample(1, "SWA", ("cold_swa",), "", 0, 0),),
        )
        self.assertEqual(len(select_targets((sample,), (0,))), 1)
        self.assertTrue(
            group_valid(
                full,
                {p.pool_id: p for p in sample.pools},
                dataclasses.replace(full, triggers=0, gpu_host_published=0),
                configured(),
            )
        )

    def test_raw_record_accepts_plain_data_without_a_runtime_projector(self):
        response = {"steps": [{"phase": "steady"}], "activity": {"running": 2}}
        record = raw_pressure_record(response, 1.0, 1.1)
        self.assertEqual(record["response"], response)
        self.assertEqual(record["client_send_s"], 1.0)
        self.assertEqual(record["client_receive_s"], 1.1)

    def test_raw_record_retains_structured_evidence_and_client_times(self):
        import json

        response = PB(steps=[{"phase": "steady"}], activity={"running": 2})
        record = raw_pressure_record(response, 1.0, 1.1, to_dict=vars)
        self.assertEqual(record["response"]["steps"][0]["phase"], "steady")
        self.assertEqual(record["response"]["activity"]["running"], 2)
        self.assertEqual(record["client_receive_s"], 1.1)
        json.dumps(record, allow_nan=False)
        with self.assertRaises(ValueError):
            raw_pressure_record(response, 2.0, 1.0, to_dict=vars)

    def test_disabled_host_disk_and_observe_only_watermarks_are_preserved(self):
        message = wire()
        current = message.ranks[0]
        full = current.groups[0]
        full.host_watermark = PB(low_ratio=0.0, high_ratio=0.0)
        full.disk_pool_id = PB(value="disk")
        full.disk_watermark = PB(low_ratio=0.0, high_ratio=0.0)
        current.pools.extend(
            [
                PB(
                    pool_id="disk",
                    tier=3,
                    total_blocks=100,
                    free_blocks=0,
                    available_blocks=0,
                    used_blocks=0,
                    pending_release_blocks=0,
                    published_host_blocks=0,
                ),
                PB(
                    pool_id="swa",
                    tier=1,
                    total_blocks=100,
                    free_blocks=0,
                    available_blocks=0,
                    used_blocks=0,
                    pending_release_blocks=0,
                    published_host_blocks=0,
                ),
            ]
        )
        current.groups.append(
            PB(
                group_id=1,
                group_type=2,
                device_pool_ids=["swa"],
                device_watermark=PB(low_ratio=0.0, high_ratio=0.0),
                watermark_trigger_count=0,
            )
        )
        sample = decode_pressure_response(message, 1.0, 1.1, (0,)).samples[0]
        pools = {pool.pool_id: pool for pool in sample.pools}
        for pool_id in ("host", "disk", "swa"):
            self.assertEqual(
                (pools[pool_id].low_ratio, pools[pool_id].high_ratio), (0.0, 0.0)
            )
        self.assertEqual(len(select_targets((sample,), (0,))), 1)
        group = sample.groups[0]
        baseline = dataclasses.replace(group, triggers=0, gpu_host_published=0)
        self.assertTrue(group_valid(group, pools, baseline, configured()))

    def test_disabled_or_invalid_full_device_watermarks_are_rejected(self):
        for low, high in ((0.0, 0.0), (0.0, 0.9), (0.9, 0.8), (0.8, 1.1)):
            message = wire()
            message.ranks[0].groups[0].device_watermark = PB(
                low_ratio=low, high_ratio=high
            )
            with self.subTest(low=low, high=high), self.assertRaises(ValueError):
                decode_pressure_response(message, 1.0, 1.1, (0,))

    def test_wire_rejects_schema_presence_tier_and_unknown_mapping(self):
        mutations = [
            lambda m: setattr(m, "schema_version", 2),
            lambda m: setattr(m.ranks[0].groups[0], "host_pool_id", None),
            lambda m: setattr(m.ranks[0].groups[0], "device_watermark", None),
            lambda m: setattr(m.ranks[0].pools[1], "tier", 1),
            lambda m: setattr(m.ranks[0].transfers[1], "pool_id", "unknown"),
            lambda m: setattr(m.ranks[0].groups[0], "group_type", 0),
            lambda m: setattr(m.ranks[0], "error", "failed"),
        ]
        for mutate in mutations:
            message = wire()
            mutate(message)
            with self.subTest(mutate=mutate), self.assertRaises(ValueError):
                decode_pressure_response(message, 1.0, 1.1, (0,))


def migration_page(
    t, d2h=0, h2d=0, *, participants=1, bounds=(0, 0), operation=2, failed_operations=0
):
    """Decode cumulative settlement endpoints, including both physical views."""
    message = wire() if participants == 1 else shared_wire(t, True)
    if participants > 2:
        message.ranks.extend(
            copy.deepcopy(message.ranks[1]) for _ in range(participants - 2)
        )
    message.requested_ranks = list(range(participants))
    for i, item in enumerate(message.ranks):
        item.world_rank = i
        item.service_instance = f"instance-{i}"
        item.sequence = t + 2
        item.sampled_monotonic_ns = int((t + i * 100) * 1e9)
        item.groups[0].watermark_trigger_count = 1
        if participants > 1:
            item.ownership.participant_world_ranks = list(range(participants))
        item.transfers = []
        for op, source, target, count in ((operation, 1, 2, d2h), (1, 2, 1, h2d)):
            for pool in item.pools:
                item.transfers.append(
                    PB(
                        group_id=0,
                        pool_id=pool.pool_id,
                        pool_tier=pool.tier,
                        operation=op,
                        source_tier=source,
                        target_tier=target,
                        copied_blocks=count,
                        copied_bytes=count * 4096,
                        published_blocks=count,
                        published_bytes=count * 4096,
                        failed_operations=failed_operations,
                    )
                )
    return decode_pressure_response(
        message, t + bounds[0], t + bounds[1], tuple(range(participants))
    ).samples


class MigrationTest(unittest.TestCase):
    def evaluator(self, *, participants=1, max_segments=4096):
        return PressureCoverage(
            migration_page(9, participants=participants), configured(), max_segments
        )

    def mock_page(self, t, *, failed=0, completed=0, participants=1):
        samples = migration_page(t, t, t, participants=participants)
        return tuple(
            dataclasses.replace(
                sample,
                mock_memory=MockMemorySample(
                    16,
                    8,
                    completed,
                    failed,
                    completed,
                    completed,
                    failed,
                    terminal_outcome_version=1,
                ),
                groups=tuple(
                    dataclasses.replace(group, triggers=0) if t == 0 else group
                    for group in sample.groups
                ),
            )
            for sample in samples
        )

    def test_zero_effective_memory_reads_use_natural_publication_and_opted_in_stores(
        self,
    ):
        for ratio, expected in ((0.0, True), (0.000000499, True), (0.0000005, False)):
            with self.subTest(ratio=ratio):

                def page(t):
                    (sample,) = self.mock_page(t, completed=t)
                    return (
                        dataclasses.replace(
                            sample,
                            mock_memory=dataclasses.replace(
                                sample.mock_memory,
                                eligible_blocks=0,
                                planned_host_blocks=0,
                                successful_h2d_blocks=0,
                            ),
                        ),
                    )

                c = PressureCoverage(
                    page(0),
                    configured(
                        evidence_policy="tree_then_mock_memory",
                        mock_memory_target_ratio=ratio,
                    ),
                )
                for t in (1, 2, 3):
                    c.add(page(t), t)
                result = c.assess_migration(1, 3)
                self.assertEqual(result.valid, expected, result.reasons)
                if expected:
                    self.assertEqual(
                        result.groups[0].h2d_evidence[0].source, "published_blocks"
                    )
                    self.assertEqual(
                        result.groups[0].d2h_evidence[0].source,
                        "mock_memory_d2h_completed_blocks",
                    )

    def test_mixed_disk_and_memory_terminal_receipts_cover_released_and_deduplicated_requests(
        self,
    ):
        cfg = configured(
            evidence_policy="tree_then_mock_disk",
            mock_cache_settings=(("mock_memory_cache", True),),
        )

        def page(t, *, released, pending=0, abandoned=0, mem=(0, 0, 0), disk=(0, 0, 0)):
            sample = self.mock_page(t)[0]
            return (
                dataclasses.replace(
                    sample,
                    local_store=LocalStoreSample(released, pending, abandoned),
                    mock_memory=MockMemorySample(0, 0, 0, 0, *mem, 1),
                    mock_cache=MockCacheSample(
                        terminal_outcome_version=1,
                        disk_write_submitted_records=disk[0],
                        disk_write_completed_records=disk[1],
                        disk_write_failed_records=disk[2],
                    ),
                ),
            )

        for tier in ("memory", "disk"):
            for outcome in (
                "deduplicated",
                "failed",
                "pending",
                "preexisting",
                "abandoned",
            ):
                with self.subTest(tier=tier, outcome=outcome):
                    c = PressureCoverage(
                        page(0, released=10, pending=int(outcome == "preexisting")), cfg
                    )
                    c.add(page(1, released=11, mem=(1, 1, 0), disk=(1, 1, 0)), 1)
                    receipt = (
                        (2, 1, 1)
                        if outcome == "failed"
                        else ((2, 1, 0) if outcome == "pending" else (1, 1, 0))
                    )
                    final = page(
                        2,
                        released=12,
                        abandoned=int(outcome == "abandoned"),
                        mem=receipt if tier == "memory" else (1, 1, 0),
                        disk=receipt if tier == "disk" else (1, 1, 0),
                    )
                    c.add(final, 2, complete=False)
                    summary = c.local_write_summary(final, 2)
                    self.assertEqual(
                        summary["valid"], outcome == "deduplicated", summary
                    )
                    self.assertEqual(
                        summary["owners"]["0"][tier]["submitted"], receipt[0]
                    )
                    if outcome == "failed":
                        self.assertIn(tier + "_write_failed", summary["reasons"])

    def test_mock_failures_survive_incomplete_partial_and_gapped_observations(self):
        for policy in ("mock_memory", "tree_then_mock_memory"):
            for gap in ("incomplete", "partial_owner", "missing_owner", "empty", "gap"):
                with self.subTest(policy=policy, gap=gap):
                    participants = 2 if gap in ("partial_owner", "missing_owner") else 1
                    evaluator = PressureCoverage(
                        self.mock_page(0, participants=participants),
                        configured(evidence_policy=policy, max_sample_gap_s=1),
                    )
                    for t in range(1, 62):
                        page = self.mock_page(
                            t,
                            failed=int(t >= 30),
                            completed=4 if t >= 11 else 0,
                            participants=participants,
                        )
                        if t == 30:
                            if gap == "gap":
                                continue
                            if gap == "partial_owner":
                                page = page[:1]
                            elif gap == "missing_owner":
                                page = page[1:]
                            elif gap == "empty":
                                page = ()
                        evaluator.add(
                            page, t, complete=not (t == 30 and gap == "incomplete")
                        )
                    self.assertTrue(evaluator.assess(1, 61).valid)
                    assessment = evaluator.assess_migration(1, 61)
                    self.assertFalse(assessment.valid)
                    self.assertIn("failed_h2d:0:0", assessment.reasons)
                    self.assertIn("failed_d2h:0:0", assessment.reasons)
                    expected_end = (
                        31 if gap in ("missing_owner", "empty", "gap") else 30
                    )
                    for evidence in (
                        assessment.groups[0].h2d_failure_evidence,
                        assessment.groups[0].d2h_failure_evidence,
                    ):
                        self.assertEqual(len(evidence), 1)
                        self.assertEqual(evidence[0].interval, (29, expected_end))
                        self.assertEqual(
                            (evidence[0].before_blocks, evidence[0].after_blocks),
                            (0, 1),
                        )

    def test_mock_failure_observations_outside_window_do_not_reject(self):
        for failure_at in (2, 30):
            with self.subTest(failure_at=failure_at):
                evaluator = PressureCoverage(
                    self.mock_page(0),
                    configured(evidence_policy="tree_then_mock_memory"),
                )
                for t in range(1, 42):
                    evaluator.add(
                        self.mock_page(
                            t,
                            failed=int(t >= failure_at),
                            completed=4 if t >= 15 else 0,
                        ),
                        t,
                        complete=t != failure_at,
                    )
                self.assertTrue(evaluator.assess_migration(10, 20).valid)

    def test_mock_successes_from_incomplete_or_partial_samples_do_not_qualify(self):
        for gap in ("incomplete", "partial"):
            with self.subTest(gap=gap):
                participants = 2 if gap == "partial" else 1
                evaluator = PressureCoverage(
                    self.mock_page(0, participants=participants),
                    configured(
                        evidence_policy="tree_then_mock_memory",
                        mock_memory_target_ratio=0.5,
                    ),
                )
                for t in range(1, 6):
                    page = self.mock_page(
                        t, completed=4 if t >= 3 else 0, participants=participants
                    )
                    if t == 3 and gap == "partial":
                        page = page[:1]
                    evaluator.add(
                        page, t, complete=not (t == 3 and gap == "incomplete")
                    )
                result = evaluator.assess_migration(1, 5)
                self.assertFalse(result.valid)
                self.assertIn("missing_h2d_load:0:0", result.reasons)
                self.assertIn("missing_d2h:0:0", result.reasons)

    def test_mock_failure_counter_reset_and_restart_during_gap_fail_closed(self):
        for fault in ("reset", "restart"):
            with self.subTest(fault=fault):
                evaluator = PressureCoverage(
                    self.mock_page(0),
                    configured(evidence_policy="tree_then_mock_memory"),
                )
                evaluator.add(self.mock_page(1, failed=1), 1, complete=False)
                sample = self.mock_page(2, failed=0 if fault == "reset" else 1)[0]
                if fault == "restart":
                    sample = dataclasses.replace(sample, instance="restarted")
                evaluator.add((sample,), 2, complete=False)
                self.assertFalse(evaluator.epoch_valid)
                self.assertFalse(evaluator.assess_migration(1, 3).valid)
                self.assertIn(
                    "counter_reset" if fault == "reset" else "inventory_discontinuity",
                    evaluator.reasons,
                )

    def test_settled_store_or_evict_and_load_qualify_each_direction(self):
        for operation in (2, 3):
            with self.subTest(operation=operation):
                evaluator = PressureCoverage(
                    (migration_page(9, operation=operation)[0],), configured()
                )
                evaluator.add(migration_page(10, operation=operation), 10)
                evaluator.add(migration_page(11, 7, 5, operation=operation), 11)
                assessment = evaluator.assess_migration(10, 20)
                self.assertTrue(assessment.valid)
                self.assertEqual(assessment.reasons, ())
                self.assertTrue(all(group.valid for group in assessment.groups))
                self.assertEqual(len(assessment.groups), 1)
                group = assessment.groups[0]
                self.assertEqual(group.d2h_intervals, ((10, 11),))
                self.assertEqual(group.h2d_intervals, ((10, 11),))

    def test_d2h_alone_cannot_qualify_migration(self):
        evaluator = self.evaluator()
        evaluator.add(migration_page(10), 10)
        evaluator.add(migration_page(11, 7, 0), 11)
        result = evaluator.assess_migration(10, 20)
        self.assertFalse(result.valid)
        self.assertEqual(result.reasons, ("missing_h2d_load:0:0",))

    def test_h2d_alone_cannot_qualify_migration(self):
        evaluator = self.evaluator()
        evaluator.add(migration_page(10), 10)
        evaluator.add(migration_page(11, 0, 7), 11)
        result = evaluator.assess_migration(10, 20)
        self.assertFalse(result.valid)
        self.assertEqual(result.reasons, ("missing_d2h:0:0",))

    def test_copy_or_bytes_without_publication_cannot_qualify(self):
        evaluator = self.evaluator()
        evaluator.add(migration_page(10), 10)
        current = migration_page(11, 7, 5)[0]
        current = dataclasses.replace(
            current,
            counters=tuple(
                (key, 0 if key[0] == "published_blocks" else value)
                for key, value in current.counters
            ),
        )
        evaluator.add((current,), 11)
        self.assertFalse(evaluator.assess_migration(10, 20).valid)

    def test_failed_or_cancelled_copy_does_not_substitute_for_publication(self):
        evaluator = self.evaluator()
        evaluator.add(migration_page(10), 10)
        current = migration_page(11, 7, 5)[0]
        current = dataclasses.replace(
            current,
            counters=tuple(
                (
                    key,
                    (
                        0
                        if key[0] in ("published_blocks", "published_bytes")
                        else 2 if key[0] == "failed_operations" else value
                    ),
                )
                for key, value in current.counters
            ),
        )
        evaluator.add((current,), 11)
        self.assertFalse(evaluator.assess_migration(10, 20).valid)

    def test_real_owner_failures_invalidate_positive_publications_through_gaps(self):
        for observation, expected_interval in (
            ("complete", (10, 11)),
            ("partial", (11, 12)),
            ("missing_owner", (11, 13)),
            ("outside_window", None),
            ("peer_only", None),
        ):
            with self.subTest(observation=observation):
                evaluator = self.evaluator(participants=2)
                evaluator.add(migration_page(10, participants=2), 10)
                evaluator.add(
                    migration_page(
                        11,
                        7,
                        5,
                        participants=2,
                        failed_operations=2 if observation == "complete" else 0,
                    ),
                    11,
                )
                current = migration_page(12, 7, 5, participants=2, failed_operations=2)
                if observation == "missing_owner":
                    evaluator.add(current[1:], 12, complete=False)
                    evaluator.add(
                        migration_page(13, 7, 5, participants=2, failed_operations=2),
                        13,
                    )
                elif observation == "partial":
                    evaluator.add(current[:1], 12, complete=False)
                    evaluator.add(
                        migration_page(13, 7, 5, participants=2, failed_operations=2),
                        13,
                    )
                elif observation == "peer_only":
                    evaluator.add(
                        (migration_page(12, 7, 5, participants=2)[0], current[1]), 12
                    )
                elif observation == "outside_window":
                    evaluator.add(migration_page(13, 7, 5, participants=2), 13)
                    evaluator.add(
                        migration_page(14, 7, 5, participants=2, failed_operations=2),
                        14,
                    )
                else:
                    evaluator.add(current, 12)
                assessment = evaluator.assess_migration(10, 12)
                self.assertEqual(assessment.valid, expected_interval is None)
                group = assessment.groups[0]
                self.assertEqual(group.d2h_evidence[0].after_blocks, 7)
                self.assertEqual(group.h2d_evidence[0].after_blocks, 5)
                if expected_interval is not None:
                    self.assertEqual(
                        assessment.reasons, ("failed_d2h:0:0", "failed_h2d:0:0")
                    )
                    for failures in (
                        group.d2h_failure_evidence,
                        group.h2d_failure_evidence,
                    ):
                        self.assertEqual(len(failures), 2)  # One per physical endpoint.
                        for failure in failures:
                            self.assertEqual(failure.interval, expected_interval)
                            self.assertEqual(failure.source, "failed_operations")
                            self.assertEqual(
                                (failure.before_blocks, failure.after_blocks), (0, 2)
                            )

    def test_publication_in_wrong_operation_or_endpoint_is_rejected(self):
        for direction in ("d2h", "h2d"):
            for mutation in ("wrong_operation", "source_pool_only"):
                with self.subTest(direction=direction, mutation=mutation):

                    def mutate(sample):
                        counters = []
                        for key, value in sample.counters:
                            if len(key) == 6 and key[0] == "published_blocks":
                                matches = (key[4], key[5]) == (
                                    (1, 2) if direction == "d2h" else (2, 1)
                                )
                                if matches and mutation == "wrong_operation":
                                    key = (
                                        *key[:3],
                                        1 if direction == "d2h" else 3,
                                        *key[4:],
                                    )
                                elif matches and key[2] == (
                                    "host" if direction == "d2h" else "gpu"
                                ):
                                    value = 0
                            counters.append((key, value))
                        return dataclasses.replace(sample, counters=tuple(counters))

                    evaluator = PressureCoverage(
                        (mutate(migration_page(9)[0]),), configured()
                    )
                    evaluator.add((mutate(migration_page(10)[0]),), 10)
                    evaluator.add((mutate(migration_page(11, 7, 5)[0]),), 11)
                    result = evaluator.assess_migration(10, 20)
                    self.assertTrue(result.epoch_valid)
                    self.assertFalse(result.valid)

    def test_outer_uncertainty_crossing_measurement_start_is_rejected(self):
        evaluator = self.evaluator()
        evaluator.add(migration_page(10, bounds=(-0.2, 0.2)), 10.2)
        evaluator.add(migration_page(11, 7, 5, bounds=(-0.1, 0.1)), 11.1)
        result = evaluator.assess_migration(10, 20)
        self.assertTrue(result.epoch_valid)
        self.assertFalse(result.valid)

    def test_outer_uncertainty_crossing_measurement_end_is_rejected(self):
        evaluator = self.evaluator()
        evaluator.add(migration_page(19, bounds=(-0.1, 0.1)), 19.1)
        evaluator.add(migration_page(20, 7, 5, bounds=(-0.2, 0.2)), 20.2)
        self.assertFalse(evaluator.assess_migration(10, 20).valid)

    def test_pre_and_post_window_publications_do_not_qualify(self):
        for left, right in ((8, 9), (20, 21)):
            with self.subTest(left=left):
                baseline = migration_page(left - 1)
                evaluator = PressureCoverage(baseline, configured())
                evaluator.add(migration_page(left), left)
                evaluator.add(migration_page(right, 7, 5), right)
                self.assertFalse(evaluator.assess_migration(10, 20).valid)

    def test_every_independent_owner_group_requires_both_directions(self):
        def page(t, peer_h2d):
            owner = migration_page(t, t - 10, t - 10)[0]
            peer = dataclasses.replace(migration_page(t, t - 10, peer_h2d)[0], rank=1)
            return owner, peer

        evaluator = PressureCoverage(page(10, 0), configured())
        evaluator.add(page(11, 0), 11)
        evaluator.add(page(12, 0), 12)
        result = evaluator.assess_migration(10, 20)
        self.assertFalse(result.valid)
        self.assertEqual(len(result.groups), 2)
        self.assertEqual(result.reasons, ("missing_h2d_load:1:0",))

    def test_four_shared_tp_views_yield_one_owner_group_without_summing(self):
        evaluator = self.evaluator(participants=4)
        evaluator.add(migration_page(10, participants=4), 10)
        evaluator.add(migration_page(11, 7, 5, participants=4), 11)
        result = evaluator.assess_migration(10, 20)
        self.assertTrue(result.valid)
        self.assertEqual(len(result.groups), 1)
        self.assertEqual(
            (result.groups[0].owner_rank, result.groups[0].owner_group_id), (0, 0)
        )
        self.assertEqual(result.groups[0].d2h_evidence[0].after_blocks, 7)
        self.assertEqual(result.groups[0].h2d_evidence[0].after_blocks, 5)

    def test_participant_publication_cannot_replace_owner_publication(self):
        evaluator = self.evaluator(participants=4)
        evaluator.add(migration_page(10, participants=4), 10)
        current = migration_page(11, 7, 5, participants=4)
        owner = migration_page(11, participants=4)[0]
        evaluator.add((owner, *current[1:]), 11)
        self.assertFalse(evaluator.assess_migration(10, 20).valid)

    def test_missing_shared_participant_breaks_delta_adjacency(self):
        for complete in (True, False):
            with self.subTest(complete=complete):
                evaluator = self.evaluator(participants=4)
                evaluator.add(migration_page(10, participants=4), 10)
                evaluator.add(
                    migration_page(11, 7, 5, participants=4)[:3], 11, complete=complete
                )
                evaluator.add(migration_page(12, 7, 5, participants=4), 12)
                self.assertFalse(evaluator.assess_migration(10, 20).valid)

    def test_stale_shared_participant_cannot_certify_owner_increment(self):
        evaluator = self.evaluator(participants=4)
        evaluator.add(migration_page(10, participants=4), 10)
        current = migration_page(11, 7, 5, participants=4)
        stale = dataclasses.replace(current[3], offset_bounds=(-3, 0))
        evaluator.add((*current[:3], stale), 11)
        self.assertFalse(evaluator.assess_migration(10, 20).valid)

    def test_counter_reset_after_missing_participant_invalidates_prior_evidence(self):
        for complete in (True, False):
            with self.subTest(complete=complete):
                evaluator = self.evaluator(participants=4)
                evaluator.add(migration_page(10, participants=4), 10)
                evaluator.add(migration_page(11, 7, 5, participants=4), 11)
                self.assertTrue(evaluator.assess_migration(10, 20).valid)
                evaluator.add(
                    migration_page(12, 10, 8, participants=4)[:3],
                    12,
                    complete=complete,
                )
                # Rejected pages cannot add intervals or advance adjacency.
                self.assertEqual(
                    evaluator.assess_migration(10, 20).groups[0].d2h_intervals,
                    ((10, 11),),
                )
                evaluator.add(migration_page(13, 9, 7, participants=4), 13)
                result = evaluator.assess_migration(10, 20)
                self.assertFalse(result.epoch_valid)
                self.assertFalse(result.valid)
                self.assertIn("counter_reset", result.reasons)
                self.assertEqual(result.groups[0].d2h_intervals, ((10, 11),))
                self.assertEqual(result.groups[0].h2d_intervals, ((10, 11),))

    def test_counter_decrease_invalidates_existing_migration_evidence(self):
        evaluator = self.evaluator()
        evaluator.add(migration_page(10), 10)
        evaluator.add(migration_page(11, 7, 5), 11)
        evaluator.add(migration_page(12, 6, 5), 12)
        result = evaluator.assess_migration(10, 20)
        self.assertFalse(result.epoch_valid)
        self.assertFalse(result.valid)
        self.assertIn("counter_reset", result.reasons)

    def test_instance_layout_and_participant_changes_invalidate_evidence(self):
        mutations = (
            lambda s: dataclasses.replace(s, instance="restarted"),
            lambda s: dataclasses.replace(
                s, pools=(dataclasses.replace(s.pools[0], capacity=101), s.pools[1])
            ),
            lambda s: dataclasses.replace(
                s, ownership=dataclasses.replace(s.ownership, participants=(0, 1))
            ),
        )
        for mutate in mutations:
            with self.subTest(mutate=mutate):
                evaluator = self.evaluator(participants=4)
                evaluator.add(migration_page(10, participants=4), 10)
                evaluator.add(migration_page(11, 7, 5, participants=4), 11)
                current = migration_page(12, 7, 5, participants=4)
                evaluator.add((mutate(current[0]), *current[1:]), 12)
                result = evaluator.assess_migration(10, 20)
                self.assertFalse(result.epoch_valid)
                self.assertFalse(result.valid)
                self.assertIn("inventory_discontinuity", result.reasons)

    def test_missing_boundary_and_gap_never_certify_a_delta(self):
        for missing in (True, False):
            with self.subTest(missing=missing):
                evaluator = self.evaluator()
                evaluator.add(migration_page(10), 10)
                if missing:
                    evaluator.add((), 11)
                evaluator.add(migration_page(13, 7, 5), 13)
                self.assertFalse(evaluator.assess_migration(10, 20).valid)

    def test_sparse_endpoint_zero_baseline_is_explicit_and_window_bounded(self):
        baseline = dataclasses.replace(
            migration_page(9)[0], counters=((("trigger", 0), 1),)
        )
        evaluator = PressureCoverage((baseline,), configured())
        first = migration_page(10)[0]
        evaluator.add(
            (dataclasses.replace(first, counters=((("trigger", 0), 1),)),), 10
        )
        evaluator.add(migration_page(11, 7, 5), 11)
        result = evaluator.assess_migration(10, 20)
        self.assertTrue(result.valid)
        evidence = result.groups[0].d2h_evidence[0]
        self.assertEqual(evidence.source, "published_blocks")
        self.assertEqual((evidence.before_blocks, evidence.after_blocks), (0, 7))
        self.assertEqual(
            evidence.zero_baseline_endpoints,
            (("published_blocks", 0, "host", 2, 1, 2),),
        )
        self.assertEqual(
            evidence.endpoint_deltas,
            ((("published_blocks", 0, "host", 2, 1, 2), 0, 7),),
        )

    def test_history_overflow_fails_closed_even_with_retained_directions(self):
        evaluator = self.evaluator(max_segments=1)
        evaluator.add(migration_page(10), 10)
        evaluator.add(migration_page(11, 7, 5), 11)
        evaluator.add(migration_page(12, 8, 6), 12)
        result = evaluator.assess_migration(10, 20)
        self.assertFalse(result.valid)
        self.assertIn("history_overflow", result.reasons)
        self.assertEqual(result.groups[0].d2h_intervals, ((11, 12),))
        self.assertEqual(len(result.groups[0].d2h_evidence), 1)

    def test_invalid_windows_are_rejected(self):
        evaluator = self.evaluator()
        for start, end in ((10, 10), (20, 10), (float("nan"), 20)):
            with self.subTest(start=start, end=end), self.assertRaises(ValueError):
                evaluator.assess_migration(start, end)


class CoverageTest(unittest.TestCase):
    def test_declared_shared_owner_preserves_raw_peers_and_owner_clock(self):
        from rtp_llm.test.continuous_pressure.diagnostics import PressureCoverage

        def page(t, baseline=False):
            message = shared_wire(t, baseline)
            return decode_pressure_response(message, t, t, (0, 1)).samples

        baseline = page(0, True)
        evaluator = PressureCoverage(baseline, configured())
        for t in range(1, 5):
            samples = page(t)
            self.assertEqual(samples[1].pools[0].used, 0)
            self.assertEqual(samples[1].pools[1].host_resident, 0)
            evaluator.add(samples, t)
        self.assertTrue(evaluator.assess(1, 4).valid)
        self.assertEqual(evaluator.intervals(1, 4), [(1, 4)])

    def test_interval_contract(self):
        from rtp_llm.test.continuous_pressure.diagnostics import (
            confirmed_segment,
            coverage,
            intersect_intervals,
            merge_intervals,
        )

        self.assertEqual(coverage([(0, 57)], 0, 60), (0.95, 3))
        self.assertEqual(merge_intervals([(0, 1), (1, 2), (2, 4)]), [(0, 4)])
        self.assertEqual(confirmed_segment(0, 4, -0.01, 0.01), (0.01, 3.99))
        self.assertEqual(intersect_intervals([(0, 60)], [(1, 59)]), [(1, 59)])

    def test_adjacent_samples_and_reset(self):
        from rtp_llm.test.continuous_pressure.diagnostics import PressureCoverage

        baseline = dataclasses.replace(
            rank(), groups=(GroupSample(0, "FULL", ("gpu",), "host", 0, 0),)
        )
        evaluator = PressureCoverage((baseline,), configured())
        for t in range(4):
            evaluator.add(
                (
                    dataclasses.replace(
                        rank(),
                        sequence=t + 2,
                        server_s=t,
                        client_send_s=t,
                        client_receive_s=t,
                    ),
                ),
                t,
            )
            if t == 2:
                self.assertFalse(evaluator.assess(0, 3).valid)
        self.assertEqual(evaluator.intervals(0, 3), [(0, 3)])
        evaluator.add(
            (
                dataclasses.replace(
                    rank(), sequence=1, server_s=4, client_send_s=4, client_receive_s=4
                ),
            ),
            4,
        )
        self.assertIn("sequence_reset", evaluator.reasons)
        self.assertEqual(evaluator.intervals(3, 4), [])

    def test_first_sample_sequence_must_exceed_prefill_baseline(self):
        from rtp_llm.test.continuous_pressure.diagnostics import PressureCoverage

        baseline = dataclasses.replace(
            rank(),
            sequence=10,
            groups=(GroupSample(0, "FULL", ("gpu",), "host", 0, 0),),
        )
        for first_sequence in (9, 10):
            with self.subTest(first_sequence=first_sequence):
                evaluator = PressureCoverage((baseline,), configured())
                for t in range(5):
                    evaluator.add(
                        (
                            dataclasses.replace(
                                rank(),
                                sequence=first_sequence + t,
                                server_s=t,
                                client_send_s=t,
                                client_receive_s=t,
                            ),
                        ),
                        t,
                    )
                self.assertFalse(evaluator.epoch_valid)
                self.assertIn("sequence_reset", evaluator.reasons)
                self.assertEqual(evaluator.intervals(0, 4), [])
                self.assertFalse(evaluator.assess(0, 4).valid)

    def test_activity_timeout(self):
        from rtp_llm.test.continuous_pressure.diagnostics import (
            ActivitySample,
            activity_valid,
        )

        sample = ActivitySample(0, 0, 5, 1, 0)
        self.assertTrue(activity_valid(sample, 10, 6))
        self.assertFalse(activity_valid(sample, 12, 6))
        self.assertFalse(activity_valid(ActivitySample(0, 0, None, 0, 1), 100, 6))

    def test_uncertainty_is_applied_only_to_segment_edges(self):
        from rtp_llm.test.continuous_pressure.diagnostics import PressureCoverage

        base = dataclasses.replace(
            rank(), groups=(GroupSample(0, "FULL", ("gpu",), "host", 0, 0),)
        )
        e = PressureCoverage((base,), configured())
        for t in range(5):
            e.add(
                (
                    dataclasses.replace(
                        rank(),
                        sequence=t + 2,
                        server_s=t,
                        client_send_s=t,
                        client_receive_s=t + 0.02,
                        offset_bounds=(-0.01, 0.01),
                    ),
                ),
                t + 0.02,
            )
        self.assertEqual(e.intervals(0, 4), [(0.01, 3.99)])

    def test_window_reports_cumulative_reasons_with_explicit_epoch_scope(self):
        from rtp_llm.test.continuous_pressure.diagnostics import PressureCoverage

        base = dataclasses.replace(
            rank(), groups=(GroupSample(0, "FULL", ("gpu",), "host", 0, 0),)
        )
        evaluator = PressureCoverage((base,), configured())
        for t in range(5):
            point = dataclasses.replace(
                rank(), sequence=t + 2, server_s=t, client_send_s=t, client_receive_s=t
            )
            if t == 0:
                point = dataclasses.replace(point, groups=base.groups)
            evaluator.add((point,), t)
        result = dataclasses.asdict(evaluator.assess(2, 4))
        self.assertTrue(result["valid"])
        self.assertEqual(result["ratio"], 1)
        self.assertNotIn("reasons", result)
        self.assertIn("pressure_lost", result["epoch_reasons"])

    def test_gap_age_missing_and_retention(self):
        from rtp_llm.test.continuous_pressure.diagnostics import PressureCoverage

        base = dataclasses.replace(
            rank(), groups=(GroupSample(0, "FULL", ("gpu",), "host", 0, 0),)
        )

        def point(t, sequence):
            return dataclasses.replace(
                rank(),
                sequence=sequence,
                server_s=t,
                client_send_s=t,
                client_receive_s=t,
            )

        e = PressureCoverage((base,), configured(), max_segments=1)
        e.add((point(0, 2),), 0)
        e.add((point(2, 3),), 2)
        self.assertEqual(e.intervals(0, 2), [(0, 2)])
        e.add((point(4.01, 4),), 4.01)
        self.assertIn("sample_gap", e.reasons)
        e.add((point(5, 5),), 8)
        self.assertIn("stale_or_unknown_age", e.reasons)
        e.add((), 9)
        e.add((point(10, 6),), 10)
        e.add((point(11, 7),), 11)
        self.assertEqual(e.intervals(0, 11), [(10, 11)])
        self.assertIn("history_overflow", e.reasons)

    def test_rank_intersection_and_endpoint_reset(self):
        from rtp_llm.test.continuous_pressure.diagnostics import PressureCoverage

        bases = tuple(
            dataclasses.replace(
                rank(r), groups=(GroupSample(0, "FULL", ("gpu",), "host", 0, 0),)
            )
            for r in (0, 1)
        )
        e = PressureCoverage(bases, configured())
        for t in range(4):
            e.add(
                tuple(
                    dataclasses.replace(
                        rank(r),
                        sequence=t + 2,
                        server_s=t,
                        client_send_s=t,
                        client_receive_s=t,
                        pools=(
                            rank(r).pools
                            if t > 0 or r == 0
                            else (
                                PoolSample("gpu", 100, 0, 0, 0.8, 0.9),
                                rank(r).pools[1],
                            )
                        ),
                    )
                    for r in (0, 1)
                ),
                t,
            )
        self.assertEqual(e.intervals(0, 3), [(1, 3)])
        message = wire()
        sample = decode_pressure_response(message, 1, 1.1, (0,)).samples[0]
        base = dataclasses.replace(
            sample,
            sequence=0,
            groups=(
                dataclasses.replace(sample.groups[0], triggers=0, gpu_host_published=0),
            ),
        )
        e = PressureCoverage((base,), configured())
        e.add((sample,), 1.1)
        message.ranks[0].sequence = 2
        message.ranks[0].sampled_monotonic_ns = 2000000000
        message.ranks[0].transfers[0].published_blocks = 6
        message.ranks[0].transfers[1].published_blocks = 100
        e.add(decode_pressure_response(message, 2, 2.1, (0,)).samples, 2.1)
        self.assertFalse(e.epoch_valid)
        self.assertIn("counter_reset", e.reasons)

    def test_clock_discontinuity_and_reset_after_missing_poll(self):
        from rtp_llm.test.continuous_pressure.diagnostics import PressureCoverage

        base = dataclasses.replace(
            rank(), groups=(GroupSample(0, "FULL", ("gpu",), "host", 0, 0),)
        )
        e = PressureCoverage((base,), configured())
        e.add(
            (
                dataclasses.replace(
                    rank(), sequence=2, server_s=0, client_send_s=0, client_receive_s=0
                ),
            ),
            0,
        )
        e.add(
            (
                dataclasses.replace(
                    rank(),
                    sequence=3,
                    server_s=1,
                    client_send_s=1.1,
                    client_receive_s=1.1,
                ),
            ),
            1.1,
        )
        self.assertIn("clock_discontinuity", e.reasons)
        self.assertEqual(e.intervals(0, 2), [])
        e.add((), 2)
        e.add(
            (
                dataclasses.replace(
                    rank(), sequence=2, server_s=3, client_send_s=3, client_receive_s=3
                ),
            ),
            3,
        )
        self.assertFalse(e.epoch_valid)
        self.assertIn("sequence_reset", e.reasons)

    def test_fanout_offset_composition(self):
        message = wire()
        message.coordinator_received_ns = 10_000_000_000
        message.coordinator_sent_ns = 10_300_000_000
        message.ranks[0].coordinator_send_ns = 10_100_000_000
        message.ranks[0].coordinator_receive_ns = 10_200_000_000
        sample = decode_pressure_response(message, 100, 100.5, (0,)).samples[0]
        self.assertAlmostEqual(sample.offset_bounds[0], 99.1)
        self.assertAlmostEqual(sample.offset_bounds[1], 99.4)


class StepClassificationTest(unittest.TestCase):
    def step(self, **changes):
        return dict(
            dict(
                step_id=1,
                complete=True,
                context_batch=1,
                decode_batch=0,
                begin_ns=10,
                end_ns=20,
                forward_enter_ns=12,
                forward_return_ns=18,
                forward_enter_cpu_ns=5,
                forward_return_cpu_ns=8,
                member_phases=["measurement"],
                member_kinds=["formal"],
                session_id="s",
                instance="i",
            ),
            **changes,
        )

    def test_gap_requires_adjacent_same_executor_steps(self):
        previous = self.step()
        current = self.step(
            step_id=2,
            begin_ns=30,
            end_ns=40,
            forward_enter_ns=32,
            forward_enter_cpu_ns=11,
        )
        self.assertTrue(results.step_eligible(previous, 10, 20))
        self.assertEqual(
            results.step_gap(previous, current), {"wall_ns": 14, "cpu_ns": 3}
        )
        for changes in (
            dict(step_id=3),
            dict(instance="other"),
            dict(session_id="other"),
            dict(forward_enter_cpu_ns=7),
            dict(forward_enter_ns=17),
        ):
            self.assertIsNone(results.step_gap(previous, dict(current, **changes)))

    def test_exclusion_categories_and_uncertain_boundaries(self):
        for changes, reason in (
            (dict(member_phases=["support"]), "support"),
            (
                dict(
                    member_phases=["measurement", "support"],
                    member_kinds=["formal"] * 2,
                    context_batch=2,
                ),
                "mixed_phase",
            ),
            (dict(member_kinds=["padding"]), "padding"),
            (dict(member_phases=[]), "unknown_membership"),
            (dict(decode_batch=1), "decode"),
            (dict(complete=False), "incomplete"),
            (dict(begin_ns=9), "boundary"),
        ):
            step = self.step(**changes)
            self.assertFalse(results.step_eligible(step, 10, 20))
            self.assertEqual(results.step_exclusion(step, 10, 20), reason)
        self.assertEqual(results.server_window(100, 200, (1.2, 2.8)), (99, 197))

    def test_distribution_is_bounded_and_labels_tail_heuristic(self):
        distribution = results.BoundedDistribution(capacity=16)
        for value in range(999):
            distribution.add(value)
        summary = distribution.summary()
        self.assertEqual(summary["count"], 999)
        self.assertEqual(summary["reservoir_size"], 16)
        self.assertEqual(summary["mean"], 499)
        self.assertEqual(summary["max"], 998)
        self.assertEqual(summary["p99_status"], "insufficient")
        distribution.add(999)
        self.assertEqual(distribution.summary()["p99_status"], "estimated")


class StepEvidenceTest(unittest.TestCase):
    def page(self, rank, ids=(1, 2), lost=False, source=True):
        return dict(
            rank_offset_bounds_s={str(rank): [0, 0]},
            response=dict(
                ranks=[
                    dict(
                        world_rank=rank,
                        service_instance="rank" + str(rank),
                        step_records_lost=lost,
                        step_history_available=True,
                        steps=[
                            dict(
                                session_id="s",
                                step_id=i,
                                sequence=i,
                                completed=True,
                                preparation_begin_ns=i * 100,
                                forward_enter_ns=i * 100 + 10,
                                forward_return_ns=i * 100 + 20,
                                end_ns=i * 100 + 20,
                                cpu_start_ns=i * 10,
                                cpu_end_ns=i * 10 + 5,
                                context_count=1,
                                source_request_ids=(
                                    ["req"] if source and rank == 0 else []
                                ),
                            )
                            for i in ids
                        ],
                    )
                ]
            ),
        )

    def make(self, capacity=4):
        analysis = results.StepAnalysis((0, 1), "s", capacity=capacity)
        analysis.window = (0, 10000)
        analysis.dispatch(dict(request_id="req", phase="measurement", kind="formal"))
        return analysis

    def test_delayed_root_joins_peer_by_id_and_independent_instance(self):
        analysis = self.make()
        analysis.consume(self.page(1))
        self.assertEqual(analysis.summary(final=False)["ranks"]["1"]["eligible"], 0)
        analysis.consume(self.page(0))
        summary = analysis.summary()
        for rank in ("0", "1"):
            self.assertEqual(summary["ranks"][rank]["eligible"], 2)
            self.assertEqual(summary["ranks"][rank]["gap_wall_ns"]["mean"], 90)
            self.assertEqual(summary["ranks"][rank]["gap_cpu_ns"]["mean"], 5)

    def test_loss_breaks_adjacency_unknown_membership_stays_excluded(self):
        analysis = self.make()
        analysis.consume(self.page(0, (1,)))
        analysis.consume(self.page(0, (2,), lost=True))
        analysis.consume(self.page(1, (3,), source=False))
        summary = analysis.summary()
        self.assertEqual(summary["ranks"]["0"]["gap_wall_ns"]["count"], 0)
        self.assertEqual(summary["ranks"]["1"]["exclusions"]["unknown_membership"], 1)

    def test_peer_loss_does_not_bridge_pending_pages(self):
        analysis = self.make()
        analysis.consume(self.page(1, (1,)))
        analysis.consume(self.page(1, (2,), lost=True))
        analysis.consume(self.page(0))
        self.assertEqual(analysis.summary()["ranks"]["1"]["gap_wall_ns"]["count"], 0)

    def test_sequence_gap_breaks_pending_adjacency_without_loss_flag(self):
        analysis = self.make()
        analysis.consume(self.page(1, (1,)))
        page = self.page(0)
        peer = self.page(1, (2,))["response"]["ranks"][0]
        peer["steps"][0]["sequence"] = 3
        page["response"]["ranks"].append(peer)
        page["rank_offset_bounds_s"]["1"] = [0, 0]
        analysis.consume(page)
        self.assertEqual(analysis.summary()["ranks"]["1"]["gap_wall_ns"]["count"], 0)

    def test_active_zero_cursor_does_not_hide_completed_and_restart_excludes(self):
        analysis = self.make()
        page = self.page(0, (1,))
        active = dict(
            page["response"]["ranks"][0]["steps"][0],
            sequence=0,
            step_id=2,
            completed=False,
        )
        page["response"]["ranks"][0]["steps"].append(active)
        analysis.consume(page)
        self.assertEqual(analysis.last_sequence[0], 1)
        analysis.consume(self.page(0, (2,)))
        page = self.page(0, (3,))
        page["response"]["ranks"][0]["service_instance"] = "restart"
        analysis.consume(page)
        summary = analysis.summary()["ranks"]["0"]
        self.assertEqual(summary["eligible"], 2)
        self.assertEqual(summary["exclusions"]["unknown_membership"], 1)

    def test_clock_uncertainty_rejects_boundary_without_subtracting_cpu_from_wall(self):
        analysis = self.make()
        analysis.window = (100, 220)
        page = self.page(0)
        page["rank_offset_bounds_s"] = {"0": [-1e-9, 1e-9]}
        analysis.consume(page)
        self.assertEqual(analysis.summary()["ranks"]["0"]["exclusions"]["boundary"], 2)
        analysis = self.make()
        page = self.page(0)
        page["response"]["ranks"][0]["steps"][1]["cpu_start_ns"] = 1000
        page["response"]["ranks"][0]["steps"][1]["cpu_end_ns"] = 1005
        analysis.consume(page)
        summary = analysis.summary()["ranks"]["0"]
        self.assertEqual(summary["gap_wall_ns"]["mean"], 90)
        self.assertEqual(summary["gap_cpu_ns"]["mean"], 985)

    def test_state_is_bounded_and_raw_precedes_analysis(self):
        analysis = self.make(capacity=2)
        for i in range(10):
            analysis.dispatch(dict(request_id=str(i), phase="support", kind="formal"))
            analysis.consume(self.page(1, (i + 1,)))
        self.assertLessEqual(len(analysis.requests), 2)
        self.assertLessEqual(sum(len(v) for v in analysis.pending.values()), 4)
        self.assertGreater(analysis.summary()["state_evictions"], 0)
        with tempfile.TemporaryDirectory() as directory:
            writer = PressureWriter(Path(directory), {})
            evidence = PressureEvidence(writer, target_ranks=(0, 1), session_id="s")
            evidence(dict(event="pressure", raw=self.page(0)))
            writer.close()
            raw = json.loads(
                (Path(directory) / "pressure/cache_samples.jsonl").read_text()
            )
            self.assertEqual(
                raw["raw"]["response"]["ranks"][0]["steps"][0]["source_request_ids"],
                ["req"],
            )


class PressureResultsTest(unittest.TestCase):
    def test_mock_frozen_policies_are_always_diagnostic_only(self):
        for policy in (
            "mock_memory",
            "tree_then_mock_memory",
            "tree_then_mock_disk",
            "tree_then_mock_remote",
        ):
            for marker in (None, False, True):
                with self.subTest(
                    policy=policy, marker=marker
                ), tempfile.TemporaryDirectory() as directory:
                    plan = self.plan()
                    if marker is not None:
                        plan["diagnostic_only"] = marker
                    writer = PressureWriter(Path(directory), {"formal_candidate": True})
                    sink = PressureEvidence(writer)
                    sink(
                        dict(
                            event="frozen",
                            session_id="s",
                            config=dict(measure_s=60, evidence_policy=policy),
                            seed_plan=plan,
                        )
                    )
                    sink(self.finished())
                    writer.close()
                    for name in ("summary.json", "manifest.json"):
                        artifact = json.loads(
                            (Path(directory) / "pressure" / name).read_text()
                        )
                        self.assertFalse(
                            artifact["classification"]["formal_config_valid"]
                        )
                        self.assertFalse(
                            artifact["validity"]["formal_acceptance_valid"]
                        )
                        self.assertIn(
                            "diagnostic_only",
                            artifact["classification"]["nonformal_reasons"],
                        )
                        if name == "manifest.json":
                            self.assertFalse(artifact["formal_candidate"])
                            self.assertIn(
                                "diagnostic_only", artifact["formal_candidate_reasons"]
                            )

    def plan(self, **changes):
        return dict(
            dict(seed_count=2, formal_config_valid=True, nonformal_reasons=[]),
            **changes,
        )

    def finished(self, **changes):
        return dict(
            dict(
                event="finished",
                failure=None,
                seed=dict(
                    planned=2,
                    submitted=2,
                    completed=2,
                    failed=0,
                    cancelled=0,
                    pending=0,
                ),
                pressure=dict(valid=True, epoch_reasons=[]),
                migration=dict(valid=True, reasons=[]),
            ),
            **changes,
        )

    def artifacts(self, *, plan=None, finished=None, manifest=None, events=()):
        with tempfile.TemporaryDirectory() as directory:
            writer = PressureWriter(Path(directory), manifest or {})
            sink = PressureEvidence(writer)
            sink(
                dict(
                    event="frozen",
                    session_id="s",
                    config=dict(measure_s=60),
                    seed_plan=self.plan() if plan is None else plan,
                )
            )
            for event in events:
                sink(event)
            sink(self.finished() if finished is None else finished)
            writer.close()
            root = Path(directory) / "pressure"
            return (
                json.loads((root / "summary.json").read_text()),
                json.loads((root / "manifest.json").read_text()),
            )

    def test_formal_acceptance_requires_all_four_independent_gates(self):
        summary, manifest = self.artifacts()
        self.assertEqual(
            summary["validity"],
            dict(
                execution_valid=True,
                full_pressure_valid=True,
                formal_config_valid=True,
                bidirectional_migration_valid=True,
                formal_acceptance_valid=True,
                pressure_valid=dict(measurement=True, profile=None, support=None),
                reuse_valid=None,
                trace_valid=None,
                trace_applicability="not_applicable",
            ),
        )
        self.assertEqual(summary["execution_mode"], "full_pressure_diagnostic_v1")
        self.assertEqual(summary["metrics"], [])
        self.assertTrue(manifest["complete"])

    def test_execution_failure_preserves_other_gates_and_reason(self):
        summary, _ = self.artifacts(finished=self.finished(failure="drain_timeout"))
        self.assertFalse(summary["validity"]["execution_valid"])
        self.assertFalse(summary["validity"]["formal_acceptance_valid"])
        for gate in (
            "full_pressure_valid",
            "formal_config_valid",
            "bidirectional_migration_valid",
        ):
            self.assertTrue(summary["validity"][gate])
        self.assertIn("drain_timeout", summary["validity_reasons"]["execution_valid"])

    def test_pressure_failure_does_not_rewrite_execution_or_migration(self):
        summary, _ = self.artifacts(
            finished=self.finished(
                pressure=dict(
                    valid=False, epoch_reasons=["unknown_activity", "sample_gap"]
                )
            )
        )
        self.assertTrue(summary["validity"]["execution_valid"])
        self.assertTrue(summary["validity"]["bidirectional_migration_valid"])
        self.assertFalse(summary["validity"]["full_pressure_valid"])
        self.assertFalse(summary["validity"]["formal_acceptance_valid"])
        self.assertEqual(
            summary["validity_reasons"]["full_pressure_valid"],
            ["unknown_activity", "sample_gap"],
        )

    def test_serialized_coverage_assessment_retains_epoch_failure_reasons(self):
        from dataclasses import asdict

        from rtp_llm.test.continuous_pressure.diagnostics import CoverageAssessment

        for reasons in (
            ("counter_reset",),
            ("inventory_discontinuity",),
            ("counter_reset", "inventory_discontinuity"),
        ):
            with self.subTest(reasons=reasons):
                pressure = asdict(
                    CoverageAssessment(
                        epoch_valid=False,
                        max_segments_per_target=4096,
                        ratio=0,
                        longest_invalid_s=60,
                        valid=False,
                        intervals=(),
                        epoch_reasons=reasons,
                    )
                )
                summary, manifest = self.artifacts(
                    finished=self.finished(pressure=pressure)
                )
                self.assertFalse(summary["validity"]["full_pressure_valid"])
                self.assertFalse(summary["validity"]["formal_acceptance_valid"])
                self.assertTrue(summary["validity"]["execution_valid"])
                self.assertTrue(summary["validity"]["formal_config_valid"])
                self.assertTrue(summary["validity"]["bidirectional_migration_valid"])
                for artifact in (summary, manifest):
                    self.assertEqual(
                        artifact["validity_reasons"]["full_pressure_valid"],
                        list(reasons),
                    )
                    self.assertEqual(
                        artifact["validity_reasons"]["formal_acceptance_valid"],
                        list(reasons),
                    )

    def test_manual_plan_stays_nonformal_after_all_seeds_complete(self):
        summary, _ = self.artifacts(
            plan=self.plan(
                formal_config_valid=False, nonformal_reasons=["manual_seed_count"]
            )
        )
        self.assertTrue(summary["validity"]["execution_valid"])
        self.assertTrue(summary["validity"]["full_pressure_valid"])
        self.assertFalse(summary["validity"]["formal_config_valid"])
        self.assertFalse(summary["validity"]["formal_acceptance_valid"])
        self.assertEqual(
            summary["classification"]["nonformal_reasons"], ["manual_seed_count"]
        )

    def test_d2h_only_migration_failure_preserves_execution_and_full_pressure(self):
        migration = dict(
            valid=False,
            reasons=["missing_h2d_load:0:3"],
            groups=[dict(owner_rank=0, owner_group_id=3)],
        )
        summary, _ = self.artifacts(finished=self.finished(migration=migration))
        self.assertTrue(summary["validity"]["execution_valid"])
        self.assertTrue(summary["validity"]["full_pressure_valid"])
        self.assertFalse(summary["validity"]["bidirectional_migration_valid"])
        self.assertFalse(summary["validity"]["formal_acceptance_valid"])
        self.assertEqual(
            summary["validity_reasons"]["bidirectional_migration_valid"],
            migration["reasons"],
        )
        self.assertEqual(summary["result"]["migration"], migration)

    def test_seed_outcome_discrepancies_fail_execution_without_hiding_counts(self):
        for change, reason in (
            (dict(completed=1), "seed_planned_completed_mismatch"),
            (dict(planned=1, completed=1, submitted=1), "seed_plan_outcome_mismatch"),
            (dict(submitted=1), "seed_planned_submitted_mismatch"),
            (dict(failed=1), "seed_failed"),
            (dict(cancelled=1), "seed_cancelled"),
            (dict(pending=1), "seed_pending"),
            (dict(pending=True), "seed_outcome_invalid"),
        ):
            with self.subTest(change=change):
                seed = dict(self.finished()["seed"], **change)
                summary, manifest = self.artifacts(finished=self.finished(seed=seed))
                self.assertFalse(summary["validity"]["execution_valid"])
                self.assertFalse(summary["validity"]["formal_acceptance_valid"])
                self.assertIn(reason, summary["validity_reasons"]["execution_valid"])
                self.assertEqual(manifest["seed_outcome"], seed)
                self.assertEqual(summary["seed_outcome"], seed)

    def test_headroom_limited_seed_uses_the_effective_completed_count(self):
        seed = dict(
            planned=48,
            submitted=48,
            completed=48,
            failed=0,
            cancelled=0,
            pending=0,
            configured_limit=1000,
            effective_seed_count=48,
            stop_reason="host_max_ratio",
            stop_host_pools=[],
        )

        summary, manifest = self.artifacts(
            plan=self.plan(seed_count=1000), finished=self.finished(seed=seed)
        )

        self.assertTrue(summary["validity"]["execution_valid"])
        self.assertEqual(summary["validity_reasons"]["execution_valid"], [])
        self.assertEqual(manifest["seed_outcome"], seed)

    def test_frozen_seed_count_requires_a_positive_integer(self):
        for count, terminal_count in (
            (True, 1),
            (1.0, 1),
            (0, 0),
            (False, 0),
            (-1, 0),
            (None, 2),
        ):
            with self.subTest(count=count):
                seed = dict(
                    planned=terminal_count,
                    submitted=terminal_count,
                    completed=terminal_count,
                    failed=0,
                    cancelled=0,
                    pending=0,
                )
                summary, manifest = self.artifacts(
                    plan=self.plan(seed_count=count), finished=self.finished(seed=seed)
                )
                self.assertFalse(summary["validity"]["execution_valid"])
                self.assertFalse(summary["validity"]["formal_config_valid"])
                self.assertFalse(summary["validity"]["formal_acceptance_valid"])
                self.assertTrue(summary["validity"]["full_pressure_valid"])
                self.assertTrue(summary["validity"]["bidirectional_migration_valid"])
                self.assertIn(
                    "seed_plan_count_invalid",
                    summary["validity_reasons"]["execution_valid"],
                )
                self.assertIn(
                    "seed_plan_count_invalid",
                    summary["validity_reasons"]["formal_config_valid"],
                )
                self.assertEqual(manifest["seed_plan"]["seed_count"], count)
                self.assertEqual(manifest["seed_outcome"], seed)

    def test_invalid_plan_count_preserves_independent_seed_failure_reasons(self):
        summary, _ = self.artifacts(
            plan=self.plan(seed_count=True),
            finished=self.finished(
                seed=dict(
                    planned=2,
                    submitted=2,
                    completed=0,
                    failed=1,
                    cancelled=1,
                    pending=1,
                )
            ),
        )
        for reason in (
            "seed_plan_count_invalid",
            "seed_planned_completed_mismatch",
            "seed_failed",
            "seed_cancelled",
            "seed_pending",
        ):
            self.assertIn(reason, summary["validity_reasons"]["execution_valid"])

    def test_finished_requires_a_successfully_processed_frozen_event(self):
        for initial_frozen, failed_event in (
            (False, False),
            (True, False),
            (False, True),
        ):
            with self.subTest(
                initial_frozen=initial_frozen, failed_event=failed_event
            ), tempfile.TemporaryDirectory() as directory:
                manifest = dict(seed_plan=self.plan())
                if initial_frozen:
                    manifest["frozen"] = dict(
                        event="frozen", seed_plan=self.plan(), session_id="old"
                    )
                writer = PressureWriter(Path(directory), manifest)
                sink = PressureEvidence(writer)
                if failed_event:
                    with self.assertRaises(KeyError):
                        sink(
                            dict(
                                event="frozen",
                                seed_plan=self.plan(),
                                session_id="s",
                                config={},
                            )
                        )
                sink(self.finished())
                writer.close()
                root = Path(directory) / "pressure"
                for name in ("manifest.json", "summary.json"):
                    artifact = json.loads((root / name).read_text())
                    self.assertFalse(artifact["validity"]["execution_valid"])
                    self.assertFalse(artifact["validity"]["formal_acceptance_valid"])
                    self.assertTrue(artifact["validity"]["formal_config_valid"])
                    self.assertTrue(artifact["validity"]["full_pressure_valid"])
                    self.assertTrue(
                        artifact["validity"]["bidirectional_migration_valid"]
                    )
                    self.assertIn(
                        "frozen_event_missing",
                        artifact["validity_reasons"]["execution_valid"],
                    )
                    self.assertEqual(artifact["seed_outcome"], self.finished()["seed"])

    def test_missing_seed_and_migration_evidence_fail_closed(self):
        event = self.finished()
        del event["seed"]
        del event["migration"]
        summary, manifest = self.artifacts(finished=event)
        self.assertFalse(summary["validity"]["execution_valid"])
        self.assertFalse(summary["validity"]["bidirectional_migration_valid"])
        self.assertEqual(
            summary["validity_reasons"]["execution_valid"], ["seed_outcome_missing"]
        )
        self.assertEqual(
            summary["validity_reasons"]["bidirectional_migration_valid"],
            ["migration_evidence_missing"],
        )
        self.assertIsNone(manifest["seed_outcome"])

    def test_initial_plan_exclusions_survive_a_later_formal_plan(self):
        summary, _ = self.artifacts(
            manifest=dict(
                seed_plan=self.plan(
                    formal_config_valid=False, nonformal_reasons=["explicit_smoke"]
                )
            )
        )
        self.assertFalse(summary["validity"]["formal_config_valid"])
        self.assertEqual(
            summary["classification"]["nonformal_reasons"], ["explicit_smoke"]
        )

    def test_malformed_seed_count_preserves_other_seed_failures(self):
        summary, _ = self.artifacts(
            finished=self.finished(
                seed=dict(
                    planned=True,
                    submitted=2,
                    completed=0,
                    failed=1,
                    cancelled=1,
                    pending=1,
                )
            )
        )
        self.assertEqual(
            summary["validity_reasons"]["execution_valid"],
            ["seed_outcome_invalid", "seed_failed", "seed_cancelled", "seed_pending"],
        )

    def test_all_independent_reasons_survive_in_stable_classification_order(self):
        earlier = [
            "grid_reuse_below_32768",
            "explicit_smoke",
            "synthetic_test_block_capacity",
            "client_inflight_below_16",
            "manual_seed_count",
            "earlier_preflight_exclusion",
        ]
        summary, manifest = self.artifacts(
            manifest=dict(formal_candidate=False, formal_candidate_reasons=earlier),
            plan=self.plan(formal_config_valid=True, nonformal_reasons=[]),
            finished=self.finished(
                failure="seed_timeout",
                seed=dict(
                    planned=2,
                    submitted=2,
                    completed=0,
                    failed=1,
                    cancelled=1,
                    pending=1,
                ),
                pressure=dict(valid=False, epoch_reasons=["pressure_gap"]),
                migration=dict(
                    valid=False, reasons=["missing_d2h:0:0", "missing_h2d_load:0:0"]
                ),
            ),
            events=[dict(event="error", reason="independent_supervisor_error")],
        )
        expected = [
            "manual_seed_count",
            "client_inflight_below_16",
            "synthetic_test_block_capacity",
            "explicit_smoke",
            "grid_reuse_below_32768",
            "earlier_preflight_exclusion",
        ]
        self.assertEqual(summary["classification"]["nonformal_reasons"], expected)
        self.assertEqual(manifest["classification"], summary["classification"])
        reasons = summary["validity_reasons"]
        for reason in (
            "seed_timeout",
            "independent_supervisor_error",
            "seed_failed",
            "seed_cancelled",
            "seed_pending",
        ):
            self.assertIn(reason, reasons["execution_valid"])
        self.assertEqual(reasons["formal_config_valid"], expected)
        self.assertEqual(reasons["full_pressure_valid"], ["pressure_gap"])
        self.assertEqual(
            reasons["bidirectional_migration_valid"],
            ["missing_d2h:0:0", "missing_h2d_load:0:0"],
        )
        self.assertEqual(manifest["validity_reasons"], reasons)
        self.assertTrue(
            all(
                summary["validity"][gate] is False
                for gate in (
                    "execution_valid",
                    "full_pressure_valid",
                    "formal_config_valid",
                    "bidirectional_migration_valid",
                    "formal_acceptance_valid",
                )
            )
        )

    def test_serialized_summary_recalculates_automatic_count_from_raw_inputs(self):
        from dataclasses import asdict

        from rtp_llm.test.continuous_pressure.diagnostics import select_targets
        from rtp_llm.test.continuous_pressure.seed_plan import build_seed_plan
        from rtp_llm.test.continuous_pressure.tests.core_test import owner

        sample = owner()
        plan = asdict(
            build_seed_plan(
                (sample,),
                select_targets((sample,), (0,)),
                cases=(dict(cache_len=32768),),
                cache_alignment=4096,
                commit_tail_tokens=4096,
                client_inflight=16,
                manual_seed_count=None,
                fill_target_tokens=102400,
                test_block_num=0,
            )
        )
        startup = dict(raw_owner=asdict(sample))
        seed = dict(
            planned=plan["seed_count"],
            submitted=plan["seed_count"],
            completed=plan["seed_count"],
            failed=0,
            cancelled=0,
            pending=0,
        )
        summary, manifest = self.artifacts(
            plan=plan,
            manifest=dict(startup_pressure=startup),
            finished=self.finished(seed=seed),
        )
        for artifact in (summary, manifest):
            self.assertEqual(artifact["seed_plan"], json.loads(json.dumps(plan)))
            self.assertEqual(
                artifact["startup_pressure"], json.loads(json.dumps(startup))
            )
            saved = artifact["seed_plan"]
            owner = artifact["startup_pressure"]["raw_owner"]
            pools = {pool["pool_id"]: pool for pool in owner["pools"]}
            lower, upper = [], []
            for group in saved["groups"]:
                blocks = (
                    saved["reusable_tokens"] + group["block_tokens"] - 1
                ) // group["block_tokens"]
                lower.append(
                    min(
                        pools[p["pool_id"]]["capacity"] // blocks
                        for p in group["device_pools"]
                    )
                    + 1
                )
                upper.append(
                    min(
                        int(pools[p["pool_id"]]["capacity"] * p["low_ratio"]) // blocks
                        for p in group["device_pools"]
                    )
                    + pools[group["host_pool"]["pool_id"]]["available"] // blocks
                )
            self.assertEqual(saved["seed_count"], max(lower))
            self.assertLessEqual(saved["seed_count"], min(upper))
            self.assertEqual(artifact["seed_outcome"], seed)
            json.dumps(artifact, allow_nan=False)

    def test_frozen_plan_is_required_and_copied_before_terminal_outcome(self):
        with tempfile.TemporaryDirectory() as directory:
            writer = PressureWriter(Path(directory), {})
            sink = PressureEvidence(writer)
            with self.assertRaisesRegex((KeyError, ValueError), "seed_plan"):
                sink(dict(event="frozen", session_id="s", config=dict(measure_s=60)))
            plan = self.plan()
            frozen = dict(
                event="frozen",
                session_id="s",
                config=dict(measure_s=60),
                seed_plan=plan,
            )
            sink(frozen)
            expected = copy.deepcopy(frozen)
            plan["nonformal_reasons"].append("mutated_after_freeze")
            sink(self.finished())
            writer.close()
            manifest = json.loads(
                (Path(directory) / "pressure/manifest.json").read_text()
            )
            self.assertEqual(manifest["frozen"], expected)
            self.assertEqual(manifest["seed_plan"], expected["seed_plan"])

    def test_profile_first_failure_does_not_mask_execution_evidence(self):
        for terminal in (
            "failed",
            "cancelled",
            "unfinished",
            "clean",
            "error",
            "shape",
        ):
            with self.subTest(
                terminal=terminal
            ), tempfile.TemporaryDirectory() as directory:
                writer = PressureWriter(Path(directory), {})
                sink = PressureEvidence(writer, profile_requested=True)
                sink(
                    dict(
                        event="frozen",
                        session_id="s",
                        config=dict(measure_s=60),
                        seed_plan=self.plan(),
                    )
                )
                request = dict(
                    event="dispatch",
                    job_id=1,
                    request_id="a",
                    phase="support",
                    dispatched_s=71,
                    input_len=16,
                )
                sink(request)
                if terminal in {"failed", "clean", "error", "shape"}:
                    sink(
                        dict(
                            event="completion",
                            job_id=1,
                            completed_s=72,
                            result=dict(
                                success=terminal != "failed",
                                input_len=16,
                                output_len=2 if terminal == "shape" else 1,
                            ),
                        )
                    )
                elif terminal == "cancelled":
                    sink(dict(event="cancellation", job_id=1, at_s=72, accepted=True))
                if terminal == "error":
                    sink(dict(event="error", reason="padding_failed"))
                sink(self.finished(failure="profile_timeout"))
                writer.close()
                summary = json.loads(
                    (Path(directory) / "pressure/summary.json").read_text()
                )
                self.assertEqual(summary["result"]["failure"], "profile_timeout")
                self.assertEqual(
                    summary["validity"]["execution_valid"], terminal == "clean"
                )

    def test_writer_rejects_nonfinite_and_unknown_stream(self):
        with tempfile.TemporaryDirectory() as directory:
            writer = PressureWriter(Path(directory), {})
            with self.assertRaises(ValueError):
                writer.append("requests", dict(latency=float("nan")))
            with self.assertRaises(ValueError):
                writer.append("other", {})
            writer.close()

    def test_unfinished_cohort_cancel_is_not_server_completion(self):
        with tempfile.TemporaryDirectory() as directory:
            writer = PressureWriter(Path(directory), {})
            sink = PressureEvidence(writer)
            sink(
                dict(
                    event="frozen",
                    session_id="s",
                    config=dict(measure_s=60),
                    seed_plan=self.plan(),
                )
            )
            sink(dict(event="phase", phase="measurement", at_s=10))
            sink(
                dict(
                    event="dispatch",
                    job_id=1,
                    request_id="a",
                    phase="measurement",
                    kind="formal",
                    dispatched_s=11,
                )
            )
            sink(dict(event="cancellation", job_id=1, at_s=72, accepted=True))
            sink(dict(event="finished", failure="drain_timeout"))
            writer.close()
            root = Path(directory) / "pressure"
            row = json.loads((root / "requests.jsonl").read_text())
            self.assertTrue(row["transport_cancel_accepted"])
            self.assertIsNone(row["completed_s"])
            summary = json.loads((root / "summary.json").read_text())
            self.assertEqual(summary["window"]["measurement_requests"], 1)
            self.assertEqual(summary["window"]["request_throughput"], 0)
            self.assertFalse(summary["validity"]["execution_valid"])

    def test_accepted_request_artifact_uses_confirmed_time_and_lag(self):
        for outcome in (
            "success",
            "failure",
            "cancellation",
            "unconfirmed_cancellation",
        ):
            with self.subTest(
                outcome=outcome
            ), tempfile.TemporaryDirectory() as directory:
                writer = PressureWriter(Path(directory), {})
                sink = PressureEvidence(writer)
                sink(
                    dict(
                        event="frozen",
                        session_id="s",
                        config=dict(measure_s=1),
                        seed_plan=self.plan(),
                    )
                )
                sink(dict(event="phase", phase="measurement", at_s=10))
                attempt = dict(
                    event="dispatch",
                    job_id=1,
                    request_id="s:measurement:1",
                    phase="measurement",
                    kind="formal",
                    attempted_dispatch_s=10.5,
                    scheduled_offset_s=0.4,
                    input_len=16,
                )
                sink(attempt)
                accepted = outcome != "unconfirmed_cancellation"
                completed = outcome in ("success", "failure")
                confirmed = (
                    dict(dispatched_s=10.75, schedule_lag_s=0.35) if accepted else {}
                )
                if not completed:
                    sink(
                        dict(
                            event="cancellation",
                            job_id=1,
                            dispatch=confirmed,
                            at_s=11.2,
                        )
                    )
                else:
                    sink(
                        dict(
                            event="completion",
                            job_id=1,
                            completed_s=11.2,
                            result=dict(
                                success=outcome == "success", input_len=16, output_len=1
                            ),
                            **confirmed,
                        )
                    )
                sink(self.finished())
                writer.close()
                root = Path(directory) / "pressure"
                row = json.loads((root / "requests.jsonl").read_text())
                self.assertEqual(row["dispatch_s"], 10.75 if accepted else None)
                self.assertEqual(row.get("schedule_lag_s"), 0.35 if accepted else None)
                self.assertEqual(row["attempted_dispatch_s"], 10.5)
                self.assertEqual(row["phase"], "measurement")
                self.assertEqual(row["success"], outcome == "success")
                summary = json.loads((root / "summary.json").read_text())
                self.assertEqual(
                    summary["window"]["measurement_requests"], int(accepted)
                )
                self.assertEqual(summary["window"]["crossing_requests"], int(completed))
                self.assertEqual(summary["window"]["in_window_completions"], 0)

    def test_rejected_dispatch_is_recorded_outside_measurement_cohort(self):
        with tempfile.TemporaryDirectory() as directory:
            writer = PressureWriter(Path(directory), {})
            sink = PressureEvidence(writer)
            sink(
                dict(
                    event="frozen",
                    session_id="s",
                    config=dict(measure_s=1),
                    seed_plan=self.plan(),
                )
            )
            sink(dict(event="phase", phase="measurement", at_s=10))
            sink(
                dict(
                    event="dispatch",
                    job_id=1,
                    request_id="s:measurement:1",
                    phase="measurement",
                    kind="formal",
                    attempted_dispatch_s=10.5,
                    input_len=16,
                )
            )
            sink(
                dict(
                    event="completion",
                    job_id=1,
                    completed_s=11,
                    result=dict(not_dispatched=True),
                )
            )
            sink(self.finished())
            writer.close()
            root = Path(directory) / "pressure"
            row = json.loads((root / "requests.jsonl").read_text())
            self.assertEqual(row["terminal"], "not_dispatched")
            self.assertEqual(row["attempted_dispatch_s"], 10.5)
            self.assertIsNone(row["dispatch_s"])
            self.assertIsNone(row["completed_s"])
            self.assertNotIn("schedule_lag_s", row)
            self.assertFalse(row["success"])
            summary = json.loads((root / "summary.json").read_text())
            self.assertEqual(summary["window"]["measurement_requests"], 0)
            self.assertEqual(summary["window"]["crossing_requests"], 0)
            self.assertFalse(summary["validity"]["execution_valid"])
            self.assertEqual(
                summary["validity_reasons"]["execution_valid"],
                ["request_not_dispatched"],
            )

    def test_late_completion_stays_in_cohort_not_throughput(self):
        window = RequestWindow(10, 70)
        window.observe(
            dict(
                phase="measurement",
                kind="formal",
                request_id="a",
                dispatch_s=69,
                completed_s=71,
                success=True,
            )
        )
        window.observe(
            dict(
                phase="support",
                kind="formal",
                request_id="b",
                dispatch_s=70,
                completed_s=71,
                success=True,
            )
        )
        summary = window.summary()
        self.assertEqual(summary["measurement_requests"], 1)
        self.assertEqual(summary["crossing_requests"], 1)
        self.assertEqual(summary["in_window_completions"], 0)
        self.assertEqual(summary["request_throughput"], 0)

    def test_partial_evidence_survives_close_without_finish(self):
        with tempfile.TemporaryDirectory() as directory:
            writer = PressureWriter(Path(directory), {"session_id": "s"})
            writer.append("cache_samples", {"rank": 0, "sequence": 1})
            writer.close()
            rows = (Path(directory) / "pressure/cache_samples.jsonl").read_text()
            self.assertEqual(json.loads(rows)["sequence"], 1)

    def test_edges_failures_cancellation_and_padding(self):
        window = RequestWindow(10, 70)
        base = dict(
            phase="measurement",
            kind="formal",
            dispatch_s=10,
            completed_s=10,
            success=False,
        )
        window.observe(base)
        window.observe(dict(base, completed_s=None, terminal="cancellation"))
        window.observe(dict(base, phase="padding", padding_for="seed"))
        window.observe(dict(base, phase="padding", padding_for="fill"))
        with self.assertRaises(ValueError):
            window.observe(dict(base, dispatch_s=70))
        self.assertEqual(window.summary()["measurement_requests"], 2)
        self.assertEqual(window.summary()["in_window_completions"], 1)
        self.assertEqual(window.summary()["in_window_successes"], 0)
        self.assertEqual(window.summary()["cancelled_requests"], 1)

    def test_cancellation_then_completion_counted_once(self):
        with tempfile.TemporaryDirectory() as directory:
            writer = PressureWriter(Path(directory), {})
            sink = PressureEvidence(writer)
            sink(
                dict(
                    event="frozen",
                    session_id="s",
                    config=dict(measure_s=60),
                    seed_plan=self.plan(),
                )
            )
            sink(dict(event="phase", phase="measurement", at_s=10))
            request = dict(
                job_id=1,
                request_id="a",
                phase="measurement",
                dispatched_s=11,
            )
            sink(dict(request, event="dispatch"))
            sink(dict(event="cancellation", job_id=1, at_s=12, accepted=True))
            sink(
                dict(
                    request,
                    event="completion",
                    completed_s=13,
                    result=dict(success=True, reuse_len=4),
                )
            )
            sink(dict(event="finished", failure=None, pressure=dict(valid=True)))
            writer.close()
            root = Path(directory) / "pressure"
            rows = [
                json.loads(s)
                for s in (root / "requests.jsonl").read_text().splitlines()
            ]
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["terminal"], "completion")
            summary = json.loads((root / "summary.json").read_text())
            self.assertEqual(summary["metrics"], [])
            self.assertEqual(summary["window"]["measurement_requests"], 1)
            self.assertEqual(summary["validity"]["trace_valid"], None)
            self.assertEqual(sink.pending, {})

    def test_abort_retains_terminal_cancellation_without_success(self):
        with tempfile.TemporaryDirectory() as directory:
            writer = PressureWriter(Path(directory), {})
            sink = PressureEvidence(writer)
            sink(
                dict(
                    event="dispatch",
                    job_id=1,
                    request_id="a",
                    phase="steady",
                    kind="formal",
                    dispatched_s=1,
                )
            )
            sink.abort("failed")
            writer.close()
            root = Path(directory) / "pressure"
            row = json.loads((root / "requests.jsonl").read_text())
            self.assertFalse(row["success"])
            self.assertIsNone(row["completed_s"])
            self.assertFalse(
                json.loads((root / "manifest.json").read_text())["complete"]
            )


class RemotePressureEvidenceTest(unittest.TestCase):
    def page(
        self,
        t,
        *,
        ratio_zero=False,
        failed=0,
        rejected=0,
        pending=0,
        faulted=False,
        quarantine=0,
        missing_runtime=False,
    ):
        from dataclasses import replace

        active = t >= 2
        return tuple(
            replace(
                s,
                mock_cache=(
                    MockCacheSample(
                        eligible_blocks=8 if active and not ratio_zero else 0,
                        planned_remote_blocks=0 if ratio_zero or not active else 4,
                        successful_remote_blocks=0 if ratio_zero or not active else 4,
                        failed_blocks=failed,
                        terminal_outcome_version=1,
                    )
                    if s.rank == 0
                    else None
                ),
                mock_remote_write=MockRemoteWriteSample(
                    submitted=2 if active else 0,
                    completed=2 if active else 0,
                    failed=failed,
                    rejected=rejected,
                    pending_requests=pending,
                    faulted=faulted,
                    quarantined_blocks=quarantine,
                ),
                remote_read_payload_bytes=7
                + (1024 * (s.rank + 1) if active and not ratio_zero else 0),
                remote_write_acknowledged_payload_bytes=1
                + (512 * (s.rank + 1) if active else 0),
                remote_runtime=(
                    None if missing_runtime else '{"version":1,"session_id":"10"}'
                ),
            )
            for s in migration_page(t, t, t, participants=2)
        )

    def coverage(self, **changes):
        c = PressureCoverage(
            self.page(0), configured(evidence_policy="tree_then_mock_remote")
        )
        c.add(self.page(1), 1)
        c.add(self.page(2, **changes), 2, complete=False)
        c.add(self.page(3, **changes), 3)
        c.finish_remote_window(
            self.page(4, **changes),
            request_reuse={
                "completed_requests": 2,
                "reused_requests": 2,
                "reused_tokens": 64,
            },
        )
        return c

    def test_enabled_local_destinations_require_all_terminal_writes_after_remote_drain(
        self,
    ):
        for tier in ("memory", "disk"):
            for outcome in ("completed", "pending", "failed", "partial_failure"):
                with self.subTest(tier=tier, outcome=outcome):
                    cfg = configured(
                        evidence_policy="tree_then_mock_remote",
                        mock_cache_settings=((f"mock_{tier}_cache", True),),
                    )

                    def page(t, submitted, completed, failed):
                        samples = list(self.page(t, ratio_zero=True))
                        owner = samples[0]
                        samples[0] = dataclasses.replace(
                            owner,
                            remote_runtime=json.dumps(
                                {"options": {f"mock_{tier}_cache": "true"}}
                            ),
                            local_store=LocalStoreSample(released_requests=min(t, 2)),
                            mock_memory=(
                                MockMemorySample(
                                    0, 0, 0, 0, submitted, completed, failed, 1
                                )
                                if tier == "memory"
                                else None
                            ),
                            mock_cache=dataclasses.replace(
                                owner.mock_cache,
                                disk_write_submitted_records=(
                                    submitted if tier == "disk" else 0
                                ),
                                disk_write_completed_records=(
                                    completed if tier == "disk" else 0
                                ),
                                disk_write_failed_records=(
                                    failed if tier == "disk" else 0
                                ),
                            ),
                        )
                        samples[1] = dataclasses.replace(
                            samples[1], remote_runtime=samples[0].remote_runtime
                        )
                        return tuple(samples)

                    c = PressureCoverage(page(0, 0, 0, 0), cfg)
                    c.add(page(1, 1, 1, 0), 1)
                    if outcome == "partial_failure":
                        c.add(page(2, 2, 1, 1)[:1], 2, complete=False)
                    c.finish_remote_window(
                        page(
                            4,
                            2,
                            2 if outcome in ("completed", "partial_failure") else 1,
                            int(outcome == "failed"),
                        ),
                        request_reuse=dict(
                            completed_requests=2, reused_requests=2, reused_tokens=64
                        ),
                    )
                    self.assertEqual(
                        c.mock_remote_summary()["valid"],
                        outcome == "completed",
                        c.mock_remote_summary(),
                    )
                    self.assertEqual(
                        c.assess_migration(1, 4).valid, outcome == "completed"
                    )
                    if outcome in ("pending", "failed"):
                        self.assertTrue(c.mock_remote_summary()["remote_drain_valid"])
                    if outcome == "partial_failure":
                        self.assertIn(
                            tier + "_write_failed", c.mock_remote_summary()["reasons"]
                        )

    def test_remote_ratio_zero_write_evidence(self):
        c = self.coverage(ratio_zero=True)
        summary = c.mock_remote_summary()
        self.assertTrue(summary["valid"], summary)
        self.assertEqual(summary["owners"]["0"]["submitted"], 2)
        self.assertIsNone(summary["owners"]["0"]["planned_remote_ratio"])
        self.assertEqual(
            summary["ranks"]["0"]["remote_write_acknowledged_payload_bytes"], 512
        )
        self.assertEqual(
            summary["ranks"]["1"]["remote_write_acknowledged_payload_bytes"], 1024
        )
        self.assertEqual(summary["remote_read_payload_bytes"], 0)
        self.assertEqual(summary["remote_write_acknowledged_payload_bytes"], 1536)

    def test_zero_denominator_requires_actual_reuse_and_half_up_zero_targets(self):
        for ratio, want in ((0, True), (0.0000004, True), (0.0000005, False)):
            c = PressureCoverage(
                self.page(0),
                configured(
                    evidence_policy="tree_then_mock_remote",
                    mock_remote_target_ratio=ratio,
                ),
            )
            c.finish_remote_window(
                self.page(4, ratio_zero=True),
                request_reuse={
                    "completed_requests": 2,
                    "reused_requests": 2,
                    "reused_tokens": 64,
                },
            )
            self.assertEqual(c.mock_remote_summary()["valid"], want)
        c = PressureCoverage(
            self.page(0), configured(evidence_policy="tree_then_mock_remote")
        )
        c.finish_remote_window(self.page(4, ratio_zero=True))
        self.assertFalse(c.mock_remote_summary()["valid"])

    def test_remote_owner_counts_once_and_actual_rank_bytes(self):
        summary = self.coverage().mock_remote_summary()
        self.assertTrue(summary["valid"], summary)
        self.assertEqual(set(summary["owners"]), {"0"})
        self.assertEqual(summary["owners"]["0"]["planned_remote_ratio"], 0.5)
        self.assertEqual(summary["owners"]["0"]["successful_remote_ratio"], 0.5)
        self.assertEqual(summary["remote_read_payload_bytes"], 3072)

    def test_remote_failure_invalidates_acceptance(self):
        for changes in (
            {"failed": 1},
            {"rejected": 1},
            {"pending": 1},
            {"faulted": True},
            {"quarantine": 1},
            {"missing_runtime": True},
        ):
            with self.subTest(changes=changes):
                self.assertFalse(
                    self.coverage(**changes).mock_remote_summary()["valid"]
                )

    def test_worker_fault_and_partial_page_failure_survive_recovery(self):
        from dataclasses import replace

        for field in ("faulted", "quarantined_blocks"):
            with self.subTest(field=field):
                c = PressureCoverage(
                    self.page(0), configured(evidence_policy="tree_then_mock_remote")
                )
                worker = self.page(2)[1]
                worker = replace(
                    worker,
                    mock_remote_write=replace(worker.mock_remote_write, **{field: 1}),
                )
                c.add((worker,), 2, complete=False)
                c.finish_remote_window(self.page(4))
                self.assertFalse(c.mock_remote_summary()["valid"])
        c = PressureCoverage(
            self.page(0), configured(evidence_policy="tree_then_mock_remote")
        )
        c.add(self.page(2, rejected=1)[:1], 2, complete=False)
        c.finish_remote_window(self.page(4))
        self.assertIn("remote_write_rejected", c.mock_remote_summary()["reasons"])
        self.assertIn("remote_counter_reset", c.mock_remote_summary()["reasons"])

    def test_remote_requires_native_capability_and_actual_final_drain(self):
        cfg = configured(evidence_policy="tree_then_mock_remote")
        with self.assertRaisesRegex(ValueError, "Remote.*capability"):
            PressureCoverage(self.page(0, missing_runtime=True), cfg)
        c = PressureCoverage(self.page(0), cfg)
        c.add(self.page(2), 2)
        self.assertFalse(c.mock_remote_summary()["valid"])


class JsonlSinkTest(unittest.TestCase):
    def test_incremental_finite_jsonl_without_retaining_events(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.jsonl"
            with JsonlSink(directory) as sink:
                sink({"event": "phase", "phase": "fill"})
                self.assertEqual(
                    json.loads(path.read_text()), {"event": "phase", "phase": "fill"}
                )
                tracemalloc.start()
                try:
                    for job_id in range(5000):
                        sink({"event": "dispatch", "job_id": job_id})
                    retained, _ = tracemalloc.get_traced_memory()
                finally:
                    tracemalloc.stop()
                self.assertLess(retained, 256 * 1024)
                with self.assertRaises(ValueError):
                    sink({"invalid": float("nan")})
            with path.open() as stream:
                self.assertEqual(sum(1 for _ in stream), 5001)
            with self.assertRaises(ValueError):
                sink({"event": "closed"})


if __name__ == "__main__":
    unittest.main()
