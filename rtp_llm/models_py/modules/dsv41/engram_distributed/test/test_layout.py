# SPDX-License-Identifier: Apache-2.0
"""Engram storage policy, shard geometry, and manifest validation."""

import base64
import unittest
from types import SimpleNamespace as NS
from unittest.mock import patch
from uuid import UUID

from rtp_llm.models_py.modules.dsv41.engram_distributed import config as c
from rtp_llm.models_py.modules.dsv41.engram_distributed import exchange as startup
from rtp_llm.models_py.modules.dsv41.engram_distributed import manifest as m
from rtp_llm.models_py.modules.dsv41.engram_distributed.config import (
    ShardConfig,
    ShardConfigError,
    StoragePolicy,
)
from rtp_llm.models_py.modules.dsv41.engram_distributed.placement import (
    DEFAULT_GRANULARITY,
    PlacementPlan,
    ShardPlan,
    plan_shards,
    segment_rows,
)


class CapacityTest(unittest.TestCase):
    def test_defaults_are_the_hard_bounds(self):
        self.assertEqual(
            c.capacity_from_environ({}),
            (c.HARD_MAX_ALLOCATION_BYTES, c.HARD_MAX_TOTAL_ALLOCATION_BYTES),
        )

    def test_explicit_local_ceiling_is_supported(self):
        self.assertEqual(
            c.capacity_from_environ(
                {
                    c.MAX_ALLOCATION_ENV: str(128 << 30),
                    c.MAX_TOTAL_ALLOCATION_ENV: str(256 << 30),
                }
            ),
            (128 << 30, 256 << 30),
        )

    def test_each_limit_can_be_lowered_independently(self):
        self.assertEqual(c.validate_capacity(1 << 30, 64 << 20), (1 << 30, 64 << 20))
        self.assertEqual(
            c.capacity_from_environ({c.MAX_ALLOCATION_ENV: "4096"}), (4096, 256 << 30)
        )

    def test_capacity_types_and_hard_bounds(self):
        for allocation, total in (
            (0, 1),
            (1, 0),
            (-1, 1),
            (True, 1),
            (1, False),
            (1.0, 1),
            (1, "1"),
            ((128 << 30) + 1, 256 << 30),
            (128 << 30, (256 << 30) + 1),
            (1 << 64, 1),
        ):
            with self.subTest(allocation=allocation, total=total):
                with self.assertRaisesRegex(c.CapacityError, "INVALID_CAPACITY_LIMIT"):
                    c.validate_capacity(allocation, total)

    def test_environment_rejects_ambiguous_or_unbounded_inputs(self):
        for value in (
            "",
            "0",
            "-1",
            "+1",
            "01",
            " 1",
            "1 ",
            "1\n",
            "1GiB",
            "1e9",
            "0x1000",
            "١",
            "1" * 10000,
            1,
            True,
            None,
        ):
            for key in (c.MAX_ALLOCATION_ENV, c.MAX_TOTAL_ALLOCATION_ENV):
                with self.subTest(key=key, value=str(value)[:20]):
                    with self.assertRaisesRegex(
                        c.CapacityError, "INVALID_CAPACITY_LIMIT"
                    ):
                        c.capacity_from_environ({key: value})

    def test_bundle_like_keys_cannot_change_limits(self):
        self.assertEqual(
            c.capacity_from_environ(
                {"budget_bytes": str(256 << 30), "max_allocation_bytes": str(128 << 30)}
            ),
            (c.HARD_MAX_ALLOCATION_BYTES, c.HARD_MAX_TOTAL_ALLOCATION_BYTES),
        )


def capacity_manifest(count):
    return NS(
        tensors=[
            NS(shards=[NS(host_id=f"host-{i}") for i in range(count)]) for _ in range(4)
        ],
    )


class LargeGpuLayoutTest(unittest.TestCase):
    def test_32_gpu_real_table_shapes_remain_single_copy(self):
        for rows in (384006168, 384016682):
            plan = plan_shards(rows, 256, 8, granularity=2 << 20)
            parts = [
                part
                for host in range(8)
                for part in segment_rows(plan, host, (0, 1, 2, 3))
            ]
            self.assertEqual(len(parts), 32)
            self.assertEqual(parts[0][1], 0)
            self.assertEqual(parts[-1][2], rows)
            self.assertTrue(
                all(left[2] == right[1] for left, right in zip(parts, parts[1:]))
            )
            self.assertEqual(
                sum(padded * (256 + 8) for _, _, _, padded in parts),
                plan.shard_bytes * 8,
            )
            self.assertTrue(
                all(padded * 8 % (2 << 20) == 0 for _, _, _, padded in parts)
            )


class ConfigTest(unittest.TestCase):
    def test_p_is_independent_of_model_parallelism_and_instance_count(self):
        for count in (2, 3, 5, 18, 36, 64):
            for instances in (1, 2, 10):
                config = ShardConfig.from_environ(
                    {
                        "ENGRAM_SHARD_COUNT": str(count),
                        "TP_SIZE": "1",
                        "DP_SIZE": "8",
                        "EP_SIZE": "8",
                        "INSTANCES": str(instances),
                    }
                )
                self.assertEqual(config.shard_count, count)
                config.validate_manifest(capacity_manifest(count))

    def test_inference_from_manifest_and_explicit_mismatch(self):
        self.assertEqual(ShardConfig.from_environ({}, inferred_count=3).shard_count, 3)
        with self.assertRaisesRegex(ShardConfigError, "COUNT_MISMATCH"):
            ShardConfig(3).validate_manifest(capacity_manifest(2))

    def test_single_host_and_invalid_counts(self):
        self.assertEqual(ShardConfig(1).shard_count, 1)
        for value in (None, True, 0, -1, 65, 2.0, "2"):
            with self.subTest(value=value), self.assertRaises(ShardConfigError):
                ShardConfig(value)
        for value in ("", "-1", "2.0", "２", " 2", 2):
            with self.subTest(value=value), self.assertRaises(ShardConfigError):
                ShardConfig.from_environ({"ENGRAM_SHARD_COUNT": value})

    def test_cancelled_mode_and_rpc_options_fail_explicitly(self):
        for key in (
            "ENGRAM_SHARD_MODE",
            "ENGRAM_SHARD_INDEX",
            "ENGRAM_SHARD_ENDPOINTS",
        ):
            with self.assertRaisesRegex(ShardConfigError, "REMOVED"):
                ShardConfig.from_environ({key: "x"}, inferred_count=2)

    def test_missing_duplicate_or_replaced_host_is_rejected(self):
        for case in ("missing", "duplicate", "replacement"):
            doc = capacity_manifest(3)
            if case == "missing":
                doc.tensors[1].shards.pop()
            if case == "duplicate":
                doc.tensors[1].shards[1].host_id = "host-0"
            if case == "replacement":
                doc.tensors[1].shards[1].host_id = "another-host"
            with self.subTest(case=case), self.assertRaises(ShardConfigError):
                ShardConfig(3).validate_manifest(doc)


class StoragePolicyTest(unittest.TestCase):
    def test_upstream_gpu_modes_preserve_handle_semantics(self):
        fabric = StoragePolicy.from_environ({"RTP_ENGRAM_STORAGE": "cuda_fabric"})
        shared = StoragePolicy.from_environ({"RTP_ENGRAM_STORAGE": "gpu_shared"})
        local = StoragePolicy.from_environ({"RTP_ENGRAM_STORAGE": "cuda_local"})
        self.assertEqual(fabric, shared)
        self.assertEqual(local.memory_kind, "gpu")
        self.assertEqual(local.backend, "cuda_ipc")
        self.assertEqual(local.default_graph_mode, "process_lifetime")

    def test_modes_are_complete_and_invalid_modes_fail_closed(self):
        for mode, shared, memory in (
            ("host", False, "host"),
            ("host_shared", True, "host"),
            ("gpu_shared", True, "gpu"),
        ):
            policy = StoragePolicy.from_environ({"RTP_ENGRAM_STORAGE": mode})
            self.assertEqual((policy.shared, policy.memory_kind), (shared, memory))
        for mode in ("", "gpu", "HOST", None, 1):
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                StoragePolicy.from_environ({"RTP_ENGRAM_STORAGE": mode})

    def test_old_selectors_fail_instead_of_silently_loading_local_tables(self):
        self.assertFalse(StoragePolicy.from_environ({}).shared)
        for memory in ("host", "gpu"):
            with self.assertRaisesRegex(ValueError, "REMOVED_ENGRAM_MEMORY_OPTION"):
                StoragePolicy.from_environ({"RTP_ENGRAM_MEMORY": memory})
        for env in ({"ENGRAM_SHARD_COUNT": "2"}, {"RTP_ENGRAM_COHORT": "a,b"}):
            with self.assertRaisesRegex(ValueError, "ENGRAM_STORAGE_REQUIRED"):
                StoragePolicy.from_environ(env)

    def test_local_mode_overrides_retained_legacy_topology_without_discovery(self):
        env = {
            "RTP_ENGRAM_STORAGE": "host",
            "ENGRAM_SHARD_COUNT": "2",
            "RTP_ENGRAM_COHORT": "unused",
        }
        with patch.object(
            c,
            "_gang_parts",
            side_effect=AssertionError("local mode entered exchange"),
        ):
            self.assertIsNone(c.build_config(env, 8))
        self.assertEqual(StoragePolicy.from_environ(env).memory_kind, "host")

    def test_shared_mode_uses_gang_without_legacy_enable_flags(self):
        for mode, memory in (("host_shared", "host"), ("gpu_shared", "gpu")):
            env = {"RTP_ENGRAM_STORAGE": mode, c.REVISION_ENV: "test-revision"}
            with patch.object(c, "_annotation_cohort", return_value=None):
                config = c.build_config(
                    env,
                    8,
                    gang_hosts=["10.0.0.1"] * 4 + ["10.0.0.2"] * 4,
                    checkpoint="/unused",
                )
            self.assertEqual(config["options"].policy.memory_kind, memory)
            self.assertEqual(len(config["group"]["members"]), 2)
            self.assertEqual(config["gpu_devices"], (0, 1, 2, 3))
            self.assertEqual(
                config["options"].policy.default_graph_mode, "process_lifetime"
            )

    def test_capacity_applies_to_physical_allocations_not_host_extent(self):
        checkpoint = NS(
            layers=[
                NS(
                    layer_id=1,
                    tensors=[
                        NS(name="weight", shape=(65536, 64)),
                        NS(name="scale", shape=(65536, 2)),
                    ],
                )
            ]
        )
        # Production names also carry their role suffix.
        for tensor in checkpoint.layers[0].tensors:
            tensor.name = "layer." + tensor.name
        plan = PlacementPlan.build(checkpoint, 0, 2, 4096, {1: (0, 1, 2, 3)})
        plan.validate_capacity(1 << 20, 4 << 20)
        self.assertEqual(plan.total_bytes, 32768 * 66)
        self.assertEqual(plan.bytes_by_location(), {i: 8192 * 66 for i in range(4)})
        with self.assertRaisesRegex(ValueError, "allocation limit"):
            plan.validate_capacity((512 << 10) - 1, 4 << 20)
        with self.assertRaisesRegex(ValueError, "host budget"):
            plan.validate_capacity(1 << 20, plan.total_bytes - 1)


REAL_ROWS = (384006168, 384016682)
DIM = 256
ALIGN_ROWS = 262144


class PlanShardsTest(unittest.TestCase):
    def test_real_geometry_is_mappable(self):
        for total in REAL_ROWS:
            plan = plan_shards(total, DIM, 4)
            self.assertEqual(plan.shard_count, 4)
            self.assertEqual(plan.shard_rows % ALIGN_ROWS, 0)
            self.assertGreaterEqual(plan.padded_rows, total)
            self.assertEqual(plan.weight_bytes % DEFAULT_GRANULARITY, 0)
            self.assertEqual(plan.scale_bytes % DEFAULT_GRANULARITY, 0)
            self.assertEqual(plan.shard_bytes % DEFAULT_GRANULARITY, 0)

    def test_padding_is_bounded(self):
        for total in REAL_ROWS:
            for count in (1, 2, 4, 8):
                plan = plan_shards(total, DIM, count)
                overhead = plan.padded_rows - total
                self.assertLess(overhead, count * ALIGN_ROWS)
                self.assertLess(overhead / total, 5e-3)

    def test_single_shard(self):
        plan = plan_shards(1000, DIM, 1)
        self.assertEqual(plan.shard_count, 1)
        self.assertEqual(plan.row_range(0), (0, 1000))

    def test_rows_cover_table_exactly_once(self):
        total = 1000
        plan = plan_shards(total, DIM, 4, granularity=8)
        covered = []
        for shard in range(plan.shard_count):
            start, end = plan.row_range(shard)
            covered.extend(range(start, end))
        self.assertEqual(covered, list(range(total)))

    def test_last_shard_holds_padding(self):
        total = 1001
        plan = plan_shards(total, DIM, 4, granularity=8)
        start, end = plan.row_range(3)
        self.assertLessEqual(end, total)
        self.assertLess(end - start, plan.shard_rows)

    def test_rejects_bad_geometry(self):
        with self.assertRaises(ValueError):
            plan_shards(0, DIM, 4)
        with self.assertRaises(ValueError):
            plan_shards(100, 250, 4)  # dim not a multiple of block
        with self.assertRaises(ValueError):
            plan_shards(100, DIM, 0)
        with self.assertRaises(ValueError):
            plan_shards(100, DIM, 4, granularity=3)

    def test_plan_is_hashable_and_immutable(self):
        plan = plan_shards(1000, DIM, 4, granularity=8)
        self.assertIsInstance(plan, ShardPlan)
        self.assertEqual(hash(plan), hash(plan_shards(1000, DIM, 4, granularity=8)))
        with self.assertRaises(Exception):
            plan.shard_rows = 1

    def test_non_power_of_two_host_counts_cover_exactly(self):
        for count in (2, 3, 5, 7, 18, 36, 64):
            for total in REAL_ROWS:
                plan = plan_shards(total, DIM, count)
                cursor = 0
                for shard in range(count):
                    begin, end = plan.row_range(shard)
                    self.assertEqual(begin, cursor)
                    self.assertLess(begin, end)
                    cursor = end
                self.assertEqual(cursor, total)

    def test_empty_shards_rejected_before_allocating(self):
        with self.assertRaisesRegex(ValueError, "empty shards"):
            plan_shards(1000, DIM, 4)

    def test_invalid_integer_geometry(self):
        for key in ("total_rows", "dim", "shard_count", "block", "granularity"):
            for bad in (0, -1, True, 1.5, "2"):
                args = dict(
                    total_rows=1048576,
                    dim=256,
                    shard_count=2,
                    block=32,
                    granularity=4096,
                )
                args[key] = bad
                with self.subTest(key=key, bad=bad), self.assertRaises(ValueError):
                    plan_shards(**args)
        plan = plan_shards(1048576, 256, 2)
        for shard in (-1, plan.shard_count, True):
            with self.subTest(shard=shard), self.assertRaises(IndexError):
                plan.row_range(shard)


def uid(n):
    return str(UUID(int=n))


def expected(**changes):
    values = dict(
        layers=((1, 3), (14, 5)),
        head_dim=32,
        model_revision="release-immutable-42",
        hash_abi="sha256:" + "b" * 64,
    )
    values.update(changes)
    return m.ExpectedLayout(**values)


GEN = uid(1)
HASH = "sha256:" + "b" * 64
LAYERS = (1, 14)


def descriptors(
    p=3,
    dim=64,
    *,
    locations=(0,),
    memory_kind="host",
    layers=None,
    granularity=4096,
    hash_abi=HASH,
):
    if layers is None:
        layers = tuple((layer, p * 65536 - 7) for layer in LAYERS)
    plans = {
        layer: plan_shards(rows, dim, p, granularity=granularity)
        for layer, rows in layers
    }
    result = []
    for shard in range(p):
        records = []
        for layer, plan in plans.items():
            segments = []
            for location, begin, end, padded in segment_rows(plan, shard, locations):
                segment = dict(location=location, row_begin=begin, row_end=end)
                for role, columns in (("weight", dim), ("scale", dim // plan.block)):
                    token = f"{shard}/{layer}/{location}/{role}".encode()
                    segment[role] = dict(
                        handle=base64.b64encode(token).decode("ascii"),
                        allocation_bytes=padded * columns,
                    )
                segments.append(segment)
            records.append(
                dict(
                    layer_id=layer,
                    rows=plan.total_rows,
                    head_dim=dim,
                    segments=segments,
                )
            )
        result.append(
            dict(
                schema_version=m.PAIRED_SHARD_SCHEMA,
                shard=shard,
                shards=p,
                generation=GEN,
                model_revision="revision",
                hash_abi=hash_abi,
                host_id=f"host-{shard}",
                memory_kind=memory_kind,
                granularity=granularity,
                layers=records,
            )
        )
    return result


def prepare(docs, *, hash_abi=HASH):
    return m.prepare_manifest(docs, len(docs), GEN, "revision", hash_abi)


class ManifestTest(unittest.TestCase):
    def test_paired_host_and_gpu_geometry(self):
        for p in (2, 3, 5, 18):
            for dim in (32, 64, 256):
                for kind, locations in (("host", (0, 1)), ("gpu", (2, 0, 3, 1))):
                    with self.subTest(p=p, dim=dim, memory_kind=kind):
                        docs = descriptors(
                            p, dim, locations=locations, memory_kind=kind
                        )
                        parsed, handles = prepare(docs[::-1])
                        self.assertEqual(
                            parsed.allocation_count, p * len(locations) * 4
                        )
                        self.assertEqual(
                            parsed.manifest.backend,
                            "fabric_gpu" if kind == "gpu" else "fabric_egm",
                        )
                        total = 0
                        tensors = {t.name: t for t in parsed.manifest.tensors}
                        for layer in LAYERS:
                            weight = tensors[f"layers.{layer}.engram.embed.weight"]
                            scale = tensors[f"layers.{layer}.engram.embed.scale"]
                            self.assertEqual(weight.row_bytes, dim)
                            self.assertEqual(scale.row_bytes, dim // 32)
                            row = 0
                            for index, (w, s) in enumerate(
                                zip(weight.shards, scale.shards)
                            ):
                                host, location = divmod(index, len(locations))
                                self.assertEqual(w.host_id, f"host-{host}")
                                self.assertEqual(s.host_id, w.host_id)
                                self.assertEqual(w.row_begin, row)
                                self.assertEqual(
                                    (s.row_begin, s.row_end), (w.row_begin, w.row_end)
                                )
                                row = w.row_end
                                for role, part, width in (
                                    ("weight", w, dim),
                                    ("scale", s, dim // 32),
                                ):
                                    self.assertEqual(
                                        handles[part.allocation_id],
                                        f"{host}/{layer}/{locations[location]}/{role}".encode(),
                                    )
                                    self.assertEqual(
                                        part.logical_bytes,
                                        (part.row_end - part.row_begin) * width,
                                    )
                                    self.assertEqual(part.allocation_bytes % 4096, 0)
                                    total += part.allocation_bytes
                            self.assertEqual(row, weight.shape[0])
                        self.assertEqual(total, parsed.physical_bytes)
                        self.assertEqual(len(handles), parsed.allocation_count)
                        self.assertGreater(
                            weight.shards[-1].allocation_bytes,
                            weight.shards[-1].logical_bytes,
                        )

    def test_mismatched_host_headers_rejected(self):
        for key, value in (
            ("schema_version", m.PAIRED_SHARD_SCHEMA + 1),
            ("model_revision", "other"),
            ("hash_abi", "sha256:" + "c" * 64),
            ("generation", uid(3)),
            ("shards", 2),
            ("shard", 0),
            ("shard", True),
            ("host_id", "host-0"),
            ("memory_kind", "gpu"),
            ("granularity", 8192),
        ):
            with self.subTest(key=key, value=value):
                docs = descriptors()
                docs[1][key] = value
                with self.assertRaises(ValueError):
                    prepare(docs)
        docs = descriptors()
        for malformed in (docs[:-1], docs + docs[:1], [docs[0], docs[0], docs[2]]):
            with self.assertRaises(ValueError):
                m.prepare_manifest(malformed, 3, GEN, "revision", HASH)

    def test_mismatched_layer_layout_rejected(self):
        for mutate in (
            lambda layers: layers.pop(),
            lambda layers: layers.append(layers[0]),
            lambda layers: layers[0].update(rows=1),
            lambda layers: layers[0].update(head_dim=32),
        ):
            docs = descriptors()
            mutate(docs[1]["layers"])
            with self.assertRaises(ValueError):
                prepare(docs)
        docs = descriptors()
        docs[0]["shard"], docs[1]["shard"] = 1, 0
        with self.assertRaisesRegex(ValueError, "geometry"):
            prepare(docs)

    def test_invalid_physical_segments_rejected(self):
        for kind, locations in (("host", (0, 1)), ("gpu", (0, 1, 2, 3))):
            for mutate in (
                lambda parts: parts[0].update(row_end=parts[0]["row_end"] - 1),
                lambda parts: parts[0].update(row_begin=True),
                lambda parts: parts[1].update(location=parts[0]["location"]),
                lambda parts: parts[0].update(location=-1),
                lambda parts: parts[0].pop("scale"),
                lambda parts: parts[0]["weight"].update(allocation_bytes=4096),
                lambda parts: parts[0]["weight"].update(handle="%%%"),
                lambda parts: parts[0]["weight"].update(
                    handle=parts[0]["scale"]["handle"]
                ),
            ):
                with self.subTest(memory_kind=kind):
                    docs = descriptors(locations=locations, memory_kind=kind)
                    mutate(docs[1]["layers"][0]["segments"])
                    with self.assertRaises(ValueError):
                        prepare(docs)

    def test_binding_checks_local_contract_and_mapping_limits(self):
        docs = descriptors()
        parsed, _ = prepare(docs)
        layout = expected(
            layers=tuple((layer, docs[0]["layers"][0]["rows"]) for layer in LAYERS),
            head_dim=64,
            model_revision="revision",
        )
        validated = m.bind_manifest(parsed, layout, GEN, m.Limits())
        self.assertIs(validated.manifest, parsed.manifest)
        for options, code in (
            (dict(limits=m.Limits(max_tensors=2)), "TENSOR_LIMIT"),
            (
                dict(limits=m.Limits(max_allocation_bytes=4096)),
                "INVALID_ALLOCATION_SIZE",
            ),
            (
                dict(
                    limits=m.Limits(
                        max_total_allocation_bytes=parsed.physical_bytes - 1
                    )
                ),
                "TOTAL_ALLOCATION_LIMIT",
            ),
            (dict(expected_generation=uid(2)), "GENERATION_MISMATCH"),
            (dict(expected_layout=expected(backend="fabric_gpu")), "BACKEND_MISMATCH"),
            (dict(expected_layout=expected()), "MODEL_CONTRACT_MISMATCH"),
            (
                dict(expected_layout=expected(model_revision="revision")),
                "INCOMPLETE_GENERATION",
            ),
        ):
            with self.subTest(code=code), self.assertRaisesRegex(m.ManifestError, code):
                m.bind_manifest(
                    parsed,
                    **dict(
                        dict(
                            expected_layout=layout,
                            expected_generation=GEN,
                            limits=m.Limits(),
                        ),
                        **options,
                    ),
                )

    def test_expected_layout_and_limits_are_validated(self):
        for change, code in (
            ({"layers": ((1, 3), (1, 5))}, "DUPLICATE_LAYER"),
            ({"head_dim": 31}, "UNSUPPORTED_LAYOUT"),
            ({"backend": "unknown"}, "UNSUPPORTED_BACKEND"),
            ({"layers": ((True, 3),)}, "INVALID_INTEGER"),
            ({"layers": ()}, "INVALID_LAYOUT"),
        ):
            with self.assertRaisesRegex(m.ManifestError, code):
                expected(**change)
        with self.assertRaisesRegex(m.ManifestError, "INVALID_LIMIT"):
            m.Limits(True)


if __name__ == "__main__":
    unittest.main()
