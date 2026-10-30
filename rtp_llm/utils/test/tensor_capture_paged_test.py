"""CPU contract/privacy tests using the real generic reservation, writer and loader."""

import copy
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

from rtp_llm.utils import tensor_capture as capture
from rtp_llm.utils import tensor_capture_indexer as indexer


class PagedTensorCaptureTest(unittest.TestCase):
    def setUp(self):
        directory = (
            os.environ.get("TEST_TMPDIR")
            or os.environ.get("TMPDIR")
            or os.environ.get("RTP_LLM_TEST_LOCAL_LAB")
            or None
        )
        self.temporary = tempfile.TemporaryDirectory(
            prefix="paged-capture-", dir=directory
        )
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        environment = mock.patch.dict(
            os.environ,
            RTP_LLM_RECORD_DIR=str(self.root / "records"),
            RTP_LLM_RECORD_SESSION="paged-canary",
        )
        environment.start()
        self.addCleanup(environment.stop)
        self.recorders = []
        self.addCleanup(self.close_recorders)
        self.identity = dict(
            schema_version=1,
            session_id="paged-canary",
            owner_id="owner",
            replica_id="dp0",
            world_rank=0,
            dp_rank=0,
            exec_id="123",
            model_role="target",
            server_role="DECODE",
            requests=[
                dict(request_id="r" + str(i + 1), phase="decode", q_len=1, kv_len=n - 1)
                for i, n in enumerate((3, 63, 69))
            ],
        )
        self.config = dict(
            enabled=True,
            authorized_capture=True,
            layers=[0],
            ops=["indexer.mqa_topk"],
            phases=["decode"],
            rank_filter="record_owner",
            max_calls=100,
            max_bundle_bytes=1000000,
            max_total_bytes=10000000,
            max_inflight_bytes=10000000,
        )
        self.lengths = torch.tensor([3, 63, 69], dtype=torch.int32)
        self.table = torch.tensor([[5, 7], [5, 7], [5, 2]], dtype=torch.int32)
        # Every nonreferenced byte contains another request's secret sentinel.
        self.pool = torch.full((8, 64, 1, 132), 231, dtype=torch.uint8)
        self.pool[5].view(-1)[: 64 * 128].fill_(17)
        self.pool[5].view(-1)[64 * 128 :].view(torch.float32).fill_(1.25)
        self.pool[2].view(-1)[: 5 * 128].fill_(29)
        self.pool[2].view(-1)[64 * 128 : 64 * 128 + 5 * 4].view(torch.float32).fill_(
            2.5
        )
        self.q = torch.ones((3, 1, 32, 128), dtype=torch.float32).to(
            torch.float8_e4m3fn
        )
        self.weights = torch.ones((3, 32), dtype=torch.float32)
        self.schedule = torch.tensor(
            [[0, 0], [0, 0], [1, 0], [2, 0], [3, 0]], dtype=torch.int32
        )
        self.environment = dict(
            deep_gemm_version="2.8.0+6d6aaad.cu132", device_capability=[9, 0], num_sms=4
        )
        self.logits = torch.full((3, 256), 913.0, dtype=torch.float32)[:, :128]
        self.indices = torch.full((3, 512), -1, dtype=torch.int32)
        for row, length in enumerate(self.lengths.tolist()):
            self.logits[row, :length] = torch.arange(length, dtype=torch.float32)
            self.indices[row, :length] = torch.arange(length, dtype=torch.int32)
        self.parameters = dict(
            k=512,
            max_seq_len=128,
            index_space="sequence_relative",
            workspace_bytes=1 << 20,
            stride0=256,
            stride1=1,
        )

    def close_recorders(self):
        for recorder in self.recorders:
            self.assertTrue(recorder.close(30), recorder.stats())
            self.assertEqual(recorder.stats()["inflight_bytes"], 0)

    def context(self, *, joint=True, config=None, identity=None):
        value = dict(self.config if config is None else config)
        value["ops"] = ["indexer.mqa_topk" if joint else "indexer.topk"]
        path = self.root / ("config-" + str(len(self.recorders)) + ".json")
        path.write_text(json.dumps(value))
        with mock.patch.dict(
            os.environ,
            RTP_LLM_RECORD_DIR=str(self.root / "records" / str(len(self.recorders))),
        ):
            recorder = capture.TensorCaptureRecorder(str(path))
        self.recorders.append(recorder)
        context = recorder.for_execution(
            json.dumps(self.identity if identity is None else identity)
        )
        self.assertIsNotNone(context, recorder.stats())
        return indexer.for_context(context), recorder

    def begin_joint(self, context, **changes):
        arguments = dict(
            layer=0,
            q_fp8=self.q,
            kv_cache_fp8=self.pool,
            weights=self.weights,
            context_lens=self.lengths.unsqueeze(1),
            block_table=self.table,
            schedule_metadata=self.schedule,
            lengths=self.lengths,
            k=512,
            block_size=64,
            max_seq_len=128,
            total_query_rows=3,
            schedule_environment=self.environment,
        )
        arguments.update(changes)
        return context.begin_paged_mqa(**arguments)

    def begin_topk(self, context, **changes):
        arguments = dict(
            layer=0,
            logits=self.logits,
            lengths=self.lengths,
            k=512,
            backend="dsv4_persistent_topk",
            parameters=self.parameters,
            total_query_rows=3,
        )
        arguments.update(changes)
        return context.begin_paged_topk(**arguments)

    def load(self, recorder):
        self.assertTrue(recorder.close(30), recorder.stats())
        paths = list((self.root / "records").rglob("manifest.json"))
        self.assertEqual(len(paths), 1, recorder.stats())
        manifest, tensors = capture.load_capture(paths[0])
        indexer.validate_paged_capture(manifest, tensors)
        return paths[0], manifest, tensors

    def joint_artifact(self):
        context, recorder = self.context()
        token = self.begin_joint(context)
        self.assertIsNotNone(token, recorder.stats())
        self.assertTrue(
            token.prepare_topk(
                self.logits, backend="dsv4_persistent_topk", parameters=self.parameters
            )
        )
        self.assertTrue(token.finish(self.indices))
        return self.load(recorder)

    def test_joint_compact_union_scale_offsets_and_no_other_request_bytes(self):
        context, recorder = self.context()
        token = self.begin_joint(context)
        self.assertIsNotNone(token, recorder.stats())
        # Mutating every borrowed source must not affect the queued artifact.
        self.pool.fill_(199)
        self.table.fill_(6)
        self.q.view(torch.uint8).fill_(88)
        self.weights.fill_(8)
        self.schedule.fill_(8)
        self.lengths.fill_(1)
        self.assertTrue(
            token.prepare_topk(
                self.logits, backend="dsv4_persistent_topk", parameters=self.parameters
            )
        )
        self.logits.fill_(199)
        self.assertTrue(token.finish(self.indices))
        self.indices.fill_(199)
        _, manifest, tensors = self.load(recorder)
        self.assertEqual(manifest["contract"], dict(name="indexer.mqa_topk", version=3))
        self.assertEqual(tensors["page_ids"].tolist(), [5, 2, -1, -1])
        self.assertEqual(tensors["page_valid_tokens"].tolist(), [64, 5, 0, 0])
        self.assertEqual(tensors["block_table"].tolist(), [[5, -1], [5, -1], [5, 2]])
        pages = tensors["packed_kv_pages"].view(4, -1)
        self.assertTrue(torch.all(pages[0, : 64 * 128] == 17))
        self.assertTrue(torch.all(pages[1, : 5 * 128] == 29))
        self.assertTrue(torch.all(pages[0, 64 * 128 :].view(torch.float32) == 1.25))
        self.assertTrue(
            torch.all(pages[1, 64 * 128 : 64 * 128 + 20].view(torch.float32) == 2.5)
        )
        self.assertFalse(torch.any(pages == 231))
        self.assertEqual(tensors["lengths"].tolist(), [3, 63, 69])
        self.assertEqual(tensors["indices"][0, :4].tolist(), [0, 1, 2, -1])
        self.assertEqual(tensors["logits"][0, :4].tolist(), [0, 1, 2, 0])
        self.assertEqual(
            manifest["tensors"]["packed_kv_pages"]["shape"], [4, 64, 1, 132]
        )

    def test_only_valid_payload_scale_and_logits_slices_are_read(self):
        context, recorder = self.context()
        original_copy = torch.Tensor.copy_
        reads = []

        def checked_copy(destination, source, *args, **kwargs):
            if source.untyped_storage()._cdata == self.pool.untyped_storage()._cdata:
                reads.append((source.storage_offset(), source.numel()))
            if source.untyped_storage()._cdata == self.logits.untyped_storage()._cdata:
                self.assertIn(source.numel(), (3, 63, 69))
            return original_copy(destination, source, *args, **kwargs)

        with mock.patch.object(torch.Tensor, "copy_", checked_copy):
            token = self.begin_joint(context)
            self.assertIsNotNone(token, recorder.stats())
            self.assertTrue(
                token.prepare_topk(
                    self.logits, backend="topk_v3", parameters=self.parameters
                )
            )
            self.assertTrue(token.finish(self.indices))
        self.assertEqual(
            reads,
            [
                (5 * 64 * 132, 64 * 128),
                (5 * 64 * 132 + 64 * 128, 256),
                (2 * 64 * 132, 5 * 128),
                (2 * 64 * 132 + 64 * 128, 20),
            ],
        )
        self.load(recorder)

    def test_budget_rejects_before_snapshot_or_d2h(self):
        context, recorder = self.context(config=dict(self.config, max_bundle_bytes=1))
        with (
            mock.patch.object(
                torch.Tensor, "copy_", side_effect=AssertionError("copy before reserve")
            ),
            mock.patch.object(
                torch.Tensor, "to", side_effect=AssertionError("D2H before reserve")
            ),
            mock.patch.object(
                indexer._PagedIndexerContract,
                "snapshot_inputs",
                side_effect=AssertionError("snapshot before reserve"),
            ) as snapshot,
        ):
            self.assertIsNone(self.begin_joint(context))
            snapshot.assert_not_called()
        self.assertEqual(recorder.stats()["skip_reasons"].get("oversize"), 1)

    def test_budget_counts_compact_capacity_not_pool(self):
        context, recorder = self.context(
            config=dict(self.config, max_bundle_bytes=150000)
        )
        large_pool = torch.full((512, 64, 1, 132), 231, dtype=torch.uint8)
        large_pool[:8].copy_(self.pool)
        token = self.begin_joint(context, kv_cache_fp8=large_pool)
        self.assertIsNotNone(token, recorder.stats())
        self.assertTrue(
            token.prepare_topk(
                self.logits, backend="topk_v3", parameters=self.parameters
            )
        )
        self.assertTrue(token.finish(self.indices))
        self.load(recorder)

    def test_topk_both_backends_preserve_valid_bits_zero_tail(self):
        for backend in ("topk_v3", "dsv4_persistent_topk"):
            with self.subTest(backend=backend):
                context, recorder = self.context(joint=False)
                self.logits[0, 0] = float("nan")
                token = self.begin_topk(context, backend=backend)
                self.assertIsNotNone(token, recorder.stats())
                self.assertTrue(token.finish(self.indices))
                self.assertTrue(recorder.close(30))
        paths = list((self.root / "records").rglob("manifest.json"))
        self.assertEqual(len(paths), 2)
        for path in paths:
            manifest, tensors = capture.load_capture(path)
            indexer.validate_paged_capture(manifest, tensors)
            self.assertTrue(torch.isnan(tensors["logits"][0, 0]))
            self.assertEqual(set(tensors), {"logits", "lengths", "indices"})
            self.assertTrue(torch.all(tensors["logits"][0, 3:] == 0))

    def test_schedule_build_provenance_without_record(self):
        old, current = "2.8.0+6d6aaad.cu132", "2.8.0+122e18b.cu132"
        cases = (
            ("2.8.0", [old], old),
            ("2.8.0", [current], current),
            (None, [current], current),
            (current, [current], current),
            (current, [], current),  # Bazel without dist-info, exact module build.
            ("2.8.0", [], None),
            (None, [], None),
            ("2.8.0", ["2.8.0"], None),
            ("2.8.0", ["2.8.0+unverified"], None),
            (current, ["2.8.0+unverified"], None),
            (old, [current], None),
            ("2.9.0", [current], None),
            ("2.8.0", [old, current], None),
        )
        for case, (module_version, versions, expected) in enumerate(cases):
            with self.subTest(module_version=module_version, versions=versions):
                site = self.root / ("site-" + str(case))
                package = site / "deep_gemm"
                package.mkdir(parents=True)
                location = package / "__init__.py"
                location.touch()
                module = SimpleNamespace(
                    __file__=str(location), __version__=module_version
                )
                for version in versions:
                    info = site / ("deep_gemm-" + version + ".dist-info")
                    info.mkdir()
                    (info / "METADATA").write_text(
                        "Metadata-Version: 2.1\nName: deep_gemm\nVersion: "
                        + version
                        + "\n"
                    )
                if expected is None:
                    with self.assertRaises(ValueError):
                        indexer.paged_schedule_version(module)
                else:
                    self.assertEqual(indexer.paged_schedule_version(module), expected)

    def test_schedule_build_provenance_rejects_wrong_owner_and_metadata_errors(self):
        location = self.root / "deep_gemm" / "__init__.py"
        module = SimpleNamespace(
            __file__=str(location), __version__="2.8.0+122e18b.cu132"
        )
        dist = SimpleNamespace(
            metadata={"Name": "deep_gemm"},
            version=module.__version__,
            locate_file=lambda name: self.root / "unrelated" / name,
        )
        with mock.patch.object(
            indexer.package_metadata, "distributions", return_value=[dist]
        ):
            with self.assertRaisesRegex(ValueError, "does not own"):
                indexer.paged_schedule_version(module)
        with mock.patch.object(
            indexer.package_metadata,
            "distributions",
            side_effect=OSError("metadata unreadable"),
        ):
            with self.assertRaises(OSError):
                indexer.paged_schedule_version(module)
        with self.assertRaisesRegex(ValueError, "package location"):
            indexer.paged_schedule_version(
                SimpleNamespace(__version__=module.__version__)
            )

    def test_verified_122e18b_schedule_roundtrip(self):
        self.environment["deep_gemm_version"] = "2.8.0+122e18b.cu132"
        _, manifest, _ = self.joint_artifact()
        self.assertEqual(
            manifest["schedule_environment"]["deep_gemm_version"],
            "2.8.0+122e18b.cu132",
        )

    def test_invalid_runtime_inputs_fail_closed_without_pool_copy(self):
        context, recorder = self.context()
        mutations = [
            dict(lengths=torch.tensor([2, 63, 69], dtype=torch.int32)),
            dict(context_lens=torch.tensor([[3], [63], [70]], dtype=torch.int32)),
            dict(block_table=torch.tensor([[8, 7], [5, 7], [5, 2]], dtype=torch.int32)),
            dict(
                block_table=torch.tensor([[-1, 7], [5, 7], [5, 2]], dtype=torch.int32)
            ),
            dict(schedule_metadata=torch.zeros((5, 2), dtype=torch.int64)),
            dict(schedule_metadata=torch.zeros((4, 2), dtype=torch.int32)),
            dict(
                schedule_environment=dict(self.environment, deep_gemm_version="unknown")
            ),
            dict(
                schedule_environment=dict(self.environment, device_capability=[12, 0])
            ),
            dict(q_fp8=self.q[:, 0]),
            dict(weights=self.weights.to(torch.bfloat16)),
            dict(total_query_rows=2),
            dict(max_seq_len=64),
            dict(k=1),
            dict(lengths=torch.zeros(3, dtype=torch.int32)),
            dict(
                schedule_environment=dict(
                    self.environment, device_capability=[9, 0], num_sms=True
                )
            ),
        ]
        original_copy = torch.Tensor.copy_

        def no_pool_copy(destination, source, *args, **kwargs):
            self.assertNotEqual(
                source.untyped_storage()._cdata, self.pool.untyped_storage()._cdata
            )
            return original_copy(destination, source, *args, **kwargs)

        with mock.patch.object(torch.Tensor, "copy_", no_pool_copy):
            for changes in mutations:
                with self.subTest(changes=list(changes)):
                    self.assertIsNone(self.begin_joint(context, **changes))
        self.assertEqual(recorder.stats()["inflight_bytes"], 0)
        self.assertFalse(list((self.root / "records").rglob("manifest.json")))

    def test_unknown_lengths_and_multitoken_identity_rejected(self):
        for changes in (
            {"kv_len": -1},
            {"kv_len": None},
            {"q_len": 2},
            {"q_len": True},
            {"phase": "prefill"},
            {"is_fake": True},
        ):
            identity = copy.deepcopy(self.identity)
            identity["requests"][0].update(changes)
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                indexer._paged_expected_lengths(identity, 3)

    def test_joint_never_falls_back_and_incomplete_stage_cancels(self):
        context, recorder = self.context()
        self.assertIsNone(self.begin_topk(context))
        token = self.begin_joint(context)
        self.assertIsNotNone(token)
        self.assertFalse(token.finish(self.indices))
        self.assertEqual(recorder.stats()["inflight_bytes"], 0)
        token = self.begin_joint(context)
        self.assertIsNotNone(token)
        self.assertFalse(
            token.prepare_topk(
                self.logits[:, :127], backend="topk_v3", parameters=self.parameters
            )
        )
        self.assertEqual(recorder.stats()["inflight_bytes"], 0)

    def test_invalid_topk_stage_parameters_cancel(self):
        context, recorder = self.context()
        for changes in (
            {"workspace_bytes": 1},
            {"index_space": "relative_to_row_start"},
            {"max_seq_len": 64},
            {"stride0": 128},
            {"k": 1024},
            {"stride1": True},
        ):
            token = self.begin_joint(context)
            self.assertIsNotNone(token)
            self.assertFalse(
                token.prepare_topk(
                    self.logits,
                    backend="topk_v3",
                    parameters=dict(self.parameters, **changes),
                )
            )
            self.assertEqual(recorder.stats()["inflight_bytes"], 0)
        token = self.begin_joint(context)
        self.assertFalse(
            token.prepare_topk(
                self.logits, backend="topk_v3_tie_break", parameters=self.parameters
            )
        )

    def test_validator_rejects_metadata_mapping_and_padding_corruption(self):
        _, manifest, tensors = self.joint_artifact()
        metadata_changes = [
            lambda m: m.update(phase="prefill"),
            lambda m: m.update(server_role="PREFILL"),
            lambda m: m.update(model_role="draft"),
            lambda m: m.update(rank=1),
            lambda m: m["contract"].update(version=2),
            lambda m: m["query_mapping"].update(coverage="partial"),
            lambda m: m["requests"][0].update(kv_len=99),
            lambda m: m["cache_layout"].update(page_capacity=3),
            lambda m: m["cache_layout"].update(head_dim=256),
            lambda m: m["schedule_environment"].update(num_sms=5),
            lambda m: m["mqa_parameters"].update(clean_logits=True),
            lambda m: m["tensors"]["page_ids"].update(dtype="int32"),
            lambda m: m.update(untrusted_metadata="secret"),
            lambda m: m["query_mapping"].update(next_n=True),
            lambda m: m["mqa_parameters"].update(clean_logits=0),
            lambda m: m["tensors"]["page_ids"].update(storage_offset=-1),
            lambda m: m["tensors"]["q_fp8"].update(original_stride=[1, 1, 1, 1]),
        ]
        for change in metadata_changes:
            damaged = copy.deepcopy(manifest)
            change(damaged)
            with self.subTest(change=change), self.assertRaises((ValueError, KeyError)):
                indexer.validate_paged_capture(damaged, tensors)
        tensor_changes = [
            ("page_ids", (1,), 5),
            ("page_ids", (2,), 7),
            ("page_valid_tokens", (0,), 3),
            ("page_valid_tokens", (2,), 1),
            ("block_table", (0, 1), 7),
            ("block_table", (0, 0), 1),
            ("packed_kv_pages", (2, 0, 0, 0), 231),
            ("logits", (0, 3), 913),
            ("kvlen_2d", (0, 0), 4),
            ("schedule_metadata", (4, 0), 2),
        ]
        for name, offset, value in tensor_changes:
            damaged = dict(tensors)
            damaged[name] = tensors[name].clone()
            damaged[name][offset] = value
            with self.subTest(name=name, offset=offset), self.assertRaises(ValueError):
                indexer.validate_paged_capture(manifest, damaged)
        for offset in (5 * 128, 64 * 128 + 5 * 4):
            damaged = dict(tensors)
            damaged["packed_kv_pages"] = tensors["packed_kv_pages"].clone()
            damaged["packed_kv_pages"][1].view(-1)[offset] = 231
            with self.subTest(offset=offset), self.assertRaises(ValueError):
                indexer.validate_paged_capture(manifest, damaged)

    def test_single_token_and_page_boundaries(self):
        for lengths in ((1, 64, 65), (64, 64, 128), (1, 1, 1)):
            with self.subTest(lengths=lengths):
                identity = copy.deepcopy(self.identity)
                for row, length in zip(identity["requests"], lengths):
                    row["kv_len"] = length - 1
                context, recorder = self.context(identity=identity)
                actual = torch.tensor(lengths, dtype=torch.int32)
                token = self.begin_joint(
                    context, lengths=actual, context_lens=actual.unsqueeze(1)
                )
                self.assertIsNotNone(token, recorder.stats())
                self.assertTrue(
                    token.prepare_topk(
                        self.logits, backend="topk_v3", parameters=self.parameters
                    )
                )
                self.assertTrue(token.finish(self.indices))
                self.assertTrue(recorder.close(30))
        paths = list((self.root / "records").rglob("manifest.json"))
        self.assertEqual(len(paths), 3)
        for path in paths:
            manifest, tensors = capture.load_capture(path)
            indexer.validate_paged_capture(manifest, tensors)
            expected = tensors["lengths"].tolist()
            self.assertEqual(
                manifest["cache_layout"]["page_capacity"],
                sum((n + 63) // 64 for n in expected),
            )

    def test_sm100_page_sizes(self):
        for block, table in (
            (32, [[0, 7, 7, 7], [1, 2, 7, 7], [3, 4, 5, 7]]),
            (128, [[0], [1], [2]]),
        ):
            with self.subTest(block=block):
                context, recorder = self.context()
                pool = torch.full((8, block, 1, 132), 42, dtype=torch.uint8)
                environment = dict(
                    self.environment,
                    device_capability=[10, 0],
                    deep_gemm_version="2.8.0",
                )
                schedule = torch.tensor(
                    [[0, 0], [1, 0], [2, 0], [3, 0], [3, 0]], dtype=torch.int32
                )
                token = self.begin_joint(
                    context,
                    kv_cache_fp8=pool,
                    block_size=block,
                    block_table=torch.tensor(table, dtype=torch.int32),
                    schedule_environment=environment,
                    schedule_metadata=schedule,
                )
                self.assertIsNotNone(token, recorder.stats())
                self.assertTrue(
                    token.prepare_topk(
                        self.logits, backend="topk_v3", parameters=self.parameters
                    )
                )
                self.assertTrue(token.finish(self.indices))
                self.assertTrue(recorder.close(30))
        for path in (self.root / "records").rglob("manifest.json"):
            manifest, tensors = capture.load_capture(path)
            indexer.validate_paged_capture(manifest, tensors)

    def test_topk_k_dispatch_and_missing_joint_metadata(self):
        for k in (512, 1024, 2048):
            context, recorder = self.context(joint=False)
            token = self.begin_topk(context, k=k, parameters=dict(self.parameters, k=k))
            self.assertIsNotNone(token, recorder.stats())
            self.assertTrue(token.finish(torch.full((3, k), -1, dtype=torch.int32)))
            self.assertTrue(recorder.close(30))
        paths = list((self.root / "records").rglob("manifest.json"))
        self.assertEqual(len(paths), 3)
        for path in paths:
            manifest, tensors = capture.load_capture(path)
            indexer.validate_paged_capture(manifest, tensors)
            manifest["cache_layout"] = {}
            with self.assertRaises(ValueError):
                indexer.validate_paged_capture(manifest, tensors)

    def test_actual_logits_shape_and_dtype_fail_without_snapshot(self):
        context, recorder = self.context()
        for logits in (
            self.logits[:, :127],
            torch.zeros((3, 129)),
            self.logits.to(torch.float64),
        ):
            token = self.begin_joint(context)
            self.assertIsNotNone(token)
            with mock.patch.object(
                indexer,
                "_paged_snapshot_logits",
                side_effect=AssertionError("invalid output must not be copied"),
            ) as snapshot:
                self.assertFalse(
                    token.prepare_topk(
                        logits, backend="topk_v3", parameters=self.parameters
                    )
                )
                snapshot.assert_not_called()
            self.assertEqual(recorder.stats()["inflight_bytes"], 0)

    def test_schedule_value_bounds_and_monotonicity(self):
        context, recorder = self.context()
        for offset, value in (
            ((4, 0), 2),
            ((0, 0), 1),
            ((2, 1), 1),
            ((2, 0), -1),
            ((3, 0), 0),
        ):
            schedule = self.schedule.clone()
            schedule[offset] = value
            with self.subTest(offset=offset, value=value):
                self.assertIsNone(self.begin_joint(context, schedule_metadata=schedule))
        self.assertEqual(recorder.stats()["inflight_bytes"], 0)

    def test_unauthorized_capture_and_unknown_identity_are_rejected(self):
        path = self.root / "unauthorized.json"
        path.write_text(json.dumps(dict(self.config, authorized_capture=False)))
        with self.assertRaisesRegex(ValueError, "authorized_capture"):
            capture.TensorCaptureRecorder(str(path))
        context, recorder = self.context()
        for changes in (
            {"server_role": "PREFILL"},
            {"server_role": "unknown"},
            {"model_role": "draft"},
            {"world_rank": True},
            {"session_id": "../escape"},
        ):
            identity = dict(self.identity, **changes)
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                indexer._paged_expected_lengths(identity, 3)
        self.assertEqual(recorder.stats()["inflight_bytes"], 0)

    def test_joint_source_lifetime_cancel_and_double_finish(self):
        import gc
        import weakref

        context, recorder = self.context()
        pool = self.pool.clone()
        reference = weakref.ref(pool)
        token = self.begin_joint(context, kv_cache_fp8=pool)
        self.assertIsNotNone(token)
        del pool
        gc.collect()
        self.assertIsNone(reference(), "token must not retain the global cache source")
        token.cancel("kernel_error")
        self.assertEqual(recorder.stats()["inflight_bytes"], 0)
        token = self.begin_joint(context)
        self.assertTrue(
            token.prepare_topk(
                self.logits, backend="topk_v3", parameters=self.parameters
            )
        )
        self.assertTrue(token.finish(self.indices))
        self.assertFalse(token.finish(self.indices))
        self.load(recorder)

    def test_graph_guard_rejects_before_device_snapshot(self):
        inputs = {"q_fp8": SimpleNamespace(device=torch.device("cuda:0"))}
        with (
            mock.patch.object(torch.cuda, "device"),
            mock.patch.object(
                torch.cuda, "is_current_stream_capturing", return_value=True
            ),
        ):
            with self.assertRaisesRegex(ValueError, "CUDA graph"):
                indexer._paged_source_devices(inputs)
        with (
            mock.patch.object(torch.cuda, "device"),
            mock.patch.object(
                torch.cuda, "is_current_stream_capturing", return_value=False
            ),
        ):
            indexer._paged_source_devices(inputs)
        with self.assertRaisesRegex(ValueError, "devices must match"):
            indexer._paged_source_devices(dict(inputs, lengths=self.lengths))

    def test_pdfusion_decode_uses_same_contract(self):
        context, recorder = self.context(
            identity=dict(self.identity, server_role="PDFUSION")
        )
        token = self.begin_joint(context)
        self.assertIsNotNone(token, recorder.stats())
        self.assertTrue(
            token.prepare_topk(
                self.logits, backend="topk_v3", parameters=self.parameters
            )
        )
        self.assertTrue(token.finish(self.indices))
        _, manifest, _ = self.load(recorder)
        self.assertEqual(manifest["phase"], "decode")
        self.assertEqual(manifest["server_role"], "PDFUSION")

    def test_filters_do_not_touch_inputs(self):
        context, recorder = self.context(config=dict(self.config, layers=[1]))
        with mock.patch.object(
            indexer,
            "_paged_expected_lengths",
            side_effect=AssertionError("selection must precede validation"),
        ):
            self.assertIsNone(self.begin_joint(context))
        self.assertEqual(recorder.stats()["skip_reasons"].get("layer_filter"), 1)

    def test_loader_rejects_payload_hash_corruption(self):
        path, manifest, _ = self.joint_artifact()
        payload = path.parent / manifest["tensors"]["packed_kv_pages"]["file"]
        data = bytearray(payload.read_bytes())
        data[0] ^= 1
        payload.write_bytes(data)
        with self.assertRaisesRegex(ValueError, "hash|checksum"):
            capture.load_capture(path)


if __name__ == "__main__":
    unittest.main()
