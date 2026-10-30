"""Model-free replay tests. CPU tests never skip.

CPU only:
  /opt/conda310/bin/python3 -m unittest rtp_llm.utils.test.tensor_capture_replay_test.TensorCaptureReplayTest -v
Real TopK CUDA (Bazel-built libraries, explicit opt-in set by the target):
  bazelisk test --config=cuda13 --config=sm120 //rtp_llm/utils/test:tensor_capture_replay_cuda_test
TensorCaptureMqaReplayCudaTest additionally requires a coordinated Bazel target
with //rtp_llm/models_py:deep_gemm (production wheel) and RTP_LLM_TEST_CAPTURE_REPLAY_CUDA=1.
Do not run CUDA classes against unverified libraries in a bare Python environment.
Once opted in, missing GPU/library or unsupported architecture FAILS, except the
real paged MQA test explicitly skips devices outside the verified SM90/SM100
families (the SM100 family includes SM103). It can run model-free against an installed RTP
wheel plus the matching capture/replay sources and a verified DeepGEMM wheel:
  RTP_LLM_TEST_CAPTURE_REPLAY_CUDA=1 /opt/conda310/bin/python3 tensor_capture_replay_test.py TensorCaptureMqaReplayCudaTest.test_paged_mqa_shared_partial_pages_reuse_and_rebuild -v
Temporary bundles use TEST_TMPDIR, TMPDIR, then RTP_LLM_TEST_LOCAL_LAB.
Without these variables, tempfile chooses the default temporary directory.
"""

import contextlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

from rtp_llm.utils import tensor_capture as capture
from rtp_llm.utils import tensor_capture_replay as replay
from rtp_llm.utils.tensor_capture_indexer import for_context


class _FloatPairReplayAdapter:
    """Test-only non-Indexer contract registered without editing the dispatcher."""

    name = "test.replay_float_pair"

    @staticmethod
    def validate(manifest, tensors):
        if manifest["backend"] != "torch_reference":
            raise replay.UnsupportedCapture("unsupported float-pair backend")
        if (
            set(manifest["inputs"]) != {"x", "y"}
            or set(manifest["outputs"]) != {"sum", "difference"}
            or set(tensors) != {"x", "y", "sum", "difference"}
        ):
            raise ValueError("invalid float-pair named tensors")
        if manifest["parameters"] != {"bias": 0.25}:
            raise ValueError("invalid float-pair parameters")
        for name, tensor in tensors.items():
            if tensor.dtype != torch.float32 or tensor.shape != tensors["x"].shape:
                raise ValueError("float-pair dtype/shape: " + name)
            if not bool(torch.isfinite(tensor).all()):
                raise replay.UnsupportedCapture("nonfinite float-pair tensor: " + name)
        expected = dict(
            sum=tensors["x"] + tensors["y"] + 0.25,
            difference=tensors["x"] - tensors["y"] - 0.25,
        )
        for name, value in expected.items():
            if not torch.equal(tensors[name], value):
                raise ValueError("float-pair numerical mismatch: " + name)
        return dict(
            comparison="all_named_float_outputs",
            compared_outputs=2,
            compared_values=sum(value.numel() for value in expected.values()),
        )


_FLOAT_PAIR_CAPTURE = capture.CaptureContract(_FloatPairReplayAdapter.name)
capture.register_capture_contract(_FLOAT_PAIR_CAPTURE)


class _ReplayFixture(unittest.TestCase):
    def setUp(self):
        lab = (
            os.environ.get("TEST_TMPDIR")
            or os.environ.get("TMPDIR")
            or os.environ.get("RTP_LLM_TEST_LOCAL_LAB")
        )

        self.temporary = tempfile.TemporaryDirectory(
            prefix="tensor-replay-test-", dir=lab
        )
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.serial = 0

    def bundle(
        self,
        logits=None,
        starts=None,
        ends=None,
        indices=None,
        backend="dsv4_per_row",
        **params
    ):
        # Exercise the real raw serializer/loader. Synthetic bounds (including
        # length zero) also cover legal operator inputs not emitted by prefill.
        if logits is None:
            logits = torch.tensor([[0.0, -3.0, -1.0, -2.0, 0.0]])
        if starts is None:
            starts = torch.tensor([1], dtype=torch.int32)
        if ends is None:
            ends = torch.tensor([4], dtype=torch.int32)
        if indices is None:
            indices = torch.tensor([[2, 1]], dtype=torch.int32)
        rows, k = indices.shape
        parameters = dict(
            k=k,
            index_space="relative_to_row_start",
            stride0=logits.stride(0),
            stride1=logits.stride(1),
        )
        if backend == "topk_v3_tie_break":
            parameters["max_seq_len"] = logits.shape[1]
        else:
            parameters.update(num_rows=rows, force_radix_sort=False)
        parameters.update(params)
        manifest = dict(
            schema_version=1,
            complete=True,
            op="indexer.topk",
            phase="prefill",
            backend=backend,
            parameters=parameters,
            query_row_start=0,
            total_query_rows=rows,
            query_row_count=rows,
            invalid_logits="zero_filled_snapshot_only",
            finite_values_checked=False,
            nonfinite_policy="valid_domain_bits_preserved_backend_semantics",
            output_contract="raw_relative_indices_minus_one_padding_backend_order",
            tensors={},
        )
        directory = self.root / ("bundle-%d" % self.serial)
        self.serial += 1
        directory.mkdir()
        tensors = dict(logits=logits, row_starts=starts, row_ends=ends, indices=indices)
        for name, tensor in tensors.items():
            desc, data = capture._encode_tensor(tensor)
            desc["file"] = name + ".bin"
            manifest["tensors"][name] = desc
            (directory / desc["file"]).write_bytes(data)
        path = directory / "manifest.json"
        path.write_text(json.dumps(manifest))
        return path

    def write_mqa_bundle(self, tensors, compute=None):
        """Use the production group writer, including its two-phase lifecycle."""
        directory = self.root / ("mqa-%d" % self.serial)
        self.serial += 1
        directory.mkdir()
        rows, _, _ = tensors["q_fp8"].shape
        columns = tensors["k_fp8"].shape[0]
        k = tensors["indices"].shape[1]
        config = directory / "config.json"
        config.write_text(
            json.dumps(
                dict(
                    enabled=True,
                    authorized_capture=True,
                    layers=[0],
                    ops=["indexer.mqa_logits", "indexer.topk"],
                    phases=["prefill"],
                    rank_filter="record_owner",
                    max_calls=1,
                    max_bundle_bytes=10000000,
                    max_total_bytes=10000000,
                    max_inflight_bytes=100000000,
                )
            )
        )
        identity = dict(
            schema_version=1,
            session_id="mqa-test",
            owner_id="owner",
            replica_id="dp0",
            world_rank=0,
            dp_rank=0,
            exec_id="1",
            model_role="target",
            requests=[
                dict(
                    phase="prefill", request_id="r1", q_len=rows, kv_len=columns - rows
                )
            ],
        )
        with mock.patch.dict(
            os.environ,
            RTP_LLM_RECORD_DIR=str(directory / "records"),
            RTP_LLM_RECORD_SESSION="mqa-test",
        ):
            recorder = capture.TensorCaptureRecorder(str(config))
            token = None
            try:
                context = for_context(recorder.for_execution(json.dumps(identity)))
                self.assertIsNotNone(context)
                token = context.begin_mqa(
                    layer=0,
                    k=k,
                    query_row_start=0,
                    total_query_rows=rows,
                    **{
                        name: tensors[name]
                        for name in (
                            "q_fp8",
                            "k_fp8",
                            "k_scale",
                            "weights",
                            "row_starts",
                            "row_ends",
                        )
                    }
                )
                self.assertIsNotNone(token)
                logits, indices = (
                    compute(tensors)
                    if compute
                    else (tensors["logits"], tensors["indices"])
                )
                self.assertTrue(
                    token.prepare_topk(
                        logits,
                        backend="dsv4_per_row",
                        parameters=dict(
                            num_rows=rows,
                            stride0=logits.stride(0),
                            stride1=logits.stride(1),
                            force_radix_sort=True,
                        ),
                    )
                )
                token.finish(indices)
            except BaseException:
                if token is not None:
                    token.cancel("kernel_error")
                raise
            finally:
                self.assertTrue(recorder.close(5))
        paths = list((directory / "records").rglob("manifest.json"))
        self.assertEqual(len(paths), 1)
        return paths[0]

    def mqa_bundle(self, dimension=128):
        q = torch.full((3, 2, dimension + 8), 2.0).to(torch.float8_e4m3fn)[
            1:, :, :dimension
        ]
        key = torch.ones(5, dimension + 8).to(torch.float8_e4m3fn)[1:, :dimension]
        scale_shape = (4,) if dimension == 128 else (4, dimension // 128)
        tensors = dict(
            q_fp8=q,
            k_fp8=key,
            k_scale=torch.full(scale_shape, 2.0),
            weights=torch.tensor([[6.0, 10.0], [6.0, 10.0]]),
            row_starts=torch.tensor([0, 0], dtype=torch.int32),
            row_ends=torch.tensor([3, 4], dtype=torch.int32),
            logits=torch.tensor([[1.0, 4.0, 2.0, float("nan")], [1.0, 4.0, 2.0, 3.0]]),
            indices=torch.tensor([[1, 2], [1, 3]], dtype=torch.int32),
        )
        return self.write_mqa_bundle(tensors)

    def indexed_bundle(
        self,
        *,
        joint=False,
        ids=(4, 0, 3),
        server_role=None,
        backend="dsv4_per_row",
        force=False,
        k=2,
        cp_size=2
    ):
        """Real version2 writer, with global frozen requests and local indexed rows."""
        directory = self.root / ("indexed-%d" % self.serial)
        self.serial += 1
        directory.mkdir()
        op = "indexer.mqa_topk" if joint else "indexer.topk"
        config = directory / "config.json"
        config.write_text(
            json.dumps(
                dict(
                    enabled=True,
                    authorized_capture=True,
                    layers=[0],
                    ops=[op],
                    phases=["prefill"],
                    rank_filter="record_owner",
                    max_calls=1,
                    max_bundle_bytes=10000000,
                    max_total_bytes=10000000,
                    max_inflight_bytes=100000000,
                )
            )
        )
        identity = dict(
            schema_version=1,
            session_id="indexed-test",
            owner_id="owner",
            replica_id="dp0",
            world_rank=0,
            dp_rank=0,
            exec_id="1",
            model_role="target",
            requests=[
                dict(phase="prefill", request_id="r1", q_len=3, kv_len=2),
                dict(phase="prefill", request_id="r2", q_len=2, kv_len=1),
            ],
        )
        if server_role is not None:
            identity["server_role"] = server_role
        rows, columns = len(ids), 8
        # Explicit oracle for requests (q=3,prefix=2), (q=2,prefix=1).
        starts = torch.tensor([0, 0, 0, 5, 5], dtype=torch.int32)[list(ids)]
        ends = torch.tensor([3, 4, 5, 7, 8], dtype=torch.int32)[list(ids)]
        logits = torch.zeros(rows + 1, columns + 3)[1:, :columns]
        indices = torch.full((rows, k), -1, dtype=torch.int32)
        for row in range(rows):
            start, end = int(starts[row]), int(ends[row])
            logits[row, start:end] = 4096.0 if joint else 1.0
            count = min(k, end - start)
            indices[row, :count] = torch.arange(count)
        query_global_ids = torch.tensor(ids, dtype=torch.int64)
        inputs = dict(
            row_starts=starts, row_ends=ends, query_global_ids=query_global_ids
        )
        outputs = dict(indices=indices)
        metadata = dict(
            query_row_count=rows,
            total_query_rows=5,
            query_mapping=dict(
                kind="indexed",
                tensor="query_global_ids",
                index_space="request_order_query_rows",
            ),
            key_domain=dict(layout="request_order_prefix_then_query", coverage="full"),
            execution_layout=dict(
                cp_rank=cp_size - 1, cp_size=cp_size, kv_cache_sharded=cp_size > 1
            ),
            capture_coverage=dict(
                scope="record_owner_local",
                local_query_rows=min(rows + 1, 5),
                local_query_row_start=int(rows < 5),
            ),
            invalid_logits="zero_filled_snapshot_only",
            finite_values_checked=False,
            nonfinite_policy="valid_domain_bits_preserved_backend_semantics",
            output_contract="raw_relative_indices_minus_one_padding_backend_order",
        )
        if joint:
            inputs = dict(
                q_fp8=torch.ones(rows, 32, 128).to(torch.float8_e4m3fn),
                k_fp8=torch.ones(columns, 128).to(torch.float8_e4m3fn),
                k_scale=torch.ones(columns),
                weights=torch.ones(rows, 32),
                **inputs
            )
            outputs = dict(logits=logits, indices=indices)
            metadata.update(
                mqa_parameters=dict(
                    backend="deep_gemm.fp8_mqa_logits",
                    clean_logits=False,
                    weights_include_query_scale=True,
                    key_scale_format="float32_per_128_channels",
                ),
                operations=[
                    dict(
                        op="indexer.mqa_logits",
                        inputs=[
                            "q_fp8",
                            "k_fp8",
                            "k_scale",
                            "weights",
                            "row_starts",
                            "row_ends",
                        ],
                        outputs=["logits"],
                    ),
                    dict(
                        op="indexer.topk",
                        inputs=["logits", "row_starts", "row_ends"],
                        outputs=["indices"],
                    ),
                ],
            )
        else:
            inputs = dict(logits=logits, **inputs)
        parameters = dict(
            k=k,
            index_space="relative_to_row_start",
            stride0=logits.stride(0),
            stride1=logits.stride(1),
        )
        if backend == "dsv4_per_row":
            parameters.update(num_rows=rows, force_radix_sort=force)
        else:
            parameters.update(max_seq_len=columns)
        with mock.patch.dict(
            os.environ,
            RTP_LLM_RECORD_DIR=str(directory / "records"),
            RTP_LLM_RECORD_SESSION="indexed-test",
        ):
            recorder = capture.TensorCaptureRecorder(str(config))
            token = None
            try:
                context = recorder.for_execution(json.dumps(identity))
                self.assertIsNotNone(context)
                token = context.begin(
                    op=op,
                    contract_version=2,
                    layer=0,
                    inputs=inputs,
                    outputs={
                        name: capture.TensorSpec(tensor.shape, tensor.dtype, "cpu")
                        for name, tensor in outputs.items()
                    },
                    backend=backend,
                    parameters=parameters,
                    metadata=metadata,
                )
                self.assertIsNotNone(token)
                # Identity and IDs must be frozen before asynchronous publication.
                identity["requests"][0]["q_len"] = 99
                query_global_ids.fill_(99)
                if joint:
                    self.assertTrue(token.capture_outputs(dict(logits=logits)))
                    self.assertEqual(
                        list((directory / "records").rglob("manifest.json")), []
                    )
                    self.assertTrue(token.finish(dict(indices=indices)))
                else:
                    self.assertTrue(token.finish(outputs))
            finally:
                if token is not None:
                    token.cancel()
                self.assertTrue(recorder.close(5))
        paths = list((directory / "records").rglob("manifest.json"))
        self.assertEqual(len(paths), 1)
        return paths[0]

    def paged_bundle(
        self,
        *,
        joint=False,
        backend="topk_v3",
        lengths=(3, 63, 69),
        k=512,
        logits=None,
        indices=None
    ):
        """Production paged writer, including shared pages and poisoned source tails."""
        directory = self.root / ("paged-%d" % self.serial)
        self.serial += 1
        directory.mkdir()
        rows, block = len(lengths), 64
        columns = ((max(lengths) + block - 1) // block + 1) * block
        if logits is None:
            logits = torch.full((rows + 1, columns + 16), float("nan"))[1:, :columns]
            for row, length in enumerate(lengths):
                logits[row, :length] = torch.arange(length, dtype=torch.float32)
        if indices is None:
            indices = torch.full((rows, k), -1, dtype=torch.int32)
            for row, length in enumerate(lengths):
                indices[row, : min(k, length)] = (
                    torch.arange(length)
                    if length <= k
                    else torch.arange(length - 1, length - k - 1, -1)
                )
        runtime_lengths = torch.tensor(lengths, dtype=torch.int32)
        parameters = dict(
            k=k,
            max_seq_len=columns,
            index_space="sequence_relative",
            workspace_bytes=1 << 20,
            stride0=logits.stride(0),
            stride1=1,
        )
        identity = dict(
            schema_version=1,
            session_id="paged-replay",
            owner_id="owner",
            replica_id="dp0",
            world_rank=0,
            dp_rank=0,
            exec_id="1",
            model_role="target",
            server_role="DECODE",
            requests=[
                dict(
                    phase="decode",
                    request_id="r%d" % (row + 1),
                    q_len=1,
                    kv_len=length - 1,
                )
                for row, length in enumerate(lengths)
            ],
        )
        config = directory / "config.json"
        config.write_text(
            json.dumps(
                dict(
                    enabled=True,
                    authorized_capture=True,
                    layers=[0],
                    ops=["indexer.mqa_topk" if joint else "indexer.topk"],
                    phases=["decode"],
                    rank_filter="record_owner",
                    max_calls=1,
                    max_bundle_bytes=10000000,
                    max_total_bytes=10000000,
                    max_inflight_bytes=100000000,
                )
            )
        )
        with mock.patch.dict(
            os.environ,
            RTP_LLM_RECORD_DIR=str(directory / "records"),
            RTP_LLM_RECORD_SESSION="paged-replay",
        ):
            recorder = capture.TensorCaptureRecorder(str(config))
            token = None
            try:
                context = for_context(recorder.for_execution(json.dumps(identity)))
                self.assertIsNotNone(context, recorder.stats())
                if joint:
                    width = columns // block
                    physical = [5, 2] + list(range(8, 8 + width - 2))
                    table = torch.tensor([physical] * rows, dtype=torch.int32)
                    pool = torch.full(
                        (max(physical) + 1, block, 1, 132), 231, dtype=torch.uint8
                    )
                    for page in physical:
                        pool[page].view(-1)[: block * 128].fill_(17)
                        pool[page].view(-1)[block * 128 :].view(torch.float32).fill_(
                            1.25
                        )
                    token = context.begin_paged_mqa(
                        layer=0,
                        q_fp8=torch.ones(rows, 1, 32, 128).to(torch.float8_e4m3fn),
                        kv_cache_fp8=pool,
                        weights=torch.ones(rows, 32),
                        context_lens=runtime_lengths[:, None],
                        block_table=table,
                        schedule_metadata=torch.tensor(
                            [[0, 0]] * 4 + [[rows, 0]], dtype=torch.int32
                        ),
                        lengths=runtime_lengths,
                        k=k,
                        block_size=block,
                        max_seq_len=columns,
                        total_query_rows=rows,
                        schedule_environment=dict(
                            deep_gemm_version="2.8.0+6d6aaad.cu132",
                            device_capability=[9, 0],
                            num_sms=4,
                        ),
                    )
                    self.assertIsNotNone(token, recorder.stats())
                    self.assertTrue(
                        token.prepare_topk(
                            logits, backend=backend, parameters=parameters
                        )
                    )
                else:
                    token = context.begin_paged_topk(
                        layer=0,
                        logits=logits,
                        lengths=runtime_lengths,
                        k=k,
                        backend=backend,
                        parameters=parameters,
                        total_query_rows=rows,
                    )
                    self.assertIsNotNone(token, recorder.stats())
                self.assertTrue(token.finish(indices))
            finally:
                if token is not None:
                    token.cancel()
                self.assertTrue(recorder.close(5), recorder.stats())
        paths = list((directory / "records").rglob("manifest.json"))
        self.assertEqual(len(paths), 1, recorder.stats())
        return paths[0]

    def replace_tensor(self, path, name, tensor):
        desc, data = capture._encode_tensor(tensor)
        desc["file"] = name + ".bin"
        (path.parent / desc["file"]).write_bytes(data)
        self.mutate(path, lambda m: m["tensors"].update({name: desc}))

    def mutate(self, path, change):
        manifest = json.loads(path.read_text())
        change(manifest)
        path.write_text(json.dumps(manifest))

    def boundary_bundle(self, backend, k, force=False):
        rows, width = 4, k + 8
        stride1 = 1 if backend == "topk_v3_tie_break" else 2
        storage = torch.zeros(rows * (width * stride1 + 8) + 3)
        logits = storage.as_strided((rows, width), (width * stride1 + 8, stride1), 3)
        starts = torch.tensor([2, 3, 1, 4], dtype=torch.int32)
        lengths = torch.tensor([0, 1, k, k + 1], dtype=torch.int32)
        ends = starts + lengths
        output = torch.full((rows, k), -1, dtype=torch.int32)
        output[1, 0] = 0
        output[2] = torch.arange(k)
        # All scores tie: smaller relative indices win, but long-row order is free.
        output[3] = torch.arange(k - 1, -1, -1)
        params = dict(force_radix_sort=force) if backend == "dsv4_per_row" else {}
        return self.bundle(logits, starts, ends, output, backend, **params)


class TensorCaptureReplayTest(_ReplayFixture):
    @contextlib.contextmanager
    def paged_runtime(
        self,
        path,
        *,
        version="2.8.0+6d6aaad.cu132",
        capability=(9, 0),
        sms=4,
        logits=None,
        topk=None,
        distribution_version=None
    ):
        manifest, tensors = capture.load_capture(path)
        if logits is None:
            logits = tensors["logits"].clone()
            for row, length in enumerate(tensors["lengths"].tolist()):
                logits[row, length:] = float("nan")
        module = SimpleNamespace(
            __file__=str(self.root / "runtime-deep-gemm" / "deep_gemm" / "__init__.py"),
            __version__=version,
            get_num_sms=mock.Mock(return_value=sms),
            fp8_paged_mqa_logits=mock.Mock(return_value=logits),
            get_paged_mqa_logits_metadata=mock.Mock(
                side_effect=lambda lengths, block, count: torch.tensor(
                    [[0, 0]] * count + [[lengths.shape[0], 0]], dtype=torch.int32
                )
            ),
        )

        def reference(scores, lengths, output, workspace, k, max_seq_len):
            self.assertEqual(workspace.dtype, torch.uint8)
            self.assertEqual(
                workspace.numel(), manifest["parameters"]["workspace_bytes"]
            )
            self.assertEqual(
                scores.stride(), tuple(manifest["tensors"]["logits"]["original_stride"])
            )
            output.fill_(-1)
            for row, length in enumerate(lengths.tolist()):
                if length <= k:
                    output[row, :length] = torch.arange(length)
                else:
                    output[row] = torch.argsort(
                        replay._score_keys(scores[row, :length]), descending=True
                    )[:k]

        distributions = (
            []
            if distribution_version is None
            else [
                SimpleNamespace(
                    metadata={"Name": "deep-gemm"},
                    version=distribution_version,
                    locate_file=lambda name: Path(module.__file__).parent.parent / name,
                )
            ]
        )
        operator = mock.Mock(side_effect=reference if topk is None else topk)
        restore, restore_paged = replay._restore, replay._restore_paged
        with mock.patch.object(
            replay, "_cuda_operator", return_value=operator
        ), mock.patch.object(
            replay,
            "_restore_paged",
            side_effect=lambda m, t, d: restore_paged(m, t, "cpu"),
        ), mock.patch.object(
            replay, "_restore", side_effect=lambda t, desc, d: restore(t, desc, "cpu")
        ), mock.patch.object(
            replay.importlib, "import_module", return_value=module
        ), mock.patch.object(
            replay.tensor_capture_indexer.package_metadata,
            "distributions",
            return_value=distributions,
        ), mock.patch.object(
            torch.cuda, "get_device_capability", return_value=capability
        ), mock.patch.object(
            torch.cuda, "synchronize"
        ):
            yield module, operator

    def test_paged_writer_loader_cpu_contracts_and_versions(self):
        for joint in (False, True):
            for backend in ("topk_v3", "dsv4_persistent_topk"):
                for k in (512, 1024, 2048):
                    with self.subTest(joint=joint, backend=backend, k=k):
                        path = self.paged_bundle(
                            joint=joint,
                            backend=backend,
                            k=k,
                            lengths=(1, 63, 64, 65, k, k + 1),
                        )
                        result = replay.replay_capture(path)
                        self.assertEqual(result["contract"]["version"], 3)
                        self.assertEqual(result["phase"], "decode")
                        self.assertFalse(result["kernel_replayed"])
                        self.assertEqual(
                            result["counts"]["selected_indices"], 193 + 2 * k
                        )
                        if joint:
                            self.assertFalse(result["mqa_kernel_replayed"])
        # Same schema envelope still routes CP to version2, never decode.
        self.assertEqual(
            replay.replay_capture(self.indexed_bundle())["contract"]["version"], 2
        )

    def test_paged_boundary_ties_do_not_impose_prefill_index_priority(self):
        for backend in ("topk_v3", "dsv4_persistent_topk"):
            path = self.paged_bundle(backend=backend, lengths=(513,))
            manifest, tensors = capture.load_capture(path)
            tensors["logits"].zero_()
            tensors["logits"][0, :513] = 7
            self.replace_tensor(path, "logits", tensors["logits"])
            self.mutate(
                path,
                lambda m: m["parameters"].update(stride0=tensors["logits"].shape[1]),
            )
            # Captured set is 1..512 (not the legacy stable-index 0..511).
            result = replay.replay_capture(path)
            self.assertEqual(result["counts"]["boundary_tie_rows"], 1)
            self.assertIn("ties_unspecified", result["selection_contract"])
            with self.paged_runtime(path):
                result = replay.replay_capture(path, "cuda")
            self.assertTrue(result["independent_topk"]["captured_score_keys_equal"])

    def test_paged_signed_zero_keys_and_valid_nan_policy(self):
        for backend in ("topk_v3", "dsv4_persistent_topk"):
            path = self.paged_bundle(backend=backend, lengths=(513,))
            _, tensors = capture.load_capture(path)
            scores = tensors["logits"]
            scores.zero_()
            scores[0, 0] = -0.0
            self.replace_tensor(path, "logits", scores)
            self.mutate(path, lambda m: m["parameters"].update(stride0=scores.shape[1]))
            replay.replay_capture(path)
            bad = tensors["indices"].clone()
            bad[0, 0] = 0
            self.replace_tensor(path, "indices", bad)
            with self.assertRaisesRegex(ValueError, "maximal candidate set"):
                replay.replay_capture(path)
            scores[0, 0] = float("nan")
            self.replace_tensor(path, "logits", scores)
            with self.assertRaisesRegex(replay.UnsupportedCapture, "valid-domain NaN"):
                replay.replay_capture(path)

    def test_paged_index_padding_bounds_order_and_score_rejections(self):
        for kind in (
            "padding",
            "duplicate",
            "negative",
            "range",
            "short_order",
            "score",
        ):
            with self.subTest(kind=kind):
                path = self.paged_bundle(lengths=(3, 513))
                _, tensors = capture.load_capture(path)
                output = tensors["indices"].clone()
                if kind == "padding":
                    output[0, 3] = 0
                elif kind == "duplicate":
                    output[1, 0] = output[1, 1]
                elif kind == "negative":
                    output[1, 0] = -1
                elif kind == "range":
                    output[1, 0] = 513
                elif kind == "short_order":
                    output[0, :3] = torch.tensor([2, 1, 0])
                else:
                    output[1, 0] = 0
                self.replace_tensor(path, "indices", output)
                with self.assertRaises(ValueError):
                    replay.replay_capture(path)

    def test_paged_tampered_metadata_and_compact_padding_rejected_before_cuda(self):
        changes = [
            lambda m: m["contract"].update(version=2),
            lambda m: m.update(phase="prefill"),
            lambda m: m["requests"][0].update(kv_len=9),
            lambda m: m["query_mapping"].update(next_n=2),
            lambda m: m["cache_layout"].update(page_capacity=99),
            lambda m: m["schedule_environment"].update(deep_gemm_version="unknown"),
            lambda m: m["parameters"].update(workspace_bytes=2**31 - 1),
            lambda m: m["tensors"]["weights"].update(storage_offset=2**40),
        ]
        for change in changes:
            path = self.paged_bundle(joint=True)
            self.mutate(path, change)
            with mock.patch.object(
                replay, "_cuda_operator"
            ) as operator, self.assertRaises(ValueError):
                replay.replay_capture(path, "cuda")
            operator.assert_not_called()
        for name, change in (
            ("lengths", lambda t: t.__setitem__(0, 4)),
            ("kvlen_2d", lambda t: t.__setitem__((0, 0), 4)),
            ("page_ids", lambda t: t.__setitem__(1, 5)),
            ("page_valid_tokens", lambda t: t.__setitem__(0, 3)),
            ("block_table", lambda t: t.__setitem__((0, -1), 5)),
            ("schedule_metadata", lambda t: t.__setitem__((0, 0), 1)),
            ("packed_kv_pages", lambda t: t[-1].fill_(1)),
            ("logits", lambda t: t.__setitem__((0, -1), -0.0)),
        ):
            with self.subTest(tensor=name):
                path = self.paged_bundle(joint=True)
                _, tensors = capture.load_capture(path)
                change(tensors[name])
                self.replace_tensor(path, name, tensors[name])
                if name == "logits":
                    self.mutate(
                        path,
                        lambda m: m["parameters"].update(
                            stride0=tensors[name].shape[1]
                        ),
                    )
                with mock.patch.object(
                    replay, "_cuda_operator"
                ) as operator, self.assertRaises(ValueError):
                    replay.replay_capture(path, "cuda")
                operator.assert_not_called()

    def test_paged_compact_remap_large_ids_shared_pages_and_byte_layout(self):
        path = self.paged_bundle(joint=True)
        manifest, tensors = capture.load_capture(path)
        high = 2**30
        table, ids = tensors["block_table"], tensors["page_ids"]
        table[table == 5] = high
        ids[ids == 5] = high
        self.replace_tensor(path, "block_table", table)
        self.replace_tensor(path, "page_ids", ids)
        self.mutate(
            path, lambda m: m["cache_layout"].update(source_page_count=high + 1)
        )
        replay.replay_capture(path)
        manifest, tensors = capture.load_capture(path)
        restored = replay._restore_paged(manifest, tensors, "cpu")
        self.assertEqual(restored["packed_kv_pages"].shape, (2, 64, 1, 132))
        self.assertEqual(
            restored["block_table"].tolist(), [[0, -1, -1], [0, -1, -1], [0, 1, -1]]
        )
        self.assertTrue(
            torch.equal(restored["packed_kv_pages"], tensors["packed_kv_pages"][:2])
        )
        self.assertTrue(
            torch.all(
                restored["packed_kv_pages"][0].view(-1)[64 * 128 :].view(torch.float32)
                == 1.25
            )
        )
        self.assertEqual(restored["logits"].stride(), (208, 1))
        self.assertEqual(restored["logits"].storage_offset(), 208)
        # Compact slot order is not required to be sorted by physical ID.
        tensors["page_ids"][:2] = tensors["page_ids"][:2].flip(0)
        tensors["page_valid_tokens"][:2] = tensors["page_valid_tokens"][:2].flip(0)
        tensors["packed_kv_pages"][:2] = tensors["packed_kv_pages"][:2].flip(0)
        replay._check_paged_schema(manifest, tensors)
        reordered = replay._restore_paged(manifest, tensors, "cpu")
        self.assertEqual(
            reordered["block_table"].tolist(), [[1, -1, -1], [1, -1, -1], [1, 0, -1]]
        )
        self.assertTrue(
            torch.equal(reordered["packed_kv_pages"][1], restored["packed_kv_pages"][0])
        )

    def test_paged_mocked_joint_replay_reuses_and_rebuilds_verified_schedules(self):
        for backend in ("topk_v3", "dsv4_persistent_topk"):
            for version, capability, sms, mode in (
                ("2.8.0+6d6aaad.cu132", (9, 0), 4, "reused"),
                ("2.8.0+122e18b.cu132", (9, 0), 4, "rebuilt"),
                ("2.8.0+6d6aaad.cu132", (10, 0), 4, "rebuilt"),
                ("2.8.0+6d6aaad.cu132", (9, 0), 6, "rebuilt"),
            ):
                with self.subTest(
                    backend=backend, mode=mode, capability=capability, sms=sms
                ):
                    path = self.paged_bundle(joint=True, backend=backend)
                    with self.paged_runtime(
                        path, version=version, capability=capability, sms=sms
                    ) as (module, operator):
                        result = replay.replay_capture(path, "cuda")
                    self.assertEqual(result["schedule"]["mode"], mode)
                    self.assertEqual(
                        module.get_paged_mqa_logits_metadata.call_count,
                        int(mode == "rebuilt"),
                    )
                    self.assertEqual(operator.call_count, 2)
                    self.assertTrue(result["mqa_kernel_replayed"])
                    self.assertTrue(result["chained_topk"]["contract_valid"])
                    self.assertEqual(result["mqa_comparison"]["valid_values"], 135)
                    self.assertEqual(result["mqa_comparison"]["max_abs"], 0)
                    args, kwargs = module.fp8_paged_mqa_logits.call_args
                    self.assertEqual(args[0].shape, (3, 1, 32, 128))
                    self.assertEqual(args[1].shape, (2, 64, 1, 132))
                    self.assertTrue(torch.equal(args[2], torch.ones(3, 32)))
                    self.assertEqual(
                        args[4].tolist(), [[0, -1, -1], [0, -1, -1], [0, 1, -1]]
                    )
                    self.assertEqual(args[5].shape, (sms + 1, 2))
                    self.assertEqual(args[6], 192)
                    self.assertEqual(kwargs, dict(clean_logits=False))

    def test_paged_runtime_unknown_abi_sm120_and_invalid_scheduler_rejected(self):
        path = self.paged_bundle(joint=True)
        for options in (
            dict(version="unknown"),
            dict(version="2.8.0"),
            dict(capability=(12, 0)),
            dict(sms=0),
        ):
            with self.subTest(options=options), self.paged_runtime(path, **options) as (
                module,
                _,
            ):
                with self.assertRaises(replay.UnsupportedCapture):
                    replay.replay_capture(path, "cuda")
                module.get_paged_mqa_logits_metadata.assert_not_called()
                module.fp8_paged_mqa_logits.assert_not_called()
        for schedule in (
            torch.zeros(7, 2, dtype=torch.int64),
            torch.zeros(7, 2, dtype=torch.int32),
        ):
            with self.paged_runtime(path, sms=6) as (module, _):
                module.get_paged_mqa_logits_metadata.return_value = schedule
                module.get_paged_mqa_logits_metadata.side_effect = None
                with self.assertRaises(ValueError):
                    replay.replay_capture(path, "cuda")
                module.fp8_paged_mqa_logits.assert_not_called()

    def test_paged_runtime_resolves_distribution_build_for_schedule_reuse(self):
        for captured in ("2.8.0", "2.8.0+6d6aaad.cu132", "2.8.0+122e18b.cu132"):
            for runtime in ("2.8.0+6d6aaad.cu132", "2.8.0+122e18b.cu132"):
                with self.subTest(captured=captured, runtime=runtime):
                    path = self.paged_bundle(joint=True)
                    self.mutate(
                        path,
                        lambda m: m["schedule_environment"].update(
                            deep_gemm_version=captured
                        ),
                    )
                    # The wheel exposes a generic module version; distribution
                    # metadata, not that string, determines schedule compatibility.
                    with self.paged_runtime(
                        path, version="2.8.0", distribution_version=runtime
                    ) as (module, _):
                        result = replay.replay_capture(path, "cuda")
                    self.assertEqual(
                        result["schedule"]["environment"]["deep_gemm_version"], runtime
                    )
                    self.assertEqual(
                        result["schedule"]["mode"],
                        "reused" if captured == runtime else "rebuilt",
                    )
                    self.assertEqual(
                        module.get_paged_mqa_logits_metadata.call_count,
                        int(captured != runtime),
                    )
                    module.fp8_paged_mqa_logits.assert_called_once()

    def test_paged_runtime_unknown_or_conflicting_distribution_rejected(self):
        path = self.paged_bundle(joint=True)
        for module_version, distribution in (
            ("2.8.0", "2.8.0+unknown.cu132"),
            ("2.8.0", "2.8.0"),
            ("2.8.0+6d6aaad.cu132", "2.8.0+122e18b.cu132"),
            ("2.7.0", "2.8.0+122e18b.cu132"),
        ):
            with self.subTest(
                module_version=module_version, distribution=distribution
            ), self.paged_runtime(
                path, version=module_version, distribution_version=distribution
            ) as (
                module,
                _,
            ):
                with self.assertRaisesRegex(
                    replay.UnsupportedCapture, "unverified runtime paged schedule ABI"
                ) as caught:
                    replay.replay_capture(path, "cuda")
                self.assertIsInstance(caught.exception.__cause__, ValueError)
                module.get_num_sms.assert_not_called()
                module.get_paged_mqa_logits_metadata.assert_not_called()
                module.fp8_paged_mqa_logits.assert_not_called()

    def test_paged_runtime_requires_imported_package_provenance(self):
        path = self.paged_bundle(joint=True)
        with self.paged_runtime(path) as (module, _):
            del module.__file__
            with self.assertRaisesRegex(replay.UnsupportedCapture, "package location"):
                replay.replay_capture(path, "cuda")
            module.fp8_paged_mqa_logits.assert_not_called()

    def test_paged_joint_numerical_and_chained_failures_are_not_hidden_by_tolerance(
        self,
    ):
        path = self.paged_bundle(joint=True, lengths=(513,))
        _, tensors = capture.load_capture(path)
        logits = tensors["logits"].clone()
        logits[0, 0] = 1.5  # Changes the selected set by one despite small logit error.
        with self.paged_runtime(path, logits=logits):
            with self.assertRaises(replay.ReplayMismatch) as caught:
                replay.replay_capture(path, "cuda", mqa_atol=2)
        self.assertTrue(caught.exception.summary["mqa_comparison"]["allclose"])
        self.assertFalse(
            caught.exception.summary["chained_topk"]["captured_score_keys_equal"]
        )
        logits = tensors["logits"].clone() + 0.25
        with self.paged_runtime(path, logits=logits):
            with self.assertRaises(replay.ReplayMismatch) as caught:
                replay.replay_capture(path, "cuda")
        self.assertFalse(caught.exception.summary["mqa_comparison"]["allclose"])
        with self.paged_runtime(path, logits=logits):
            self.assertTrue(
                replay.replay_capture(path, "cuda", mqa_atol=0.25)["mqa_comparison"][
                    "allclose"
                ]
            )

    def test_paged_nonfinite_inputs_and_infinite_score_keys(self):
        for name in ("q_fp8", "weights", "payload", "scale"):
            with self.subTest(name=name):
                path = self.paged_bundle(joint=True)
                _, tensors = capture.load_capture(path)
                tensor_name = name if name in tensors else "packed_kv_pages"
                tensor = tensors[tensor_name]
                if name == "payload":
                    tensor[0].view(-1)[:1].view(torch.float8_e4m3fn).fill_(float("nan"))
                elif name == "scale":
                    tensor[0].view(-1)[64 * 128 :].view(torch.float32)[0] = float("inf")
                else:
                    tensor.flatten()[0] = float("nan")
                self.replace_tensor(path, tensor_name, tensor)
                with self.assertRaisesRegex(
                    replay.UnsupportedCapture, "nonfinite paged MQA"
                ):
                    replay.replay_capture(path)
        path = self.paged_bundle(lengths=(513,))
        _, tensors = capture.load_capture(path)
        logits = tensors["logits"]
        logits[0, 0] = -float("inf")
        logits[0, 512] = float("inf")
        self.replace_tensor(path, "logits", logits)
        self.mutate(path, lambda m: m["parameters"].update(stride0=logits.shape[1]))
        replay.replay_capture(path)

    def test_paged_mocked_joint_accepts_tie_equivalence_but_checks_short_order(self):
        path = self.paged_bundle(joint=True, lengths=(513,))
        _, tensors = capture.load_capture(path)
        logits = tensors["logits"].clone()
        logits[0, :513] = 1
        self.replace_tensor(path, "logits", logits)
        self.mutate(path, lambda m: m["parameters"].update(stride0=logits.shape[1]))

        def different_tie(scores, lengths, output, workspace, k, columns):
            output[0] = torch.arange(k)  # Captured indices select 1..512.

        with self.paged_runtime(path, topk=different_tie):
            result = replay.replay_capture(path, "cuda")
        for comparison in ("independent_topk", "chained_topk"):
            self.assertFalse(result[comparison]["selected_sets_equal"])
            self.assertTrue(result[comparison]["captured_score_keys_equal"])
            self.assertTrue(result[comparison]["contract_valid"])

    def test_paged_mocked_native_failures_preserve_error_classification(self):
        path = self.paged_bundle(joint=True)
        for error, expected in (
            (RuntimeError("Unsupported architecture"), replay.UnsupportedCapture),
            (NotImplementedError("paged MQA unavailable"), replay.UnsupportedCapture),
            (RuntimeError("CUDA illegal memory access"), RuntimeError),
        ):
            with self.paged_runtime(path) as (module, _):
                module.fp8_paged_mqa_logits.side_effect = error
                with self.assertRaises(expected) as caught:
                    replay.replay_capture(path, "cuda")
                if expected is RuntimeError:
                    self.assertIs(caught.exception, error)
        with self.paged_runtime(path), mock.patch.object(
            replay.importlib, "import_module", side_effect=ImportError("no deep_gemm")
        ):
            with self.assertRaises(replay.ReplayUnavailable):
                replay.replay_capture(path, "cuda")
        for bad in (
            None,
            torch.zeros(3, 192, dtype=torch.float64),
            torch.zeros(1, 192),
        ):
            with self.paged_runtime(path) as (module, _):
                module.fp8_paged_mqa_logits.return_value = bad
                with self.assertRaises(ValueError):
                    replay.replay_capture(path, "cuda")

    def test_paged_independent_and_chained_topk_contract_failures(self):
        path = self.paged_bundle(joint=True)
        _, tensors = capture.load_capture(path)
        for failing_call in (1, 2):
            calls = []

            def bad_topk(scores, lengths, output, workspace, k, columns):
                calls.append(1)
                output.copy_(tensors["indices"])
                if len(calls) == failing_call:
                    output[0, :3] = torch.tensor([2, 1, 0])

            with self.paged_runtime(path, topk=bad_topk):
                with self.assertRaises(ValueError) as caught:
                    replay.replay_capture(path, "cuda", mqa_atol=1e10)
            if failing_call == 2:
                self.assertIsInstance(caught.exception, replay.ReplayMismatch)
                self.assertFalse(
                    caught.exception.summary["chained_topk"]["contract_valid"]
                )
            else:
                self.assertNotIsInstance(caught.exception, replay.ReplayMismatch)

    def test_fixture_prefers_bazel_directory_and_cleans_up(self):
        names = ("TEST_TMPDIR", "TMPDIR", "RTP_LLM_TEST_LOCAL_LAB")
        directories = [self.root / name for name in names]
        for directory in directories:
            directory.mkdir()
        for index, directory in enumerate(directories + [self.root]):
            environment = {
                name: str(path)
                for name, path in zip(names[index:], directories[index:])
            }
            with self.subTest(environment=environment):
                fixture = _ReplayFixture()
                self.addCleanup(fixture.doCleanups)
                with mock.patch.dict(
                    os.environ, environment, clear=True
                ), mock.patch.object(tempfile, "gettempdir", return_value=str(self.root)):
                    fixture.setUp()
                self.assertEqual(fixture.root.parent, directory)
                self.assertTrue(fixture.root.is_dir())
                fixture.doCleanups()
                self.assertFalse(fixture.root.exists())
                self.assertTrue(directory.is_dir())

    def test_cpu_is_contract_validation_and_does_not_import_ops(self):
        with mock.patch.object(
            replay.importlib, "import_module", side_effect=AssertionError("ops import")
        ), mock.patch.object(
            torch.cuda, "is_available", side_effect=AssertionError("CUDA probe")
        ), mock.patch.object(
            torch, "topk", side_effect=AssertionError("not an oracle")
        ):
            result = replay.replay_capture(self.bundle())
        self.assertFalse(result["kernel_replayed"])
        self.assertEqual(result["mode"], "cpu_contract_validation")
        self.assertEqual(result["counts"]["selected_indices"], 2)

    def test_lengths_zero_short_equal_long_and_nonzero_starts(self):
        for backend, k in (
            ("dsv4_per_row", 1),
            ("dsv4_per_row", 3),
            ("topk_v3_tie_break", 512),
            ("topk_v3_tie_break", 1024),
            ("topk_v3_tie_break", 2048),
        ):
            with self.subTest(backend=backend, k=k):
                result = replay.replay_capture(self.boundary_bundle(backend, k))
                self.assertEqual(result["counts"]["empty_rows"], 1)
                self.assertEqual(result["counts"]["selected_indices"], 1 + 2 * k)
                self.assertEqual(result["counts"]["boundary_tie_rows"], 1)

    def test_k_larger_than_width(self):
        result = replay.replay_capture(
            self.bundle(
                torch.zeros(1, 2),
                torch.tensor([0], dtype=torch.int32),
                torch.tensor([2], dtype=torch.int32),
                torch.tensor([[0, 1, -1, -1]], dtype=torch.int32),
            )
        )
        self.assertEqual(result["counts"]["padding_indices"], 2)

    def test_duplicate_missing_out_of_range_negative_and_suboptimal(self):
        for indices in ([[1, 1]], [[1, -1]], [[1, 3]], [[1, -2]], [[0, 1]]):
            with self.subTest(indices=indices), self.assertRaises(ValueError):
                replay.replay_capture(
                    self.bundle(indices=torch.tensor(indices, dtype=torch.int32))
                )

    def test_padding_and_short_row_order(self):
        for indices in ([[0, -2]], [[-1, 0]], [[0, 0]]):
            with self.subTest(indices=indices), self.assertRaises(ValueError):
                replay.replay_capture(
                    self.bundle(
                        torch.zeros(1, 1),
                        torch.tensor([0], dtype=torch.int32),
                        torch.tensor([1], dtype=torch.int32),
                        torch.tensor(indices, dtype=torch.int32),
                    )
                )
        with self.assertRaisesRegex(ValueError, "sequential"):
            replay.replay_capture(
                self.bundle(
                    torch.zeros(1, 2),
                    torch.tensor([0], dtype=torch.int32),
                    torch.tensor([2], dtype=torch.int32),
                    torch.tensor([[1, 0]], dtype=torch.int32),
                )
            )

    def test_ties_use_smaller_index_not_output_order(self):
        logits = torch.tensor([[9.0, 7.0, 7.0, 7.0, -1.0]])
        for indices, valid in (([1, 0], True), ([0, 2], False), ([1, 2], False)):
            path = self.bundle(
                logits,
                torch.tensor([0], dtype=torch.int32),
                torch.tensor([5], dtype=torch.int32),
                torch.tensor([indices], dtype=torch.int32),
            )
            if valid:
                self.assertEqual(replay.replay_capture(path)["status"], "validated")
            else:
                with self.assertRaisesRegex(ValueError, "maximal candidate"):
                    replay.replay_capture(path)

    def test_dsv4_current_source_ties_not_legacy_arrival_slot_order(self):
        # The pre-stabilization insertion branch used candidate-slot i < j,
        # not token indices. Both current dispatches require smaller indices.
        cases = (
            ([0.0] * 4, [2, 1, 0], [3, 2, 1]),
            ([-7.0] * 4, [2, 1, 0], [3, 2, 1]),
            ([-0.0, 0.0, -0.0, 0.0], [3, 1, 0], [3, 1, 2]),
            ([0.0] * 2049, [2, 1, 0], [2048, 2, 1]),
        )
        for force in (False, True):
            for values, expected, legacy in cases:
                with self.subTest(force=force, length=len(values), prefix=values[:4]):
                    for indices, valid in ((expected, True), (legacy, False)):
                        path = self.bundle(
                            torch.tensor([values]),
                            torch.tensor([0], dtype=torch.int32),
                            torch.tensor([len(values)], dtype=torch.int32),
                            torch.tensor([indices], dtype=torch.int32),
                            force_radix_sort=force,
                        )
                        if valid:
                            replay.replay_capture(path)
                        else:
                            with self.assertRaisesRegex(
                                ValueError, "index tie-break.*provenance"
                            ):
                                replay.replay_capture(path)

    def test_signed_zero_and_infinities(self):
        logits = torch.tensor([[-float("inf"), -0.0, 0.0, float("inf")]])
        for indices, valid in (([2, 3], True), ([1, 3], False), ([0, 3], False)):
            path = self.bundle(
                logits,
                torch.tensor([0], dtype=torch.int32),
                torch.tensor([4], dtype=torch.int32),
                torch.tensor([indices], dtype=torch.int32),
            )
            if valid:
                replay.replay_capture(path)
            else:
                with self.assertRaises(ValueError):
                    replay.replay_capture(path)

    def test_nan_explicitly_unsupported_not_silently_passed(self):
        for bits in (0x7FC00001, -0x003FFFFF):
            logits = torch.tensor([[0.0, 1.0, 0.0]])
            logits.view(torch.int32)[0, 1] = bits
            with self.subTest(bits=bits), self.assertRaisesRegex(
                replay.UnsupportedCapture, "NaN"
            ):
                replay.replay_capture(
                    self.bundle(
                        logits,
                        torch.tensor([1], dtype=torch.int32),
                        torch.tensor([2], dtype=torch.int32),
                        torch.tensor([[0]], dtype=torch.int32),
                    )
                )

    def test_v3_effective_length_clamped_by_saved_max_seq_len(self):
        logits = torch.zeros(1, 520)
        logits[0, 514:] = 100
        output = torch.full((1, 512), -1, dtype=torch.int32)
        output[0, :2] = torch.tensor([0, 1])
        path = self.bundle(
            logits,
            torch.tensor([0], dtype=torch.int32),
            torch.tensor([520], dtype=torch.int32),
            output,
            "topk_v3_tie_break",
            max_seq_len=2,
        )
        self.assertEqual(replay.replay_capture(path)["counts"]["selected_indices"], 2)

    def test_bounds_and_invalid_domain_zero_policy(self):
        for starts, ends in (([-1], [4]), ([3], [2]), ([0], [6])):
            with self.subTest(starts=starts, ends=ends), self.assertRaisesRegex(
                ValueError, "bounds"
            ):
                replay.replay_capture(
                    self.bundle(
                        starts=torch.tensor(starts, dtype=torch.int32),
                        ends=torch.tensor(ends, dtype=torch.int32),
                    )
                )
        for invalid in (123.0, float("nan")):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(
                ValueError, "zero-filled"
            ):
                replay.replay_capture(
                    self.bundle(logits=torch.tensor([[invalid, -3.0, -1.0, -2.0, 0.0]]))
                )

    def test_tensor_types_shapes_and_missing_extra_tensors(self):
        cases = (
            dict(logits=torch.zeros(1, 5, dtype=torch.float16)),
            dict(starts=torch.tensor([1], dtype=torch.int64)),
            dict(ends=torch.tensor([[4]], dtype=torch.int32)),
            dict(indices=torch.tensor([[2, 1]], dtype=torch.int64)),
        )
        for kwargs in cases:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                replay.replay_capture(self.bundle(**kwargs))
        for change in (
            lambda m: m["tensors"].pop("row_starts"),
            lambda m: m["tensors"]["indices"].update(shape=[2, 1]),
        ):
            path = self.bundle()
            self.mutate(path, change)
            with self.assertRaises(ValueError):
                replay.replay_capture(path)

    def test_metadata_tampering(self):
        mutations = [
            lambda m: m.update(op="other"),
            lambda m: m.update(phase="decode"),
            lambda m: m.update(schema_version=2),
            lambda m: m.update(complete=False),
            lambda m: m.update(query_row_count=2),
            lambda m: m.update(query_row_start=1),
            lambda m: m.update(finite_values_checked=True),
            lambda m: m.update(nonfinite_policy="ignore"),
            lambda m: m.update(output_contract="sorted"),
            lambda m: m["parameters"].update(stride0=7),
            lambda m: m["parameters"].update(num_rows=2),
            lambda m: m["parameters"].update(force_radix_sort=1),
            lambda m: m["parameters"].update(k=True),
            lambda m: m["parameters"].update(k=0),
            lambda m: m["parameters"].update(index_space="absolute"),
            lambda m: m["parameters"].update(extra=1),
            lambda m: m["parameters"].pop("stride1"),
            lambda m: m["tensors"]["logits"].update(nbytes=1),
        ]
        for index, change in enumerate(mutations):
            with self.subTest(index=index):
                path = self.bundle()
                self.mutate(path, change)
                with self.assertRaises(ValueError):
                    replay.replay_capture(path)

    def test_checksum_tampering(self):
        path = self.bundle()
        binary = path.parent / "indices.bin"
        binary.write_bytes(bytes(len(binary.read_bytes())))
        with self.assertRaisesRegex(ValueError, "checksum"):
            replay.replay_capture(path)

    def test_unsupported_backend_and_v3_k(self):
        for backend in ("torch", "unknown", "topk_v3_tie_break"):
            with self.subTest(backend=backend), self.assertRaises(
                replay.UnsupportedCapture
            ):
                replay.replay_capture(self.bundle(backend=backend))

    def test_layout_restoration_and_rejection(self):
        path = self.boundary_bundle("dsv4_per_row", 3)
        manifest, tensors = capture.load_capture(path)
        desc = manifest["tensors"]["logits"]
        restored = replay._restore(tensors["logits"], desc, "cpu")
        self.assertEqual(restored.stride(), tuple(desc["original_stride"]))
        self.assertEqual(restored.storage_offset(), 3)
        self.assertTrue(torch.equal(restored, tensors["logits"]))
        for name, change in (
            ("logits", dict(original_stride=[1, 0])),
            ("indices", dict(original_stride=[6, 2])),
            ("logits", dict(storage_offset=2**40)),
        ):
            with self.subTest(name=name, change=change):
                path = self.bundle()

                def mutate(m):
                    m["tensors"][name].update(change)
                    if name == "logits" and "original_stride" in change:
                        m["parameters"].update(stride0=1, stride1=0)

                self.mutate(path, mutate)
                with self.assertRaises(replay.UnsupportedCapture):
                    replay.replay_capture(path)

    def test_cuda_dispatch_wiring_on_cpu_not_a_kernel_test(self):
        # Mock only device allocation/dispatch to check orchestration without
        # claiming to execute CUDA. Real kernels have a separate opt-in class.
        restore = replay._restore
        for backend, k in (("dsv4_per_row", 3), ("topk_v3_tie_break", 512)):
            for behavior in ("write", "unwritten", "failure"):
                with self.subTest(backend=backend, behavior=behavior):
                    path = self.boundary_bundle(backend, k, force=True)
                    manifest, tensors = capture.load_capture(path)

                    def operator(logits, starts, ends, output, *params):
                        self.assertEqual(
                            logits.stride(),
                            tuple(manifest["tensors"]["logits"]["original_stride"]),
                        )
                        self.assertEqual(logits.storage_offset(), 3)
                        self.assertTrue(bool((output == -2).all()))
                        saved = manifest["parameters"]
                        expected = (
                            (4, saved["stride0"], saved["stride1"], k, True)
                            if backend == "dsv4_per_row"
                            else (k, saved["max_seq_len"])
                        )
                        self.assertEqual(params, expected)
                        if behavior == "failure":
                            raise RuntimeError("synthetic launch error")
                        if behavior == "write":
                            output.copy_(tensors["indices"])
                            output[3] = output[3].flip(0)

                    with mock.patch.object(
                        replay, "_cuda_operator", return_value=operator
                    ), mock.patch.object(
                        replay,
                        "_restore",
                        side_effect=lambda t, d, device: restore(t, d, "cpu"),
                    ), mock.patch.object(
                        torch.cuda, "synchronize"
                    ):
                        if behavior == "write":
                            result = replay.replay_capture(path, "cuda")
                            self.assertFalse(result["raw_order_equal"])
                        elif behavior == "unwritten":
                            with self.assertRaisesRegex(
                                ValueError, "row 0: invalid -1 padding"
                            ):
                                replay.replay_capture(path, "cuda")
                        else:
                            with self.assertRaisesRegex(
                                RuntimeError, "synthetic launch"
                            ):
                                replay.replay_capture(path, "cuda")

    def test_v3_invalid_parameters_and_layout(self):
        for value in (0, -1, 521, True, 1.5):
            path = self.boundary_bundle("topk_v3_tie_break", 512)
            self.mutate(path, lambda m: m["parameters"].update(max_seq_len=value))
            with self.subTest(value=value), self.assertRaisesRegex(
                ValueError, "max_seq_len"
            ):
                replay.replay_capture(path)
        path = self.boundary_bundle("topk_v3_tie_break", 512)

        def mutate(m):
            m["parameters"].update(stride0=1048, stride1=2)
            m["tensors"]["logits"]["original_stride"] = [1048, 2]

        self.mutate(path, mutate)
        with self.assertRaisesRegex(replay.UnsupportedCapture, "stride1"):
            replay.replay_capture(path)

    def test_cli_summary_and_unavailable_exit_codes(self):
        path = self.bundle()
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(replay.main([str(path)]), 0)
        self.assertFalse(json.loads(output.getvalue())["kernel_replayed"])
        with mock.patch.object(
            torch.cuda, "is_available", return_value=False
        ), contextlib.redirect_stderr(io.StringIO()) as error:
            self.assertEqual(replay.main([str(path), "--device", "cuda"]), 3)
        self.assertEqual(json.loads(error.getvalue())["status"], "unavailable")
        with mock.patch.object(
            torch.cuda, "is_available", return_value=True
        ), mock.patch.object(torch.version, "cuda", "test"), mock.patch.object(
            replay.importlib,
            "import_module",
            side_effect=ImportError("missing library"),
        ), self.assertRaisesRegex(
            replay.ReplayUnavailable, "operator unavailable"
        ):
            replay.replay_capture(path, "cuda")
        self.mutate(path, lambda m: m.update(op="other"))
        with contextlib.redirect_stderr(io.StringIO()) as error:
            self.assertEqual(replay.main([str(path)]), 1)
        self.assertEqual(json.loads(error.getvalue())["status"], "failed")
        self.mutate(path, lambda m: m.update(op="indexer.topk", backend="unknown"))
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(replay.main([str(path)]), 2)

    def test_mqa_real_writer_cpu_roundtrip_and_layout(self):
        for dimension in (128, 256):
            with self.subTest(dimension=dimension):
                path = self.mqa_bundle(dimension)
                manifest, tensors = capture.load_capture(path)
                self.assertEqual(manifest["schema_version"], 2)
                self.assertEqual(
                    set(tensors),
                    {
                        "q_fp8",
                        "k_fp8",
                        "k_scale",
                        "weights",
                        "logits",
                        "indices",
                        "row_starts",
                        "row_ends",
                    },
                )
                with mock.patch.object(
                    replay.importlib,
                    "import_module",
                    side_effect=AssertionError("native import"),
                ), mock.patch.object(
                    torch.cuda, "is_available", side_effect=AssertionError("CUDA probe")
                ):
                    result = replay.replay_capture(path)
                self.assertFalse(result["mqa_kernel_replayed"])
                self.assertFalse(result["kernel_replayed"])
                self.assertEqual(result["mode"], "cpu_contract_validation")
                self.assertEqual(float(tensors["logits"][0, 3]), 0.0)
                for name in ("q_fp8", "k_fp8"):
                    desc = manifest["tensors"][name]
                    restored = replay._restore(tensors[name], desc, "cpu")
                    self.assertEqual(restored.stride(), tuple(desc["original_stride"]))
                    self.assertGreater(restored.storage_offset(), 0)
                    self.assertTrue(
                        torch.equal(restored.float(), tensors[name].float())
                    )

    def test_mqa_metadata_and_schema_rejections(self):
        mutations = (
            lambda m: m.update(schema_version=1),
            lambda m: m.update(op="indexer.topk"),
            lambda m: m.pop("mqa_parameters"),
            lambda m: m["operations"][0].update(outputs=["indices"]),
            lambda m: m["mqa_parameters"].update(clean_logits=True),
            lambda m: m["mqa_parameters"].update(weights_include_query_scale=False),
            lambda m: m["mqa_parameters"].update(weights_include_query_scale=1),
            lambda m: m["mqa_parameters"].update(key_scale_format="ue8m0"),
            lambda m: m["mqa_parameters"].update(q_scale=2),
            lambda m: m["tensors"].pop("weights"),
        )
        for change in mutations:
            path = self.mqa_bundle()
            self.mutate(path, change)
            with self.assertRaises(ValueError):
                replay.replay_capture(path)
        path = self.mqa_bundle()
        self.mutate(path, lambda m: m["mqa_parameters"].update(backend="unknown"))
        with self.assertRaisesRegex(
            replay.ReplayUnavailable, "unsupported MQA backend"
        ):
            replay.replay_capture(path)

    def test_mqa_tensor_validation(self):
        for name, tensor in (
            ("q_fp8", torch.ones(2, 2, 128)),
            ("q_fp8", torch.ones(2, 2, 64).to(torch.float8_e4m3fn)),
            ("k_fp8", torch.ones(4, 128).to(torch.float8_e5m2)),
            ("k_fp8", torch.ones(3, 128).to(torch.float8_e4m3fn)),
            ("k_scale", torch.ones(4, 1)),
            ("k_scale", torch.ones(4, dtype=torch.float16)),
            ("k_scale", torch.full((4,), float("nan"))),
            ("weights", torch.ones(2, 3)),
            ("weights", torch.full((2, 2), float("inf"))),
            ("q_fp8", torch.full((2, 2, 128), float("nan")).to(torch.float8_e4m3fn)),
        ):
            with self.subTest(name=name, shape=tensor.shape, dtype=tensor.dtype):
                path = self.mqa_bundle()
                self.replace_tensor(path, name, tensor)
                with self.assertRaises(ValueError):
                    replay.replay_capture(path)
        path = self.mqa_bundle(256)
        self.replace_tensor(path, "k_scale", torch.ones(4))
        with self.assertRaisesRegex(ValueError, "k_scale"):
            replay.replay_capture(path)

    def test_mqa_cuda_orchestration_mocked_on_cpu_not_kernel_validation(self):
        for scenario in (
            "equal",
            "close_changed_set",
            "drift_same_set",
            "nan",
            "wrong_shape",
            "wrong_dtype",
            "bad_chain",
            "library_missing",
            "unsupported_device",
        ):
            with self.subTest(scenario=scenario):
                path = self.mqa_bundle()
                manifest, tensors = capture.load_capture(path)
                result_logits = tensors["logits"].clone()
                result_logits[0, 3] = float(
                    "nan"
                )  # Unspecified outside ks/ke: never compare or clean it.
                if scenario == "close_changed_set":
                    result_logits[1, 2] = 3.1
                elif scenario == "drift_same_set":
                    result_logits[1, 1] += 0.5
                elif scenario == "nan":
                    result_logits[0, 0] = float("nan")
                elif scenario == "wrong_shape":
                    result_logits = result_logits[:, :3]
                elif scenario == "wrong_dtype":
                    result_logits = result_logits.half()
                topk_calls = []

                def topk(logits, starts, ends, output, *params):
                    topk_calls.append(logits.clone())
                    self.assertEqual(params, (2, 4, 1, 2, True))
                    self.assertTrue(bool((output == -2).all()))
                    output.copy_(tensors["indices"])
                    if len(topk_calls) == 2 and scenario == "close_changed_set":
                        output[1] = torch.tensor([1, 2])
                    if len(topk_calls) == 2 and scenario == "bad_chain":
                        output[1] = torch.tensor([1, 1])

                def mqa(q, kv, weights, starts, ends, *, clean_logits):
                    if scenario == "unsupported_device":
                        raise NotImplementedError("synthetic unsupported architecture")
                    self.assertFalse(clean_logits)
                    self.assertTrue(torch.equal(q.float(), tensors["q_fp8"].float()))
                    self.assertEqual(
                        q.stride(),
                        tuple(manifest["tensors"]["q_fp8"]["original_stride"]),
                    )
                    self.assertTrue(
                        torch.equal(kv[0].float(), tensors["k_fp8"].float())
                    )
                    self.assertTrue(torch.equal(kv[1], tensors["k_scale"]))
                    self.assertTrue(bool((kv[1] == 2).all()))
                    self.assertTrue(torch.equal(weights, tensors["weights"]))
                    self.assertTrue(torch.equal(starts, tensors["row_starts"]))
                    self.assertTrue(torch.equal(ends, tensors["row_ends"]))
                    return result_logits

                mock_mqa = mock.Mock(side_effect=mqa)
                module = mock.Mock(fp8_mqa_logits=mock_mqa)
                restore = replay._restore
                with mock.patch.object(
                    replay, "_cuda_operator", return_value=topk
                ), mock.patch.object(
                    replay,
                    "_restore",
                    side_effect=lambda t, d, device: restore(t, d, "cpu"),
                ), mock.patch.object(
                    torch.cuda, "synchronize"
                ), mock.patch.object(
                    replay.importlib,
                    "import_module",
                    return_value=module,
                    side_effect=(
                        ImportError("missing deep_gemm")
                        if scenario == "library_missing"
                        else None
                    ),
                ) as importer:
                    if scenario == "equal":
                        result = replay.replay_capture(path, "cuda")
                        self.assertEqual(result["mqa_comparison"]["max_abs"], 0)
                        self.assertEqual(result["mqa_comparison"]["valid_values"], 7)
                        self.assertTrue(result["chained_topk"]["selected_sets_equal"])
                    elif scenario in (
                        "close_changed_set",
                        "drift_same_set",
                        "bad_chain",
                    ):
                        with self.assertRaises(replay.ReplayMismatch) as error:
                            replay.replay_capture(
                                path,
                                "cuda",
                                mqa_atol=2 if scenario == "close_changed_set" else 0,
                            )
                        result = error.exception.summary
                        self.assertEqual(result["status"], "failed")
                        self.assertGreaterEqual(
                            result["chained_topk"]["min_captured_score_gap"], 0
                        )
                        if scenario == "close_changed_set":
                            self.assertTrue(result["mqa_comparison"]["allclose"])
                            self.assertEqual(
                                result["chained_topk"]["differing_rows"], 1
                            )
                            self.assertEqual(
                                result["chained_topk"]["symmetric_difference_count"], 2
                            )
                        elif scenario == "drift_same_set":
                            self.assertFalse(result["mqa_comparison"]["allclose"])
                            self.assertTrue(
                                result["chained_topk"]["selected_sets_equal"]
                            )
                        else:
                            self.assertFalse(result["chained_topk"]["contract_valid"])
                    elif scenario == "library_missing":
                        with self.assertRaisesRegex(
                            replay.ReplayUnavailable, "DeepGEMM"
                        ):
                            replay.replay_capture(path, "cuda")
                    else:
                        with self.assertRaises(ValueError):
                            replay.replay_capture(path, "cuda")
                importer.assert_called_once_with("deep_gemm")
                self.assertEqual(
                    mock_mqa.call_count, 0 if scenario == "library_missing" else 1
                )
                self.assertTrue(torch.equal(topk_calls[0], tensors["logits"]))
                if len(topk_calls) == 2:
                    self.assertTrue(bool(torch.isnan(topk_calls[1][0, 3])))
                    self.assertTrue(torch.equal(topk_calls[1][1], result_logits[1]))

    def test_mqa_native_runtime_error_classification_and_cli(self):
        path = self.mqa_bundle()
        _, tensors = capture.load_capture(path)
        cases = (
            ("Unsupported architecture", True),
            ("Unsupported architecture: sm_120", True),
            (
                "Assertion error (csrc/apis/attention.hpp:177): Unsupported architecture",
                True,
            ),
            (
                "Assertion error (csrc/apis/attention.hpp:123): false: Unsupported architecture\nC++ trace (most recent call first):",
                True,
            ),
            ("Assertion error (csrc/apis/attention.hpp:123): arch_major == 10", True),
            (
                "Assertion error (csrc/apis/attention.hpp:123): jit->device.get_arch_major() == 9",
                True,
            ),
            (
                "Assertion error (csrc/apis/attention.hpp:123): jit->device.get_arch_major() == 10",
                True,
            ),
            ("CUDA out of memory. Tried to allocate 2.00 GiB", False),
            ("CUDA error: unspecified launch failure", False),
            (
                "Assertion error (deep_jit/cache/disk.hpp:104): not home.empty(): HOME environment variable must not be empty",
                False,
            ),
            ("Unsupported architecture or scaling factor types", False),
            (
                "Assertion error (csrc/apis/attention.hpp:123): arch_major == 10 and next_n == 1",
                False,
            ),
            ("CUDA launch failed\nC++ trace: Unsupported architecture", False),
        )
        restore = replay._restore
        for message, unsupported in cases:
            with self.subTest(message=message):
                native_error = RuntimeError(message)
                mqa = mock.Mock(side_effect=native_error)
                expected = replay.UnsupportedCapture if unsupported else RuntimeError
                with self.assertRaises(expected) as raised:
                    replay._call_mqa(mqa, tensors)
                if unsupported:
                    self.assertIs(raised.exception.__cause__, native_error)
                    for name in ("q_fp8", "k_fp8", "k_scale", "weights"):
                        self.assertIn(name + "_shape=", str(raised.exception))
                        self.assertIn(name + "_stride=", str(raised.exception))
                else:
                    self.assertIs(raised.exception, native_error)

                def topk(logits, starts, ends, output, *params):
                    output.copy_(tensors["indices"])

                with mock.patch.object(
                    replay, "_cuda_operator", return_value=topk
                ), mock.patch.object(
                    replay, "_mqa_operator", return_value=mqa
                ), mock.patch.object(
                    replay,
                    "_restore",
                    side_effect=lambda t, d, device: restore(t, d, "cpu"),
                ), mock.patch.object(
                    torch.cuda, "synchronize"
                ), contextlib.redirect_stderr(
                    io.StringIO()
                ) as output:
                    self.assertEqual(
                        replay.main([str(path), "--device", "cuda"]),
                        2 if unsupported else 1,
                    )
                self.assertEqual(
                    json.loads(output.getvalue())["status"],
                    "unsupported" if unsupported else "failed",
                )

    def test_mqa_writer_compute_failure_preserves_root_cause(self):
        _, tensors = capture.load_capture(self.mqa_bundle())
        compute = mock.Mock(side_effect=RuntimeError("original native failure"))
        with self.assertRaisesRegex(RuntimeError, "original native failure"):
            self.write_mqa_bundle(tensors, compute)
        compute.assert_called_once()

    def test_mqa_missing_exports_and_unknown_topk_are_unavailable(self):
        path = self.mqa_bundle()
        self.mutate(path, lambda m: m.update(backend="unknown"))
        with self.assertRaisesRegex(replay.ReplayUnavailable, "unsupported backend"):
            replay.replay_capture(path)
        for module in (object(), mock.Mock(fp8_mqa_logits=None)):
            with mock.patch.object(
                replay.importlib, "import_module", return_value=module
            ), self.assertRaisesRegex(replay.ReplayUnavailable, "DeepGEMM"):
                replay._mqa_operator()

    def test_mqa_zero_scale_has_no_extra_reader_restriction(self):
        path = self.mqa_bundle()
        self.replace_tensor(path, "k_scale", torch.zeros(4))
        # CPU validates the recorded TopK, not the mathematical MQA relation.
        self.assertFalse(replay.replay_capture(path)["mqa_kernel_replayed"])

    def test_mqa_valid_domain_error_metrics_and_relative_tolerance(self):
        tensors = dict(
            logits=torch.tensor([[0.0, 2.0, float("inf"), 999.0]]),
            row_starts=torch.tensor([0]),
            row_ends=torch.tensor([3]),
        )
        actual = torch.tensor([[0.0, 2.25, float("inf"), float("nan")]])
        measured = replay._compare_mqa(tensors, actual, atol=0, rtol=0.125)
        self.assertTrue(measured["allclose"])
        self.assertEqual(measured["max_abs"], 0.25)
        self.assertEqual(measured["max_rel"], 0.125)
        self.assertEqual(measured["valid_values"], 3)
        actual[0, 0] = 0.1
        measured = replay._compare_mqa(tensors, actual, atol=0, rtol=0.125)
        self.assertFalse(measured["allclose"])
        self.assertEqual(measured["max_rel"], "inf")
        json.dumps(measured, allow_nan=False)
        tensors["row_starts"][0] = 3
        measured = replay._compare_mqa(tensors, actual, atol=0, rtol=0)
        self.assertEqual(measured["valid_values"], 0)
        self.assertEqual(measured["max_abs"], 0)

    def test_mqa_tolerances_and_cli_diagnostics(self):
        path = self.mqa_bundle()
        for value in (-1, float("nan"), float("inf"), True):
            for name in ("mqa_atol", "mqa_rtol"):
                with self.subTest(name=name, value=value), self.assertRaisesRegex(
                    ValueError, "tolerances"
                ):
                    replay.replay_capture(path, **{name: value})
        summary = dict(
            status="failed",
            mqa_comparison=dict(max_abs=0.1),
            chained_topk=dict(differing_rows=1),
        )
        with mock.patch.object(
            replay, "replay_capture", side_effect=replay.ReplayMismatch(summary)
        ) as run, contextlib.redirect_stderr(io.StringIO()) as output:
            self.assertEqual(
                replay.main(
                    [
                        str(path),
                        "--device",
                        "cuda",
                        "--mqa-atol",
                        "0.1",
                        "--mqa-rtol",
                        "0.02",
                    ]
                ),
                1,
            )
        self.assertEqual(json.loads(output.getvalue()), summary)
        run.assert_called_once_with(str(path), "cuda", mqa_atol=0.1, mqa_rtol=0.02)

    def test_indexed_real_writer_loader_cpu_roundtrip(self):
        for joint in (False, True):
            for role in (None, "PREFILL", "PDFUSION"):
                with self.subTest(joint=joint, role=role):
                    path = self.indexed_bundle(joint=joint, server_role=role)
                    manifest, tensors = capture.load_capture(path)
                    self.assertEqual(manifest["schema_version"], 3)
                    self.assertEqual(
                        manifest["contract"], dict(name=manifest["op"], version=2)
                    )
                    self.assertNotIn("query_row_start", manifest)
                    self.assertEqual(tensors["query_global_ids"].tolist(), [4, 0, 3])
                    self.assertEqual(tensors["query_global_ids"].dtype, torch.int64)
                    self.assertEqual(manifest["requests"][0]["q_len"], 3)
                    self.assertEqual(
                        manifest["tensors"]["logits"]["original_stride"], [11, 1]
                    )
                    self.assertEqual(
                        manifest["tensors"]["logits"]["storage_offset"], 11
                    )
                    self.assertEqual(
                        set(manifest["outputs"]),
                        {"logits", "indices"} if joint else {"indices"},
                    )
                    if role is None:
                        self.assertNotIn("server_role", manifest)
                    else:
                        self.assertEqual(manifest["server_role"], role)
                    with mock.patch.object(
                        replay,
                        "_cuda_operator",
                        side_effect=AssertionError("native TopK"),
                    ), mock.patch.object(
                        replay,
                        "_mqa_operator",
                        side_effect=AssertionError("native MQA"),
                    ), mock.patch.object(
                        torch.cuda,
                        "is_available",
                        side_effect=AssertionError("CUDA probe"),
                    ):
                        result = replay.replay_capture(path)
                    self.assertEqual(result["counts"]["rows"], 3)
                    self.assertEqual(result["schema_version"], 3)
                    self.assertFalse(result["kernel_replayed"])
                    if joint:
                        self.assertFalse(result["mqa_kernel_replayed"])

    def test_indexed_ids_rehashed_tampering_rejected(self):
        cases = [
            torch.tensor([4, 4, 3]),
            torch.tensor([-1, 0, 3]),
            torch.tensor([5, 0, 3]),
            torch.tensor([3, 0, 4]),
            torch.tensor([4, 0, 3], dtype=torch.int32),
            torch.tensor([[4, 0, 3]]),
            torch.tensor([4, 0]),
            torch.tensor([], dtype=torch.int64),
        ]
        for joint in (False, True):
            for ids in cases:
                with self.subTest(joint=joint, ids=ids):
                    path = self.indexed_bundle(joint=joint)
                    self.replace_tensor(path, "query_global_ids", ids)
                    # Valid descriptor/checksum is insufficient to prove semantics.
                    capture.load_capture(path)
                    with self.assertRaises(ValueError):
                        replay.replay_capture(path)

    def test_indexed_mapping_metadata_rejections(self):
        cases = [
            ("query_row_start", 0),
            ("query_row_count", 2),
            ("total_query_rows", 6),
            (
                "query_mapping",
                dict(
                    kind="contiguous",
                    tensor="query_global_ids",
                    index_space="request_order_query_rows",
                ),
            ),
            (
                "query_mapping",
                dict(
                    kind="indexed",
                    tensor="row_starts",
                    index_space="request_order_query_rows",
                ),
            ),
            (
                "query_mapping",
                dict(
                    kind="indexed",
                    tensor="query_global_ids",
                    index_space="local_query_rows",
                ),
            ),
            ("key_domain", dict(layout="shard_order", coverage="full")),
            (
                "key_domain",
                dict(layout="request_order_prefix_then_query", coverage="local"),
            ),
            ("execution_layout", dict(cp_rank=-1, cp_size=2, kv_cache_sharded=True)),
            ("execution_layout", dict(cp_rank=2, cp_size=2, kv_cache_sharded=True)),
            ("execution_layout", dict(cp_rank=0, cp_size=0, kv_cache_sharded=False)),
            ("execution_layout", dict(cp_rank=0, cp_size=True, kv_cache_sharded=False)),
            ("execution_layout", dict(cp_rank=0, cp_size=2)),
            (
                "execution_layout",
                dict(cp_rank=0, cp_size=2, kv_cache_sharded=True, extra=0),
            ),
            ("execution_layout", dict(cp_rank=False, cp_size=2, kv_cache_sharded=True)),
            ("execution_layout", dict(cp_rank=0, cp_size=2, kv_cache_sharded=1)),
            (
                "capture_coverage",
                dict(scope="global", local_query_rows=4, local_query_row_start=1),
            ),
            (
                "capture_coverage",
                dict(
                    scope="record_owner_local",
                    local_query_rows=3,
                    local_query_row_start=1,
                ),
            ),
            (
                "capture_coverage",
                dict(
                    scope="record_owner_local",
                    local_query_rows=4,
                    local_query_row_start=-1,
                ),
            ),
            (
                "capture_coverage",
                dict(
                    scope="record_owner_local",
                    local_query_rows=True,
                    local_query_row_start=0,
                ),
            ),
            (
                "capture_coverage",
                dict(
                    scope="record_owner_local",
                    local_query_rows=6,
                    local_query_row_start=0,
                ),
            ),
            (
                "capture_coverage",
                dict(
                    scope="record_owner_local",
                    local_query_rows=4,
                    local_query_row_start=False,
                ),
            ),
            ("capture_coverage", dict(scope="record_owner_local", local_query_rows=4)),
        ]
        for key, value in cases:
            with self.subTest(field=key, value=value):
                path = self.indexed_bundle()
                self.mutate(path, lambda m: m.update({key: value}))
                with self.assertRaises(ValueError):
                    replay.replay_capture(path)
        for key in (
            "query_mapping",
            "key_domain",
            "execution_layout",
            "capture_coverage",
            "query_row_count",
            "total_query_rows",
            "requests",
        ):
            with self.subTest(missing=key):
                path = self.indexed_bundle()
                self.mutate(path, lambda m: m.pop(key))
                with self.assertRaises(ValueError):
                    replay.replay_capture(path)

    def test_indexed_frozen_requests_and_ranges_rejections(self):
        for field, value in (
            ("q_len", 4),
            ("kv_len", 3),
            ("q_len", -1),
            ("kv_len", True),
        ):
            with self.subTest(field=field, value=value):
                path = self.indexed_bundle()
                self.mutate(path, lambda m: m["requests"][0].update({field: value}))
                with self.assertRaises(ValueError):
                    replay.replay_capture(path)
        for name, values in (("row_starts", [4, 0, 5]), ("row_ends", [7, 3, 7])):
            with self.subTest(tensor=name):
                path = self.indexed_bundle()
                self.replace_tensor(path, name, torch.tensor(values, dtype=torch.int32))
                with self.assertRaises(ValueError):
                    replay.replay_capture(path)

    def test_indexed_requires_complete_global_key_domain(self):
        # The selected rows need only the first request, but saved keys must still
        # include the second request. Neither missing nor extra keys are legal.
        for joint in (False, True):
            for columns in (5, 9):
                with self.subTest(joint=joint, columns=columns):
                    path = self.indexed_bundle(joint=joint, ids=(0, 2))
                    _, tensors = capture.load_capture(path)
                    logits = torch.zeros(2, columns)
                    logits[:, : min(columns, 8)] = tensors["logits"][
                        :, : min(columns, 8)
                    ]
                    self.replace_tensor(path, "logits", logits)
                    self.mutate(
                        path,
                        lambda m: m["parameters"].update(stride0=columns, stride1=1),
                    )
                    if joint:
                        self.replace_tensor(
                            path,
                            "k_fp8",
                            torch.ones(columns, 128).to(torch.float8_e4m3fn),
                        )
                        self.replace_tensor(path, "k_scale", torch.ones(columns))
                    capture.load_capture(path)
                    with self.assertRaisesRegex(ValueError, "key domain"):
                        replay.replay_capture(path)
        path = self.indexed_bundle(ids=(0,))
        # Changing an unselected request's prefix must also invalidate N.
        self.mutate(path, lambda m: m["requests"][1].update(kv_len=2))
        with self.assertRaisesRegex(ValueError, "key domain"):
            replay.replay_capture(path)

    def test_indexed_exact_interface_and_fixed_operations(self):
        for joint in (False, True):
            with self.subTest(joint=joint):
                path = self.indexed_bundle(joint=joint)

                def move_id_to_output(m):
                    m["inputs"].remove("query_global_ids")
                    m["outputs"].append("query_global_ids")

                self.mutate(path, move_id_to_output)
                capture.load_capture(path)
                with self.assertRaisesRegex(ValueError, "interface"):
                    replay.replay_capture(path)
        path = self.indexed_bundle(joint=True)
        self.mutate(
            path, lambda m: m["operations"][0]["inputs"].append("query_global_ids")
        )
        with self.assertRaisesRegex(ValueError, "operation wiring"):
            replay.replay_capture(path)
        path = self.indexed_bundle(joint=True)
        self.mutate(
            path,
            lambda m: m["outputs"].remove("logits") or m["inputs"].append("logits"),
        )
        with self.assertRaisesRegex(ValueError, "interface"):
            replay.replay_capture(path)

    def test_indexed_backend_layout_finite_and_math_validation(self):
        for joint in (False, True):
            with self.subTest(joint=joint):
                path = self.indexed_bundle(joint=joint)
                self.mutate(path, lambda m: m.update(backend="unknown"))
                with self.assertRaises(
                    (replay.UnsupportedCapture, replay.ReplayUnavailable)
                ):
                    replay.replay_capture(path)
                path = self.indexed_bundle(joint=joint)
                self.replace_tensor(
                    path,
                    "indices",
                    torch.tensor([[1, 2], [0, 1], [0, 1]], dtype=torch.int32),
                )
                with self.assertRaisesRegex(ValueError, "tie-break"):
                    replay.replay_capture(path)
                path = self.indexed_bundle(joint=joint)
                self.mutate(
                    path,
                    lambda m: m["tensors"]["row_starts"].update(original_stride=[2]),
                )
                with self.assertRaisesRegex(replay.UnsupportedCapture, "contiguous"):
                    replay.replay_capture(path)
        for name in ("q_fp8", "k_fp8", "k_scale", "weights", "logits"):
            with self.subTest(nonfinite=name):
                path = self.indexed_bundle(joint=True)
                _, tensors = capture.load_capture(path)
                tensor = tensors[name].float()
                tensor.reshape(-1)[5 if name == "logits" else 0] = float("nan")
                self.replace_tensor(path, name, tensor.to(tensors[name].dtype))
                if name == "logits":
                    self.mutate(
                        path, lambda m: m["parameters"].update(stride0=8, stride1=1)
                    )
                with self.assertRaises(replay.UnsupportedCapture):
                    replay.replay_capture(path)

    def test_indexed_contract_metadata_not_legacy_disguised(self):
        changes = [
            dict(phase="decode"),
            dict(finite_values_checked=True),
            dict(invalid_logits="preserved"),
            dict(nonfinite_policy="finite"),
            dict(output_contract="absolute_indices"),
            dict(query_row_count=True),
            dict(total_query_rows=5.0),
            dict(contract=dict(name="indexer.topk", version=1)),
            dict(contract=dict(name="indexer.topk", version=3)),
            dict(schema_version=1),
        ]
        for fields in changes:
            with self.subTest(fields=fields):
                path = self.indexed_bundle()
                self.mutate(path, lambda m: m.update(fields))
                with self.assertRaises(ValueError):
                    replay.replay_capture(path)
        for field, value in (
            ("clean_logits", True),
            ("weights_include_query_scale", False),
            ("key_scale_format", "per_key"),
        ):
            with self.subTest(mqa_parameter=field):
                path = self.indexed_bundle(joint=True)
                self.mutate(path, lambda m: m["mqa_parameters"].update({field: value}))
                with self.assertRaises(ValueError):
                    replay.replay_capture(path)

    def test_indexed_single_cp_and_single_row_boundaries(self):
        for ids in ((0,), (4,), (4, 0, 3, 1, 2)):
            with self.subTest(ids=ids):
                path = self.indexed_bundle(ids=ids, cp_size=1)
                # Full local coverage starts at zero, including the final global ID.
                self.mutate(
                    path,
                    lambda m: m.update(
                        capture_coverage=dict(
                            scope="record_owner_local",
                            local_query_rows=len(ids),
                            local_query_row_start=0,
                        )
                    ),
                )
                self.assertEqual(
                    replay.replay_capture(path)["counts"]["rows"], len(ids)
                )

    def test_server_role_cannot_disguise_decode_as_prefill(self):
        for make in (
            self.bundle,
            self.mqa_bundle,
            self.indexed_bundle,
            lambda: self.indexed_bundle(joint=True),
        ):
            for role in ("DECODE", "prefill", "unknown", None, 1):
                with self.subTest(factory=make, role=role):
                    path = make()
                    self.mutate(
                        path, lambda m: m.update(server_role=role, phase="prefill")
                    )
                    with self.assertRaisesRegex(ValueError, "server_role"):
                        replay.replay_capture(path)

    def test_indexed_cuda_dispatch_on_cpu_excludes_ids_and_collectives(self):
        for joint in (False, True):
            with self.subTest(joint=joint):
                path = self.indexed_bundle(joint=joint)
                manifest, tensors = capture.load_capture(path)
                restored_names = []

                def restore(tensor, desc, device):
                    self.assertEqual(device, "cuda")
                    restored_names.append(desc["file"])
                    return tensor.clone()

                def topk(operator, actual_manifest, restored):
                    self.assertEqual(actual_manifest["schema_version"], 3)
                    self.assertNotIn("query_global_ids", restored)
                    return tensors["indices"].clone()

                def mqa(q, key, weights, starts, ends, *, clean_logits):
                    self.assertFalse(clean_logits)
                    self.assertEqual(q.shape, (3, 32, 128))
                    self.assertEqual(key[0].shape, (8, 128))
                    self.assertTrue(torch.equal(starts, tensors["row_starts"]))
                    self.assertTrue(torch.equal(ends, tensors["row_ends"]))
                    return tensors["logits"].clone()

                with mock.patch.object(
                    replay, "_restore", side_effect=restore
                ), mock.patch.object(
                    replay, "_cuda_operator", return_value=mock.Mock()
                ), mock.patch.object(
                    replay, "_run_topk", side_effect=topk
                ) as run, mock.patch.object(
                    replay, "_mqa_operator", return_value=mqa
                ) as get_mqa, mock.patch.object(
                    torch.cuda, "synchronize"
                ), mock.patch.object(
                    torch.distributed,
                    "all_gather",
                    side_effect=AssertionError("collective"),
                ), mock.patch.object(
                    torch.distributed,
                    "all_to_all_single",
                    side_effect=AssertionError("collective"),
                ):
                    result = replay.replay_capture(path, "cuda")
                self.assertNotIn("query_global_ids.bin", restored_names)
                self.assertEqual(result["schema_version"], 3)
                self.assertEqual(run.call_count, 2 if joint else 1)
                self.assertEqual(get_mqa.call_count, int(joint))
                if joint:
                    self.assertTrue(result["mqa_comparison"]["allclose"])
                    self.assertTrue(result["chained_topk"]["selected_sets_equal"])

    def write_float_pair_bundle(self, *, staged=False):
        directory = self.root / ("float-pair-%d" % self.serial)
        self.serial += 1
        directory.mkdir()
        config = directory / "config.json"
        config.write_text(
            json.dumps(
                dict(
                    enabled=True,
                    authorized_capture=True,
                    layers=[0],
                    ops=[_FLOAT_PAIR_CAPTURE.name],
                    phases=["prefill"],
                    rank_filter="record_owner",
                    max_calls=1,
                    max_bundle_bytes=100000,
                    max_total_bytes=100000,
                    max_inflight_bytes=1000000,
                )
            )
        )
        identity = dict(
            schema_version=1,
            session_id="float-pair",
            owner_id="owner",
            replica_id="dp0",
            world_rank=0,
            dp_rank=0,
            exec_id="1",
            model_role="target",
            requests=[dict(phase="prefill", request_id="r1", q_len=3, kv_len=0)],
        )
        x = torch.arange(12, dtype=torch.float32).reshape(3, 4)[:, 1::2]
        y = torch.full_like(x, 0.5)
        outputs = dict(sum=x + y + 0.25, difference=x - y - 0.25)
        expected = {
            name: tensor.clone() for name, tensor in dict(x=x, y=y, **outputs).items()
        }
        with mock.patch.dict(
            os.environ,
            RTP_LLM_RECORD_DIR=str(directory / "records"),
            RTP_LLM_RECORD_SESSION="float-pair",
        ):
            recorder = capture.TensorCaptureRecorder(str(config))
            token = None
            try:
                context = recorder.for_execution(json.dumps(identity))
                self.assertIsNotNone(context)
                self.assertEqual(context.identity["schema_version"], 1)
                token = context.begin(
                    op=_FLOAT_PAIR_CAPTURE.name,
                    layer=0,
                    inputs=dict(x=x, y=y),
                    outputs={
                        name: capture.TensorSpec(tensor.shape, tensor.dtype, "cpu")
                        for name, tensor in outputs.items()
                    },
                    backend="torch_reference",
                    parameters=dict(bias=0.25),
                )
                self.assertIsNotNone(token)
                x.fill_(99)
                y.fill_(99)
                if staged:
                    self.assertTrue(token.capture_outputs(dict(sum=outputs["sum"])))
                    self.assertEqual(
                        list((directory / "records").rglob("manifest.json")), []
                    )
                    outputs["sum"].fill_(99)
                    self.assertTrue(
                        token.finish(dict(difference=outputs["difference"]))
                    )
                else:
                    self.assertTrue(token.finish(outputs))
                for tensor in outputs.values():
                    tensor.fill_(99)
            finally:
                if token is not None:
                    token.cancel()
                self.assertTrue(recorder.close(5))
        paths = list((directory / "records").rglob("manifest.json"))
        self.assertEqual(len(paths), 1)
        return paths[0], expected

    def test_schema3_public_capture_bins_loader_registered_replay_roundtrip(self):
        with mock.patch.dict(replay._REPLAY_CONTRACTS):
            replayer = mock.Mock(
                side_effect=AssertionError("CPU validation must not launch")
            )
            replay.register_replay_contract(
                _FLOAT_PAIR_CAPTURE.name,
                1,
                validator=_FloatPairReplayAdapter.validate,
                replayer=replayer,
            )
            for staged in (False, True):
                with self.subTest(staged=staged):
                    path, expected = self.write_float_pair_bundle(staged=staged)
                    self.assertEqual(
                        {p.name for p in path.parent.glob("*.bin")},
                        {name + ".bin" for name in expected},
                    )
                    manifest, tensors = capture.load_capture(path)
                    self.assertEqual(manifest["schema_version"], 3)
                    self.assertEqual(
                        manifest["contract"],
                        dict(name=_FLOAT_PAIR_CAPTURE.name, version=1),
                    )
                    self.assertEqual(manifest["inputs"], ["x", "y"])
                    self.assertEqual(manifest["outputs"], ["sum", "difference"])
                    self.assertEqual(
                        manifest["tensors"]["x"]["original_stride"], [4, 2]
                    )
                    self.assertEqual(manifest["tensors"]["x"]["storage_offset"], 1)
                    self.assertNotIn("mqa_parameters", manifest)
                    for name in expected:
                        self.assertEqual(tensors[name].dtype, torch.float32)
                        self.assertTrue(tensors[name].is_contiguous())
                        torch.testing.assert_close(
                            tensors[name], expected[name], rtol=0, atol=0
                        )
                    # Replay needs neither writer authorization nor a model/native library.
                    with mock.patch.dict(os.environ, {}, clear=True), mock.patch.object(
                        torch.cuda,
                        "is_available",
                        side_effect=AssertionError("CUDA probe"),
                    ), mock.patch.object(
                        replay.importlib,
                        "import_module",
                        side_effect=AssertionError("native import"),
                    ):
                        result = replay.replay_capture(path)
                        with contextlib.redirect_stdout(io.StringIO()) as output:
                            self.assertEqual(replay.main([str(path)]), 0)
                    self.assertEqual(json.loads(output.getvalue()), result)
                    self.assertEqual(result["compared_outputs"], 2)
                    self.assertEqual(result["compared_values"], 12)
                    self.assertEqual(result["mode"], "cpu_contract_validation")
                    self.assertFalse(result["kernel_replayed"])
                    self.assertNotIn("requests", result)
                    replayer.assert_not_called()

    def test_schema3_replay_registration_does_not_authorize_capture(self):
        config = self.root / "unauthorized.json"
        config.write_text(
            json.dumps(
                dict(
                    enabled=True,
                    authorized_capture=False,
                    ops=[_FLOAT_PAIR_CAPTURE.name],
                )
            )
        )
        with mock.patch.dict(replay._REPLAY_CONTRACTS):
            replay.register_replay_contract(
                _FLOAT_PAIR_CAPTURE.name, 1, validator=_FloatPairReplayAdapter.validate
            )
            with self.assertRaisesRegex(
                ValueError, "authorized_capture=true is required"
            ):
                capture.TensorCaptureRecorder(str(config))
        self.assertEqual(list(self.root.rglob("*.bin")), [])
        self.assertEqual(list(self.root.rglob("manifest.json")), [])

    def test_schema3_real_metadata_validation_precedes_adapter_dispatch(self):
        path, _ = self.write_float_pair_bundle()
        saved = path.read_text()
        with mock.patch.dict(replay._REPLAY_CONTRACTS):
            validator = mock.Mock(wraps=_FloatPairReplayAdapter.validate)
            replayer = mock.Mock(
                side_effect=AssertionError("invalid metadata must not launch")
            )
            replay.register_replay_contract(
                _FLOAT_PAIR_CAPTURE.name, 1, validator=validator, replayer=replayer
            )
            mutations = (
                lambda m: m.update(op="other"),
                lambda m: m["contract"].update(version=True),
                lambda m: m.update(inputs=["x", "x"]),
                lambda m: m.update(outputs=["sum", "x"]),
                lambda m: m.update(outputs=["sum"]),
                lambda m: m.update(parameters=[]),
                lambda m: m["tensors"]["sum"].update(file="../sum.bin"),
            )
            for index, mutate in enumerate(mutations):
                with self.subTest(index=index):
                    path.write_text(saved)
                    self.mutate(path, mutate)
                    with self.assertRaises(ValueError):
                        replay.replay_capture(path, "cuda")
                    validator.assert_not_called()
                    replayer.assert_not_called()
            # Unknown contracts load, but cannot authorize a callback or GPU execution.
            path.write_text(saved)
            self.mutate(
                path,
                lambda m: m.update(
                    op="untrusted.module.function",
                    contract=dict(name="untrusted.module.function", version=1),
                ),
            )
            capture.load_capture(path)
            with self.assertRaises(replay.UnsupportedCapture):
                replay.replay_capture(path, "cuda")
            validator.assert_not_called()
            replayer.assert_not_called()
            path.write_text(saved)
            self.mutate(
                path,
                lambda m: m.update(kernel_replayed=True, mode="cuda_kernel_replay"),
            )
            result = replay.replay_capture(path)
            self.assertFalse(result["kernel_replayed"])
            self.assertEqual(result["mode"], "cpu_contract_validation")
            replayer.assert_not_called()

    def test_schema3_real_each_output_mismatch_not_hidden_by_mqa_tolerances(self):
        path, expected = self.write_float_pair_bundle()
        with mock.patch.dict(replay._REPLAY_CONTRACTS):
            replayer = mock.Mock(
                side_effect=AssertionError("invalid captured output must not launch")
            )
            replay.register_replay_contract(
                _FLOAT_PAIR_CAPTURE.name,
                1,
                validator=_FloatPairReplayAdapter.validate,
                replayer=replayer,
            )
            for name in ("sum", "difference"):
                with self.subTest(output=name):
                    self.replace_tensor(path, name, expected[name] + 0.125)
                    capture.load_capture(
                        path
                    )  # Valid checksum/descriptor, wrong numerical output.
                    with self.assertRaisesRegex(
                        ValueError, "numerical mismatch: " + name
                    ):
                        replay.replay_capture(
                            path, "cuda", mqa_atol=1000, mqa_rtol=1000
                        )
                    replayer.assert_not_called()
                    self.replace_tensor(path, name, expected[name])
            self.assertFalse(replay.replay_capture(path)["kernel_replayed"])

    def float_pair_fixture(self, x=None, y=None):
        # Focused loader-return fixture, separate from the real public round-trip.
        # File/descriptor/envelope validation is outside this fixture's coverage;
        # this boundary tests replay dispatch and adapter semantics only.
        if x is None:
            x = torch.tensor([[1.5, -2.0], [0.0, 4.25]])
        if y is None:
            y = torch.tensor([[0.5, 3.0], [-1.0, 0.25]])
        tensors = dict(x=x, y=y, sum=x + y + 0.25, difference=x - y - 0.25)
        manifest = dict(
            schema_version=3,
            complete=True,
            op=_FloatPairReplayAdapter.name,
            contract=dict(name=_FloatPairReplayAdapter.name, version=1),
            inputs=["x", "y"],
            outputs=["sum", "difference"],
            backend="torch_reference",
            parameters=dict(bias=0.25),
        )
        for name, tensor in tensors.items():
            tensors[name] = tensor.contiguous()
        registry = mock.patch.dict(replay._REPLAY_CONTRACTS)
        registry.start()
        self.addCleanup(registry.stop)
        loader = mock.patch.object(
            replay, "load_capture", return_value=(manifest, tensors)
        )
        loader.start()
        self.addCleanup(loader.stop)
        return manifest, tensors

    def test_schema3_float_multi_output_adapter_without_dispatcher_changes(self):
        for x, y in (
            (torch.tensor([[1.5, -2.0], [0.0, 4.25]]), torch.ones(2, 2)),
            (torch.tensor(1.5), torch.tensor(-0.5)),
            (torch.empty(0, 2), torch.empty(0, 2)),
            (torch.arange(12, dtype=torch.float32).reshape(3, 4).t(), torch.ones(4, 3)),
        ):
            with self.subTest(shape=x.shape), mock.patch.dict(replay._REPLAY_CONTRACTS):
                manifest, tensors = self.float_pair_fixture(x, y)
                # Named outputs are not positional, and there is no TopK K/index tensor.
                manifest["outputs"].reverse()
                replay.register_replay_contract(
                    _FloatPairReplayAdapter.name,
                    1,
                    validator=_FloatPairReplayAdapter.validate,
                )
                with mock.patch.object(
                    replay.importlib,
                    "import_module",
                    side_effect=AssertionError("import"),
                ), mock.patch.object(
                    torch.cuda, "is_available", side_effect=AssertionError("CUDA probe")
                ), mock.patch.object(
                    replay,
                    "_check_schema",
                    side_effect=AssertionError("Indexer validator"),
                ):
                    result = replay.replay_capture("loader-return-fixture")
                self.assertEqual(result["status"], "validated")
                self.assertEqual(result["schema_version"], 3)
                self.assertEqual(result["contract"], manifest["contract"])
                self.assertEqual(result["compared_outputs"], 2)
                self.assertEqual(result["compared_values"], 2 * x.numel())
                self.assertNotIn("k", result)
                self.assertFalse(result["kernel_replayed"])
                self.assertEqual(result["mode"], "cpu_contract_validation")
                json.dumps(result, allow_nan=False)

    def test_schema3_each_float_output_is_numerically_checked(self):
        manifest, tensors = self.float_pair_fixture()
        replay.register_replay_contract(
            _FloatPairReplayAdapter.name, 1, validator=_FloatPairReplayAdapter.validate
        )
        for name in manifest["outputs"]:
            original = tensors[name]
            for replacement, error in (
                (original.double(), "dtype/shape: " + name),
                (original[:1], "dtype/shape: " + name),
                (torch.full_like(original, float("nan")), "nonfinite"),
                (torch.full_like(original, float("inf")), "nonfinite"),
            ):
                with self.subTest(output=name, error=error), self.assertRaisesRegex(
                    ValueError, error
                ):
                    tensors[name] = replacement
                    replay.replay_capture("loader-return-fixture")
            tensors[name] = original
        self.assertEqual(
            replay.replay_capture("loader-return-fixture")["compared_outputs"], 2
        )

    def test_schema3_versions_dispatch_and_cuda_callback_wiring_on_cpu(self):
        manifest, tensors = self.float_pair_fixture()
        validators, replayers = [], []
        for version in (1, 2):
            validator = mock.Mock(wraps=_FloatPairReplayAdapter.validate)
            # Orchestration only: the mock explicitly does not claim a kernel replay.
            replayer = mock.Mock(
                return_value=dict(
                    status="validated", kernel_replayed=False, adapter_version=version
                )
            )
            validators.append(validator)
            replayers.append(replayer)
            replay.register_replay_contract(
                _FloatPairReplayAdapter.name,
                version,
                validator=validator,
                replayer=replayer,
            )
        for version in (1, 2):
            manifest["contract"]["version"] = version
            result = replay.replay_capture(
                "loader-return-fixture", "cuda", mqa_atol=0.1, mqa_rtol=0.2
            )
            self.assertEqual(result["adapter_version"], version)
            self.assertFalse(result["kernel_replayed"])
            validators[version - 1].assert_called_once_with(manifest, tensors)
            args, kwargs = replayers[version - 1].call_args
            self.assertEqual(args[0], manifest)
            self.assertIs(args[1], tensors)
            self.assertEqual(kwargs["device"], "cuda")
            self.assertEqual(kwargs["options"], dict(mqa_atol=0.1, mqa_rtol=0.2))
            self.assertFalse(kwargs["summary"]["kernel_replayed"])
            self.assertEqual(kwargs["summary"]["contract"], manifest["contract"])

    def test_schema3_unknown_contract_version_and_backend_are_unsupported(self):
        manifest, _ = self.float_pair_fixture()
        replay.register_replay_contract(
            _FloatPairReplayAdapter.name, 1, validator=_FloatPairReplayAdapter.validate
        )
        for name, version, backend in (
            ("untrusted.module.function", 1, "torch_reference"),
            (_FloatPairReplayAdapter.name, 2, "torch_reference"),
            (_FloatPairReplayAdapter.name, 1, "untrusted.backend"),
        ):
            manifest.update(
                op=name, contract=dict(name=name, version=version), backend=backend
            )
            for device in ("cpu", "cuda"):
                with self.subTest(
                    name=name, version=version, backend=backend, device=device
                ), mock.patch.object(
                    replay.importlib,
                    "import_module",
                    side_effect=AssertionError("import"),
                ), mock.patch.object(
                    torch.cuda, "is_available", side_effect=AssertionError("CUDA probe")
                ):
                    with self.assertRaises(replay.UnsupportedCapture):
                        replay.replay_capture("loader-return-fixture", device)
                    with contextlib.redirect_stderr(io.StringIO()) as output:
                        self.assertEqual(
                            replay.main(["loader-return-fixture", "--device", device]),
                            2,
                        )
                    summary = json.loads(output.getvalue())
                    self.assertEqual(summary["status"], "unsupported")
                    self.assertNotIn("kernel_replayed", summary)

    def test_schema3_validation_only_adapter_and_cpu_status(self):
        self.float_pair_fixture()
        validator = mock.Mock(
            return_value=dict(kernel_replayed=True, mode="cuda_kernel_replay")
        )
        replay.register_replay_contract(
            _FloatPairReplayAdapter.name, 1, validator=validator
        )
        result = replay.replay_capture("loader-return-fixture")
        self.assertFalse(result["kernel_replayed"])
        self.assertEqual(result["mode"], "cpu_contract_validation")
        with self.assertRaisesRegex(replay.ReplayUnavailable, "no CUDA replayer"):
            replay.replay_capture("loader-return-fixture", "cuda")
        with contextlib.redirect_stderr(io.StringIO()) as output:
            self.assertEqual(
                replay.main(["loader-return-fixture", "--device", "cuda"]), 3
            )
        self.assertEqual(json.loads(output.getvalue())["status"], "unavailable")

    def test_schema3_validator_runs_before_replayer_and_errors_keep_cause(self):
        manifest, tensors = self.float_pair_fixture()
        replayer = mock.Mock(side_effect=RuntimeError("adapter launch failed"))
        replay.register_replay_contract(
            _FloatPairReplayAdapter.name,
            1,
            validator=_FloatPairReplayAdapter.validate,
            replayer=replayer,
        )
        for key, value, error in (
            ("parameters", {}, "parameters"),
            ("outputs", ["sum", "renamed"], "named tensors"),
        ):
            original = manifest[key]
            manifest[key] = value
            with self.assertRaisesRegex(ValueError, error):
                replay.replay_capture("loader-return-fixture", "cuda")
            manifest[key] = original
        replayer.assert_not_called()
        replay.replay_capture("loader-return-fixture", "cpu")
        replayer.assert_not_called()
        with self.assertRaisesRegex(RuntimeError, "adapter launch failed") as raised:
            replay.replay_capture("loader-return-fixture", "cuda")
        self.assertIs(raised.exception, replayer.side_effect)
        with contextlib.redirect_stderr(io.StringIO()) as output:
            self.assertEqual(
                replay.main(["loader-return-fixture", "--device", "cuda"]), 1
            )
        self.assertEqual(json.loads(output.getvalue())["status"], "failed")

    def test_schema3_options_are_adapter_owned_and_do_not_imply_kernel_execution(self):
        manifest, _ = self.float_pair_fixture()
        replayer = mock.Mock(side_effect=lambda m, t, **kwargs: kwargs["summary"])
        replay.register_replay_contract(
            _FloatPairReplayAdapter.name,
            1,
            validator=_FloatPairReplayAdapter.validate,
            replayer=replayer,
        )
        # These MQA-only options have no meaning for this non-Indexer contract.
        result = replay.replay_capture(
            "loader-return-fixture", "cuda", mqa_atol=-1, mqa_rtol=None
        )
        self.assertFalse(result["kernel_replayed"])
        self.assertEqual(result["mode"], "cpu_contract_validation")
        self.assertEqual(
            replayer.call_args.kwargs["options"], dict(mqa_atol=-1, mqa_rtol=None)
        )
        validator, replayer = mock.Mock(), mock.Mock()
        validate_options = mock.Mock(side_effect=ValueError("adapter options rejected"))
        replay.register_replay_contract(
            _FloatPairReplayAdapter.name,
            2,
            validator=validator,
            replayer=replayer,
            validate_options=validate_options,
        )
        manifest["contract"]["version"] = 2
        with self.assertRaisesRegex(ValueError, "adapter options rejected"):
            replay.replay_capture("loader-return-fixture", "cuda", mqa_atol=0.5)
        validate_options.assert_called_once_with(dict(mqa_atol=0.5, mqa_rtol=0.0))
        validator.assert_not_called()
        replayer.assert_not_called()

    def test_replay_contract_registration_rejects_invalid_or_duplicate_keys(self):
        with mock.patch.dict(replay._REPLAY_CONTRACTS):
            validator = mock.Mock(return_value={})
            for name, version in (
                ("", 1),
                ([], 1),
                ("test bad", 1),
                ("test", True),
                ("test", 0),
                ("test", -1),
                ("test", 1.5),
            ):
                with self.subTest(name=name, version=version), self.assertRaises(
                    ValueError
                ):
                    replay.register_replay_contract(name, version, validator=validator)
            for callbacks in (
                dict(validator=None),
                dict(validator=validator, replayer=42),
                dict(validator=validator, validate_options=42),
            ):
                with self.assertRaises(ValueError):
                    replay.register_replay_contract("test.invalid", 1, **callbacks)
            replay.register_replay_contract("test.register", 1, validator=validator)
            with self.assertRaisesRegex(ValueError, "already registered"):
                replay.register_replay_contract(
                    "test.register", 1, validator=lambda m, t: {}
                )
            self.assertIs(replay._REPLAY_CONTRACTS[("test.register", 1)][0], validator)
            for name, version in replay._LEGACY_CONTRACTS.values():
                with self.assertRaisesRegex(ValueError, "already registered"):
                    replay.register_replay_contract(name, version, validator=validator)

    def test_replay_loader_failure_never_invokes_registered_callbacks(self):
        self.float_pair_fixture()
        validator, replayer = mock.Mock(), mock.Mock()
        replay.register_replay_contract(
            _FloatPairReplayAdapter.name, 1, validator=validator, replayer=replayer
        )
        error = ValueError("tensor checksum mismatch")
        with mock.patch.object(
            replay, "load_capture", side_effect=error
        ), self.assertRaisesRegex(ValueError, "checksum") as raised:
            replay.replay_capture("loader-return-fixture", "cuda")
        self.assertIs(raised.exception, error)
        validator.assert_not_called()
        replayer.assert_not_called()

    def test_real_capture_round_trip_no_reader_authorization(self):
        config = self.root / "capture.json"
        config.write_text(
            json.dumps(
                dict(
                    enabled=True,
                    authorized_capture=True,
                    layers=[0],
                    ops=["indexer.topk"],
                    phases=["prefill"],
                    rank_filter="record_owner",
                    max_calls=1,
                    max_bundle_bytes=100000,
                    max_total_bytes=100000,
                    max_inflight_bytes=1000000,
                )
            )
        )
        identity = dict(
            schema_version=1,
            session_id="test",
            owner_id="owner",
            replica_id="dp0",
            world_rank=0,
            dp_rank=0,
            exec_id="1",
            model_role="target",
            requests=[dict(phase="prefill", request_id="r1", q_len=2, kv_len=1)],
        )
        with mock.patch.dict(
            os.environ,
            RTP_LLM_RECORD_DIR=str(self.root / "records"),
            RTP_LLM_RECORD_SESSION="test",
        ):
            recorder = capture.TensorCaptureRecorder(str(config))
            try:
                context = for_context(recorder.for_execution(json.dumps(identity)))
                self.assertIsNotNone(context)
                token = context.begin_topk(
                    layer=0,
                    logits=torch.tensor([[2.0, 1.0, 9.0], [1.0, 3.0, 2.0]]),
                    row_starts=torch.tensor([0, 0], dtype=torch.int32),
                    row_ends=torch.tensor([2, 3], dtype=torch.int32),
                    k=2,
                    backend="dsv4_per_row",
                    parameters=dict(
                        num_rows=2, stride0=3, stride1=1, force_radix_sort=False
                    ),
                    query_row_start=0,
                    total_query_rows=2,
                )
                self.assertIsNotNone(token)
                token.finish(torch.tensor([[0, 1], [2, 1]], dtype=torch.int32))
            finally:
                self.assertTrue(recorder.close(5))
        paths = list((self.root / "records").rglob("manifest.json"))
        self.assertEqual(len(paths), 1)
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(replay.replay_capture(paths[0])["counts"]["rows"], 2)


@unittest.skipUnless(
    os.environ.get("RTP_LLM_TEST_CAPTURE_REPLAY_CUDA") == "1",
    "CUDA replay not requested; opt in with RTP_LLM_TEST_CAPTURE_REPLAY_CUDA=1",
)
class TensorCaptureReplayCudaTest(_ReplayFixture):
    def test_paged_decode_topk_backends(self):
        for backend in ("topk_v3", "dsv4_persistent_topk"):
            for k in (512, 1024, 2048):
                with self.subTest(backend=backend, k=k):
                    path = self.paged_bundle(
                        backend=backend, k=k, lengths=(1, k - 1, k, k + 1)
                    )
                    result = replay.replay_capture(path, "cuda")
                    self.assertTrue(result["kernel_replayed"])
                    self.assertTrue(
                        result["independent_topk"]["captured_score_keys_equal"]
                    )

    def check_kernel(self, backend, k, force=False):
        result = replay.replay_capture(self.boundary_bundle(backend, k, force), "cuda")
        self.assertTrue(result["kernel_replayed"])
        self.assertEqual(result["mode"], "cuda_kernel_replay")

    def test_dsv4_insertion(self):
        self.check_kernel("dsv4_per_row", 3)

    def test_dsv4_forced_radix(self):
        self.check_kernel("dsv4_per_row", 3, force=True)

    def test_dsv4_unique_negative_scores(self):
        for force in (False, True):
            with self.subTest(force=force):
                result = replay.replay_capture(
                    self.bundle(force_radix_sort=force), "cuda"
                )
                self.assertTrue(result["kernel_replayed"])

    def test_v3_tie_break(self):
        self.check_kernel("topk_v3_tie_break", 512)

    def test_indexed_cp_dsv4_insertion_and_radix(self):
        for force in (False, True):
            with self.subTest(force=force):
                result = replay.replay_capture(self.indexed_bundle(force=force), "cuda")
                self.assertTrue(result["kernel_replayed"])
                self.assertEqual(result["schema_version"], 3)

    def test_indexed_cp_v3(self):
        path = self.indexed_bundle(backend="topk_v3_tie_break", k=512)
        self.assertTrue(replay.replay_capture(path, "cuda")["kernel_replayed"])


@unittest.skipUnless(
    os.environ.get("RTP_LLM_TEST_CAPTURE_REPLAY_CUDA") == "1",
    "MQA CUDA replay requires explicit opt-in and Bazel-built DeepGEMM/native libraries",
)
class TensorCaptureMqaReplayCudaTest(_ReplayFixture):
    def test_paged_mqa_shared_partial_pages_reuse_and_rebuild(self):
        self.assertTrue(torch.cuda.is_available(), "real paged MQA test requires CUDA")
        capability = torch.cuda.get_device_capability()
        if capability[0] not in (9, 10):
            self.skipTest(
                "real paged MQA requires verified SM90/SM100-family hardware "
                "(SM103 supported); found SM%d%d" % capability
            )
        deep_gemm = replay.importlib.import_module("deep_gemm")
        version = replay.tensor_capture_indexer.paged_schedule_version(deep_gemm)
        num_sms = deep_gemm.get_num_sms()
        environment = dict(
            deep_gemm_version=version,
            device_capability=list(capability),
            num_sms=num_sms,
        )
        lengths = torch.tensor([3, 63, 69, 513], dtype=torch.int32, device="cuda")
        rows, block, dimension, heads, columns, k = 4, 64, 128, 32, 640, 512
        physical = [5, 2, 8, 9, 10, 11, 12, 13, 14, 15]
        table = torch.tensor([physical] * rows, dtype=torch.int32, device="cuda")
        pool = torch.full((17, block, 1, 132), 231, dtype=torch.uint8, device="cuda")
        for logical, page in enumerate(physical):
            blob = pool[page].view(-1)
            blob[: block * dimension].view(torch.float8_e4m3fn).fill_(1)
            blob[block * dimension :].view(torch.float32).copy_(
                torch.arange(
                    logical * block + 1,
                    (logical + 1) * block + 1,
                    dtype=torch.float32,
                    device="cuda",
                )
                / dimension
            )
        q = torch.ones(rows, 1, heads, dimension, device="cuda").to(torch.float8_e4m3fn)
        weights = torch.ones(rows, heads, device="cuda")
        context_lens = lengths[:, None]
        schedule = deep_gemm.get_paged_mqa_logits_metadata(context_lens, block, num_sms)
        topk = replay._cuda_operator("topk_v3")
        workspace = torch.empty(1 << 20, dtype=torch.uint8, device="cuda")
        indices = torch.empty(rows, k, dtype=torch.int32, device="cuda")
        identity = dict(
            schema_version=1,
            session_id="paged-real",
            owner_id="owner",
            replica_id="dp0",
            world_rank=0,
            dp_rank=0,
            exec_id="1",
            model_role="target",
            server_role="DECODE",
            requests=[
                dict(
                    phase="decode",
                    request_id="r%d" % (row + 1),
                    q_len=1,
                    kv_len=length - 1,
                )
                for row, length in enumerate(lengths.tolist())
            ],
        )
        config = self.root / "paged-real-config.json"
        config.write_text(
            json.dumps(
                dict(
                    enabled=True,
                    authorized_capture=True,
                    layers=[0],
                    ops=["indexer.mqa_topk"],
                    phases=["decode"],
                    rank_filter="record_owner",
                    max_calls=1,
                    max_bundle_bytes=10000000,
                    max_total_bytes=10000000,
                    max_inflight_bytes=100000000,
                )
            )
        )
        with mock.patch.dict(
            os.environ,
            RTP_LLM_RECORD_DIR=str(self.root / "records"),
            RTP_LLM_RECORD_SESSION="paged-real",
        ):
            recorder = capture.TensorCaptureRecorder(str(config))
            token = None
            try:
                context = for_context(recorder.for_execution(json.dumps(identity)))
                self.assertIsNotNone(context, recorder.stats())
                token = context.begin_paged_mqa(
                    layer=0,
                    q_fp8=q,
                    kv_cache_fp8=pool,
                    weights=weights,
                    context_lens=context_lens,
                    block_table=table,
                    schedule_metadata=schedule,
                    lengths=lengths,
                    k=k,
                    block_size=block,
                    max_seq_len=columns,
                    total_query_rows=rows,
                    schedule_environment=environment,
                )
                self.assertIsNotNone(token, recorder.stats())
                logits = deep_gemm.fp8_paged_mqa_logits(
                    q,
                    pool,
                    weights,
                    context_lens,
                    table,
                    schedule,
                    columns,
                    clean_logits=False,
                )
                # Q and K are ones; scale=(token+1)/128 and folded weights=1.
                # This oracle is exact, independent of DeepGEMM and replay.
                for row, length in enumerate(lengths.tolist()):
                    expected = (
                        torch.arange(1, length + 1, dtype=torch.float32, device="cuda")
                        * heads
                    )
                    torch.testing.assert_close(
                        logits[row, :length], expected, rtol=0, atol=0
                    )
                parameters = dict(
                    k=k,
                    max_seq_len=columns,
                    index_space="sequence_relative",
                    workspace_bytes=workspace.numel(),
                    stride0=logits.stride(0),
                    stride1=1,
                )
                self.assertTrue(
                    token.prepare_topk(logits, backend="topk_v3", parameters=parameters)
                )
                topk(logits, lengths, indices, workspace, k, columns)
                self.assertTrue(token.finish(indices))
            finally:
                if token is not None:
                    token.cancel()
                self.assertTrue(recorder.close(30), recorder.stats())
        paths = list((self.root / "records").rglob("manifest.json"))
        self.assertEqual(len(paths), 1, recorder.stats())
        path = paths[0]
        manifest, tensors = capture.load_capture(path)
        self.assertEqual(manifest["contract"], dict(name="indexer.mqa_topk", version=3))
        self.assertEqual(manifest["schedule_environment"], environment)
        self.assertEqual(tensors["page_ids"].tolist(), physical[:9] + [-1] * 4)
        self.assertEqual(
            tensors["page_valid_tokens"].tolist(), [64] * 8 + [1] + [0] * 4
        )
        self.assertEqual(tensors["block_table"][:, 0].tolist(), [5] * rows)
        for mode in ("reused", "rebuilt"):
            with self.subTest(schedule=mode):
                if mode == "rebuilt":
                    # A genuine alternate schedule plus matching metadata remains
                    # a valid artifact, but cannot be reused at the current SM count.
                    alternate_sms = max(1, num_sms // 2) if num_sms > 1 else 2
                    alternate = deep_gemm.get_paged_mqa_logits_metadata(
                        context_lens, block, alternate_sms
                    )
                    self.replace_tensor(path, "schedule_metadata", alternate.cpu())
                    self.mutate(
                        path,
                        lambda m: m["schedule_environment"].update(
                            num_sms=alternate_sms
                        ),
                    )
                self.assertFalse(replay.replay_capture(path)["kernel_replayed"])
                result = replay.replay_capture(path, "cuda")
                self.assertTrue(result["kernel_replayed"])
                self.assertTrue(result["mqa_kernel_replayed"])
                self.assertEqual(
                    result["schedule"],
                    dict(mode=mode, environment=environment, compact_pages=9),
                )
                self.assertEqual(result["mqa_comparison"]["valid_values"], 648)
                self.assertEqual(result["mqa_comparison"]["max_abs"], 0)
                for stage in ("independent_topk", "chained_topk"):
                    self.assertTrue(result[stage]["contract_valid"])
                    self.assertTrue(result[stage]["selected_sets_equal"])
                    self.assertTrue(result[stage]["captured_score_keys_equal"])

    def check_group_replay(self, rows, columns):
        # No availability skip: opted-in missing/unsupported hardware or libraries
        # must fail with the real cause. Run only via the coordinated Bazel target.
        topk = replay._cuda_operator("dsv4_per_row")
        mqa = replay._mqa_operator()
        heads, dimension, k = 32, 128, 3
        key_values = torch.ones(columns, device="cuda")
        key_values[:3] = torch.tensor([2.0, 4.0, 3.0], device="cuda")
        tensors = dict(
            q_fp8=torch.ones(rows, heads, dimension, device="cuda").to(
                torch.float8_e4m3fn
            ),
            k_fp8=key_values[:, None]
            .expand(columns, dimension)
            .contiguous()
            .to(torch.float8_e4m3fn),
            k_scale=torch.full((columns,), 0.5, device="cuda"),
            weights=torch.full((rows, heads), 1.5, device="cuda"),
            row_starts=torch.zeros(rows, dtype=torch.int32, device="cuda"),
            row_ends=torch.arange(
                columns - rows + 1, columns + 1, dtype=torch.int32, device="cuda"
            ),
            indices=torch.empty(rows, k, dtype=torch.int32, device="cuda"),
        )

        def compute(t):
            logits = replay._call_mqa(mqa, t)
            expected = key_values * (dimension * heads * 0.5 * 1.5)
            for row in range(rows):
                end = int(t["row_ends"][row])
                torch.testing.assert_close(
                    logits[row, :end], expected[:end], rtol=0, atol=0
                )
            topk(
                logits,
                t["row_starts"],
                t["row_ends"],
                t["indices"],
                rows,
                logits.stride(0),
                logits.stride(1),
                k,
                True,
            )
            return logits, t["indices"]

        path = self.write_mqa_bundle(tensors, compute)
        result = replay.replay_capture(path, "cuda")
        self.assertTrue(result["mqa_kernel_replayed"])
        self.assertTrue(result["independent_topk"]["selected_sets_equal"])
        self.assertTrue(result["chained_topk"]["selected_sets_equal"])
        self.assertEqual(result["mqa_comparison"]["max_abs"], 0)

    def test_indexed_cp_joint_multi_request_prefix(self):
        # Four rows satisfy DeepGEMM's 128/heads block size with heads=32.
        # Offline replay uses full keys and local queries, never collectives.
        path = self.indexed_bundle(joint=True, ids=(4, 0, 3, 1), force=True)
        result = replay.replay_capture(path, "cuda")
        self.assertEqual(result["schema_version"], 3)
        self.assertTrue(result["mqa_kernel_replayed"])
        self.assertTrue(result["chained_topk"]["selected_sets_equal"])
        self.assertEqual(result["mqa_comparison"]["max_abs"], 0)

    def test_model_free_deepgemm_group_replay(self):
        self.check_group_replay(rows=128, columns=256)

    def test_non_aligned_key_count_small_query_tail(self):
        # Four query rows meet the 128/heads query blocking constraint; the
        # key dimension has no such N-alignment contract. Do not pad N for mocks.
        self.check_group_replay(rows=4, columns=257)


if __name__ == "__main__":
    unittest.main()
