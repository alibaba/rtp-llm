import copy
import hashlib
import inspect
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import unittest
import weakref
from contextlib import nullcontext
from pathlib import Path
from unittest import mock

import torch

from rtp_llm.utils import tensor_capture as capture
from rtp_llm.utils import tensor_capture_indexer as indexer_capture

_FLOAT_PAIR = capture.CaptureContract(
    "test.float_pair", metadata_fields=frozenset({"label"})
)
capture.register_capture_contract(_FLOAT_PAIR)


class _PackedInputContract(capture.CaptureContract):
    def input_specs(self, inputs, identity, metadata):
        return {"packed": capture.TensorSpec((2, 3), torch.float32)}

    def snapshot_inputs(self, inputs, tensors, identity, metadata):
        self.snapshot("direct", inputs["direct"], tensors)
        self.snapshot("packed", inputs["packed"][:2, :3], tensors)


_PACKED_INPUT = _PackedInputContract(
    "test.packed_input", metadata_fields=frozenset({"label"})
)
capture.register_capture_contract(_PACKED_INPUT)


def _float_pair(context, value):
    """A second real call site: registration + generic API, no Indexer knowledge."""
    token = context.begin(
        op=_FLOAT_PAIR.name,
        layer=0,
        inputs={"value": value},
        outputs={
            "positive": capture.TensorSpec(value.shape, value.dtype, value.device),
            "negative": capture.TensorSpec(value.shape, value.dtype, value.device),
        },
        backend="torch",
        parameters={"offset": 1},
    )
    try:
        outputs = {"positive": value + 1, "negative": value - 1}
    except BaseException:
        if token is not None:
            token.cancel()
        raise
    if token is not None:
        token.finish(outputs)
    return outputs


# A liveness watchdog for concurrent CI IO/scheduling, not a performance assertion.
# Timeout-contract tests separately keep their explicit close(0.01) checks.
_COMPLETION_TIMEOUT = 30


def _temporary_directory(prefix):
    directory = (
        os.environ.get("TEST_TMPDIR")
        or os.environ.get("TMPDIR")
        or os.environ.get("RTP_LLM_TEST_LOCAL_LAB")
    )
    return tempfile.TemporaryDirectory(prefix=prefix, dir=directory)


class TensorCaptureTest(unittest.TestCase):
    def setUp(self):
        self.temporary = _temporary_directory("tensor-capture-test-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.record_dir = self.root / "records"
        self.config_path = self.root / "config.json"
        self.config = dict(
            enabled=True,
            authorized_capture=True,
            layers=[0, 1, 0],
            ops=["indexer.topk"],
            phases=["prefill"],
            rank_filter="record_owner",
            max_calls=100,
            max_bundle_bytes=1000000,
            max_total_bytes=10000000,
            max_inflight_bytes=10000000,
        )
        self.identity = dict(
            schema_version=1,
            session_id="canary-session",
            owner_id="123-456",
            replica_id="dp0",
            world_rank=0,
            dp_rank=0,
            exec_id="123456789",
            model_role="target",
            requests=[dict(phase="prefill", request_id="r1", q_len=2, kv_len=1)],
        )
        self.environment = mock.patch.dict(
            os.environ,
            RTP_LLM_RECORD_DIR=str(self.record_dir),
            RTP_LLM_RECORD_SESSION="canary-session",
        )
        self.environment.start()
        self.addCleanup(self.environment.stop)
        self.recorders = []

    def tearDown(self):
        for recorder in self.recorders:
            self._assert_recorder_closed(recorder)

    def _writer_diagnostics(self, recorder):
        thread = recorder._thread
        frame = sys._current_frames().get(thread.ident) if thread is not None else None
        stack = (
            "".join(traceback.format_stack(frame))
            if frame is not None
            else "<no writer frame>"
        )
        return "stats=%r\nwriter stack:\n%s" % (recorder.stats(), stack)

    def _assert_recorder_drained(self, recorder, timeout=_COMPLETION_TIMEOUT):
        # _release notifies only after snapshot storage is discarded. Waiting on
        # that state keeps successful artifact tests independent of fsync latency.
        with recorder._condition:
            drained = recorder._condition.wait_for(
                lambda: recorder._pending == 0, timeout
            )
        if not drained:
            self.fail("writer did not drain: " + self._writer_diagnostics(recorder))
        self.assertEqual(recorder.stats()["inflight_bytes"], 0)

    def _assert_recorder_closed(self, recorder, timeout=_COMPLETION_TIMEOUT):
        deadline = time.monotonic() + timeout
        self._assert_recorder_drained(recorder, timeout)
        # Drain and join share one deadline; no retries or unbounded waits.
        if not recorder.close(max(0, deadline - time.monotonic())):
            self.fail("writer did not exit: " + self._writer_diagnostics(recorder))
        self.assertEqual(recorder.stats()["pending"], 0)
        self.assertEqual(recorder.stats()["inflight_bytes"], 0)
        if recorder._thread is not None:
            self.assertFalse(recorder._thread.is_alive())

    def test_temporary_directory_priority_and_cleanup(self):
        directories = {
            name: str(self.root / name)
            for name in ("TEST_TMPDIR", "TMPDIR", "RTP_LLM_TEST_LOCAL_LAB")
        }
        for directory in directories.values():
            Path(directory).mkdir()

        for removed, expected in (
            ((), directories["TEST_TMPDIR"]),
            (("TEST_TMPDIR",), directories["TMPDIR"]),
            (("TEST_TMPDIR", "TMPDIR"), directories["RTP_LLM_TEST_LOCAL_LAB"]),
        ):
            environment = {k: v for k, v in directories.items() if k not in removed}
            with (
                self.subTest(removed=removed),
                mock.patch.dict(os.environ, environment, clear=True),
            ):
                with _temporary_directory("directory-selection-") as name:
                    path = Path(name)
                    self.assertEqual(path.parent, Path(expected))
                    self.assertTrue(path.is_dir())
                self.assertFalse(path.exists())
                self.assertTrue(Path(expected).is_dir())

        with (
            mock.patch.dict(os.environ, {}, clear=True),
            mock.patch.object(tempfile, "TemporaryDirectory") as temporary,
        ):
            self.assertIs(
                _temporary_directory("default-selection-"), temporary.return_value
            )
            temporary.assert_called_once_with(prefix="default-selection-", dir=None)

    def recorder(self, config=None, **kwargs):
        self.config_path.write_text(
            json.dumps(self.config if config is None else config)
        )
        recorder = capture.TensorCaptureRecorder(str(self.config_path), **kwargs)
        self.recorders.append(recorder)
        return recorder

    def context(self, recorder=None, identity=None):
        recorder = recorder or self.recorder()
        context = recorder.for_execution(
            json.dumps(self.identity if identity is None else identity)
        )
        self.assertIsNotNone(context)
        return indexer_capture.for_context(context)

    def begin(self, context, **kwargs):
        values = dict(
            layer=0,
            logits=torch.tensor([[1.0, 2.0, 999.0], [3.0, 4.0, 5.0]]),
            row_starts=torch.tensor([0, 0], dtype=torch.int32),
            row_ends=torch.tensor([2, 3], dtype=torch.int32),
            k=2,
            backend="dsv4_per_row",
            parameters={"num_rows": 2, "force_radix_sort": False},
            query_row_start=0,
            total_query_rows=2,
        )
        values.update(kwargs)
        return context.begin_topk(**values)

    def output(self):
        return torch.tensor([[1, 0], [2, 1]], dtype=torch.int32)

    def manifests(self):
        return sorted(
            self.record_dir.glob("owner-*/tensor-capture/sink-*/call-*/manifest.json")
        )

    def test_disabled_has_no_environment_files_threads_or_cuda(self):
        with (
            mock.patch.dict(
                os.environ, {"RTP_LLM_RECORD_DIR": "", "RTP_LLM_RECORD_SESSION": ""}
            ),
            mock.patch.object(
                threading.Thread, "start", side_effect=AssertionError("thread")
            ),
            mock.patch.object(
                torch.cuda, "is_initialized", side_effect=AssertionError("cuda")
            ),
        ):
            recorder = self.recorder({})
            self.assertIsNone(recorder.for_execution("not json"))
            self.assertTrue(recorder.close())
        self.assertFalse(self.record_dir.exists())
        self.assertEqual(recorder.stats()["skip_reasons"], {"disabled": 1})

    def test_config_enabled_validates_without_recorder_side_effects(self):
        with (
            mock.patch.dict(
                os.environ, RTP_LLM_RECORD_DIR="", RTP_LLM_RECORD_SESSION=""
            ),
            mock.patch.object(
                threading.Thread, "start", side_effect=AssertionError("thread")
            ),
            mock.patch.object(
                torch.cuda, "is_initialized", side_effect=AssertionError("cuda")
            ),
        ):
            for value, expected in (
                ({}, False),
                ({"enabled": False}, False),
                (self.config, True),
            ):
                self.config_path.write_text(json.dumps(value))
                self.assertIs(capture.config_enabled(str(self.config_path)), expected)
            self.config_path.write_text('{"enabled": 1}')
            with self.assertRaises(ValueError):
                capture.config_enabled(str(self.config_path))
        self.assertFalse(self.record_dir.exists())

    def test_strict_configuration(self):
        changes = [
            {"enabled": 1},
            {"authorized_capture": 1},
            {"authorized_capture": False},
            {"layers": [-1]},
            {"layers": [True]},
            {"layers": []},
            {"layers": "0"},
            {"ops": ["indexer.mqa_logits"]},
            {"ops": []},
            {"phases": []},
            {"phases": ["unknown"]},
            {"phases": [True]},
            {"phases": "decode"},
            {"rank_filter": "all"},
            {"unexpected": 1},
            {"max_calls": 0},
            {"max_bundle_bytes": -1},
            {"max_total_bytes": 1.5},
            {"max_inflight_bytes": True},
        ]
        for change in changes:
            with self.subTest(change=change):
                with self.assertRaises(ValueError):
                    self.recorder(dict(self.config, **change))
        for field in (
            "authorized_capture",
            "max_calls",
            "rank_filter",
            "phases",
            "layers",
        ):
            with self.subTest(missing=field):
                config = dict(self.config)
                config.pop(field)
                with self.assertRaises(ValueError):
                    self.recorder(config)
        self.config_path.write_text('{"enabled":false,"enabled":true}')
        with self.assertRaises(ValueError):
            capture.TensorCaptureRecorder(str(self.config_path))
        self.assertFalse(self.record_dir.exists())
        recorder = self.recorder()
        self.assertEqual(recorder._config["layers"], [0, 1])

    def test_required_environment(self):
        for directory, session in (
            ("", "s"),
            ("relative", "s"),
            (str(self.record_dir), ""),
            (str(self.record_dir), "../s"),
            (str(self.root / ".." / "bad"), "s"),
        ):
            with self.subTest(directory=directory, session=session):
                with mock.patch.dict(
                    os.environ,
                    RTP_LLM_RECORD_DIR=directory,
                    RTP_LLM_RECORD_SESSION=session,
                ):
                    with self.assertRaises(ValueError):
                        self.recorder()
        self.assertFalse(self.record_dir.exists())

    def test_record_environment_file_or_symlink_rejected(self):
        blocked = self.root / "not-a-directory"
        blocked.write_text("file")
        link = self.root / "directory-link"
        link.symlink_to(self.root)
        for path in (blocked, blocked / "child", link / "child"):
            with self.subTest(path=path):
                with mock.patch.dict(os.environ, RTP_LLM_RECORD_DIR=str(path)):
                    with self.assertRaises(ValueError):
                        self.recorder()

    def test_owner_rank_only_and_identity_validation(self):
        nonowner = self.recorder(tp_rank=1)
        self.assertIsNone(nonowner.for_execution(json.dumps(self.identity)))
        self.assertEqual(nonowner.stats()["skip_reasons"]["non_owner_rank"], 1)
        self.assertFalse(self.record_dir.exists())
        recorder = self.recorder()
        changes = [
            {"schema_version": 2},
            {"schema_version": True},
            {"session_id": "other"},
            {"owner_id": "../escape"},
            {"owner_id": "a/b"},
            {"replica_id": ""},
            {"world_rank": 1},
            {"world_rank": True},
            {"dp_rank": -1},
            {"exec_id": 1},
            {"exec_id": "../../x"},
            {"model_role": "draft"},
            {"requests": []},
        ]
        for change in changes:
            with self.subTest(change=change):
                self.assertIsNone(
                    recorder.for_execution(json.dumps(dict(self.identity, **change)))
                )
        for change in (
            {"is_fake": True},
            {"is_fake": 0},
            {"phase": "decode"},
            {"request_id": "customer-id"},
            {"q_len": 0},
            {"q_len": True},
            {"kv_len": -1},
            {"sequence_id": "s1"},
        ):
            identity = copy.deepcopy(self.identity)
            identity["requests"][0].update(change)
            self.assertIsNone(recorder.for_execution(json.dumps(identity)))
        self.assertFalse(self.record_dir.exists())
        self.context(recorder)
        changed_owner = dict(self.identity, owner_id="another-owner")
        self.assertIsNone(recorder.for_execution(json.dumps(changed_owner)))
        self.assertEqual(recorder.stats()["skip_reasons"]["owner_mismatch"], 1)

    def test_owner_can_have_nonzero_world_rank_and_all_rows_must_be_legal(self):
        recorder = self.recorder(world_rank=4)
        identity = copy.deepcopy(self.identity)
        identity.update(world_rank=4, dp_rank=2, replica_id="dp2")
        context = self.context(recorder, identity)
        self.begin(context).finish(self.output())
        self._assert_recorder_closed(recorder)
        manifest, _ = capture.load_capture(self.manifests()[0])
        self.assertEqual((manifest["rank"], manifest["dp_rank"]), (4, 2))
        other = self.recorder()
        identity = copy.deepcopy(self.identity)
        identity["requests"].append(
            dict(phase="prefill", request_id="r2", q_len=1, kv_len=0, is_fake=True)
        )
        self.assertIsNone(other.for_execution(json.dumps(identity)))
        identity["requests"][-1].update(is_fake=False, phase="decode")
        self.assertIsNone(other.for_execution(json.dumps(identity)))

    def test_frozen_identity_unknown_fields_not_exported(self):
        identity = copy.deepcopy(self.identity)
        identity["prompt"] = "SECRET-PROMPT"
        identity["requests"][0]["trace_id"] = "SECRET-TRACE"
        identity["requests"][0]["sequence_id"] = 7
        context = self.context(identity=identity)
        identity["requests"][0]["q_len"] = 999
        with self.assertRaises(TypeError):
            context.identity["requests"][0]["q_len"] = 999
        self.begin(context).finish(self.output())
        self._assert_recorder_closed(context._recorder)
        text = self.manifests()[0].read_text()
        self.assertNotIn("SECRET", text)
        self.assertNotIn("trace_id", text)
        self.assertEqual(json.loads(text)["requests"][0]["sequence_id"], 7)

    def test_mask_snapshot_before_after_alias_and_layout(self):
        context = self.context()
        base = torch.arange(24, dtype=torch.float32).reshape(4, 6)
        logits = base[1:3, 1::2]
        logits[0, 2] = float("nan")
        logits[0, 0] = float("nan")
        original = logits.clone()
        starts, ends = (
            torch.tensor([0, 0], dtype=torch.int32),
            torch.tensor([2, 3], dtype=torch.int32),
        )
        token = self.begin(context, logits=logits, row_starts=starts, row_ends=ends)
        self.assertIsNotNone(token)
        torch.testing.assert_close(logits, original, equal_nan=True)
        logits.fill_(99)
        starts.fill_(99)
        ends.fill_(99)
        backing = torch.zeros((2, 4), dtype=torch.int32)
        output = backing[:, 1::2]
        output.copy_(self.output())
        token.finish(output)
        output.fill_(77)
        token.finish(output)
        token.cancel()
        self._assert_recorder_closed(context._recorder)
        manifest, tensors = capture.load_capture(self.manifests()[0])
        original[0, 2] = 0
        torch.testing.assert_close(tensors["logits"], original, equal_nan=True)
        torch.testing.assert_close(tensors["indices"], self.output())
        self.assertEqual(manifest["tensors"]["logits"]["original_stride"], [6, 2])
        self.assertEqual(manifest["tensors"]["logits"]["storage_offset"], 7)
        self.assertEqual(manifest["tensors"]["indices"]["original_stride"], [4, 2])
        self.assertFalse(manifest["finite_values_checked"])
        self.assertEqual(manifest["parameters"]["index_space"], "relative_to_row_start")
        self.assertEqual(context._recorder.stats()["inflight_bytes"], 0)
        self.assertEqual(context._recorder.stats()["completed"], 1)
        for path in self.record_dir.rglob("*"):
            if path.is_file():
                self.assertEqual(path.stat().st_mode & 0o777, 0o600)
            elif path.name.startswith("sink-") or path.name.startswith("call-"):
                self.assertEqual(path.stat().st_mode & 0o777, 0o700)

    def test_prefix_chunk_across_requests(self):
        identity = copy.deepcopy(self.identity)
        identity["requests"].append(
            dict(phase="prefill", request_id="r2", q_len=3, kv_len=2)
        )
        context = self.context(identity=identity)
        logits = torch.arange(24, dtype=torch.float32).reshape(3, 8)
        token = self.begin(
            context,
            logits=logits,
            query_row_start=1,
            total_query_rows=5,
            row_starts=torch.tensor([0, 3, 3], dtype=torch.int32),
            row_ends=torch.tensor([3, 6, 7], dtype=torch.int32),
        )
        self.assertIsNotNone(token)
        token.finish(torch.zeros((3, 2), dtype=torch.int32))
        self._assert_recorder_closed(context._recorder)
        manifest, tensors = capture.load_capture(self.manifests()[0].parent)
        self.assertEqual(manifest["query_row_start"], 1)
        self.assertEqual(manifest["total_query_rows"], 5)
        expected = logits.clone()
        expected[0, 3:] = 0
        expected[1, :3] = 0
        expected[1, 6:] = 0
        expected[2, :3] = 0
        expected[2, 7:] = 0
        torch.testing.assert_close(tensors["logits"], expected)

    def test_prefix_chunk_waits_for_delayed_manifest_publication(self):
        entered, release = threading.Event(), threading.Event()
        waiting, finished = threading.Event(), threading.Event()
        errors = []
        atomic_file = capture._atomic_file
        assert_drained = self._assert_recorder_drained

        def blocked_publish(path, data):
            if path.name == "manifest.json":
                entered.set()
                if not release.wait(_COMPLETION_TIMEOUT):
                    raise AssertionError("test did not release manifest publication")
            return atomic_file(path, data)

        def observed_wait(recorder, timeout=_COMPLETION_TIMEOUT):
            waiting.set()
            return assert_drained(recorder, timeout)

        def run_prefix_test():
            try:
                # Exercise the exact failing success path, including loaded values.
                self.test_prefix_chunk_across_requests()
            except BaseException as error:
                errors.append(error)
            finally:
                finished.set()

        with (
            mock.patch.object(capture, "_atomic_file", side_effect=blocked_publish),
            mock.patch.object(
                self, "_assert_recorder_drained", side_effect=observed_wait
            ),
        ):
            worker = threading.Thread(target=run_prefix_test)
            try:
                worker.start()
                self.assertTrue(entered.wait(_COMPLETION_TIMEOUT))
                self.assertTrue(waiting.wait(_COMPLETION_TIMEOUT))
                recorder = self.recorders[0]
                # Reproduce close timing out on healthy-but-incomplete IO. No
                # sleeps: only the controller may release the publication gate.
                self.assertFalse(recorder.close(0.01))
                self.assertFalse(finished.is_set())
                stats = recorder.stats()
                self.assertEqual(stats["pending"], 1)
                self.assertGreater(stats["inflight_bytes"], 0)
                self.assertEqual(stats["completed"], 0)
                self.assertEqual(stats["io_error"], 0)
                self.assertEqual(self.manifests(), [])
            finally:
                release.set()
                worker.join(_COMPLETION_TIMEOUT)
        self.assertFalse(worker.is_alive())
        self.assertEqual(errors, [])
        self.assertTrue(finished.is_set())
        stats = recorder.stats()
        self.assertEqual(stats["completed"], 1)
        self.assertEqual(stats["pending"], 0)
        self.assertEqual(stats["inflight_bytes"], 0)
        self.assertEqual(stats["io_error"], 0)
        self.assertEqual(stats["writer_error"], 0)
        self.assertFalse(recorder._thread.is_alive())

    def test_completion_watchdog_still_rejects_stalled_writer(self):
        context = self.context()
        recorder = context._recorder
        token = self.begin(context)
        try:
            with self.assertRaisesRegex(
                AssertionError, "writer did not drain"
            ) as failure:
                self._assert_recorder_closed(recorder, timeout=0.01)
            self.assertIn("'pending': 1", str(failure.exception))
            self.assertIn("writer stack:", str(failure.exception))
            self.assertEqual(recorder.stats()["pending"], 1)
            self.assertGreater(recorder.stats()["inflight_bytes"], 0)
        finally:
            token.cancel()
        self._assert_recorder_closed(recorder)

    def test_reject_mapping_dtypes_parameters_and_output(self):
        context = self.context()
        changes = [
            {"query_row_start": 1},
            {"total_query_rows": 3},
            {"query_row_start": -1},
            {"logits": torch.ones(2, 2)},
            {"logits": torch.ones(2, 3, dtype=torch.float16)},
            {"logits": torch.empty(0, 3)},
            {"row_starts": torch.zeros(2, dtype=torch.int64)},
            {"row_starts": torch.tensor([0, 1], dtype=torch.int32)},
            {"row_ends": torch.tensor([1, 3], dtype=torch.int32)},
            {"row_ends": torch.tensor([2, 4], dtype=torch.int32)},
            {"row_ends": torch.tensor([-1, 3], dtype=torch.int32)},
            {"row_ends": torch.tensor([2], dtype=torch.int32)},
            {"k": 0},
            {"k": True},
            {"backend": "../bad"},
            {"parameters": {"k": 3}},
            {"parameters": {"index_space": "absolute"}},
            {"parameters": {"x": float("nan")}},
            {"parameters": {"x": [1, 2]}},
            {"parameters": {"x": "business text"}},
        ]
        for change in changes:
            with self.subTest(keys=list(change)):
                self.assertIsNone(self.begin(context, **change))
                self.assertEqual(context._recorder.stats()["inflight_bytes"], 0)
        self.assertIsNone(self.begin(context, layer=8))
        self.begin(context).finish(torch.zeros(2, 2))
        self.begin(context).finish(torch.zeros(2, 1, dtype=torch.int32))
        self.assertEqual(context._recorder.stats()["pending"], 0)
        self.assertEqual(self.manifests(), [])

    def test_zero_prefix_padding_and_cancel(self):
        identity = copy.deepcopy(self.identity)
        identity["requests"][0].update(q_len=1, kv_len=0)
        context = self.context(identity=identity)
        token = self.begin(
            context,
            logits=torch.tensor([[-3.0]]),
            row_starts=torch.tensor([0], dtype=torch.int32),
            row_ends=torch.tensor([1], dtype=torch.int32),
            total_query_rows=1,
        )
        self.assertIsNotNone(token)
        token.finish(torch.tensor([[0, -1]], dtype=torch.int32))
        self._assert_recorder_closed(context._recorder)
        _, tensors = capture.load_capture(self.manifests()[0])
        self.assertEqual(tensors["indices"].tolist(), [[0, -1]])
        other = self.context()
        token = self.begin(other)
        token.cancel()
        token.cancel()
        token.finish(self.output())
        self.assertEqual(other._recorder.stats()["pending"], 0)
        self.assertEqual(other._recorder.stats()["reserved_bytes"], 0)
        self.assertEqual(other._recorder.stats()["skip_reasons"]["kernel_error"], 1)
        token = self.begin(other)
        del token
        self.assertEqual(other._recorder.stats()["pending"], 0)

    def test_raw_encoding_bfloat_float8_integers_noncontiguous_empty_scalar(self):
        examples = [
            torch.arange(24, dtype=torch.int32).reshape(4, 6)[1:, ::2],
            torch.empty((2, 0, 3), dtype=torch.float32),
            torch.tensor(42, dtype=torch.int64),
            torch.tensor([True, False]),
            torch.tensor([-128, 127], dtype=torch.int8),
            torch.tensor([0, 32767, -32768, -1], dtype=torch.int16).view(
                torch.bfloat16
            ),
        ]
        for name in capture._DTYPES:
            if name.startswith("float8"):
                examples.append(
                    torch.arange(256, dtype=torch.uint8).view(capture._DTYPES[name])
                )
        for number, tensor in enumerate(examples):
            with self.subTest(dtype=tensor.dtype, shape=tensor.shape):
                desc, data = capture._encode_tensor(tensor)
                self.assertEqual(desc["original_stride"], list(tensor.stride()))
                self.assertEqual(desc["storage_offset"], tensor.storage_offset())
                self.assertEqual(desc["nbytes"], tensor.numel() * tensor.element_size())
                directory = self.root / ("raw-%d" % number)
                directory.mkdir()
                desc["file"] = "value.bin"
                (directory / "value.bin").write_bytes(data)
                (directory / "manifest.json").write_text(
                    json.dumps(
                        dict(schema_version=1, complete=True, tensors={"value": desc})
                    )
                )
                _, restored = capture.load_capture(directory)
                expected = tensor.contiguous().reshape(-1).view(torch.uint8)
                self.assertTrue(
                    torch.equal(
                        restored["value"].reshape(-1).view(torch.uint8), expected
                    )
                )
                self.assertEqual(list(restored["value"].shape), list(tensor.shape))
        desc, data = capture._encode_tensor(
            torch.tensor([0x01020304], dtype=torch.int32)
        )
        self.assertEqual(data, b"\x04\x03\x02\x01")
        self.assertEqual(desc["sha256"], hashlib.sha256(data).hexdigest())

    def test_loader_rejects_corruption_incomplete_paths_and_bounds(self):
        context = self.context()
        self.begin(context).finish(self.output())
        self._assert_recorder_closed(context._recorder)
        path = self.manifests()[0]
        original = json.loads(path.read_text())
        tensor = path.parent / "logits.bin"
        data = tensor.read_bytes()
        tensor.write_bytes(b"x" + data[1:])
        with self.assertRaisesRegex(ValueError, "checksum"):
            capture.load_capture(path)
        tensor.write_bytes(data + b"x")
        with self.assertRaises(ValueError):
            capture.load_capture(path)
        tensor.write_bytes(data)
        mutations = [
            lambda m: m.update(complete=False),
            lambda m: m.update(schema_version=99),
            lambda m: m["tensors"]["logits"].update(file="../logits.bin"),
            lambda m: m["tensors"]["logits"].update(file=str(tensor)),
            lambda m: m["tensors"]["logits"].update(dtype="pickle"),
            lambda m: m["tensors"]["logits"].update(shape=[-1]),
            lambda m: m["tensors"]["logits"].update(shape=[True, 3]),
            lambda m: m["tensors"]["logits"].update(shape=[2**40, 3]),
            lambda m: m["tensors"]["logits"].update(shape=[2, 4]),
            lambda m: m["tensors"]["logits"].update(original_stride=[-1, 1]),
            lambda m: m["tensors"]["logits"].update(storage_offset=-1),
            lambda m: m["tensors"]["logits"].update(byte_order="big"),
            lambda m: m["tensors"]["logits"].update(nbytes=2**40),
            lambda m: m["tensors"]["logits"].update(sha256="wrong"),
        ]
        for mutate in mutations:
            manifest = copy.deepcopy(original)
            mutate(manifest)
            path.write_text(json.dumps(manifest))
            with self.assertRaises(ValueError):
                capture.load_capture(path)
        path.write_text(json.dumps(original))
        tensor.unlink()
        outside = self.root / "outside.bin"
        outside.write_bytes(data)
        tensor.symlink_to(outside)
        with self.assertRaises((OSError, ValueError)):
            capture.load_capture(path)
        alias = self.root / "alias"
        alias.symlink_to(path.parent)
        with self.assertRaises(ValueError):
            capture.load_capture(alias)

    def test_budgets_reserved_before_copy(self):
        for field, reason in (
            ("max_bundle_bytes", "oversize"),
            ("max_total_bytes", "total_budget"),
            ("max_inflight_bytes", "inflight_budget"),
        ):
            with self.subTest(field=field):
                context = self.context(self.recorder(dict(self.config, **{field: 1})))
                with mock.patch.object(
                    capture,
                    "_snapshot",
                    side_effect=AssertionError("copied before budget"),
                ):
                    self.assertIsNone(self.begin(context))
                self.assertEqual(context._recorder.stats()["skip_reasons"][reason], 1)
                self.assertEqual(context._recorder.stats()["selected"], 0)
        context = self.context(self.recorder(dict(self.config, max_calls=1)))
        token = self.begin(context)
        self.assertIsNone(self.begin(context))
        token.cancel()
        self.assertIsNone(self.begin(context))
        self.assertEqual(context._recorder.stats()["skip_reasons"]["max_calls"], 2)

    def test_total_and_inflight_budget_lifetime(self):
        context = self.context()
        recorder = context._recorder
        token = self.begin(context)
        size = recorder.stats()["reserved_bytes"]
        recorder._config["max_total_bytes"] = size
        self.assertIsNone(self.begin(context))
        token.cancel()
        recorder._config["max_total_bytes"] = 10 * size
        recorder._config["max_inflight_bytes"] = capture._INFLIGHT_FACTOR * size
        token = self.begin(context)
        self.assertIsNone(self.begin(context))
        entered, release = threading.Event(), threading.Event()
        original = recorder._write_bundle

        def blocked(*args):
            entered.set()
            self.assertTrue(release.wait(_COMPLETION_TIMEOUT))
            return original(*args)

        with mock.patch.object(recorder, "_write_bundle", side_effect=blocked):
            try:
                token.finish(self.output())
                self.assertTrue(entered.wait(_COMPLETION_TIMEOUT))
                self.assertEqual(recorder.stats()["queue_size"], 0)
                self.assertEqual(
                    recorder.stats()["inflight_bytes"], capture._INFLIGHT_FACTOR * size
                )
                self.assertIsNone(self.begin(context))
                self.assertFalse(recorder.close(0.01))
            finally:
                release.set()
            self._assert_recorder_closed(recorder)
        self.assertEqual(recorder.stats()["inflight_bytes"], 0)
        self.assertEqual(recorder.stats()["reserved_bytes"], size)
        self.assertEqual(recorder.stats()["completed"], 1)
        # A successfully published bundle permanently consumes total budget.
        other = self.context()
        first = self.begin(other)
        other._recorder._config["max_total_bytes"] = other._recorder.stats()[
            "reserved_bytes"
        ]
        first.finish(self.output())
        self._assert_recorder_drained(other._recorder)
        self.assertIsNone(self.begin(other))
        self.assertEqual(other._recorder.stats()["skip_reasons"]["total_budget"], 1)

    def test_bounded_queue_and_enqueue_race(self):
        context = self.context()
        recorder = context._recorder
        self.assertEqual(recorder._queue.maxsize, 32)
        entered, release = threading.Event(), threading.Event()
        original = recorder._write_bundle

        def blocked(*args):
            entered.set()
            self.assertTrue(release.wait(_COMPLETION_TIMEOUT))
            return original(*args)

        with mock.patch.object(recorder, "_write_bundle", side_effect=blocked):
            try:
                self.begin(context).finish(self.output())
                self.assertTrue(entered.wait(_COMPLETION_TIMEOUT))
                late_token = self.begin(context)
                for _ in range(32):
                    token = self.begin(context)
                    self.assertIsNotNone(token)
                    token.finish(self.output())
                self.assertIsNone(self.begin(context))
                late_token.finish(self.output())
                self.assertEqual(recorder.stats()["skip_reasons"]["queue_full"], 2)
                self.assertEqual(recorder.stats()["pending"], 33)
            finally:
                release.set()
            self._assert_recorder_closed(recorder)
        self.assertEqual(recorder.stats()["completed"], 33)
        self.assertEqual(recorder.stats()["inflight_bytes"], 0)

    def test_thread_and_io_failure_fail_open_without_manifest(self):
        recorder = self.recorder()
        with mock.patch.object(
            threading.Thread, "start", side_effect=RuntimeError("no threads")
        ):
            self.assertIsNone(recorder.for_execution(json.dumps(self.identity)))
        self.assertIsNone(recorder.for_execution(json.dumps(self.identity)))
        self.assertEqual(recorder.stats()["writer_error"], 1)
        context = self.context()
        original = capture._atomic_file

        def fail(path, data):
            if path.name == "row_ends.bin":
                raise OSError("disk full")
            return original(path, data)

        with mock.patch.object(capture, "_atomic_file", side_effect=fail):
            self.begin(context).finish(self.output())
            self._assert_recorder_closed(context._recorder)
        self.assertEqual(self.manifests(), [])
        self.assertEqual(context._recorder.stats()["io_error"], 1)
        self.assertEqual(context._recorder.stats()["bytes_published"], 0)
        self.assertGreater(context._recorder.stats()["reserved_bytes"], 0)
        self.assertEqual(context._recorder.stats()["inflight_bytes"], 0)
        self.assertTrue(list(self.record_dir.rglob("row_starts.bin")))

    def test_manifest_publish_failure_keeps_incomplete_and_total_charge(self):
        context = self.context()
        recorder = context._recorder
        token = self.begin(context)
        size = recorder.stats()["reserved_bytes"]
        recorder._config["max_total_bytes"] = size
        replace = os.replace

        def fail_manifest(source, destination):
            if destination.name == "manifest.json":
                raise OSError("manifest rename failed")
            return replace(source, destination)

        with mock.patch.object(os, "replace", side_effect=fail_manifest):
            token.finish(self.output())
            self._assert_recorder_drained(recorder)
        self.assertEqual(self.manifests(), [])
        self.assertEqual(list(self.record_dir.rglob("*.tmp")), [])
        self.assertEqual(recorder.stats()["io_error"], 1)
        self.assertEqual(recorder.stats()["bytes_published"], 0)
        self.assertEqual(recorder.stats()["reserved_bytes"], size)
        self.assertIsNone(self.begin(context))
        self.assertEqual(recorder.stats()["skip_reasons"]["total_budget"], 1)

    def test_snapshot_errors_and_close_during_copy(self):
        context = self.context()
        recorder = context._recorder
        with mock.patch.object(
            capture, "_snapshot", side_effect=RuntimeError("copy failed")
        ):
            self.assertIsNone(self.begin(context))
        self.assertEqual(recorder.stats()["pending"], 0)
        token = self.begin(context)
        with mock.patch.object(
            capture, "_snapshot", side_effect=RuntimeError("output copy failed")
        ):
            token.finish(self.output())
        self.assertEqual(recorder.stats()["pending"], 0)
        entered, release = threading.Event(), threading.Event()
        original = capture._snapshot
        result = []

        def blocked(tensor):
            if tensor.dtype == torch.float32:
                entered.set()
                self.assertTrue(release.wait(_COMPLETION_TIMEOUT))
            return original(tensor)

        with mock.patch.object(capture, "_snapshot", side_effect=blocked):
            worker = threading.Thread(target=lambda: result.append(self.begin(context)))
            try:
                worker.start()
                self.assertTrue(entered.wait(_COMPLETION_TIMEOUT))
                self.assertFalse(recorder.close(0.01))
                self.assertGreater(recorder.stats()["inflight_bytes"], 0)
            finally:
                release.set()
                worker.join(_COMPLETION_TIMEOUT)
        self.assertFalse(worker.is_alive())
        self.assertIsNotNone(result[0])
        result[0].finish(self.output())
        self._assert_recorder_closed(recorder)
        self.assertIsNone(self.begin(context))
        self.assertIsNone(recorder.for_execution(json.dumps(self.identity)))
        self.assertEqual(recorder.stats()["inflight_bytes"], 0)
        self.assertEqual(self.manifests(), [])

    def test_two_model_instances_do_not_overwrite_and_metadata(self):
        with (
            mock.patch.object(torch.cuda, "is_initialized", return_value=False),
            mock.patch.object(
                torch.cuda, "is_available", side_effect=AssertionError("cuda probe")
            ),
            mock.patch.object(
                torch.cuda,
                "get_device_properties",
                side_effect=AssertionError("cuda init"),
            ),
        ):
            contexts = [self.context(), self.context()]
            for context in contexts:
                self.begin(context).finish(self.output())
            for context in contexts:
                self._assert_recorder_closed(context._recorder)
        paths = self.manifests()
        self.assertEqual(len(paths), 2)
        self.assertNotEqual(paths[0].parent.parent, paths[1].parent.parent)
        for path in paths:
            metadata = json.loads((path.parent.parent / "run.json").read_text())
            self.assertEqual(metadata["owner_id"], "123-456")
            self.assertIsNone(metadata["model_provenance"])
            self.assertIsNone(metadata["gpu"])
            self.assertEqual(
                metadata["config_sha256"],
                hashlib.sha256(capture._json_bytes(metadata["config"])).hexdigest(),
            )

    def test_capture_directory_symlink_refused(self):
        self.record_dir.mkdir()
        outside = self.root / "outside"
        outside.mkdir()
        (self.record_dir / "owner-123-456").symlink_to(outside)
        recorder = self.recorder()
        recorder.for_execution(json.dumps(self.identity))
        self._assert_recorder_closed(recorder)
        self.assertEqual(recorder.stats()["writer_error"], 1)
        self.assertEqual(list(outside.iterdir()), [])

    def test_close_is_bounded_during_sink_initialization(self):
        recorder = self.recorder()
        entered, release = threading.Event(), threading.Event()
        original = recorder._initialize_sink

        def blocked():
            entered.set()
            self.assertTrue(release.wait(_COMPLETION_TIMEOUT))
            return original()

        with mock.patch.object(recorder, "_initialize_sink", side_effect=blocked):
            try:
                self.context(recorder)
                self.assertTrue(entered.wait(_COMPLETION_TIMEOUT))
                self.assertFalse(recorder.close(0.01))
                self.assertTrue(recorder._thread.daemon)
            finally:
                release.set()
            self._assert_recorder_closed(recorder)

    def test_sink_init_failure_drains_queued_payload(self):
        recorder = self.recorder()
        entered, release = threading.Event(), threading.Event()

        def blocked_failure():
            entered.set()
            self.assertTrue(release.wait(_COMPLETION_TIMEOUT))
            raise OSError("metadata failed")

        with mock.patch.object(
            recorder, "_initialize_sink", side_effect=blocked_failure
        ):
            try:
                context = self.context(recorder)
                self.assertTrue(entered.wait(_COMPLETION_TIMEOUT))
                self.begin(context).finish(self.output())
                self.assertGreater(recorder.stats()["inflight_bytes"], 0)
            finally:
                release.set()
            self._assert_recorder_closed(recorder)
        self.assertEqual(recorder.stats()["inflight_bytes"], 0)
        self.assertEqual(recorder.stats()["reserved_bytes"], 0)
        self.assertTrue(recorder.stats()["failed"])
        self.assertEqual(self.manifests(), [])

    def test_concurrent_selection_limits_and_exact_bundle_boundary(self):
        recorder = self.recorder(dict(self.config, max_calls=5))
        context = self.context(recorder)
        token = self.begin(context)
        size = recorder.stats()["reserved_bytes"]
        token.cancel()
        recorder._config["max_bundle_bytes"] = size
        barrier = threading.Barrier(9)
        failures = []

        def select():
            try:
                barrier.wait(_COMPLETION_TIMEOUT)
                selected = self.begin(context)
                if selected is not None:
                    selected.finish(self.output())
            except BaseException as error:
                failures.append(error)

        workers = [threading.Thread(target=select) for _ in range(8)]
        for worker in workers:
            worker.start()
        barrier.wait(_COMPLETION_TIMEOUT)
        for worker in workers:
            worker.join(_COMPLETION_TIMEOUT)
            self.assertFalse(worker.is_alive())
        self.assertEqual(failures, [])
        self._assert_recorder_closed(recorder)
        stats = recorder.stats()
        self.assertEqual(stats["selected"], 5)
        self.assertEqual(stats["completed"], 4)
        self.assertEqual(stats["inflight_bytes"], 0)
        self.assertEqual(stats["reserved_bytes"], 4 * size)
        self.assertEqual(stats["skip_reasons"]["max_calls"], 4)

    def test_writer_reader_payload_limit_before_reservation_or_copy(self):
        context = self.context()
        recorder = context._recorder
        # logits=24, ranges=8+8, indices=16. All individual files fit 55.
        payload = 56
        with (
            mock.patch.object(capture, "_MAX_FILE_BYTES", payload - 1),
            mock.patch.object(recorder, "_reserve") as reserve,
            mock.patch.object(capture, "_snapshot") as snapshot,
        ):
            self.assertIsNone(self.begin(context))
            reserve.assert_not_called()
            snapshot.assert_not_called()
        self.assertEqual(recorder.stats()["selected"], 0)
        self.assertEqual(recorder.stats()["skip_reasons"]["payload_limit"], 1)
        with mock.patch.object(capture, "_MAX_FILE_BYTES", payload):
            token = self.begin(context)
            self.assertIsNotNone(token)
            token.finish(self.output())
            self._assert_recorder_closed(recorder)
            path = self.manifests()[0]
            manifest, tensors = capture.load_capture(path)
            self.assertEqual(
                sum(d["nbytes"] for d in manifest["tensors"].values()), payload
            )
            torch.testing.assert_close(tensors["indices"], self.output())
        self.assertEqual(recorder.stats()["completed"], 1)
        self.assertEqual(
            recorder.stats()["bytes_published"], payload + path.stat().st_size
        )
        with (
            mock.patch.object(capture, "_MAX_FILE_BYTES", payload - 1),
            mock.patch.object(
                capture, "_read_regular", wraps=capture._read_regular
            ) as read,
        ):
            with self.assertRaisesRegex(ValueError, "capture exceeds loader limit"):
                capture.load_capture(path)
            # Reject the aggregate before reading ANY tensor payload.
            self.assertEqual(read.call_count, 1)
            self.assertEqual(read.call_args.args[0], path)

    def test_same_token_finish_finish_and_finish_cancel_races(self):
        for first, second in (
            ("finish", "finish"),
            ("finish", "cancel"),
            ("cancel", "finish"),
        ):
            with self.subTest(first=first, second=second):
                context = self.context()
                recorder = context._recorder
                token = self.begin(context)
                entered, release = threading.Event(), threading.Event()
                second_started, second_done = threading.Event(), threading.Event()
                errors = []
                pending_at_boundary = []
                target = capture if first == "finish" else recorder
                name = "_snapshot" if first == "finish" else "_release"
                original = getattr(target, name)

                def blocked(*args, **kwargs):
                    # Refund now holds the recorder condition through the entire
                    # transaction; querying stats from the waiting thread deadlocks.
                    pending_at_boundary.append(recorder._pending)
                    entered.set()
                    if not release.wait(_COMPLETION_TIMEOUT):
                        raise AssertionError("test did not release token operation")
                    return original(*args, **kwargs)

                def invoke(operation, competing=False):
                    try:
                        if competing:
                            second_started.set()
                        if operation == "finish":
                            token.finish(self.output())
                        else:
                            token.cancel()
                    except BaseException as error:
                        errors.append(error)
                    finally:
                        if competing:
                            second_done.set()

                with mock.patch.object(target, name, side_effect=blocked) as operation:
                    workers = [
                        threading.Thread(target=invoke, args=(first,)),
                        threading.Thread(target=invoke, args=(second, True)),
                    ]
                    try:
                        workers[0].start()
                        self.assertTrue(entered.wait(_COMPLETION_TIMEOUT))
                        workers[1].start()
                        self.assertTrue(second_started.wait(_COMPLETION_TIMEOUT))
                        self.assertFalse(second_done.is_set())
                        self.assertEqual(pending_at_boundary, [1])
                    finally:
                        release.set()
                        for worker in workers:
                            if worker.ident is not None:
                                worker.join(_COMPLETION_TIMEOUT)
                    self.assertTrue(all(not worker.is_alive() for worker in workers))
                    self.assertEqual(operation.call_count, 1)
                self.assertEqual(errors, [])
                self._assert_recorder_closed(recorder)
                stats = recorder.stats()
                self.assertEqual(stats["selected"], 1)
                self.assertEqual(stats["completed"], int(first == "finish"))
                self.assertEqual(stats["pending"], 0)
                self.assertEqual(stats["inflight_bytes"], 0)
                if first == "cancel":
                    self.assertEqual(stats["reserved_bytes"], 0)
                    self.assertEqual(stats["bytes_published"], 0)

    def test_close_during_output_snapshot_retains_reservation(self):
        context = self.context()
        recorder = context._recorder
        token = self.begin(context)
        before = recorder.stats()
        entered, release = threading.Event(), threading.Event()
        original = capture._snapshot
        errors = []

        def blocked(tensor):
            snapshot = original(tensor)
            entered.set()
            if not release.wait(_COMPLETION_TIMEOUT):
                raise AssertionError("test did not release output snapshot")
            return snapshot

        def finish():
            try:
                token.finish(self.output())
            except BaseException as error:
                errors.append(error)

        with mock.patch.object(capture, "_snapshot", side_effect=blocked):
            worker = threading.Thread(target=finish)
            try:
                worker.start()
                self.assertTrue(entered.wait(_COMPLETION_TIMEOUT))
                self.assertFalse(recorder.close(0.01))
                self.assertEqual(recorder.stats()["pending"], 1)
                self.assertEqual(
                    recorder.stats()["inflight_bytes"], before["inflight_bytes"]
                )
                self.assertEqual(
                    recorder.stats()["reserved_bytes"], before["reserved_bytes"]
                )
                self.assertTrue(token._tensors)
            finally:
                release.set()
                worker.join(_COMPLETION_TIMEOUT)
        self.assertFalse(worker.is_alive())
        self.assertEqual(errors, [])
        self._assert_recorder_closed(recorder)
        self.assertEqual(token._tensors, {})
        self.assertEqual(recorder.stats()["pending"], 0)
        self.assertEqual(recorder.stats()["inflight_bytes"], 0)
        self.assertEqual(recorder.stats()["reserved_bytes"], 0)
        self.assertEqual(self.manifests(), [])

    def test_reader_rejects_manifest_symlink_fifo_and_duplicate_keys(self):
        context = self.context()
        self.begin(context).finish(self.output())
        self._assert_recorder_closed(context._recorder)
        path = self.manifests()[0]
        original = path.read_text()
        alias = self.root / "alias-manifest"
        alias.mkdir()
        (alias / "manifest.json").symlink_to(path)
        with self.assertRaisesRegex(ValueError, "unsafe manifest path"):
            capture.load_capture(alias)
        for marker, replacement in (
            ('"complete":true', '"complete":false,"complete":true'),
            ('"dtype":"float32"', '"dtype":"float16","dtype":"float32"'),
        ):
            with self.subTest(marker=marker):
                self.assertIn(marker, original)
                path.write_text(original.replace(marker, replacement, 1))
                with self.assertRaisesRegex(ValueError, "duplicate JSON key"):
                    capture.load_capture(path)
        path.write_text(original)
        # A subprocess timeout keeps a missing O_NONBLOCK regression from hanging UT.
        fifo_dir = self.root / "fifo"
        fifo_dir.mkdir()
        fifo = fifo_dir / "manifest.json"
        os.mkfifo(fifo, 0o600)
        program = (
            "import sys\n"
            "from rtp_llm.utils.tensor_capture import load_capture\n"
            "try:\n"
            "    load_capture(sys.argv[1])\n"
            "except ValueError as error:\n"
            "    assert 'size/type' in str(error), str(error)\n"
            "else:\n"
            "    raise AssertionError('FIFO accepted')\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", program, str(fifo)],
            capture_output=True,
            text=True,
            timeout=20,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    @unittest.skipUnless(
        os.environ.get("RTP_LLM_TEST_TENSOR_CAPTURE_CUDA") == "1",
        "opt in with RTP_LLM_TEST_TENSOR_CAPTURE_CUDA=1",
    )
    def test_cuda_nondefault_stream_noncontiguous_owned_snapshot(self):
        self.assertTrue(
            torch.cuda.is_available(), "CUDA opt-in requires an available GPU"
        )
        context = self.context()
        producer = torch.cuda.Stream()
        self.assertNotEqual(
            producer.cuda_stream, torch.cuda.default_stream().cuda_stream
        )
        with torch.cuda.stream(producer):
            # Allocation, producer writes, snapshots and subsequent overwrites are
            # deliberately on the same nondefault stream, without a pre-copy sync.
            backing = torch.empty((2, 6), dtype=torch.float32, device="cuda")
            logits = backing[:, ::2]
            logits.fill_(7)
            logits[0, 2] = 99
            ranges = torch.empty((2, 4), dtype=torch.int32, device="cuda")
            starts, ends = ranges[:, 0], ranges[:, 2]
            starts.zero_()
            ends[0] = 2
            ends[1] = 3
            token = self.begin(context, logits=logits, row_starts=starts, row_ends=ends)
            self.assertIsNotNone(token)
            logits.fill_(101)
            ranges.fill_(102)
            output_backing = torch.empty((2, 4), dtype=torch.int32, device="cuda")
            output = output_backing[:, ::2]
            output[:, 0] = 1
            output[:, 1] = 0
            token.finish(output)
            output.fill_(103)
        producer.synchronize()
        self._assert_recorder_closed(context._recorder)
        manifest, tensors = capture.load_capture(self.manifests()[0])
        torch.testing.assert_close(
            tensors["logits"], torch.tensor([[7.0, 7.0, 0.0], [7.0, 7.0, 7.0]])
        )
        torch.testing.assert_close(
            tensors["row_starts"], torch.tensor([0, 0], dtype=torch.int32)
        )
        torch.testing.assert_close(
            tensors["row_ends"], torch.tensor([2, 3], dtype=torch.int32)
        )
        torch.testing.assert_close(
            tensors["indices"], torch.tensor([[1, 0], [1, 0]], dtype=torch.int32)
        )
        self.assertEqual(manifest["tensors"]["logits"]["original_stride"], [6, 2])
        self.assertEqual(manifest["tensors"]["indices"]["original_stride"], [4, 2])
        self.assertTrue(all(t.device.type == "cpu" for t in tensors.values()))
        self.assertEqual(
            logits.cpu().tolist(), [[101.0, 101.0, 101.0], [101.0, 101.0, 101.0]]
        )
        self.assertEqual(output.cpu().tolist(), [[103, 103], [103, 103]])

    def _mqa_context(self, identity=None, **config):
        return self.context(
            self.recorder(
                dict(self.config, ops=["indexer.mqa_logits", "indexer.topk"], **config)
            ),
            identity,
        )

    def _mqa_arguments(self, rows=2, columns=3, heads=2, dimension=128):
        # Sliced layouts exercise raw bits (including FP8 NaNs), source strides
        # and storage offsets without requiring an MQA kernel on the test host.
        q = (
            torch.arange(rows * heads * dimension * 2)
            .to(torch.uint8)
            .view(torch.float8_e4m3fn)
        )
        keys = (
            torch.arange(columns * dimension * 2)
            .to(torch.uint8)
            .view(torch.float8_e4m3fn)
        )
        scales = torch.arange(
            columns * (dimension // 128) * 2, dtype=torch.float32
        ).reshape(columns, -1)[:, ::2]
        return dict(
            layer=0,
            q_fp8=q.reshape(rows, heads, dimension * 2)[:, :, 1::2],
            k_fp8=keys.reshape(columns, dimension * 2)[:, 1::2],
            k_scale=scales[:, 0] if dimension == 128 else scales,
            weights=torch.arange(rows * heads * 2, dtype=torch.float32).reshape(
                rows, heads * 2
            )[:, ::2],
            row_starts=torch.zeros(rows, dtype=torch.int32),
            row_ends=torch.arange(columns - rows + 1, columns + 1, dtype=torch.int32),
            k=2,
            query_row_start=0,
            total_query_rows=rows,
        )

    def _assert_bits_equal(self, actual, expected):
        self.assertEqual(actual.dtype, expected.dtype)
        self.assertEqual(actual.shape, expected.shape)
        self.assertTrue(
            torch.equal(
                actual.contiguous().reshape(-1).view(torch.uint8),
                expected.contiguous().reshape(-1).view(torch.uint8),
            )
        )

    def test_mqa_config_and_no_topk_fallback(self):
        for ops in (
            ["indexer.mqa_logits"],
            ["indexer.topk", "indexer.mqa_logits"],
            ["indexer.mqa_logits", "indexer.topk", "indexer.topk"],
            ["indexer.mqa_topk", "indexer.topk"],
        ):
            with self.subTest(ops=ops), self.assertRaises(ValueError):
                self.recorder(dict(self.config, ops=ops))
        joint = self._mqa_context()
        self.assertTrue(joint.mqa_enabled)
        canonical = self.context(
            self.recorder(dict(self.config, ops=["indexer.mqa_topk"]))
        )
        self.assertTrue(canonical.mqa_enabled)
        combined = self.context(
            self.recorder(dict(self.config, ops=[_FLOAT_PAIR.name, "indexer.mqa_topk"]))
        )
        self.assertTrue(combined.mqa_enabled)
        self.assertTrue(combined.context.is_selected(_FLOAT_PAIR.name))
        topk = self.context()
        self.assertFalse(topk.mqa_enabled)
        with mock.patch.object(capture, "_snapshot") as snapshot:
            self.assertIsNone(self.begin(joint))
            self.assertIsNone(topk.begin_mqa(**self._mqa_arguments()))
            snapshot.assert_not_called()
        self.assertEqual(joint._recorder.stats()["selected"], 0)
        self.begin(topk).finish(self.output())
        self._assert_recorder_closed(topk._recorder)
        manifest, _ = capture.load_capture(self.manifests()[0])
        self.assertEqual(manifest["schema_version"], 1)
        self.assertEqual(manifest["op"], "indexer.topk")
        self.assertNotIn("mqa_parameters", manifest)
        identity = dict(self.identity, schema_version=2)
        self.assertIsNone(joint._recorder.for_execution(json.dumps(identity)))

    def test_mqa_eight_tensor_roundtrip_and_single_logits(self):
        for dimension in (128, 256):
            with self.subTest(dimension=dimension):
                context = self._mqa_context()
                inputs = self._mqa_arguments(dimension=dimension)
                expected = {
                    name: tensor.clone()
                    for name, tensor in inputs.items()
                    if isinstance(tensor, torch.Tensor)
                }
                token = context.begin_mqa(**inputs)
                self.assertIsNotNone(token)
                # MQA input ownership is established before any producer reuse.
                for name, tensor in inputs.items():
                    if isinstance(tensor, torch.Tensor):
                        if tensor.dtype == torch.float8_e4m3fn:
                            tensor.view(torch.uint8).zero_()
                        else:
                            tensor.zero_()
                logits = torch.tensor(
                    [[float("nan"), -2.0, float("nan")], [-3.0, -4.0, -5.0]]
                )
                untouched = logits.clone()
                parameters = {"force_radix_sort": False, "stride0": 3, "stride1": 1}
                self.assertTrue(
                    token.prepare_topk(
                        logits, backend="dsv4_per_row", parameters=parameters
                    )
                )
                torch.testing.assert_close(logits, untouched, equal_nan=True)
                parameters["force_radix_sort"] = True
                expected["logits"] = untouched
                expected["logits"][0, 2] = 0
                logits.fill_(999)
                output = torch.empty((2, 4), dtype=torch.int32)[:, 1::2]
                output.copy_(self.output())
                expected["indices"] = output.clone()
                token.finish(output)
                output.fill_(999)
                self._assert_recorder_closed(context._recorder)
                path = next(context._recorder._sink.glob("call-*/manifest.json"))
                manifest, tensors = capture.load_capture(path)
                self.assertEqual(set(tensors), set(expected))
                self.assertEqual(len(tensors), 8)
                for name, tensor in tensors.items():
                    self._assert_bits_equal(tensor, expected[name])
                self.assertEqual(manifest["schema_version"], 2)
                self.assertEqual(manifest["op"], "indexer.mqa_topk")
                self.assertEqual(manifest["backend"], "dsv4_per_row")
                self.assertEqual(manifest["parameters"]["k"], 2)
                self.assertFalse(manifest["parameters"]["force_radix_sort"])
                self.assertEqual(
                    manifest["mqa_parameters"],
                    dict(
                        backend="deep_gemm.fp8_mqa_logits",
                        clean_logits=False,
                        weights_include_query_scale=True,
                        key_scale_format="float32_per_128_channels",
                    ),
                )
                self.assertEqual(manifest["operations"][0]["outputs"], ["logits"])
                self.assertIn("logits", manifest["operations"][1]["inputs"])
                self.assertEqual(len(list(path.parent.glob("*.bin"))), 8)
                self.assertEqual(len(list(path.parent.glob("*logits*.bin"))), 1)
                self.assertEqual(manifest["tensors"]["q_fp8"]["storage_offset"], 1)
                self.assertEqual(
                    manifest["tensors"]["q_fp8"]["original_stride"],
                    list(inputs["q_fp8"].stride()),
                )
                self.assertEqual(context._recorder.stats()["completed"], 1)

    def test_mqa_prefix_mapping_and_full_keys_across_chunk(self):
        identity = copy.deepcopy(self.identity)
        identity["requests"].append(
            dict(phase="prefill", request_id="r2", q_len=3, kv_len=2)
        )
        context = self._mqa_context(identity)
        inputs = self._mqa_arguments(rows=3, columns=8)
        inputs.update(
            query_row_start=1,
            total_query_rows=5,
            row_starts=torch.tensor([0, 3, 3], dtype=torch.int32),
            row_ends=torch.tensor([3, 6, 7], dtype=torch.int32),
        )
        for replacement in (
            {"k_fp8": inputs["k_fp8"][:7], "k_scale": inputs["k_scale"][:7]},
            {"row_ends": torch.tensor([3, 4, 5], dtype=torch.int32)},
            {"total_query_rows": 3},
        ):
            self.assertIsNone(context.begin_mqa(**dict(inputs, **replacement)))
        token = context.begin_mqa(**inputs)
        self.assertIsNotNone(token)
        logits = torch.arange(24, dtype=torch.float32).reshape(3, 8)
        self.assertTrue(
            token.prepare_topk(
                logits, backend="topk_v3_tie_break", parameters={"max_seq_len": 5}
            )
        )
        token.finish(torch.zeros(3, 2, dtype=torch.int32))
        self._assert_recorder_closed(context._recorder)
        manifest, tensors = capture.load_capture(self.manifests()[0])
        self.assertEqual(manifest["query_row_start"], 1)
        self.assertEqual(manifest["total_query_rows"], 5)
        self.assertEqual(tensors["k_fp8"].shape, (8, 128))
        expected = logits.clone()
        expected[0, 3:] = 0
        expected[1, :3] = expected[1, 6:] = 0
        expected[2, :3] = expected[2, 7:] = 0
        torch.testing.assert_close(tensors["logits"], expected)
        self.assertEqual(
            context._recorder.stats()["skip_reasons"]["invalid_mqa_contract"], 3
        )

    def test_mqa_contract_validation(self):
        context = self._mqa_context()
        inputs = self._mqa_arguments()
        changes = [
            {"q_fp8": None},
            {"q_fp8": torch.empty(2, 2, 128)},
            {"q_fp8": torch.empty(2, 128, dtype=torch.float8_e4m3fn)},
            {"q_fp8": torch.empty(2, 0, 128, dtype=torch.float8_e4m3fn)},
            {"q_fp8": torch.empty(0, 2, 128, dtype=torch.float8_e4m3fn)},
            {"q_fp8": torch.empty(2, 2, 127, dtype=torch.float8_e4m3fn)},
            {"q_fp8": torch.empty(2, 2, 129, dtype=torch.float8_e4m3fn)},
            {"k_fp8": None},
            {"k_fp8": torch.empty(3, 128, dtype=torch.float8_e5m2)},
            {"k_fp8": torch.empty(3, 256, dtype=torch.float8_e4m3fn)},
            {"k_fp8": torch.empty(1, 3, 128, dtype=torch.float8_e4m3fn)},
            {"k_fp8": inputs["k_fp8"][:2]},
            {"k_scale": None},
            {"k_scale": torch.ones(3, dtype=torch.float16)},
            {"k_scale": torch.ones(3, 1)},
            {"k_scale": torch.ones(2)},
            {"weights": torch.ones(2, 2, dtype=torch.float16)},
            {"weights": torch.ones(2, 1)},
            {"weights": torch.ones(1, 2)},
            {"weights": torch.ones(2, 2, device="meta")},
            {"weights": torch.ones(2, 2).to_sparse()},
            {"row_starts": torch.zeros(2, dtype=torch.int64)},
            {"row_starts": torch.tensor([0, 1], dtype=torch.int32)},
            {"row_ends": torch.tensor([2, 4], dtype=torch.int32)},
            {"row_ends": torch.tensor([2], dtype=torch.int32)},
            {"query_row_start": -1},
            {"query_row_start": 1},
            {"total_query_rows": 3},
            {"k": 0},
            {"k": True},
            {"k": -1},
        ]
        for change in changes:
            with self.subTest(fields=list(change)):
                self.assertIsNone(context.begin_mqa(**dict(inputs, **change)))
                self.assertEqual(context._recorder.stats()["pending"], 0)
                self.assertEqual(context._recorder.stats()["inflight_bytes"], 0)
                self.assertEqual(context._recorder.stats()["reserved_bytes"], 0)
        self.assertEqual(
            context._recorder.stats()["skip_reasons"]["invalid_mqa_contract"],
            len(changes),
        )
        larger = self._mqa_arguments(dimension=256)
        for scale in (torch.ones(3), torch.ones(3, 1), torch.ones(3, 3)):
            self.assertIsNone(context.begin_mqa(**dict(larger, k_scale=scale)))
        self.assertIsNone(context.begin_mqa(**dict(inputs, layer=8)))
        self.assertEqual(self.manifests(), [])

    def test_mqa_complete_payload_and_quota_before_copy(self):
        context = self._mqa_context()
        inputs = self._mqa_arguments()
        payload = (
            sum(
                t.numel() * t.element_size()
                for t in inputs.values()
                if isinstance(t, torch.Tensor)
            )
            + 2 * (3 + 2) * 4
        )
        with (
            mock.patch.object(capture, "_MAX_FILE_BYTES", payload - 1),
            mock.patch.object(context._recorder, "_reserve") as reserve,
            mock.patch.object(capture, "_snapshot") as snapshot,
        ):
            self.assertIsNone(context.begin_mqa(**inputs))
            self.assertIsNone(self.begin(context))
            reserve.assert_not_called()
            snapshot.assert_not_called()
        with mock.patch.object(capture, "_MAX_FILE_BYTES", payload):
            token = context.begin_mqa(**inputs)
            self.assertIsNotNone(token)
            size = context._recorder.stats()["reserved_bytes"]
            self.assertEqual(
                context._recorder.stats()["inflight_bytes"],
                size * capture._INFLIGHT_FACTOR,
            )
            # Exercise the maximum number of long, bounded backend scalars.
            parameters = {("p%d" % i).ljust(128, "x"): "v" * 128 for i in range(32)}
            self.assertTrue(
                token.prepare_topk(
                    torch.ones(2, 3), backend="dsv4_per_row", parameters=parameters
                )
            )
            token.finish(self.output())
            self._assert_recorder_closed(context._recorder)
            manifest, _ = capture.load_capture(self.manifests()[0])
            self.assertEqual(
                sum(d["nbytes"] for d in manifest["tensors"].values()), payload
            )
        for field, budget, reason in (
            ("max_bundle_bytes", size - 1, "oversize"),
            ("max_total_bytes", size - 1, "total_budget"),
            (
                "max_inflight_bytes",
                size * capture._INFLIGHT_FACTOR - 1,
                "inflight_budget",
            ),
        ):
            other = self._mqa_context(**{field: budget})
            with mock.patch.object(capture, "_snapshot") as snapshot:
                self.assertIsNone(other.begin_mqa(**inputs))
                snapshot.assert_not_called()
            self.assertEqual(other._recorder.stats()["skip_reasons"][reason], 1)
            self.assertEqual(other._recorder.stats()["selected"], 0)

    def test_mqa_budget_includes_all_prefix_keys_not_just_query_chunk(self):
        identity = copy.deepcopy(self.identity)
        identity["requests"][0].update(q_len=1, kv_len=256)
        context = self._mqa_context(identity)
        inputs = self._mqa_arguments(rows=1, columns=257, heads=1)
        input_bytes = sum(
            t.numel() * t.element_size()
            for t in inputs.values()
            if isinstance(t, torch.Tensor)
        )
        # This would fit if K were incorrectly trimmed to the single query row.
        truncated_budget = input_bytes - inputs["k_fp8"].numel() + 128 + (257 + 2) * 4
        with (
            mock.patch.object(capture, "_MAX_FILE_BYTES", truncated_budget),
            mock.patch.object(capture, "_snapshot") as snapshot,
        ):
            self.assertIsNone(context.begin_mqa(**inputs))
            snapshot.assert_not_called()
        self.assertEqual(context._recorder.stats()["selected"], 0)
        self.assertEqual(context._recorder.stats()["skip_reasons"]["payload_limit"], 1)

    def test_mqa_snapshot_failures_cancel_whole_group(self):
        inputs = self._mqa_arguments()
        context = self._mqa_context()
        original = capture._snapshot

        def fail_fp8(tensor):
            if tensor.dtype == torch.float8_e4m3fn:
                raise RuntimeError("copy failed")
            return original(tensor)

        with mock.patch.object(capture, "_snapshot", side_effect=fail_fp8):
            self.assertIsNone(context.begin_mqa(**inputs))
        self.assertEqual(
            context._recorder.stats()["skip_reasons"]["input_snapshot_error"], 1
        )
        token = context.begin_mqa(**inputs)
        self.assertIsNotNone(token)
        with mock.patch.object(
            capture, "_snapshot", side_effect=RuntimeError("logits copy failed")
        ):
            self.assertFalse(
                token.prepare_topk(
                    torch.ones(2, 3), backend="dsv4_per_row", parameters={}
                )
            )
        self.assertEqual(
            context._recorder.stats()["skip_reasons"]["logits_snapshot_error"], 1
        )
        token.finish(self.output())
        token.cancel()
        self.assertIsNone(self.begin(context))
        self.assertEqual(context._recorder.stats()["pending"], 0)
        self.assertEqual(context._recorder.stats()["reserved_bytes"], 0)
        self.assertEqual(self.manifests(), [])

    def test_mqa_prepare_contract_and_missing_prepare(self):
        context = self._mqa_context()
        inputs = self._mqa_arguments()
        for kwargs in (
            {"logits": torch.ones(2, 4)},
            {"logits": torch.ones(1, 3)},
            {"logits": torch.ones(2, 3, dtype=torch.float16)},
            {"logits": None},
            {"logits": torch.ones(2, 3, device="meta")},
            {"backend": "../bad"},
            {"parameters": {"k": 3}},
            {"parameters": {"index_space": "absolute"}},
        ):
            token = context.begin_mqa(**inputs)
            args = dict(logits=torch.ones(2, 3), backend="dsv4_per_row", parameters={})
            args.update(kwargs)
            with mock.patch.object(capture, "_snapshot") as snapshot:
                self.assertFalse(token.prepare_topk(**args))
                snapshot.assert_not_called()
            token.finish(self.output())
            token.cancel()
            self.assertEqual(context._recorder.stats()["pending"], 0)
        token = context.begin_mqa(**inputs)
        with mock.patch.object(capture, "_snapshot") as snapshot:
            token.finish(self.output())
            snapshot.assert_not_called()
        self.assertFalse(
            token.prepare_topk(torch.ones(2, 3), backend="dsv4_per_row", parameters={})
        )
        token = context.begin_mqa(**inputs)
        self.assertTrue(
            token.prepare_topk(torch.ones(2, 3), backend="dsv4_per_row", parameters={})
        )
        with mock.patch.object(capture, "_snapshot") as snapshot:
            self.assertFalse(
                token.prepare_topk(
                    torch.ones(2, 3), backend="dsv4_per_row", parameters={}
                )
            )
            snapshot.assert_not_called()
        token.finish(self.output())
        token = context.begin_mqa(**inputs)
        token.cancel()
        self.assertFalse(
            token.prepare_topk(torch.ones(2, 3), backend="dsv4_per_row", parameters={})
        )
        self.assertEqual(context._recorder.stats()["pending"], 0)
        self.assertEqual(context._recorder.stats()["reserved_bytes"], 0)
        self.assertEqual(self.manifests(), [])

    def test_mqa_invalid_output_and_late_calls_do_not_republish(self):
        context = self._mqa_context()
        for output in (
            torch.ones(2, 2),
            torch.ones(2, 1, dtype=torch.int32),
            torch.ones(2, 2, dtype=torch.int32, device="meta"),
        ):
            token = context.begin_mqa(**self._mqa_arguments())
            self.assertTrue(
                token.prepare_topk(
                    torch.ones(2, 3), backend="dsv4_per_row", parameters={}
                )
            )
            with mock.patch.object(capture, "_snapshot") as snapshot:
                token.finish(output)
                snapshot.assert_not_called()
            self.assertEqual(context._recorder.stats()["pending"], 0)
        self.assertEqual(self.manifests(), [])
        token = context.begin_mqa(**self._mqa_arguments())
        self.assertTrue(
            token.prepare_topk(torch.ones(2, 3), backend="dsv4_per_row", parameters={})
        )
        token.finish(self.output())
        self.assertFalse(
            token.prepare_topk(torch.ones(2, 3), backend="dsv4_per_row", parameters={})
        )
        token.finish(self.output())
        token.cancel()
        self._assert_recorder_closed(context._recorder)
        self.assertEqual(context._recorder.stats()["completed"], 1)
        self.assertEqual(len(self.manifests()), 1)

    def test_mqa_v2_loader_retains_hash_path_and_aggregate_limits(self):
        context = self._mqa_context()
        token = context.begin_mqa(**self._mqa_arguments())
        self.assertTrue(
            token.prepare_topk(torch.ones(2, 3), backend="dsv4_per_row", parameters={})
        )
        token.finish(self.output())
        self._assert_recorder_closed(context._recorder)
        path = self.manifests()[0]
        manifest, _ = capture.load_capture(path)
        original = path.read_bytes()
        manifest["tensors"]["q_fp8"]["file"] = "../q_fp8.bin"
        path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "unsafe tensor filename"):
            capture.load_capture(path)
        path.write_bytes(original)
        raw_path = path.parent / "q_fp8.bin"
        data = raw_path.read_bytes()
        raw_path.write_bytes(bytes([data[0] ^ 1]) + data[1:])
        with self.assertRaisesRegex(ValueError, "checksum"):
            capture.load_capture(path)
        raw_path.write_bytes(data)
        payload = sum(desc["nbytes"] for desc in manifest["tensors"].values())
        with mock.patch.object(capture, "_MAX_FILE_BYTES", payload - 1):
            with self.assertRaisesRegex(ValueError, "capture exceeds loader limit"):
                capture.load_capture(path)
        with mock.patch.object(capture, "_MAX_FILE_BYTES", payload):
            loaded, tensors = capture.load_capture(path)
            self.assertEqual(loaded["schema_version"], 2)
            self.assertEqual(len(tensors), 8)

    def test_mqa_prepare_serializes_with_prepare_finish_cancel(self):
        for competing in ("prepare", "finish", "cancel"):
            with self.subTest(competing=competing):
                context = self._mqa_context()
                token = context.begin_mqa(**self._mqa_arguments())
                entered, release = threading.Event(), threading.Event()
                started, done = threading.Event(), threading.Event()
                original = capture._snapshot
                logits = torch.ones(2, 3)
                errors, prepared, competing_results = [], [], []

                def blocked(tensor):
                    if tensor is logits:
                        entered.set()
                        if not release.wait(_COMPLETION_TIMEOUT):
                            raise AssertionError("test did not release logits copy")
                    return original(tensor)

                def first():
                    try:
                        prepared.append(
                            token.prepare_topk(
                                logits, backend="dsv4_per_row", parameters={}
                            )
                        )
                    except BaseException as error:
                        errors.append(error)

                def second():
                    started.set()
                    try:
                        if competing == "prepare":
                            competing_results.append(
                                token.prepare_topk(
                                    logits, backend="dsv4_per_row", parameters={}
                                )
                            )
                        elif competing == "finish":
                            token.finish(self.output())
                        else:
                            token.cancel()
                    except BaseException as error:
                        errors.append(error)
                    finally:
                        done.set()

                with mock.patch.object(
                    capture, "_snapshot", side_effect=blocked
                ) as snapshot:
                    workers = [
                        threading.Thread(target=first),
                        threading.Thread(target=second),
                    ]
                    try:
                        workers[0].start()
                        self.assertTrue(entered.wait(_COMPLETION_TIMEOUT))
                        workers[1].start()
                        self.assertTrue(started.wait(_COMPLETION_TIMEOUT))
                        self.assertFalse(done.is_set())
                        self.assertEqual(context._recorder.stats()["pending"], 1)
                    finally:
                        release.set()
                        for worker in workers:
                            if worker.ident is not None:
                                worker.join(_COMPLETION_TIMEOUT)
                    self.assertTrue(all(not w.is_alive() for w in workers))
                    self.assertEqual(
                        snapshot.call_count, 2 if competing == "finish" else 1
                    )
                self.assertEqual(errors, [])
                self.assertEqual(prepared, [True])
                if competing == "prepare":
                    self.assertEqual(competing_results, [False])
                self._assert_recorder_closed(context._recorder)
                self.assertEqual(
                    context._recorder.stats()["completed"], int(competing == "finish")
                )

    def test_close_during_mqa_prepare_preserves_copy_reservation(self):
        context = self._mqa_context()
        token = context.begin_mqa(**self._mqa_arguments())
        entered, release = threading.Event(), threading.Event()
        before = context._recorder.stats()
        original = capture._snapshot
        results = []

        def blocked(tensor):
            snapshot = original(tensor)
            entered.set()
            if not release.wait(_COMPLETION_TIMEOUT):
                raise AssertionError("test did not release logits copy")
            return snapshot

        with mock.patch.object(capture, "_snapshot", side_effect=blocked):
            worker = threading.Thread(
                target=lambda: results.append(
                    token.prepare_topk(
                        torch.ones(2, 3), backend="dsv4_per_row", parameters={}
                    )
                )
            )
            try:
                worker.start()
                self.assertTrue(entered.wait(_COMPLETION_TIMEOUT))
                self.assertFalse(context._recorder.close(0.01))
                self.assertEqual(
                    context._recorder.stats()["inflight_bytes"],
                    before["inflight_bytes"],
                )
                self.assertEqual(context._recorder.stats()["pending"], 1)
            finally:
                release.set()
                worker.join(_COMPLETION_TIMEOUT)
        self.assertFalse(worker.is_alive())
        self.assertEqual(results, [False])
        self._assert_recorder_closed(context._recorder)
        self.assertEqual(context._recorder.stats()["reserved_bytes"], 0)
        self.assertEqual(self.manifests(), [])

    @unittest.skipUnless(
        os.environ.get("RTP_LLM_TEST_TENSOR_CAPTURE_CUDA") == "1",
        "opt in with RTP_LLM_TEST_TENSOR_CAPTURE_CUDA=1",
    )
    def test_cuda_mqa_joint_snapshot_on_nondefault_producer_stream(self):
        self.assertTrue(
            torch.cuda.is_available(), "CUDA opt-in requires an available GPU"
        )
        context = self._mqa_context()
        inputs = self._mqa_arguments()
        expected = {
            name: tensor.clone()
            for name, tensor in inputs.items()
            if isinstance(tensor, torch.Tensor)
        }
        producer = torch.cuda.Stream()
        with torch.cuda.stream(producer):
            for name, tensor in expected.items():
                shape = list(tensor.shape)
                shape[-1] *= 2
                gpu = torch.empty(shape, dtype=tensor.dtype, device="cuda")[..., ::2]
                gpu.copy_(tensor)
                inputs[name] = gpu
            token = context.begin_mqa(**inputs)
            self.assertIsNotNone(token)
            for name in expected:
                tensor = inputs[name]
                if tensor.dtype == torch.float8_e4m3fn:
                    tensor.view(torch.uint8).zero_()
                else:
                    tensor.zero_()
            logits = torch.empty(2, 6, dtype=torch.float32, device="cuda")[:, ::2]
            logits.fill_(7)
            self.assertTrue(
                token.prepare_topk(logits, backend="dsv4_per_row", parameters={})
            )
            logits.fill_(101)
            output = torch.empty(2, 4, dtype=torch.int32, device="cuda")[:, ::2]
            output.zero_()
            token.finish(output)
            output.fill_(102)
        producer.synchronize()
        self._assert_recorder_closed(context._recorder)
        manifest, tensors = capture.load_capture(self.manifests()[0])
        expected["logits"] = torch.tensor([[7.0, 7.0, 0.0], [7.0, 7.0, 7.0]])
        expected["indices"] = torch.zeros(2, 2, dtype=torch.int32)
        self.assertEqual(manifest["schema_version"], 2)
        self.assertEqual(set(tensors), set(expected))
        for name, tensor in tensors.items():
            self._assert_bits_equal(tensor, expected[name])
            self.assertEqual(manifest["tensors"][name]["original_stride"][-1], 2)
        self.assertEqual(context._recorder.stats()["completed"], 1)

    def _assert_capture_interrupt(
        self, recorder, operation, interruption, snapshots, release_count=1
    ):
        release = recorder._release

        def checked_release(*args, **kwargs):
            # The propagated traceback must not retain a temporary CPU copy
            # after the quota has become available to the next capture.
            self.assertTrue(all(reference() is None for reference in snapshots))
            return release(*args, **kwargs)

        with mock.patch.object(
            recorder, "_release", side_effect=checked_release
        ) as released:
            try:
                operation()
            except BaseException as caught:
                self.assertIs(caught, interruption)
                self.assertIsNotNone(caught.__traceback__)
            else:
                self.fail("capture swallowed the interruption")
            self.assertEqual(released.call_count, release_count)
        stats = recorder.stats()
        self.assertEqual(stats["pending"], 0)
        self.assertEqual(stats["inflight_bytes"], 0)
        self.assertEqual(stats["reserved_bytes"], 0)
        self.assertEqual(stats["completed"], 0)
        self.assertEqual(stats["bytes_published"], 0)
        self.assertTrue(all(reference() is None for reference in snapshots))
        self._assert_recorder_closed(recorder)
        self.assertEqual(self.manifests(), [])

    def test_begin_baseexception_releases_before_propagation(self):
        for error_type in (KeyboardInterrupt, SystemExit):
            for joint in (False, True):
                for stage in (
                    "descriptor",
                    "reserve",
                    "before_copy",
                    "after_copy",
                    "last_copy",
                    "manifest",
                    "token_allocation",
                ):
                    with self.subTest(
                        error=error_type.__name__, joint=joint, stage=stage
                    ):
                        context = self._mqa_context() if joint else self.context()
                        recorder = context._recorder
                        arguments = self._mqa_arguments() if joint else None
                        operation = (
                            (lambda: context.begin_mqa(**arguments))
                            if joint
                            else (lambda: self.begin(context))
                        )
                        interruption = error_type("capture interrupted")
                        snapshots = []
                        original_snapshot = capture._snapshot
                        calls = 0

                        def snapshot(tensor):
                            nonlocal calls
                            calls += 1
                            if stage == "before_copy":
                                raise interruption
                            owned = original_snapshot(tensor)
                            snapshots.append(weakref.ref(owned))
                            if stage == "after_copy" or (
                                stage == "last_copy" and calls == (6 if joint else 3)
                            ):
                                raise interruption
                            return owned

                        target, name = {
                            "descriptor": (capture, "_descriptor"),
                            "reserve": (recorder, "_reserve"),
                            "manifest": (context.context, "_manifest"),
                            "token_allocation": (capture.CaptureToken, "__init__"),
                        }.get(stage, (None, None))
                        injection = (
                            mock.patch.object(target, name, side_effect=interruption)
                            if target is not None
                            else nullcontext()
                        )
                        with (
                            mock.patch.object(
                                capture, "_snapshot", side_effect=snapshot
                            ),
                            injection,
                        ):
                            self._assert_capture_interrupt(
                                recorder,
                                operation,
                                interruption,
                                snapshots,
                                release_count=int(
                                    stage not in ("descriptor", "reserve")
                                ),
                            )
                        self.assertEqual(
                            recorder.stats()["selected"],
                            int(stage not in ("descriptor", "reserve")),
                        )

    def test_token_prepare_and_finish_baseexception_cleanup(self):
        for error_type in (KeyboardInterrupt, SystemExit):
            for action in ("prepare", "topk_finish", "joint_finish"):
                for stage in (
                    "descriptor",
                    "before_copy",
                    "after_copy",
                    "after_snapshot",
                ):
                    with self.subTest(
                        error=error_type.__name__, action=action, stage=stage
                    ):
                        context = (
                            self.context()
                            if action == "topk_finish"
                            else self._mqa_context()
                        )
                        token = (
                            self.begin(context)
                            if action == "topk_finish"
                            else context.begin_mqa(**self._mqa_arguments())
                        )
                        logits = torch.ones(2, 3)
                        if action == "joint_finish":
                            self.assertTrue(
                                token.prepare_topk(
                                    logits, backend="dsv4_per_row", parameters={}
                                )
                            )
                        snapshots = [
                            weakref.ref(tensor) for tensor in token._tensors.values()
                        ]
                        interruption = error_type("token interrupted")
                        original_snapshot = capture._snapshot
                        original_logits = indexer_capture._snapshot_logits

                        def snapshot(tensor):
                            if stage == "before_copy":
                                raise interruption
                            owned = original_snapshot(tensor)
                            snapshots.append(weakref.ref(owned))
                            if stage == "after_copy":
                                raise interruption
                            return owned

                        def after_logits(*args):
                            original_logits(*args)
                            raise interruption

                        if stage == "descriptor":
                            target, name, effect = capture, "_descriptor", interruption
                        elif stage == "after_snapshot" and action == "prepare":
                            target, name, effect = (
                                indexer_capture,
                                "_snapshot_logits",
                                after_logits,
                            )
                        elif stage == "after_snapshot":
                            target, name, effect = (
                                context._recorder,
                                "_enqueue",
                                interruption,
                            )
                        else:
                            target, name, effect = None, None, None
                        operation = (
                            (
                                lambda: token.prepare_topk(
                                    logits, backend="dsv4_per_row", parameters={}
                                )
                            )
                            if action == "prepare"
                            else (lambda: token.finish(self.output()))
                        )
                        injection = (
                            mock.patch.object(target, name, side_effect=effect)
                            if target is not None
                            else nullcontext()
                        )
                        with (
                            mock.patch.object(
                                capture, "_snapshot", side_effect=snapshot
                            ),
                            injection,
                        ):
                            self._assert_capture_interrupt(
                                context._recorder, operation, interruption, snapshots
                            )
                        self.assertTrue(token._done)
                        self.assertEqual(token._tensors, {})
                        self.assertIsNone(token._manifest)
                        token.cancel()
                        self.assertEqual(context._recorder.stats()["pending"], 0)

    def test_cancel_accounting_interrupt_cannot_leak_or_double_release(self):
        for error_type in (KeyboardInterrupt, SystemExit):
            for action in ("cancel", "unprepared_finish", "closed_prepare"):
                with self.subTest(error=error_type.__name__, action=action):
                    context = self._mqa_context()
                    token = context.begin_mqa(**self._mqa_arguments())
                    snapshots = [
                        weakref.ref(tensor) for tensor in token._tensors.values()
                    ]
                    interruption = error_type("skip accounting interrupted")
                    if action == "closed_prepare":
                        self.assertFalse(context._recorder.close(0.01))

                    def operation():
                        if action == "closed_prepare":
                            return token.prepare_topk(
                                torch.ones(2, 3), backend="dsv4_per_row", parameters={}
                            )
                        if action == "unprepared_finish":
                            return token.finish(self.output())
                        return token.cancel()

                    with mock.patch.object(
                        context._recorder, "note_skip", side_effect=interruption
                    ):
                        self._assert_capture_interrupt(
                            context._recorder, operation, interruption, snapshots
                        )
                    token.cancel()
                    self.assertEqual(context._recorder.stats()["pending"], 0)

    def test_interrupted_token_construction_keeps_begin_as_owner(self):
        for error_type in (KeyboardInterrupt, SystemExit):
            with self.subTest(error=error_type.__name__):
                context = self._mqa_context()
                interruption = error_type("token lock allocation interrupted")
                initialize = capture.CaptureToken.__init__

                def interrupted_init(token, *args, **kwargs):
                    with mock.patch.object(
                        capture.threading, "Lock", side_effect=interruption
                    ):
                        initialize(token, *args, **kwargs)

                with (
                    mock.patch.object(
                        capture.CaptureToken, "__init__", interrupted_init
                    ),
                    mock.patch.object(sys, "unraisablehook") as unraisable,
                ):
                    self._assert_capture_interrupt(
                        context._recorder,
                        lambda: context.begin_mqa(**self._mqa_arguments()),
                        interruption,
                        [],
                    )
                    unraisable.assert_not_called()

    def _generic_context(self, **config):
        recorder = self.recorder(dict(self.config, ops=[_FLOAT_PAIR.name], **config))
        return recorder.for_execution(json.dumps(self.identity))

    def _generic_begin(self, context, **changes):
        arguments = dict(
            op=_FLOAT_PAIR.name,
            layer=0,
            inputs={"value": torch.ones(2)},
            outputs={
                name: capture.TensorSpec((2,), torch.float32)
                for name in ("positive", "negative")
            },
            backend="torch",
            parameters={},
        )
        arguments.update(changes)
        return context.begin(**arguments)

    def test_generic_real_float_multioutput_roundtrip(self):
        context = self._generic_context()
        value = torch.arange(12, dtype=torch.float64).reshape(2, 6)[:, 1::2]
        original = value.clone()
        outputs = _float_pair(context, value)
        expected = {name: tensor.clone() for name, tensor in outputs.items()}
        value.fill_(99)
        for tensor in outputs.values():
            tensor.fill_(100)
        self._assert_recorder_closed(context._recorder)
        manifest, tensors = capture.load_capture(self.manifests()[0])
        self.assertEqual(manifest["schema_version"], 3)
        self.assertEqual(manifest["contract"], {"name": _FLOAT_PAIR.name, "version": 1})
        self.assertEqual(manifest["op"], _FLOAT_PAIR.name)
        self.assertEqual(manifest["inputs"], ["value"])
        self.assertEqual(manifest["outputs"], ["positive", "negative"])
        self.assertEqual(manifest["parameters"], {"offset": 1})
        self.assertEqual(manifest["requests"], self.identity["requests"])
        self.assertNotIn("query_row_start", manifest)
        self.assertEqual(manifest["tensors"]["value"]["original_stride"], [6, 2])
        torch.testing.assert_close(tensors["value"], original)
        for name in expected:
            torch.testing.assert_close(tensors[name], expected[name])
        self.assertFalse(hasattr(context, "begin_topk"))
        self.assertFalse(hasattr(context, "begin_mqa"))

    def test_generic_unknown_op_and_selector_fail_closed(self):
        for ops in (
            ["unknown.capture"],
            [_FLOAT_PAIR.name, "unknown.capture"],
            [_FLOAT_PAIR.name] * 2,
        ):
            with self.subTest(ops=ops), self.assertRaises(ValueError):
                self.recorder(dict(self.config, ops=ops))
        context = self._generic_context()
        with mock.patch.object(capture, "_snapshot") as snapshot:
            with self.assertRaisesRegex(ValueError, "unknown"):
                self._generic_begin(context, op="unknown.capture")
            self.assertIsNone(self._generic_begin(context, op="indexer.topk"))
            self.assertIsNone(self._generic_begin(context, layer=100))
            snapshot.assert_not_called()
        self.assertEqual(context._recorder.stats()["selected"], 0)

    def test_generic_registry_duplicate_and_concurrent_registration(self):
        with self.assertRaisesRegex(ValueError, "duplicate"):
            capture.register_capture_contract(_FLOAT_PAIR)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            capture.register_capture_contract(capture.CaptureContract("indexer.topk"))
        name = "test.concurrent." + self.root.name
        results = []
        lock = threading.Lock()
        barrier = threading.Barrier(4)

        def register():
            barrier.wait(timeout=5)
            try:
                capture.register_capture_contract(capture.CaptureContract(name))
                result = "registered"
            except ValueError:
                result = "duplicate"
            with lock:
                results.append(result)

        threads = [threading.Thread(target=register) for _ in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(10)
            self.assertFalse(thread.is_alive())
        self.assertEqual(sorted(results), ["duplicate"] * 3 + ["registered"])
        for field in (
            "owner_id",
            "contract",
            "tensors",
            "parameters",
            "call_id",
            "max_calls",
        ):
            with self.subTest(field=field), self.assertRaises(ValueError):
                capture.register_capture_contract(
                    capture.CaptureContract(
                        "test.bad." + field, metadata_fields=frozenset({field})
                    )
                )

    def test_metadata_snapshot_is_bounded_before_encoding_or_reservation(self):
        initialized = threading.Event()
        initialize = capture.TensorCaptureRecorder._initialize_sink

        def initialize_sink(recorder):
            initialize(recorder)
            initialized.set()

        # Run metadata encoding must finish before the module-wide encoder mock.
        with mock.patch.object(
            capture.TensorCaptureRecorder, "_initialize_sink", new=initialize_sink
        ):
            context = self._generic_context()
            self.assertTrue(
                initialized.wait(_COMPLETION_TIMEOUT),
                self._writer_diagnostics(context._recorder),
            )
        cyclic = []
        cyclic.append(cyclic)
        deep = None
        for _ in range(66):
            deep = [deep]
        for value in (
            "x" * (capture._MAX_JSON + 1),
            "\U0001f600" * (capture._MAX_JSON // 12),
            [None] * capture._MAX_JSON,
            -(1 << (capture._MAX_JSON * 8)),
            cyclic,
            deep,
            float("inf"),
            object(),
        ):
            with (
                self.subTest(kind=type(value).__name__),
                mock.patch.object(
                    capture,
                    "_json_bytes",
                    side_effect=AssertionError("encoded oversized metadata"),
                ) as encoder,
                mock.patch.object(context._recorder, "_reserve") as reserve,
                mock.patch.object(capture, "_snapshot") as snapshot,
            ):
                self.assertIsNone(
                    self._generic_begin(context, metadata={"label": value})
                )
                encoder.assert_not_called()
                reserve.assert_not_called()
                snapshot.assert_not_called()
        self.assertEqual(context._recorder.stats()["pending"], 0)
        self.assertEqual(context._recorder.stats()["selected"], 0)

    def test_metadata_snapshot_exact_json_budget_and_owned_containers(self):
        values = (
            {},
            {"label": []},
            {"label": "ascii"},
            {"label": '"\\\b\f\n\r\t\x00\x1f\x7f\u00e9\u4e2d\U0001f600\ud800'},
            {"label": [None, True, False, -0.0, 1.25e100, -(2**63 - 1), 2**63 - 1]},
            {"label": {"nested": (1, 2, {"x": "y"})}},
        )
        for value in values:
            encoded = capture._json_bytes(value)
            with self.subTest(value=repr(value)):
                clean, size = capture._snapshot_metadata(value, len(encoded))
                self.assertEqual(size, len(encoded))
                self.assertEqual(capture._json_bytes(clean), encoded)
                with self.assertRaises(ValueError):
                    capture._snapshot_metadata(value, len(encoded) - 1)
        original = {"label": {"nested": [1, 2]}}
        clean, _ = capture._snapshot_metadata(original, capture._MAX_JSON)
        original["label"]["nested"].append(3)
        self.assertEqual(clean, {"label": {"nested": [1, 2]}})

    def test_token_subclass_constructor_failure_releases_reservation_once(self):
        for error_type in (RuntimeError, KeyboardInterrupt, SystemExit):
            with self.subTest(error=error_type.__name__):
                interruption = error_type("after base token construction")
                constructed = []

                class FailingToken(capture.CaptureToken):
                    def __init__(self, *args, **kwargs):
                        super().__init__(*args, **kwargs)
                        constructed.append(self)
                        raise interruption

                name = "test.constructor." + self.root.name + error_type.__name__
                capture.register_capture_contract(
                    capture.CaptureContract(name, token_type=FailingToken)
                )
                recorder = self.recorder(dict(self.config, ops=[name]))
                context = recorder.for_execution(json.dumps(self.identity))
                with mock.patch.object(
                    recorder, "_release", wraps=recorder._release
                ) as released:
                    if error_type is RuntimeError:
                        self.assertIsNone(self._generic_begin(context, op=name))
                    else:
                        with self.assertRaises(error_type) as raised:
                            self._generic_begin(context, op=name)
                        self.assertIs(raised.exception, interruption)
                    self.assertEqual(released.call_count, 1)
                    self.assertEqual(recorder.stats()["pending"], 0)
                    self.assertEqual(recorder.stats()["inflight_bytes"], 0)
                    self.assertEqual(recorder.stats()["reserved_bytes"], 0)
                    self.assertEqual(constructed[0]._tensors, {})
                    # Destructor and explicit cancellation both see begin's completed release.
                    constructed[0].__del__()
                    constructed[0].cancel()
                    constructed.clear()
                    self.assertEqual(released.call_count, 1)
                self._assert_recorder_closed(recorder)
        self.assertEqual(self.manifests(), [])

    def _interrupt_accounting_line(self, function, statement, operation, error):
        lines, start = inspect.getsourcelines(function)
        line = start + next(i for i, text in enumerate(lines) if statement in text)
        previous = sys.gettrace()

        def interrupt(frame, event, arg):
            if (
                frame.f_code is function.__code__
                and event == "line"
                and frame.f_lineno == line
            ):
                raise error
            return interrupt

        try:
            sys.settrace(interrupt)
            with self.assertRaises(type(error)) as raised:
                operation()
        finally:
            sys.settrace(previous)
        self.assertIs(raised.exception, error)

    def test_reservation_acquisition_interrupt_rolls_back_partial_accounting(self):
        for error_type in (KeyboardInterrupt, SystemExit):
            for statement in (
                'self._counts["selected"] +=',
                "self._reserved +=",
                "self._inflight +=",
                "self._pending +=",
                "reservation.active = True",
                "return call",
            ):
                with self.subTest(error=error_type.__name__, statement=statement):
                    recorder = self.recorder()
                    held = capture._CaptureReservation(recorder, 17, {})
                    recorder._reserve(held)
                    before = recorder.stats()
                    reservation = capture._CaptureReservation(recorder, 31, {})
                    try:
                        self._interrupt_accounting_line(
                            capture.TensorCaptureRecorder._reserve,
                            statement,
                            lambda: recorder._reserve(reservation),
                            error_type("partial acquisition"),
                        )
                        self.assertFalse(reservation.active)
                        self.assertFalse(reservation.release())
                        for key in (
                            "selected",
                            "reserved_bytes",
                            "inflight_bytes",
                            "pending",
                        ):
                            self.assertEqual(recorder.stats()[key], before[key])
                        self.assertEqual(
                            recorder._reserve(reservation), before["selected"]
                        )
                        self.assertTrue(reservation.release())
                        self.assertFalse(reservation.release())
                        self.assertTrue(held.release())
                        self.assertTrue(recorder.close(0))
                    finally:
                        # A regression must fail assertions, not strand teardown.
                        with recorder._condition:
                            recorder._reserved = recorder._inflight = (
                                recorder._pending
                            ) = 0

    def test_reservation_release_interrupt_completes_once(self):
        boundaries = (
            (capture._CaptureReservation.release, "self.tensors.clear()"),
            (capture._CaptureReservation.release, "self.manifest = None"),
            (capture._CaptureReservation.release, "recorder._release("),
            (capture._CaptureReservation.release, "self.active = False"),
            (capture._CaptureReservation.release, "return True"),
            (capture.TensorCaptureRecorder._release, "self._inflight -="),
            (capture.TensorCaptureRecorder._release, "self._pending -="),
            (capture.TensorCaptureRecorder._release, "self._reserved -="),
            (capture.TensorCaptureRecorder._release, "self._condition.notify_all()"),
        )
        for error_type in (KeyboardInterrupt, SystemExit):
            for retain_total in (False, True):
                for function, statement in boundaries:
                    if retain_total and statement == "self._reserved -=":
                        continue
                    with self.subTest(
                        error=error_type.__name__,
                        retain_total=retain_total,
                        function=function.__qualname__,
                        statement=statement,
                    ):
                        recorder = self.recorder()
                        held = capture._CaptureReservation(recorder, 17, {})
                        recorder._reserve(held)
                        reservation = capture._CaptureReservation(
                            recorder, 31, {"value": torch.ones(2)}
                        )
                        snapshot = weakref.ref(reservation.tensors["value"])
                        recorder._reserve(reservation)
                        reservation.queued = retain_total
                        reservation.manifest = {"complete": False}
                        try:
                            if retain_total:
                                self.assertFalse(reservation.release())
                                self.assertTrue(reservation.active)
                            self._interrupt_accounting_line(
                                function,
                                statement,
                                lambda: reservation.release(
                                    writer=retain_total, retain_total=retain_total
                                ),
                                error_type("partial refund"),
                            )
                            self.assertFalse(reservation.active)
                            self.assertIsNone(snapshot())
                            self.assertIsNone(reservation.manifest)
                            self.assertFalse(reservation.release(writer=True))
                            self.assertEqual(
                                recorder.stats()["reserved_bytes"],
                                17 + 31 * retain_total,
                            )
                            self.assertEqual(
                                recorder.stats()["inflight_bytes"],
                                17 * capture._INFLIGHT_FACTOR,
                            )
                            self.assertEqual(recorder.stats()["pending"], 1)
                            self.assertTrue(held.release())
                            self.assertTrue(recorder.close(0))
                        finally:
                            with recorder._condition:
                                recorder._reserved = recorder._inflight = (
                                    recorder._pending
                                ) = 0

    def test_interrupt_after_reservation_acquisition_is_released(self):
        context = self._generic_context()
        recorder = context._recorder
        reserve = recorder._reserve
        interruption = KeyboardInterrupt("after quota acquisition")

        def acquire_then_interrupt(*args):
            self.assertIsNotNone(reserve(*args))
            raise interruption

        with mock.patch.object(
            recorder, "_reserve", side_effect=acquire_then_interrupt
        ):
            self._assert_capture_interrupt(
                recorder, lambda: self._generic_begin(context), interruption, []
            )

    def test_enqueue_handoff_interrupt_preserves_writer_owned_bundle(self):
        for boundary in ("enqueue", "put_nowait", "_put"):
            for error_type in (RuntimeError, KeyboardInterrupt, SystemExit):
                with self.subTest(boundary=boundary, error=error_type.__name__):
                    context = self._generic_context()
                    recorder = context._recorder
                    token = self._generic_begin(context)
                    size = recorder.stats()["reserved_bytes"]
                    before = set(self.manifests())
                    entered, release = threading.Event(), threading.Event()
                    write = recorder._write_bundle
                    target = recorder if boundary == "enqueue" else recorder._queue
                    method = "_enqueue" if boundary == "enqueue" else boundary
                    enqueue = getattr(target, method)
                    interruption = error_type("after accepted enqueue")

                    def enqueue_then_interrupt(*args):
                        enqueue(*args)
                        raise interruption

                    def blocked_write(*args):
                        entered.set()
                        self.assertTrue(release.wait(_COMPLETION_TIMEOUT))
                        return write(*args)

                    outputs = {"positive": torch.ones(2), "negative": torch.zeros(2)}
                    with (
                        mock.patch.object(
                            target, method, side_effect=enqueue_then_interrupt
                        ),
                        mock.patch.object(
                            recorder, "_write_bundle", side_effect=blocked_write
                        ),
                        mock.patch.object(
                            recorder, "_release", wraps=recorder._release
                        ) as released,
                    ):
                        try:
                            if error_type is RuntimeError:
                                self.assertFalse(token.finish(outputs))
                            else:
                                with self.assertRaises(error_type) as raised:
                                    token.finish(outputs)
                                self.assertIs(raised.exception, interruption)
                            self.assertTrue(entered.wait(_COMPLETION_TIMEOUT))
                            self.assertTrue(token._done)
                            token.cancel()
                            self.assertEqual(recorder.stats()["pending"], 1)
                            self.assertEqual(
                                recorder.stats()["inflight_bytes"],
                                capture._INFLIGHT_FACTOR * size,
                            )
                            self.assertEqual(released.call_count, 0)
                            self.assertEqual(
                                set(token._reservation.tensors),
                                {"value", "positive", "negative"},
                            )
                        finally:
                            release.set()
                        self._assert_recorder_closed(recorder)
                        self.assertEqual(released.call_count, 1)
                    self.assertEqual(recorder.stats()["completed"], 1)
                    self.assertEqual(recorder.stats()["reserved_bytes"], size)
                    self.assertEqual(recorder.stats()["io_error"], 0)
                    paths = set(self.manifests()) - before
                    self.assertEqual(len(paths), 1)
                    _, tensors = capture.load_capture(paths.pop())
                    torch.testing.assert_close(tensors["value"], torch.ones(2))
                    for name, tensor in outputs.items():
                        torch.testing.assert_close(tensors[name], tensor)

    def test_enqueue_interrupt_before_insertion_refunds_reservation(self):
        for error_type in (RuntimeError, KeyboardInterrupt, SystemExit):
            with self.subTest(error=error_type.__name__):
                context = self._generic_context()
                recorder = context._recorder
                token = self._generic_begin(context)
                interruption = error_type("before queue insertion")
                outputs = {"positive": torch.ones(2), "negative": torch.zeros(2)}
                with mock.patch.object(
                    recorder._queue, "put_nowait", side_effect=interruption
                ):
                    if error_type is RuntimeError:
                        self.assertFalse(token.finish(outputs))
                    else:
                        with self.assertRaises(error_type) as raised:
                            token.finish(outputs)
                        self.assertIs(raised.exception, interruption)
                self.assertFalse(token._reservation.queued)
                token.cancel()
                self._assert_recorder_closed(recorder)
                self.assertEqual(recorder.stats()["reserved_bytes"], 0)
        self.assertEqual(self.manifests(), [])

    def test_enqueue_interrupt_after_writer_completion_does_not_refund_total(self):
        context = self._generic_context()
        recorder = context._recorder
        token = self._generic_begin(context)
        size = recorder.stats()["reserved_bytes"]
        enqueue = recorder._enqueue
        interruption = KeyboardInterrupt("writer already completed")

        def enqueue_then_interrupt(*args):
            self.assertTrue(enqueue(*args))
            self._assert_recorder_drained(recorder)
            raise interruption

        with (
            mock.patch.object(recorder, "_enqueue", side_effect=enqueue_then_interrupt),
            mock.patch.object(
                recorder, "_release", wraps=recorder._release
            ) as released,
        ):
            with self.assertRaises(KeyboardInterrupt) as raised:
                token.finish({"positive": torch.ones(2), "negative": torch.zeros(2)})
            self.assertIs(raised.exception, interruption)
            token.cancel()
            self._assert_recorder_closed(recorder)
            self.assertEqual(released.call_count, 1)
        self.assertEqual(recorder.stats()["completed"], 1)
        self.assertEqual(recorder.stats()["reserved_bytes"], size)
        self.assertEqual(recorder.stats()["inflight_bytes"], 0)
        _, tensors = capture.load_capture(self.manifests()[0])
        self.assertEqual(set(tensors), {"value", "positive", "negative"})

    def test_generic_metadata_is_owned_and_cannot_override_envelope(self):
        context = self._generic_context()
        for field in (
            "exec_id",
            "schema_version",
            "contract",
            "tensors",
            "parameters",
            "max_total_bytes",
            "undeclared",
        ):
            with (
                self.subTest(field=field),
                mock.patch.object(capture, "_snapshot") as snapshot,
            ):
                self.assertIsNone(self._generic_begin(context, metadata={field: "bad"}))
                snapshot.assert_not_called()
        metadata, parameters = {"label": {"case": "original"}}, {"offset": 1}
        token = self._generic_begin(context, metadata=metadata, parameters=parameters)
        metadata["label"]["case"] = "changed"
        parameters["offset"] = 999
        token.finish({"positive": torch.ones(2), "negative": torch.zeros(2)})
        self._assert_recorder_closed(context._recorder)
        manifest, _ = capture.load_capture(self.manifests()[0])
        self.assertEqual(manifest["label"], {"case": "original"})
        self.assertEqual(manifest["parameters"], {"offset": 1})
        self.assertEqual(manifest["exec_id"], self.identity["exec_id"])

    def test_generic_staged_outputs_publish_only_complete_bundle(self):
        context = self._generic_context()
        token = self._generic_begin(context)
        self.assertTrue(token.capture_outputs({"positive": torch.ones(2)}))
        self.assertEqual(context._recorder.stats()["pending"], 1)
        self.assertEqual(self.manifests(), [])
        self.assertTrue(token.capture_outputs({"negative": torch.zeros(2)}))
        self.assertTrue(token.finish({}))
        self._assert_recorder_closed(context._recorder)
        self.assertEqual(len(self.manifests()), 1)
        self.assertFalse(token.finish({}))
        token.cancel()
        self.assertEqual(context._recorder.stats()["completed"], 1)

    def test_generic_invalid_or_missing_outputs_cancel_entire_bundle(self):
        for output in (
            {},
            {"positive": torch.ones(2)},
            {"positive": torch.ones(2), "unknown": torch.ones(2)},
            {"positive": torch.ones(3), "negative": torch.ones(2)},
            {"positive": torch.ones(2, dtype=torch.int32), "negative": torch.ones(2)},
        ):
            with self.subTest(output=list(output)):
                context = self._generic_context()
                token = self._generic_begin(context)
                with mock.patch.object(capture, "_snapshot") as snapshot:
                    self.assertFalse(token.finish(output))
                    snapshot.assert_not_called()
                token.cancel()
                self.assertEqual(context._recorder.stats()["pending"], 0)
                self.assertEqual(context._recorder.stats()["reserved_bytes"], 0)
        context = self._generic_context()
        token = self._generic_begin(context)
        self.assertTrue(token.capture_outputs({"positive": torch.ones(2)}))
        self.assertFalse(token.capture_outputs({"positive": torch.ones(2)}))
        self.assertEqual(context._recorder.stats()["pending"], 0)
        self.assertEqual(self.manifests(), [])

    def test_generic_budget_counts_all_outputs_before_any_copy(self):
        context = self._generic_context()
        for budget in (8, 16, 23):
            with (
                self.subTest(budget=budget),
                mock.patch.object(capture, "_MAX_FILE_BYTES", budget),
                mock.patch.object(capture, "_snapshot") as snapshot,
            ):
                self.assertIsNone(self._generic_begin(context))
                snapshot.assert_not_called()
        self.assertEqual(context._recorder.stats()["selected"], 0)
        with mock.patch.object(capture, "_MAX_FILE_BYTES", 24):
            token = self._generic_begin(context)
        self.assertIsNotNone(token)
        token.cancel()
        self.assertEqual(context._recorder.stats()["reserved_bytes"], 0)

    def test_generic_contract_validation_and_metadata_bounds_before_copy(self):
        context = self._generic_context()
        for change in (
            {"inputs": {"value": None}},
            {"inputs": {"positive": torch.ones(2)}},
            {"inputs": {"../bad": torch.ones(2)}},
            {"outputs": {}},
            {"outputs": {"positive": torch.ones(2)}},
            {
                "outputs": {
                    "o%d" % i: capture.TensorSpec((), torch.float32) for i in range(33)
                }
            },
            {"parameters": {"bad": float("nan")}},
            {"parameters": {"bad": {"nested": 1}}},
            {"backend": "../unsafe"},
            {"metadata": {"label": "x" * capture._MAX_JSON}},
        ):
            with (
                self.subTest(fields=list(change)),
                mock.patch.object(capture, "_snapshot") as snapshot,
            ):
                self.assertIsNone(self._generic_begin(context, **change))
                snapshot.assert_not_called()
        self.assertEqual(context._recorder.stats()["selected"], 0)
        self.assertEqual(context._recorder.stats()["pending"], 0)

    def test_generic_stage_parameter_error_and_close_release_reservation(self):
        for change in ({"parameters": {"bad": []}}, {"backend": "../bad"}):
            context = self._generic_context()
            token = self._generic_begin(context)
            with mock.patch.object(capture, "_snapshot") as snapshot:
                self.assertFalse(
                    token.capture_outputs({"positive": torch.ones(2)}, **change)
                )
                snapshot.assert_not_called()
            self.assertEqual(context._recorder.stats()["pending"], 0)
            self.assertEqual(context._recorder.stats()["reserved_bytes"], 0)
        context = self._generic_context()
        token = self._generic_begin(context)
        self.assertTrue(
            token.capture_outputs(
                {"positive": torch.ones(2)},
                backend="alternate",
                parameters={"offset": 2},
            )
        )
        self.assertFalse(context._recorder.close(0.01))
        with mock.patch.object(capture, "_snapshot") as snapshot:
            self.assertFalse(token.finish({"negative": torch.ones(2)}))
            snapshot.assert_not_called()
        self.assertEqual(context._recorder.stats()["reserved_bytes"], 0)
        self.assertEqual(self.manifests(), [])

    def test_generic_scalar_empty_and_invalid_spec(self):
        context = self._generic_context()
        token = self._generic_begin(
            context,
            inputs={},
            outputs={
                "positive": capture.TensorSpec((), torch.float32),
                "negative": capture.TensorSpec((0, 2), torch.bfloat16),
            },
        )
        token.finish(
            {
                "positive": torch.tensor(2.0),
                "negative": torch.empty(0, 2, dtype=torch.bfloat16),
            }
        )
        self._assert_recorder_closed(context._recorder)
        _, tensors = capture.load_capture(self.manifests()[0])
        self.assertEqual(tensors["positive"].shape, ())
        self.assertEqual(tensors["negative"].shape, (0, 2))
        for shape in ((-1,), (True,), (1,) * 17):
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                capture.TensorSpec(shape, torch.float32)

    def test_generic_v3_loader_rejects_interface_tampering(self):
        context = self._generic_context()
        _float_pair(context, torch.ones(2))
        self._assert_recorder_closed(context._recorder)
        path = self.manifests()[0]
        original = json.loads(path.read_text())
        changes = (
            {"contract": {"name": _FLOAT_PAIR.name, "version": True}},
            {"op": "different.contract"},
            {"backend": "../unsafe"},
            {"inputs": ["value", "value"]},
            {"outputs": ["positive"]},
            {"outputs": ["value", "positive", "negative"]},
            {"parameters": {"bad": [1, 2]}},
        )
        for change in changes:
            with self.subTest(change=change):
                path.write_text(json.dumps(dict(original, **change)))
                with self.assertRaises(ValueError):
                    capture.load_capture(path)
        unknown = copy.deepcopy(original)
        unknown.update(
            op="unregistered.reader",
            contract={"name": "unregistered.reader", "version": 1},
        )
        path.write_text(json.dumps(unknown))
        manifest, _ = capture.load_capture(path)
        self.assertEqual(manifest["op"], "unregistered.reader")

    def test_generic_snapshot_exception_releases_before_propagation(self):
        for stage in ("input", "staged", "finish"):
            for error_type in (RuntimeError, KeyboardInterrupt, SystemExit):
                with self.subTest(stage=stage, error=error_type.__name__):
                    context = self._generic_context()
                    token = None if stage == "input" else self._generic_begin(context)
                    snapshots = (
                        []
                        if token is None
                        else [weakref.ref(x) for x in token._tensors.values()]
                    )
                    interruption = error_type("snapshot interrupted")
                    original_snapshot = capture._snapshot

                    def fail_after_copy(tensor):
                        owned = original_snapshot(tensor)
                        snapshots.append(weakref.ref(owned))
                        raise interruption

                    operation = (
                        (lambda: self._generic_begin(context))
                        if stage == "input"
                        else (
                            (lambda: token.capture_outputs({"positive": torch.ones(2)}))
                            if stage == "staged"
                            else (
                                lambda: token.finish(
                                    {
                                        "positive": torch.ones(2),
                                        "negative": torch.ones(2),
                                    }
                                )
                            )
                        )
                    )
                    with mock.patch.object(
                        capture, "_snapshot", side_effect=fail_after_copy
                    ):
                        if error_type is RuntimeError:
                            self.assertFalse(operation())
                            self.assertEqual(context._recorder.stats()["pending"], 0)
                            self.assertEqual(
                                context._recorder.stats()["reserved_bytes"], 0
                            )
                            self.assertTrue(
                                all(reference() is None for reference in snapshots)
                            )
                        else:
                            self._assert_capture_interrupt(
                                context._recorder, operation, interruption, snapshots
                            )

    def _indexed_arguments(self, joint=False, **config):
        identity = copy.deepcopy(self.identity)
        identity["requests"].append(
            dict(phase="prefill", request_id="r2", q_len=3, kv_len=2)
        )
        context = (
            self._mqa_context(identity, **config)
            if joint
            else self.context(self.recorder(dict(self.config, **config)), identity)
        )
        arguments = dict(
            layer=0,
            k=2,
            query_row_start=None,
            total_query_rows=5,
            query_global_ids=torch.tensor([99, 4, 99, 0, 99, 2, 99], dtype=torch.int64)[
                1::2
            ],
            row_starts=torch.tensor([3, 0, 3], dtype=torch.int32),
            row_ends=torch.tensor([8, 2, 6], dtype=torch.int32),
            local_query_rows=4,
            local_query_row_start=1,
            cp_rank=6,
            cp_size=8,
            kv_cache_sharded=True,
        )
        if joint:
            arguments = dict(self._mqa_arguments(rows=3, columns=8), **arguments)
        else:
            arguments.update(
                logits=torch.arange(24, dtype=torch.float32).reshape(3, 8),
                backend="dsv4_per_row",
                parameters={"num_rows": 3},
            )
        return context, arguments

    def test_indexed_roundtrip_owned_ids_prefix_and_legacy_envelopes(self):
        for joint in (False, True):
            for sharded in (False, True):
                with self.subTest(joint=joint, sharded=sharded):
                    context, arguments = self._indexed_arguments(joint)
                    arguments["kv_cache_sharded"] = sharded
                    token = (context.begin_mqa if joint else context.begin_topk)(
                        **arguments
                    )
                    self.assertIsNotNone(token)
                    arguments["query_global_ids"].fill_(99)
                    arguments["row_starts"].zero_()
                    if joint:
                        self.assertTrue(
                            token.prepare_topk(
                                torch.arange(24, dtype=torch.float32).reshape(3, 8),
                                backend="dsv4_per_row",
                                parameters={"num_rows": 3},
                            )
                        )
                    token.finish(torch.zeros(3, 2, dtype=torch.int32))
                    self._assert_recorder_closed(context._recorder)
                    path = next(context._recorder._sink.glob("call-*/manifest.json"))
                    manifest, tensors = capture.load_capture(path)
                    self.assertEqual(manifest["schema_version"], 3)
                    self.assertEqual(
                        manifest["contract"],
                        dict(
                            name="indexer.mqa_topk" if joint else "indexer.topk",
                            version=2,
                        ),
                    )
                    self.assertNotIn("query_row_start", manifest)
                    self.assertEqual(
                        manifest["query_mapping"],
                        dict(
                            kind="indexed",
                            tensor="query_global_ids",
                            index_space="request_order_query_rows",
                        ),
                    )
                    self.assertEqual(
                        manifest["key_domain"],
                        dict(
                            layout="request_order_prefix_then_query",
                            coverage="full",
                        ),
                    )
                    self.assertEqual(
                        manifest["execution_layout"],
                        dict(
                            cp_rank=6,
                            cp_size=8,
                            kv_cache_sharded=sharded,
                        ),
                    )
                    self.assertEqual(
                        manifest["capture_coverage"],
                        dict(
                            scope="record_owner_local",
                            local_query_rows=4,
                            local_query_row_start=1,
                        ),
                    )
                    self.assertIn("query_global_ids", manifest["inputs"])
                    self.assertNotIn("query_global_ids", manifest["outputs"])
                    self.assertEqual(tensors["query_global_ids"].tolist(), [4, 0, 2])
                    self.assertEqual(
                        manifest["tensors"]["query_global_ids"]["nbytes"], 24
                    )
                    self.assertEqual(
                        manifest["tensors"]["query_global_ids"]["original_stride"], [2]
                    )
                    self.assertEqual(
                        manifest["tensors"]["query_global_ids"]["storage_offset"], 1
                    )
                    self.assertEqual(len(tensors), 9 if joint else 5)
                    self.assertIsNone(
                        indexer_capture.validate_query_mapping(
                            manifest, tensors, manifest
                        )
                    )
                    if joint:
                        self.assertEqual(tensors["k_fp8"].shape, (8, 128))
                    self.assertTrue((tensors["logits"][1, 2:] == 0).all())
                    data = (path.parent / "query_global_ids.bin").read_bytes()
                    self.assertEqual(
                        hashlib.sha256(data).hexdigest(),
                        manifest["tensors"]["query_global_ids"]["sha256"],
                    )
                    (path.parent / "query_global_ids.bin").write_bytes(
                        bytes([data[0] ^ 1]) + data[1:]
                    )
                    with self.assertRaisesRegex(ValueError, "checksum"):
                        capture.load_capture(path)
        for joint in (False, True):
            context = self._mqa_context() if joint else self.context()
            token = (
                context.begin_mqa(**self._mqa_arguments())
                if joint
                else self.begin(context)
            )
            if joint:
                self.assertTrue(
                    token.prepare_topk(
                        torch.ones(2, 3), backend="dsv4_per_row", parameters={}
                    )
                )
            token.finish(self.output())
            self._assert_recorder_closed(context._recorder)
            manifest, tensors = capture.load_capture(
                next(context._recorder._sink.glob("call-*/manifest.json"))
            )
            self.assertEqual(manifest["schema_version"], 2 if joint else 1)
            self.assertNotIn("contract", manifest)
            self.assertNotIn("query_mapping", manifest)
            self.assertNotIn("query_global_ids", tensors)
            indexer_capture.validate_query_mapping(manifest, tensors, manifest)

    def test_indexed_chunk_boundaries_zero_prefix_and_single_row_tail(self):
        context, arguments = self._indexed_arguments()
        identity = copy.deepcopy(self.identity)
        identity["requests"] = [
            dict(phase="prefill", request_id="r1", q_len=2, kv_len=0),
            dict(phase="prefill", request_id="r2", q_len=3, kv_len=2),
        ]
        context = self.context(identity=identity)
        ids = torch.tensor([4, 0, 2], dtype=torch.int64)
        starts = torch.tensor([2, 0, 2], dtype=torch.int32)
        ends = torch.tensor([7, 1, 5], dtype=torch.int32)
        for offset, stop in ((0, 2), (2, 3)):
            values = dict(
                arguments,
                query_global_ids=ids[offset:stop],
                row_starts=starts[offset:stop],
                row_ends=ends[offset:stop],
                logits=torch.ones(stop - offset, 7),
                local_query_row_start=offset,
                local_query_rows=3,
                parameters={"num_rows": stop - offset},
            )
            token = context.begin_topk(**values)
            self.assertIsNotNone(token)
            token.finish(torch.zeros(stop - offset, 2, dtype=torch.int32))
        self._assert_recorder_closed(context._recorder)
        paths = sorted(context._recorder._sink.glob("call-*/manifest.json"))
        self.assertEqual(len(paths), 2)
        captured = []
        for path in paths:
            manifest, tensors = capture.load_capture(path)
            indexer_capture.validate_query_mapping(manifest, tensors, manifest)
            captured.extend(tensors["query_global_ids"].tolist())
        self.assertEqual(captured, [4, 0, 2])

    def test_indexed_invalid_mapping_precedes_large_payload_snapshot(self):
        changes = (
            {"query_global_ids": torch.tensor([4, 0, 2], dtype=torch.int32)},
            {"query_global_ids": torch.tensor([4, 0], dtype=torch.int64)},
            {"query_global_ids": torch.tensor([-1, 0, 2])},
            {"query_global_ids": torch.tensor([5, 0, 2])},
            {"query_global_ids": torch.tensor([4, 0, 4])},
            {"query_global_ids": None},
            {"query_row_start": 0},
            {"total_query_rows": 4},
            {"row_starts": torch.tensor([0, 0, 3], dtype=torch.int32)},
            {"row_ends": torch.tensor([8, 3, 6], dtype=torch.int32)},
            {"local_query_row_start": 2},
            {"local_query_row_start": -1},
            {"local_query_rows": 6},
            {"local_query_rows": None},
            {"cp_rank": 8},
            {"cp_rank": True},
            {"cp_size": 0},
            {"cp_size": True},
            {"kv_cache_sharded": 1},
        )
        for joint in (False, True):
            context, arguments = self._indexed_arguments(joint)
            begin = context.begin_mqa if joint else context.begin_topk
            for change in changes:
                with self.subTest(joint=joint, fields=list(change)):
                    changed = dict(arguments, **change)
                    with mock.patch.object(
                        capture, "_snapshot", wraps=capture._snapshot
                    ) as snapshot:
                        self.assertIsNone(begin(**changed))
                    for call in snapshot.call_args_list:
                        self.assertEqual(call.args[0].ndim, 1)
                        self.assertIn(call.args[0].dtype, (torch.int32, torch.int64))
                    self.assertEqual(context._recorder.stats()["pending"], 0)
                    self.assertEqual(context._recorder.stats()["reserved_bytes"], 0)
                    self.assertEqual(context._recorder.stats()["inflight_bytes"], 0)
            self._assert_recorder_closed(context._recorder)
            self.assertEqual(self.manifests(), [])

    def test_indexed_shared_helper_rejects_metadata_and_mapping_tampering(self):
        context, arguments = self._indexed_arguments()
        token = context.begin_topk(**arguments)
        metadata = copy.deepcopy(token._manifest)
        tensors = dict(token._tensors)
        token.cancel()
        changes = (
            lambda m: m.update(query_row_start=0),
            lambda m: m.pop("key_domain"),
            lambda m: m["query_mapping"].update(kind="contiguous"),
            lambda m: m["query_mapping"].update(tensor="row_ends"),
            lambda m: m["key_domain"].update(coverage="local"),
            lambda m: m["execution_layout"].update(cp_rank=-1),
            lambda m: m["execution_layout"].update(extra=1),
            lambda m: m["capture_coverage"].update(scope="all_ranks"),
            lambda m: m["capture_coverage"].update(local_query_rows=True),
            lambda m: m.update(total_query_rows=3),
            lambda m: m["requests"][0].update(q_len=True),
            lambda m: m["requests"][0].update(kv_len=-1),
        )
        for change in changes:
            changed = copy.deepcopy(metadata)
            change(changed)
            with self.subTest(change=change), self.assertRaises(ValueError):
                indexer_capture.validate_query_mapping(changed, tensors, changed)
        for replacement in (
            torch.tensor([4, 0, 4]),
            torch.tensor([5, 0, 2]),
            torch.tensor([-1, 0, 2]),
            torch.tensor([4, 0, 2], dtype=torch.int32),
            torch.empty(3, dtype=torch.int64, device="meta"),
            None,
        ):
            with self.assertRaises(ValueError):
                indexer_capture.validate_query_mapping(
                    metadata, dict(tensors, query_global_ids=replacement), metadata
                )
        self.assertEqual(tensors["query_global_ids"].tolist(), [4, 0, 2])

    def test_shared_mapping_requires_complete_key_domain_for_all_requests(self):
        identity = copy.deepcopy(self.identity)
        identity["requests"].append(
            dict(phase="prefill", request_id="r2", q_len=3, kv_len=2)
        )
        for indexed in (False, True):
            metadata = dict(query_row_count=1, total_query_rows=5)
            mapping = dict(
                row_starts=torch.tensor([0], dtype=torch.int32),
                row_ends=torch.tensor([2], dtype=torch.int32),
            )
            if indexed:
                metadata.update(
                    query_mapping=dict(
                        kind="indexed",
                        tensor="query_global_ids",
                        index_space="request_order_query_rows",
                    ),
                    key_domain=dict(
                        layout="request_order_prefix_then_query",
                        coverage="full",
                    ),
                    execution_layout=dict(cp_rank=0, cp_size=8, kv_cache_sharded=True),
                    capture_coverage=dict(
                        scope="record_owner_local",
                        local_query_rows=1,
                        local_query_row_start=0,
                    ),
                )
                mapping["query_global_ids"] = torch.tensor([0], dtype=torch.int64)
            else:
                metadata["query_row_start"] = 0
            # The capture path validates mapping snapshots before copying payloads.
            self.assertIsNone(
                indexer_capture.validate_query_mapping(identity, mapping, metadata)
            )
            for names in (("logits",), ("k_fp8",), ("logits", "k_fp8")):
                tensors = dict(mapping)
                for name in names:
                    tensors[name] = (
                        torch.zeros(1, 8)
                        if name == "logits"
                        else torch.zeros(8, 128).to(torch.float8_e4m3fn)
                    )
                self.assertIsNone(
                    indexer_capture.validate_query_mapping(identity, tensors, metadata)
                )
                for name in names:
                    # Test each operand independently, including when the other
                    # operand still has the correct global key width.
                    for columns in (3, 7, 9):
                        malformed = dict(tensors)
                        malformed[name] = (
                            torch.zeros(1, columns)
                            if name == "logits"
                            else torch.zeros(columns, 128).to(torch.float8_e4m3fn)
                        )
                        with self.subTest(
                            indexed=indexed, names=names, name=name, columns=columns
                        ):
                            with self.assertRaisesRegex(ValueError, "key domain"):
                                indexer_capture.validate_query_mapping(
                                    identity, malformed, metadata
                                )
                # No selected row belongs to r2. Its prefix still contributes to N.
                changed = copy.deepcopy(identity)
                changed["requests"][1]["kv_len"] += 1
                with self.subTest(indexed=indexed, names=names, unselected_prefix=True):
                    with self.assertRaisesRegex(ValueError, "key domain"):
                        indexer_capture.validate_query_mapping(
                            changed, tensors, metadata
                        )

    def test_indexed_budget_counts_ids_and_full_keys_before_copy(self):
        for joint in (False, True):
            context, arguments = self._indexed_arguments(joint)
            begin = context.begin_mqa if joint else context.begin_topk
            payload = sum(
                t.numel() * t.element_size()
                for t in arguments.values()
                if isinstance(t, torch.Tensor)
            )
            payload += 3 * 2 * 4 + (3 * 8 * 4 if joint else 0)
            with (
                mock.patch.object(capture, "_MAX_FILE_BYTES", payload - 1),
                mock.patch.object(capture, "_snapshot") as snapshot,
            ):
                self.assertIsNone(begin(**arguments))
                snapshot.assert_not_called()
            with mock.patch.object(capture, "_MAX_FILE_BYTES", payload):
                token = begin(**arguments)
                self.assertIsNotNone(token)
                size = context._recorder.stats()["reserved_bytes"]
                token.cancel()
            for field, limit in (
                ("max_bundle_bytes", size - 1),
                ("max_total_bytes", size - 1),
                ("max_inflight_bytes", size * capture._INFLIGHT_FACTOR - 1),
            ):
                limited, values = self._indexed_arguments(joint, **{field: limit})
                with mock.patch.object(capture, "_snapshot") as snapshot:
                    self.assertIsNone(
                        (limited.begin_mqa if joint else limited.begin_topk)(**values)
                    )
                    snapshot.assert_not_called()

    def test_indexed_empty_and_unselected_do_not_copy(self):
        for joint in (False, True):
            context, arguments = self._indexed_arguments(joint)
            begin = context.begin_mqa if joint else context.begin_topk
            empty = dict(arguments)
            for name, value in empty.items():
                if isinstance(value, torch.Tensor) and name not in ("k_fp8", "k_scale"):
                    empty[name] = value[:0]
            empty.update(local_query_rows=0, local_query_row_start=0)
            with mock.patch.object(capture, "_snapshot") as snapshot:
                self.assertIsNone(begin(**empty))
                self.assertIsNone(begin(**dict(arguments, layer=99)))
                snapshot.assert_not_called()
        recorder = self.recorder({})
        with mock.patch.object(capture, "_snapshot") as snapshot:
            self.assertIsNone(recorder.for_execution("not parsed when disabled"))
            snapshot.assert_not_called()
        nonowner = self.recorder(tp_rank=1)
        self.assertIsNone(nonowner.for_execution(json.dumps(self.identity)))
        self.assertEqual(nonowner.stats()["skip_reasons"]["non_owner_rank"], 1)

    def test_indexed_snapshot_failures_and_interrupts_release_group(self):
        for joint in (False, True):
            context, arguments = self._indexed_arguments(joint)
            begin = context.begin_mqa if joint else context.begin_topk
            for error_type in (RuntimeError, KeyboardInterrupt):
                for failing_name in (
                    "query_global_ids",
                    "row_ends",
                    "k_fp8" if joint else "logits",
                ):
                    original = capture._snapshot
                    error = error_type("indexed snapshot interruption")

                    def snapshot(tensor):
                        if tensor is arguments[failing_name]:
                            raise error
                        return original(tensor)

                    with mock.patch.object(capture, "_snapshot", side_effect=snapshot):
                        if error_type is RuntimeError:
                            self.assertIsNone(begin(**arguments))
                        else:
                            with self.assertRaises(KeyboardInterrupt) as raised:
                                begin(**arguments)
                            self.assertIs(raised.exception, error)
                    self.assertEqual(context._recorder.stats()["pending"], 0)
                    self.assertEqual(context._recorder.stats()["reserved_bytes"], 0)
                    self.assertEqual(context._recorder.stats()["inflight_bytes"], 0)
            if joint:
                self.assertIsNone(self.begin(context))
            self._assert_recorder_closed(context._recorder)
            self.assertEqual(self.manifests(), [])

    def test_contract_versions_share_selection_and_never_fall_back(self):
        name = "test.versions." + self.root.name
        alias = ((name + ".first", name + ".second"),)
        for version in (2, 1):
            capture.register_capture_contract(
                capture.CaptureContract(
                    name,
                    version=version,
                    aliases=alias,
                    selection_group=name,
                )
            )
        self.assertEqual(capture._resolve_ops(list(alias[0])), [name])
        self.assertEqual(capture._contract(name).version, 1)
        self.assertEqual(capture._contract(name, 2).version, 2)
        for version in (0, -1, True, 3, "2"):
            with self.assertRaises(ValueError):
                capture._contract(name, version)
        for changes in (dict(aliases=()), dict(selection_group=None)):
            values = dict(version=3, aliases=alias, selection_group=name)
            values.update(changes)
            with self.assertRaisesRegex(ValueError, "selection semantics"):
                capture.register_capture_contract(
                    capture.CaptureContract(name, **values)
                )
        with self.assertRaisesRegex(ValueError, "duplicate"):
            capture.register_capture_contract(
                capture.CaptureContract(
                    name, version=2, aliases=alias, selection_group=name
                )
            )
        with self.assertRaisesRegex(ValueError, "duplicate"):
            capture.register_capture_contract(
                capture.CaptureContract(name + ".other", aliases=alias)
            )
        context = self.context().context
        with mock.patch.object(capture, "_snapshot") as snapshot:
            with self.assertRaises(ValueError):
                context.begin(
                    op="indexer.topk",
                    contract_version=99,
                    layer=0,
                    inputs={},
                    outputs={},
                    backend="test",
                    parameters={},
                )
            snapshot.assert_not_called()
        only_v2 = name + ".only_v2"
        capture.register_capture_contract(capture.CaptureContract(only_v2, version=2))
        self.assertEqual(capture._resolve_ops([only_v2]), [only_v2])
        self.assertFalse(context.is_selected(only_v2))
        with self.assertRaises(ValueError):
            capture._contract(only_v2)

    def test_indexed_generic_begin_cannot_bypass_mapping_interface(self):
        for joint in (False, True):
            context, arguments = self._indexed_arguments(joint)
            token = (context.begin_mqa if joint else context.begin_topk)(**arguments)
            metadata = {
                key: copy.deepcopy(value)
                for key, value in token._manifest.items()
                if key in token._contract.metadata_fields
            }
            inputs = dict(token._tensors)
            outputs = dict(token._outputs)
            token.cancel()
            op = "indexer.mqa_topk" if joint else "indexer.topk"
            variants = []
            missing_ids = dict(inputs)
            del missing_ids["query_global_ids"]
            variants.append((missing_ids, metadata, outputs))
            variants.append((inputs, {}, outputs))
            variants.append((dict(inputs, extra=torch.ones(1)), metadata, outputs))
            variants.append(
                (
                    inputs,
                    metadata,
                    dict(outputs, extra=capture.TensorSpec((1,), torch.float32)),
                )
            )
            contiguous = dict(metadata, query_row_start=0)
            variants.append((inputs, contiguous, outputs))
            for in_values, meta, out_values in variants:
                with mock.patch.object(capture, "_snapshot") as snapshot:
                    self.assertIsNone(
                        context.context.begin(
                            op=op,
                            contract_version=2,
                            layer=0,
                            inputs=in_values,
                            outputs=out_values,
                            backend="pending",
                            parameters={},
                            metadata=meta,
                        )
                    )
                    snapshot.assert_not_called()
                self.assertEqual(context._recorder.stats()["pending"], 0)

    def _decode_identity(self):
        identity = copy.deepcopy(self.identity)
        identity.update(server_role="DECODE")
        identity["requests"] = [
            dict(phase="decode", request_id="r1", q_len=1, kv_len=0),
            dict(phase="decode", request_id="r2", q_len=1, kv_len=17),
        ]
        return identity

    def test_phase_configuration_normalization_and_cached_query(self):
        for phases, normalized in (
            (["prefill"], ["prefill"]),
            (["decode"], ["decode"]),
            (["decode", "prefill", "decode"], ["prefill", "decode"]),
            (["decode", "decode"], ["decode"]),
        ):
            self.config_path.write_text(json.dumps(dict(self.config, phases=phases)))
            with (
                mock.patch.object(threading.Thread, "start") as start,
                mock.patch.object(torch.cuda, "is_initialized") as cuda,
            ):
                self.assertEqual(
                    capture._config(self.config_path)["phases"], normalized
                )
                self.assertTrue(capture.config_enabled(self.config_path))
                for phase in ("prefill", "decode"):
                    self.assertEqual(
                        capture.config_enabled(self.config_path, phase=phase),
                        phase in normalized,
                    )
                for phase in ("", "unknown", True, 1, [], {}):
                    with self.assertRaises(ValueError):
                        capture.config_enabled(self.config_path, phase=phase)
                start.assert_not_called()
                cuda.assert_not_called()
        self.config_path.write_text("{}")
        self.assertFalse(capture.config_enabled(self.config_path, phase="decode"))
        self.assertFalse(self.record_dir.exists())

    def test_pd_server_roles_phase_identity_and_manifest(self):
        for phase in ("prefill", "decode"):
            for role in (None, "PREFILL", "DECODE", "PDFUSION"):
                with self.subTest(phase=phase, role=role):
                    identity = (
                        self._decode_identity()
                        if phase == "decode"
                        else copy.deepcopy(self.identity)
                    )
                    identity.pop("server_role", None)
                    if role is not None:
                        identity["server_role"] = role
                    recorder = self.recorder(
                        dict(
                            self.config,
                            phases=["prefill", "decode"],
                            ops=[_FLOAT_PAIR.name],
                        )
                    )
                    valid = role in (
                        None,
                        "PDFUSION",
                        "PREFILL" if phase == "prefill" else "DECODE",
                    )
                    if not valid:
                        with mock.patch.object(recorder, "_start") as start:
                            self.assertIsNone(
                                recorder.for_execution(json.dumps(identity), phase)
                            )
                            start.assert_not_called()
                        self.assertEqual(
                            recorder.stats()["skip_reasons"]["invalid_context"], 1
                        )
                        continue
                    context = recorder.for_execution(json.dumps(identity), phase)
                    self.assertIsNotNone(context)
                    self.assertEqual(context.phase, phase)
                    self.assertEqual(context.identity["phase"], phase)
                    self.assertEqual(context.identity.get("server_role"), role)
                    with self.assertRaises(TypeError):
                        context.identity["phase"] = "unknown"
                    _float_pair(context, torch.ones(2))
                    self._assert_recorder_closed(recorder)
                    manifest, _ = capture.load_capture(
                        next(recorder._sink.glob("call-*/manifest.json"))
                    )
                    self.assertEqual(manifest["phase"], phase)
                    self.assertEqual(manifest.get("server_role"), role)
                    self.assertEqual("server_role" in manifest, role is not None)
                    self.assertEqual(manifest["model_role"], "target")
                    self.assertEqual(manifest["requests"], identity["requests"])
        recorder = self.recorder()
        for role in ("prefill", "UNKNOWN", "", None, True, 1, [], {}):
            self.assertIsNone(
                recorder.for_execution(
                    json.dumps(dict(self.identity, server_role=role))
                )
            )
        self.assertEqual(recorder.stats()["skip_reasons"]["invalid_context"], 8)
        for field in ("server_role", "phase"):
            with self.assertRaises(ValueError):
                capture.register_capture_contract(
                    capture.CaptureContract(
                        "test.pd." + field + self.root.name,
                        metadata_fields=frozenset({field}),
                    )
                )

    def test_decode_identity_rejects_mixed_special_and_inconsistent_phase(self):
        recorder = self.recorder(dict(self.config, phases=["prefill", "decode"]))
        identity = self._decode_identity()
        changes = (
            lambda x: x.update(phase="prefill"),
            lambda x: x.update(phase=None),
            lambda x: x["requests"][0].update(phase="prefill"),
            lambda x: x["requests"][0].update(phase="verify"),
            lambda x: x["requests"][0].update(q_len=2),
            lambda x: x["requests"][0].update(q_len=0),
            lambda x: x["requests"][0].update(q_len=True),
            lambda x: x["requests"][0].update(kv_len=-1),
            lambda x: x["requests"][0].update(kv_len=True),
            lambda x: x["requests"][0].update(is_fake=True),
            lambda x: x.update(model_role="draft"),
        )
        with (
            mock.patch.object(recorder, "_start") as start,
            mock.patch.object(capture, "_snapshot") as snapshot,
        ):
            for change in changes:
                changed = copy.deepcopy(identity)
                change(changed)
                self.assertIsNone(recorder.for_execution(json.dumps(changed)))
            for phase in ("prefill", "", "unknown", True, [], {}):
                self.assertIsNone(
                    recorder.for_execution(json.dumps(identity), phase=phase)
                )
            start.assert_not_called()
            snapshot.assert_not_called()
        for explicit in (False, True):
            valid = dict(identity, phase="decode") if explicit else identity
            self.assertIsNotNone(
                recorder.for_execution(json.dumps(valid), phase="decode")
            )

    def test_phase_filter_runs_before_sink_and_producer_phase_is_checked(self):
        for selected, identity in (
            (["prefill"], self._decode_identity()),
            (["decode"], self.identity),
        ):
            recorder = self.recorder(dict(self.config, phases=selected))
            with (
                mock.patch.object(recorder, "_start") as start,
                mock.patch.object(capture, "_snapshot") as snapshot,
            ):
                self.assertIsNone(recorder.for_execution(json.dumps(identity)))
                self.assertEqual(recorder.stats()["skip_reasons"], {"phase_filter": 1})
                self.assertIsNone(
                    recorder.for_execution(json.dumps(identity), phase=selected[0])
                )
                self.assertEqual(recorder.stats()["skip_reasons"]["invalid_context"], 1)
                start.assert_not_called()
                snapshot.assert_not_called()
            self.assertIsNone(recorder._thread)
        disabled = self.recorder({})
        self.assertIsNone(disabled.for_execution("not parsed", phase="decode"))
        self.assertFalse(self.record_dir.exists())

    def test_decode_dp_owners_have_independent_recorders_and_budgets(self):
        for dp_rank, world_rank in ((0, 0), (1, 4)):
            identity = self._decode_identity()
            identity.update(
                dp_rank=dp_rank,
                world_rank=world_rank,
                replica_id="dp%d" % dp_rank,
                owner_id="owner%d" % dp_rank,
            )
            config = dict(
                self.config, phases=["decode"], ops=[_FLOAT_PAIR.name], max_calls=1
            )
            nonowner = self.recorder(config, tp_rank=1, world_rank=world_rank)
            self.assertIsNone(nonowner.for_execution(json.dumps(identity), "decode"))
            self.assertEqual(nonowner.stats()["skip_reasons"], {"non_owner_rank": 1})
            recorder = self.recorder(config, tp_rank=0, world_rank=world_rank)
            context = recorder.for_execution(json.dumps(identity), "decode")
            self.assertIsNotNone(context)
            _float_pair(context, torch.ones(1))
            _float_pair(context, torch.ones(1))
            self._assert_recorder_closed(recorder)
            self.assertEqual(recorder.stats()["completed"], 1)
            self.assertEqual(recorder.stats()["skip_reasons"]["max_calls"], 1)
            manifest, _ = capture.load_capture(
                next(recorder._sink.glob("call-*/manifest.json"))
            )
            self.assertEqual(
                (manifest["dp_rank"], manifest["world_rank"]), (dp_rank, world_rank)
            )
        self.assertEqual(len(self.manifests()), 2)

    def _packed_arguments(self, **config):
        recorder = self.recorder(dict(self.config, ops=[_PACKED_INPUT.name], **config))
        context = recorder.for_execution(json.dumps(self.identity))
        self.assertIsNotNone(context)
        return context, dict(
            op=_PACKED_INPUT.name,
            layer=0,
            backend="test",
            parameters={},
            inputs={
                "packed": torch.arange(64, dtype=torch.float32).reshape(8, 8)[:, ::2],
                "direct": torch.arange(8, dtype=torch.float32)[1::3],
            },
            outputs={"out": capture.TensorSpec((2,), torch.float32)},
            metadata={"label": "compact"},
        )

    def test_transformed_input_roundtrip_single_reservation_and_descriptors(self):
        context, args = self._packed_arguments()
        original = capture._snapshot
        expected = args["inputs"]["packed"][:2, :3].clone()

        def snapshot(source):
            self.assertEqual(context._recorder.stats()["pending"], 1)
            self.assertGreater(context._recorder.stats()["reserved_bytes"], 0)
            return original(source)

        with (
            mock.patch.object(
                context._recorder, "_reserve", wraps=context._recorder._reserve
            ) as reserve,
            mock.patch.object(capture, "_snapshot", side_effect=snapshot),
        ):
            token = context.begin(**args)
            self.assertIsNotNone(token)
            self.assertEqual(reserve.call_count, 1)
            size = context._recorder.stats()["reserved_bytes"]
            self.assertEqual(
                context._recorder.stats()["inflight_bytes"],
                size * capture._INFLIGHT_FACTOR,
            )
            args["inputs"]["packed"].fill_(99)
            token.finish({"out": torch.zeros(2)})
            reserve.assert_called_once()
        self._assert_recorder_closed(context._recorder)
        manifest, tensors = capture.load_capture(
            next(context._recorder._sink.glob("call-*/manifest.json"))
        )
        torch.testing.assert_close(tensors["packed"], expected)
        packed_desc = manifest["tensors"]["packed"]
        self.assertEqual(packed_desc["shape"], [2, 3])
        self.assertEqual(packed_desc["original_stride"], [3, 1])
        self.assertEqual(packed_desc["storage_offset"], 0)
        direct_desc = manifest["tensors"]["direct"]
        self.assertEqual(direct_desc["original_stride"], [3])
        self.assertEqual(direct_desc["storage_offset"], 1)
        self.assertEqual(set(manifest["inputs"]), {"packed", "direct"})
        self.assertEqual(sum(d["nbytes"] for d in manifest["tensors"].values()), 44)

    def test_transformed_global_pool_not_charged_as_artifact(self):
        context, args = self._packed_arguments()
        # Zero-stride source is logically larger than the full artifact limit,
        # but allocating/serializing the pool is unnecessary for this contract.
        args["inputs"]["packed"] = torch.ones(1).expand(capture._MAX_FILE_BYTES, 4)
        with mock.patch.object(capture, "_MAX_FILE_BYTES", 44):
            token = context.begin(**args)
            self.assertIsNotNone(token)
            self.assertEqual(token._tensors["packed"].shape, (2, 3))
            self.assertEqual(token._tensors["packed"].untyped_storage().nbytes(), 24)
            token.cancel()
        self.assertEqual(context._recorder.stats()["pending"], 0)

    def test_transformed_budget_and_metadata_rejected_before_snapshot(self):
        context, args = self._packed_arguments()
        token = context.begin(**args)
        size = context._recorder.stats()["reserved_bytes"]
        token.cancel()
        for field, limit in (
            ("max_bundle_bytes", size - 1),
            ("max_total_bytes", size - 1),
            ("max_inflight_bytes", capture._INFLIGHT_FACTOR * size - 1),
        ):
            limited, values = self._packed_arguments(**{field: limit})
            with mock.patch.object(_PackedInputContract, "snapshot_inputs") as snapshot:
                self.assertIsNone(limited.begin(**values))
                snapshot.assert_not_called()
            self.assertEqual(limited._recorder.stats()["pending"], 0)
        with (
            mock.patch.object(capture, "_MAX_FILE_BYTES", 43),
            mock.patch.object(_PackedInputContract, "snapshot_inputs") as snapshot,
        ):
            self.assertIsNone(context.begin(**args))
            snapshot.assert_not_called()
        for metadata in ({"label": "x" * (capture._MAX_JSON + 1)}, {"phase": "decode"}):
            with (
                mock.patch.object(_PackedInputContract, "input_specs") as plan,
                mock.patch.object(context._recorder, "_reserve") as reserve,
            ):
                self.assertIsNone(context.begin(**dict(args, metadata=metadata)))
                plan.assert_not_called()
                reserve.assert_not_called()
        with mock.patch.object(_PackedInputContract, "input_specs") as plan:
            self.assertIsNone(context.begin(**dict(args, layer=99)))
            plan.assert_not_called()

    def test_transformed_invalid_plans_and_source_metadata_before_reservation(self):
        context, args = self._packed_arguments()
        plans = (
            None,
            [],
            {"missing": capture.TensorSpec((2, 3), torch.float32)},
            {"out": capture.TensorSpec((2,), torch.float32)},
            {"packed": torch.ones(2, 3)},
            {"packed": capture.TensorSpec((2, 3), torch.float32, "cuda")},
        )
        for plan in plans:
            with (
                mock.patch.object(
                    _PackedInputContract, "input_specs", return_value=plan
                ),
                mock.patch.object(context._recorder, "_reserve") as reserve,
                mock.patch.object(_PackedInputContract, "snapshot_inputs") as snapshot,
            ):
                self.assertIsNone(context.begin(**args))
                reserve.assert_not_called()
                snapshot.assert_not_called()
        for source in (None, torch.ones(2, 3).to_sparse(), torch.ones((1,) * 17)):
            values = dict(args, inputs=dict(args["inputs"], packed=source))
            with (
                mock.patch.object(_PackedInputContract, "input_specs") as plan,
                mock.patch.object(context._recorder, "_reserve") as reserve,
            ):
                self.assertIsNone(context.begin(**values))
                plan.assert_not_called()
                reserve.assert_not_called()
        for error_type in (ValueError, KeyboardInterrupt):
            error = error_type("plan failed")
            with (
                mock.patch.object(
                    _PackedInputContract, "input_specs", side_effect=error
                ),
                mock.patch.object(context._recorder, "_reserve") as reserve,
            ):
                if error_type is ValueError:
                    self.assertIsNone(context.begin(**args))
                else:
                    with self.assertRaises(KeyboardInterrupt) as raised:
                        context.begin(**args)
                    self.assertIs(raised.exception, error)
                reserve.assert_not_called()

    def test_transformed_invalid_snapshot_shape_layout_ownership_and_names(self):
        context, args = self._packed_arguments()
        values = (
            torch.ones(1, 3),
            torch.ones(2, 3, dtype=torch.float16),
            torch.ones(3, 2).t(),
            torch.ones(2, 3).to_sparse(),
            torch.empty(2, 3, device="meta"),
            torch.ones(20)[1:7].reshape(2, 3),
            torch.ones(20)[:6].reshape(2, 3),
            None,
        )
        # Same-sized source alias must not become an owned capture snapshot.
        args["inputs"]["packed"] = torch.ones(2, 3)
        values += (args["inputs"]["packed"].detach(),)
        original = _PackedInputContract.snapshot_inputs
        for value in values:

            def malformed(contract, inputs, tensors, identity, metadata):
                original(contract, inputs, tensors, identity, metadata)
                tensors["packed"] = value

            with mock.patch.object(
                _PackedInputContract, "snapshot_inputs", new=malformed
            ):
                self.assertIsNone(context.begin(**args))
            self.assertEqual(context._recorder.stats()["pending"], 0)
            self.assertEqual(context._recorder.stats()["reserved_bytes"], 0)
            self.assertEqual(context._recorder.stats()["inflight_bytes"], 0)
        for name in ("extra", "out", "missing"):

            def wrong_names(contract, inputs, tensors, identity, metadata):
                original(contract, inputs, tensors, identity, metadata)
                if name == "missing":
                    del tensors["direct"]
                else:
                    tensors[name] = torch.zeros(1)

            with mock.patch.object(
                _PackedInputContract, "snapshot_inputs", new=wrong_names
            ):
                self.assertIsNone(context.begin(**args))
            self.assertEqual(context._recorder.stats()["pending"], 0)
        self._assert_recorder_closed(context._recorder)
        self.assertEqual(self.manifests(), [])

    def test_transformed_snapshot_errors_interrupts_close_and_cancel_release_once(self):
        context, args = self._packed_arguments()
        original = _PackedInputContract.snapshot_inputs
        for error_type in (RuntimeError, KeyboardInterrupt, SystemExit):
            error = error_type("snapshot interrupted")

            def fail(contract, inputs, tensors, identity, metadata):
                original(contract, inputs, tensors, identity, metadata)
                raise error

            with (
                mock.patch.object(_PackedInputContract, "snapshot_inputs", new=fail),
                mock.patch.object(
                    context._recorder, "_release", wraps=context._recorder._release
                ) as release,
            ):
                if error_type is RuntimeError:
                    self.assertIsNone(context.begin(**args))
                else:
                    with self.assertRaises(error_type) as raised:
                        context.begin(**args)
                    self.assertIs(raised.exception, error)
                release.assert_called_once()
            self.assertEqual(context._recorder.stats()["pending"], 0)
            self.assertEqual(context._recorder.stats()["reserved_bytes"], 0)
        token = context.begin(**args)
        self.assertIsNotNone(token)
        with mock.patch.object(
            context._recorder, "_release", wraps=context._recorder._release
        ) as release:
            token.cancel()
            token.cancel()
            token.finish({"out": torch.zeros(2)})
            release.assert_called_once()
        token = context.begin(**args)
        self.assertIsNotNone(token)
        # Close may precede token completion; finish must release, not publish.
        context._recorder.close(0.01)
        token.finish({"out": torch.zeros(2)})
        self._assert_recorder_closed(context._recorder)
        self.assertEqual(self.manifests(), [])
        self.assertEqual(context._recorder.stats()["reserved_bytes"], 0)

    def test_transformed_token_construction_interrupt_drops_storage_before_refund(self):
        for error_type in (KeyboardInterrupt, SystemExit):
            context, args = self._packed_arguments()
            original = capture._snapshot
            snapshots = []

            def snapshot(source):
                owned = original(source)
                snapshots.append(weakref.ref(owned))
                return owned

            error = error_type("token allocation interrupted")
            with (
                mock.patch.object(capture, "_snapshot", side_effect=snapshot),
                mock.patch.object(capture.CaptureToken, "__init__", side_effect=error),
            ):
                self._assert_capture_interrupt(
                    context._recorder, lambda: context.begin(**args), error, snapshots
                )

    def test_transformed_close_during_snapshot_retains_single_reservation(self):
        context, args = self._packed_arguments()
        original = _PackedInputContract.snapshot_inputs

        def close_during_snapshot(contract, inputs, tensors, identity, metadata):
            original(contract, inputs, tensors, identity, metadata)
            self.assertEqual(context._recorder.stats()["pending"], 1)
            context._recorder.close(0.01)
            self.assertEqual(context._recorder.stats()["pending"], 1)
            self.assertGreater(context._recorder.stats()["inflight_bytes"], 0)

        with mock.patch.object(
            _PackedInputContract, "snapshot_inputs", new=close_during_snapshot
        ):
            token = context.begin(**args)
        self.assertIsNotNone(token)
        token.finish({"out": torch.zeros(2)})
        self._assert_recorder_closed(context._recorder)
        self.assertEqual(context._recorder.stats()["reserved_bytes"], 0)
        self.assertEqual(self.manifests(), [])

    def test_transformed_empty_and_explicit_cpu_specs(self):
        for shape in ((0, 3), (2, 3)):
            context, args = self._packed_arguments()
            spec = capture.TensorSpec(shape, torch.float32, "cpu")

            def snapshot(contract, inputs, tensors, identity, metadata):
                tensors["direct"] = capture._snapshot(inputs["direct"])
                tensors["packed"] = torch.zeros(shape)

            with (
                mock.patch.object(
                    _PackedInputContract, "input_specs", return_value={"packed": spec}
                ),
                mock.patch.object(
                    _PackedInputContract, "snapshot_inputs", new=snapshot
                ),
            ):
                token = context.begin(**args)
            self.assertIsNotNone(token)
            token.finish({"out": torch.ones(2)})
            self._assert_recorder_closed(context._recorder)
            manifest, tensors = capture.load_capture(
                next(context._recorder._sink.glob("call-*/manifest.json"))
            )
            self.assertEqual(tuple(tensors["packed"].shape), shape)
            self.assertEqual(manifest["tensors"]["packed"]["nbytes"], spec.nbytes)

    def test_stats_snapshot_is_independent_and_reasons_bounded(self):
        recorder = self.recorder({})
        for i in range(100):
            recorder.note_skip("reason_%d" % i)
        recorder.note_skip("request data must not appear")
        stats = recorder.stats()
        self.assertLessEqual(len(stats["skip_reasons"]), 65)
        stats["skip_reasons"].clear()
        self.assertEqual(recorder.stats()["skipped"], 101)
        self.assertTrue(recorder.stats()["skip_reasons"])


if __name__ == "__main__":
    unittest.main()
