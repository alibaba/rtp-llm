"""CPU contract tests: do not exercise CUDA kernels or real FST I/O."""

import queue
import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from rtp_llm.models_py import fastsafetensors_read_spec as reader
from rtp_llm.models_py import v4_fst_diagnostic as diag


class TraceTest(unittest.TestCase):

    def test_fenced_wo_a_audit_compares_selected_tp_rows_and_never_repairs(self):
        from safetensors.torch import save_file

        class AttentionFP8(diag.torch.nn.Module):
            def __init__(self, value, layer):
                super().__init__()
                self._checkpoint_owned = True
                self.layer_id, self.tp_size, self.tp_rank = layer, 2, 1
                self.wo_a_s = value

        trace = diag.V4FstTrace("fenced", 1, "cpu")
        source = diag.torch.arange(24, dtype=diag.torch.uint8).reshape(4, 6)
        module_type = diag.torch.float8_e8m0fnu
        source = source.view(module_type)
        with tempfile.TemporaryDirectory() as d:
            path = str(Path(d) / "selected.safetensors")
            names = [f"layers.{i}.attn.wo_a.scale" for i in (0, 1)]
            save_file({k: source.clone() for k in names}, path)
            owners = diag.torch.nn.ModuleList(
                [AttentionFP8(source[2:].clone(), i) for i in (0, 1)]
            )
            catalog = SimpleNamespace(entries=[(k, path) for k in names])
            trace.bind_wo_a_route(owners, catalog, [path])
            # Source is correct, one byte changes at collective return, and
            # the same byte reaches the sharded ReadSpec and physical owner.
            spec = reader.ReadSpec(names[0], names[0], (4, 6),
                                   slices=(slice(2, 4), slice(None)))
            returned = source.clone()
            returned.view(diag.torch.uint8)[3, 1] ^= 1
            fb = SimpleNamespace(
                key_to_rank_lidx={names[0]: (1, 0)},
                pg=SimpleNamespace(rank=lambda: 1),
                rank_loaders={1: [SimpleNamespace(tensors={
                    names[0]: SimpleNamespace(get_raw=lambda: source)
                })]},
                get_tensor=mock.Mock(return_value=returned),
            )
            with self.assertLogs(diag.log) as route:
                checkpoint = trace.observe_wo_a_reader(fb, names[0], spec, False)
                for _, value in trace.wo_a_dispatch([(names[0], checkpoint)]):
                    owners[0].wo_a_s.copy_(value.tensor)
            fb.get_tensor.assert_called_once_with(names[0])
            for line, stage, differences, offset in zip(route.output, (
                "native_file_buffer", "get_tensor", "after_readspec", "after_dispatch",
            ), (0, 1, 1, 1), (-1, 19, 7, 7), strict=True):
                self.assertIn(f"stage={stage}", line)
                self.assertIn(f"mismatches={differences} first_offset={offset}", line)
                self.assertNotIn(path, line)
            owners[0].wo_a_s.copy_(source[2:])
            with self.assertLogs(diag.log) as logs:
                trace.verify_wo_a_owners(owners, catalog, [path])
            self.assertIn("layers=2", logs.output[-1])
            owners[1].wo_a_s.view(diag.torch.uint8)[0, 0] ^= 1
            with self.assertRaisesRegex(RuntimeError, "mismatch rank=1 layer=1"):
                trace.verify_wo_a_owners(owners, catalog, [path])
            self.assertNotEqual(
                owners[1].wo_a_s.view(diag.torch.uint8)[0, 0].item(), 12
            )
            with self.assertRaisesRegex(RuntimeError, "audit input"):
                trace.verify_wo_a_owners(owners, catalog, [])

    def test_read_failure_logs_rank_batch_and_ordinal_without_payload(self):
        trace = diag.V4FstTrace("trace", 1, "cpu")
        failure = RuntimeError("private error")
        with self.assertLogs(diag.log, level="INFO") as messages:
            with self.assertRaises(RuntimeError) as caught:
                with trace.watch_keys(16, ["private-a", "private-b"]) as keys:
                    self.assertEqual(next(keys), "private-a")
                    self.assertEqual(next(keys), "private-b")
                    raise failure
        self.assertIs(caught.exception, failure)
        self.assertIn("rank=1 batch=16 key_ordinal=1 reason=RuntimeError", messages.output[-1])
        self.assertNotIn("private", "\n".join(messages.output))

    def test_producer_failure_preserves_exception_without_a_producer_fence(self):
        failure = RuntimeError("private error")
        class Reader:
            def _load_single_batch(self, batch, files):
                raise failure
        trace = diag.V4FstTrace("fenced", 0, "cpu")
        with self.assertLogs(diag.log, level="INFO") as messages, mock.patch.object(
            diag.torch.cuda, "synchronize"
        ) as sync, self.assertRaises(RuntimeError) as caught:
            trace.reader_class(Reader, [])()._load_single_batch(18, [])
        sync.assert_not_called()
        self.assertIs(caught.exception, failure)
        self.assertIn("stage=producer_raised batch=18", messages.output[-1])
        self.assertNotIn("private", "\n".join(messages.output))

    def test_producer_queued_failure_is_not_reported_as_normal_return(self):
        class Reader:
            def _load_single_batch(self, batch, files):
                self.error_event = SimpleNamespace(is_set=lambda: True)

        trace = diag.V4FstTrace("fenced", 1, "cpu")
        with self.assertLogs(diag.log, level="INFO") as logs, mock.patch.object(
            diag.torch.cuda, "synchronize"
        ) as sync:
            trace.reader_class(Reader, [])()._load_single_batch(18, [])
        self.assertIn("stage=producer_returned_error_event batch=18", logs.output[-1])
        sync.assert_not_called()

    def test_rejected_hash_router_compares_only_its_small_physical_tensor(self):
        from safetensors.torch import save_file

        torch = diag.torch
        key = "layers.2.ffn.gate.tid2eid"
        source = torch.tensor([[0, 1], [1, 0]])
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "private_checkpoint_name.safetensors")
            save_file(
                {key: source, "private_other_tensor": torch.tensor([98765])}, path
            )
            catalog = lambda: SimpleNamespace(entries=((key, path),))
            observed = source.clone()
            observed[1, 1] = 3
            error = diag.V4RouterRangeError(2, 2, observed, 0, 3)
            t = diag.V4FstTrace("trace", 1, "cpu")
            with self.assertLogs(diag.log, level="ERROR") as logs:
                t.compare_bad_router(error, catalog, [path])
            msg = logs.output[0]
            self.assertIn("layer=2", msg)
            self.assertIn("source_range=[0,1] equal=False", msg)
            self.assertIn("observed_sha256=", msg)
            for private in (tmp, "private", "98765", key):
                self.assertNotIn(private, msg)
            with self.assertLogs(diag.log, level="ERROR") as logs:
                t.compare_bad_router(error, catalog, [])
            self.assertIn("unavailable", logs.output[0])
            self.assertNotIn(tmp, logs.output[0])
            save_file({key: observed}, path)
            with self.assertLogs(diag.log, level="ERROR") as logs:
                t.compare_bad_router(error, catalog, [path])
            self.assertIn("source_range=[0,3] equal=True", logs.output[0])

    def test_other_failure_or_oversize_router_never_reads_the_checkpoint(self):
        catalog = mock.Mock()
        trace = diag.V4FstTrace("trace", 1, "cpu")
        trace.compare_bad_router(ValueError("private path"), catalog, [])
        value = diag.torch.empty((1025, 1024), device="meta", dtype=diag.torch.int64)
        with self.assertLogs(diag.log, level="ERROR"):
            trace.compare_bad_router(
                diag.V4RouterRangeError(2, 2, value, 0, 8), catalog, []
            )
        catalog.assert_not_called()

    def test_metadata_no_paths_or_tensor_payload(self):
        files = [
            "/private/credentials/model-00005-of-00048.safetensors",
            "/private/model/custom-shard.safetensors",
        ]

        class Producer:
            def _load_single_batch(self, batch, paths):
                return 9

        t = diag.V4FstTrace("fenced", 1, "cuda:1")
        with self.assertLogs(diag.log, level="INFO") as logs, mock.patch.object(
            diag.torch.cuda, "synchronize"
        ) as sync:
            self.assertEqual(
                t.reader_class(Producer, files)()._load_single_batch(4, files), 9
            )
            sync.assert_not_called()
        text = "\n".join(logs.output)
        self.assertIn("batch=4 shards=5/48,selected:2", text)
        for forbidden in ("private", "credentials", "safetensors", "custom-shard"):
            self.assertNotIn(forbidden, text)

    def exercise_consumer(self, mode, *, fail_at=None):
        actions = []
        fb = mock.Mock()
        fb.close.side_effect = lambda: actions.append("close")
        timer = lambda *args: nullcontext(SimpleNamespace(elapsed_ms=0, batch_id=None))

        class Parent:
            def __init__(self):
                self.batch_queue = queue.Queue()
                self.batch_queue.put(
                    SimpleNamespace(
                        fb=fb,
                        keys=[],
                        batch_id=7,
                        add_filenames_time=0,
                        copy_files_time=0,
                    )
                )
                self.queue_size, self.consumer_processed = 1, None
                self._log_message = lambda *_args, **_kwargs: None

        traced = reader.build_read_spec_parallel_loader(Parent, timer)(
            read_spec_resolver=None
        )
        if mode is not None:
            traced._rtp_v4_fst_trace = diag.V4FstTrace(mode, 0, "cuda:0")

        def sync(_device):
            actions.append("sync")
            if actions.count("sync") == fail_at:
                raise RuntimeError("device fault")

        def read(*args):
            actions.append("dispatch")
            yield "private tensor", object()

        with mock.patch.object(
            reader, "iter_read_spec_tensors", read
        ), mock.patch.object(diag.torch.cuda, "synchronize", sync):
            if fail_at:
                with self.assertRaisesRegex(
                    RuntimeError, "device fault"
                ), self.assertLogs(diag.log, level="ERROR") as logged:
                    list(traced._consume_single_batch())
                self.assertIn("stage=consumer_before_release batch=7", logged.output[0])
            else:
                self.assertEqual(len(list(traced._consume_single_batch())), 1)
        return actions

    def test_ordinary_is_untouched_and_trace_does_not_synchronize(self):
        self.assertEqual(self.exercise_consumer(None), ["dispatch", "close"])
        self.assertEqual(self.exercise_consumer("trace"), ["dispatch", "close"])

    def test_fenced_order(self):
        self.assertEqual(
            self.exercise_consumer("fenced"),
            ["sync", "dispatch", "sync", "close", "sync"],
        )

    def test_fence_fault_still_releases_batch(self):
        self.assertEqual(
            self.exercise_consumer("fenced", fail_at=2),
            ["sync", "dispatch", "sync", "close"],
        )

    def test_invalid_mode_fails_before_querying_cuda(self):
        with mock.patch.dict(diag.os.environ, {"RTP_V4_XGRAMMAR_FST_DIAG": "bad"}):
            with self.assertRaisesRegex(ValueError, "trace or fenced"):
                diag.V4FstTrace.from_env("cuda")


if __name__ == "__main__":
    unittest.main()
