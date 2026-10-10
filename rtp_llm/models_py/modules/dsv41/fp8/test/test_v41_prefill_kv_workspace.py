"""Interleaved KV reuse, invalidation, and CUDA foreach ordering."""

import importlib.util
import unittest
import weakref
from pathlib import Path
from unittest.mock import patch

import torch

spec = importlib.util.spec_from_file_location(
    "kv_workspace", Path(__file__).resolve().parents[1] / "_v41_prefill_kv_workspace.py"
)
workspace = importlib.util.module_from_spec(spec)
spec.loader.exec_module(workspace)


def reference(globals_by_req, swa):
    return torch.cat([t for (g, _), sw in zip(globals_by_req, swa) for t in (g, sw)])


def inputs(batch=3, device="cuda"):
    globals_by_req = [
        (
            torch.full(
                ([16384, 32768, 65536][i % 3], 512),
                i + 1,
                dtype=torch.bfloat16,
                device=device,
            ),
            None,
        )
        for i in range(batch)
    ]
    swa = [
        torch.full(
            ([255, 639, 2175, 4223][i % 4], 512),
            -i - 1,
            dtype=torch.bfloat16,
            device=device,
        )
        for i in range(batch)
    ]
    return globals_by_req, swa


class CPUFallbackTest(unittest.TestCase):
    def test_cpu_never_retains_batch_workspace(self):
        shared = {}
        globals_by_req = [(torch.ones(3, 4), None), (torch.ones(5, 4) * 2, None)]
        swa = [torch.zeros(2, 4), torch.zeros(1, 4)]
        output = workspace.combine_kv(shared, globals_by_req, swa)
        self.assertTrue(torch.equal(output, reference(globals_by_req, swa)))
        self.assertNotIn("prefill_kv_workspace", shared)

    def test_single_request_reuse_still_works(self):
        shared = {}
        globals_by_req = [(torch.ones(3, 4), None)]
        swa = [torch.zeros(2, 4)]
        first = workspace.combine_kv(shared, globals_by_req, swa)
        swa[0].fill_(7)
        self.assertIs(workspace.combine_kv(shared, globals_by_req, swa), first)
        self.assertTrue(torch.equal(first, reference(globals_by_req, swa)))


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class CUDAWorkspaceTest(unittest.TestCase):
    def test_small_mixed_batch_negative_cache_and_readmission(self):
        shared = {}
        globals_by_req = [
            (
                torch.full(
                    ([127, 511, 1021][i % 3], 512),
                    i + 1,
                    device="cuda",
                    dtype=torch.bfloat16,
                ),
                None,
            )
            for i in range(32)
        ]
        swa = [
            torch.zeros(
                ([0, 13, 67, 129][i % 4], 512), device="cuda", dtype=torch.bfloat16
            )
            for i in range(32)
        ]
        with patch.object(
            torch, "_foreach_copy_", side_effect=AssertionError("small batch foreach")
        ):
            for change in ("first", "layout", "order", "source"):
                if change == "layout":
                    swa[0] = torch.ones((19, 1024), device="cuda")[:, ::2]
                elif change == "order":
                    globals_by_req.reverse()
                    swa.reverse()
                elif change == "source":
                    globals_by_req[0] = (globals_by_req[0][0].clone(), None)
                output = workspace.combine_kv(shared, globals_by_req, swa)
                self.assertTrue(torch.equal(output, reference(globals_by_req, swa)))
                self.assertIsNone(shared["prefill_kv_workspace"][1])
                self.assertIsNone(shared["prefill_kv_workspace"][2])
                ref = weakref.ref(output)
                del output
                self.assertIsNone(ref())
        # A new, profitable source group must not inherit the cached rejection.
        globals_by_req, swa = inputs(32)
        output = workspace.combine_kv(shared, globals_by_req, swa)
        self.assertIs(shared["prefill_kv_workspace"][2], output)
        self.assertTrue(torch.equal(output, reference(globals_by_req, swa)))
        self.assertIs(workspace.combine_kv(shared, globals_by_req, swa), output)

    def test_raw_bf16_bits_and_empty_tail(self):
        shared = {}
        globals_by_req, swa = inputs()
        swa[0] = swa[0][:0]
        output = workspace.combine_kv(shared, globals_by_req, swa)
        # Include signed zero, NaN payloads, infinities and arbitrary mantissas.
        for sw in swa:
            sw.view(torch.int16).random_(-32768, 32767)
        self.assertIs(workspace.combine_kv(shared, globals_by_req, swa), output)
        self.assertTrue(
            torch.equal(
                output.view(torch.uint8),
                reference(globals_by_req, swa).view(torch.uint8),
            )
        )

    def test_misaligned_contiguous_source_falls_back(self):
        shared = {}
        globals_by_req, swa = inputs()
        workspace.combine_kv(shared, globals_by_req, swa)
        base = torch.ones(swa[0].numel() + 1, device="cuda", dtype=torch.bfloat16)
        swa[0] = base[1:].view_as(swa[0])
        self.assertTrue(swa[0].is_contiguous())
        output = workspace.combine_kv(shared, globals_by_req, swa)
        self.assertNotIn("prefill_kv_workspace", shared)
        self.assertTrue(torch.equal(output, reference(globals_by_req, swa)))

    def test_mixed_batch32_exact_bytes_and_no_swa_retention(self):
        shared = {}
        globals_by_req, swa = inputs(32)
        output = workspace.combine_kv(shared, globals_by_req, swa)
        self.assertTrue(
            torch.equal(
                output.view(torch.uint8),
                reference(globals_by_req, swa).view(torch.uint8),
            )
        )
        previous = [weakref.ref(t) for t in swa]
        swa = [torch.full_like(t, i + 97) for i, t in enumerate(swa)]
        self.assertTrue(all(ref() is None for ref in previous))
        self.assertIs(workspace.combine_kv(shared, globals_by_req, swa), output)
        self.assertTrue(
            torch.equal(
                output.view(torch.uint8),
                reference(globals_by_req, swa).view(torch.uint8),
            )
        )
        for i, (g, _) in enumerate(globals_by_req):
            self.assertTrue(torch.all(g == i + 1).item())

    def test_source_replacement_reordering_shape_change_and_clear(self):
        shared = {}
        globals_by_req, swa = inputs()
        output = workspace.combine_kv(shared, globals_by_req, swa)
        for mutation in ("source", "order", "shape"):
            with self.subTest(mutation=mutation):
                old = weakref.ref(output)
                del output
                if mutation == "source":
                    old_source = weakref.ref(globals_by_req[1][0])
                    globals_by_req[1] = (
                        torch.full_like(globals_by_req[1][0], 53),
                        None,
                    )
                elif mutation == "order":
                    globals_by_req.reverse()
                    swa.reverse()
                else:
                    swa[0] = swa[0][:-1]
                output = workspace.combine_kv(shared, globals_by_req, swa)
                self.assertIsNone(old())
                if mutation == "source":
                    self.assertIsNone(old_source())
                self.assertTrue(torch.equal(output, reference(globals_by_req, swa)))
        old = weakref.ref(output)
        del output
        shared.pop("prefill_kv_workspace")
        self.assertIsNone(old())

    def test_unsupported_layout_dtype_and_small_global_drop_cache(self):
        for reason in ("strided", "dtype", "small", "copy_volume"):
            with self.subTest(reason=reason):
                shared = {}
                globals_by_req, swa = inputs()
                output = workspace.combine_kv(shared, globals_by_req, swa)
                old = weakref.ref(output)
                del output
                if reason == "strided":
                    swa[0] = torch.zeros(
                        (255, 1024), device="cuda", dtype=torch.bfloat16
                    )[:, ::2]
                elif reason == "dtype":
                    swa[0] = swa[0].float()
                elif reason == "small":
                    globals_by_req = [(g[:1], None) for g, _ in globals_by_req]
                else:
                    swa = [torch.zeros_like(g) for g, _ in globals_by_req]
                    swa[0] = torch.cat((swa[0], swa[0]))
                output = workspace.combine_kv(shared, globals_by_req, swa)
                self.assertIsNone(old())
                if reason == "small":
                    self.assertIsNone(shared["prefill_kv_workspace"][2])
                    for sw in swa:
                        sw.fill_(18)
                    output = workspace.combine_kv(shared, globals_by_req, swa)
                else:
                    self.assertNotIn("prefill_kv_workspace", shared)
                self.assertTrue(torch.equal(output, reference(globals_by_req, swa)))

    def test_batch_single_batch_transition(self):
        shared = {}
        globals_by_req, swa = inputs()
        workspace.combine_kv(shared, globals_by_req, swa)
        for gs, ss in ((globals_by_req[:1], [swa[0][:128]]), (globals_by_req, swa)):
            output = workspace.combine_kv(shared, gs, ss)
            self.assertTrue(torch.equal(output, reference(gs, ss)))

    def test_nondefault_stream_order_and_consumer_snapshot(self):
        globals_by_req, swa = inputs()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        shared = {}
        with torch.cuda.stream(stream):
            first = workspace.combine_kv(shared, globals_by_req, swa)
            for sw in swa:
                sw.fill_(91)
            self.assertIs(workspace.combine_kv(shared, globals_by_req, swa), first)
            snapshot = first.clone()
            expected = reference(globals_by_req, swa)
            for sw in swa:
                sw.fill_(17)
            self.assertIs(workspace.combine_kv(shared, globals_by_req, swa), first)
            expected_updated = reference(globals_by_req, swa)
        torch.cuda.current_stream().wait_stream(stream)
        self.assertTrue(torch.equal(snapshot, expected))
        self.assertTrue(torch.equal(first, expected_updated))

    def test_graph_replay_updates_same_output(self):
        globals_by_req, swa = inputs()
        shared = {}
        output = workspace.combine_kv(shared, globals_by_req, swa)
        workspace.combine_kv(shared, globals_by_req, swa)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            workspace.combine_kv(shared, globals_by_req, swa)
        for value in (39, -13):
            for sw in swa:
                sw.fill_(value)
            graph.replay()
            self.assertTrue(torch.equal(output, reference(globals_by_req, swa)))


if __name__ == "__main__":
    unittest.main(verbosity=2)
