"""Exact math and stable graph reuse for heterogeneous high-pressure prefills."""

import itertools
import os
import shutil
import unittest
from unittest import mock

import torch
from rtp_llm.models_py.triton_kernels.kimi_kda.chunk import chunk_kda
from rtp_llm.models_py.triton_kernels.kimi_kda.full_reuse_graph import (
    _GRAPHS,
    chunk_kda_full_reuse_graph,
    graph_stats,
)


class FullReuseGraphTest(unittest.TestCase):
    def setUp(self):
        self.assertTrue(torch.cuda.is_available(), "GPU validation must not skip")
        self.env = mock.patch.dict(os.environ, {"GLM53_KDA_FULL_REUSE_GRAPH": "1"})
        self.env.start()
        self.addCleanup(self.env.stop)
        print(
            "GPU tools:",
            {name: shutil.which(name) for name in ("nsys", "ncu", "compute-sanitizer")},
        )
        torch.manual_seed(531015)
        torch.backends.cuda.matmul.allow_tf32 = False

    def inputs(self, lengths, heads=2):
        shape = (1, sum(lengths), heads, 128)
        return dict(
            q=torch.randn(shape, device="cuda", dtype=torch.bfloat16) * 0.1,
            k=torch.randn(shape, device="cuda", dtype=torch.bfloat16) * 0.1,
            v=torch.randn(shape, device="cuda", dtype=torch.bfloat16) * 0.1,
            g=torch.randn(shape, device="cuda", dtype=torch.bfloat16) * 0.1 - 4,
            beta=torch.rand(shape[:3], device="cuda"),
            initial_state=torch.randn(len(lengths), heads, 128, 128, device="cuda")
            * 0.1,
            cu_seqlens=torch.tensor(
                [0, *itertools.accumulate(lengths)], device="cuda", dtype=torch.int32
            ),
            A_log=torch.randn(heads, device="cuda") * 0.1,
            dt_bias=torch.randn(heads * 128, device="cuda") * 0.1,
        )

    def compare(self, values):
        expected = chunk_kda(
            **values,
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
            use_gate_in_kernel=True,
            safe_gate=True,
            lower_bound=-5.0,
            return_intermediate_states=True,
            fuse_state_recurrence=False,
            chunk_size=64,
        )
        actual = chunk_kda_full_reuse_graph(**values)
        self.assertIsNotNone(actual, "eligible cache tails must not fall back")
        for a, b in zip(actual, expected):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        return actual

    def test_exact_ragged_outputs_and_changed_weights(self):
        for lengths in (
            (1,),
            (31, 63, 64, 65, 129, 257, 512),
            (418,) * 7 + (419,) * 32,
            (75,) * 64,
            (513, 1024, 2048),
        ):
            with self.subTest(lengths=lengths):
                values = self.inputs(lengths)
                actual = self.compare(values)
                frozen = tuple(t.clone() for t in actual)
                for name in (
                    "q",
                    "k",
                    "v",
                    "g",
                    "beta",
                    "initial_state",
                    "A_log",
                    "dt_bias",
                ):
                    values[name].mul_(0.75)
                self.compare(values)
                for a, b in zip(actual, frozen):
                    torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_request_order_and_group_capacity_do_not_recapture(self):
        layouts = [
            (418,) * 7 + (419,) * 32,
            tuple(reversed((418,) * 7 + (419,) * 32)),
            (418,) * 8 + (419,) * 31,
            (418,) * 4 + (419,) * 35,
            (418,) * 2 + (419,) * 37,
            (418,) * 21 + (419,) * 18,
        ]
        self.compare(self.inputs(layouts[0]))
        captures = graph_stats()["captures"]
        for lengths in layouts[1:]:
            self.compare(self.inputs(lengths))
            self.assertEqual(
                graph_stats()["captures"],
                captures,
                "request order/frequency within capacity must reuse graph",
            )

    def test_heterogeneous_batch_compositions_reuse_length_groups(self):
        lengths = (31, 63, 65, 75, 129, 257, 419, 512)
        self.compare(self.inputs(lengths * 2, heads=16))
        self.compare(self.inputs(lengths, heads=16))
        captured = graph_stats()["captures"]
        for order in (lengths, lengths[::-1], lengths[2:] + lengths[:2]):
            self.compare(self.inputs(order, heads=16))
            self.assertEqual(graph_stats()["captures"], captured)
        # Changing weights and request order must not leak old outputs/states.
        self.assertGreater(graph_stats()["grouped_calls"], 0)

    def test_normalization_configuration_survives_capacity_padding(self):
        for lengths in ((419,) * 9, (129,) * 17, (256,) * 16, (256,) * 17):
            with self.subTest(lengths=lengths):
                self.compare(self.inputs(lengths, heads=16))

    def test_high_pressure_39_sequences_heads32(self):
        values = self.inputs((418,) * 7 + (419,) * 32, heads=32)
        self.compare(values)
        captured = graph_stats()["captures"]
        for _ in range(3):
            values["A_log"].add_(0.01)
            self.compare(values)
            self.assertEqual(graph_stats()["captures"], captured)
        print("High pressure graph stats:", graph_stats())

    def test_streams_and_cached_metadata_lifetimes(self):
        values = self.inputs((75, 129, 75))
        outputs = self.compare(values)
        frozen = tuple(t.clone() for t in outputs)
        # Churn FLA's external four-entry tensor cache; graph metadata must be
        # held by its own workspace, never only by that unrelated cache.
        for length in range(81, 88):
            self.compare(self.inputs((length,)))
        self.compare(values)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            self.compare(self.inputs((129, 75, 75)))
        torch.cuda.current_stream().wait_stream(stream)
        for a, b in zip(outputs, frozen):
            torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_base_group_graph_prefix_overflow_and_owned_outputs(self):
        lengths = tuple(range(80, 420, 17)) * 3
        with mock.patch.dict(os.environ, {"GLM53_KDA_BASE_GROUP_GRAPH": "1"}):
            values = self.inputs(lengths, heads=16)
            outputs = self.compare(values)
            frozen = tuple(t.clone() for t in outputs)
            self.compare(self.inputs(lengths[::-1], heads=16))
            self.compare(self.inputs(tuple(range(80, 420, 17)) * 2, heads=16))
            for a, b in zip(outputs, frozen):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
            self.assertGreater(graph_stats().get("base_replays", 0), 0)

    def test_explicit_opt_out(self):
        with mock.patch.dict(os.environ, {"GLM53_KDA_FULL_REUSE_GRAPH": "0"}):
            self.assertIsNone(chunk_kda_full_reuse_graph(**self.inputs((75,))))


if __name__ == "__main__":
    unittest.main()
