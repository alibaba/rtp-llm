"""Qualify explicit rotary computation without changing default construction."""

import unittest
from unittest.mock import patch

import torch
from rtp_llm.models_py.modules.dsv4 import rope
from rtp_llm.models_py.modules.dsv4.fp8.test import (
    test_v41_cuda_frequency_reference_sm120 as frequency_scope,
)
from rtp_llm.models_py.modules.dsv4.fp8.test.v41_output_precision import (
    output_projection_reference,
)
from rtp_llm.models_py.modules.dsv4.fp8.test.v41_tp8_projection import tp8_projection
from rtp_llm.tools.dsv41_output_projection_replay import execute_case, immutable_sources
from rtp_llm.tools.dsv41_query_owner_replay import tensor_digest


def parameters(owner):
    return (
        owner._rope_dim,
        owner._rope_max_seq_len,
        owner._rope_o_seq_len,
        owner._rope_base,
        owner._rope_factor,
        owner._rope_beta_fast,
        owner._rope_beta_slow,
    )


class RotaryComputeDeviceSM120Test(frequency_scope.CUDAFrequencySM120Test):
    @torch.inference_mode()
    def test_explicit_memo_origins_and_ambient_defaults(self):
        default = torch.get_default_device()
        differing_tables = 0
        for ratio in (0, 1, 2):
            owner = frequency_scope.make_layer(2 if ratio else 0, ratio)
            args, device = parameters(owner), owner.freqs_cis.device
            with patch.object(rope, "_FREQS_CIS_CACHE", {}):
                legacy = rope.precompute_freqs_cis(*args, device=device)
                legacy_key = (*args, str(device))
                self.assertIs(rope._FREQS_CIS_CACHE[legacy_key], legacy)
                with torch.device(device):
                    cpu = rope.precompute_freqs_cis(
                        *args, device=device, compute_device="cpu"
                    )
                    released = self.official.precompute_freqs_cis(*args)
                self.exact(legacy, cpu)
                self.assertIsNot(legacy, cpu)
                cuda = rope.precompute_freqs_cis(
                    *args, device=device, compute_device="cuda"
                )
                self.exact(released, cuda)
                self.assertIsNot(cuda, cpu)
                self.assertIs(
                    cuda,
                    rope.precompute_freqs_cis(
                        *args, device=device, compute_device=device
                    ),
                )
                self.assertIs(
                    cpu,
                    rope.precompute_freqs_cis(
                        *args, device=device, compute_device=torch.device("cpu:0")
                    ),
                )
                self.assertIs(legacy, rope.precompute_freqs_cis(*args, device=device))
                differing_tables += tensor_digest(cpu) != tensor_digest(cuda)
                self.assertEqual(len(rope._FREQS_CIS_CACHE), 3)
                with self.assertRaisesRegex(ValueError, "CPU or CUDA"):
                    rope.precompute_freqs_cis(
                        *args, device=device, compute_device="meta"
                    )
            self.assertEqual(default, torch.get_default_device())
        self.assertGreater(differing_tables, 0)
        print(
            'DSV41_ROPE_COMPUTE_MEMO {"ratios": 3, "origins": 2, "ambient_defaults": 2}',
            flush=True,
        )

    @torch.inference_mode()
    def test_explicit_reset_held_outputs_and_restoration(self):
        with tp8_projection(True), output_projection_reference(
            True, kernel=self.kernel
        ):
            for case in self.cases:
                owner = self.owners[case["layer"]]
                original = owner.freqs_cis
                sources = immutable_sources(owner, self.shards[case["layer"]])
                before = {name: tensor_digest(value) for name, value in sources.items()}
                try:
                    device = original.device
                    owner.reset_rope_cache(device, compute_device=device)
                    self.exact(case["released_freqs_full"], owner.freqs_cis)
                    self.exact(case["outputs"]["released"], self.projection(case))
                    self.assertEqual(
                        before,
                        {name: tensor_digest(value) for name, value in sources.items()},
                    )
                finally:
                    owner.freqs_cis = original
                self.exact(case["outputs"]["candidate"], self.projection(case))
        print(
            'DSV41_ROPE_COMPUTE_HELD {"cases": 3, "outputs_exact": 3, "restorations": 3}',
            flush=True,
        )

    @torch.inference_mode()
    def test_explicit_component_graphs_with_changing_inputs(self):
        for case in (self.cases[0], self.cases[2]):
            owner = self.owners[case["layer"]]
            original = owner.freqs_cis
            rows = len(case["identity"]["input_ids"])
            attention = (
                case["attention_before"].cuda().reshape(rows, 64, 512).contiguous()
            )
            before = attention.clone()
            positions = (
                torch.arange(rows, device="cuda", dtype=torch.int64)
                + case["identity"]["start_pos"]
            )
            try:
                owner.reset_rope_cache(original.device, compute_device=original.device)
                with tp8_projection(True), output_projection_reference(
                    True, kernel=self.kernel
                ):

                    def execute():
                        return owner._project_output(
                            attention.clone(), owner.freqs_cis[positions]
                        )

                    execute()
                    execute()
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        result = execute()
                    for step in range(3):
                        attention.copy_(before * (1 + 0.25 * step))
                        positions.copy_(
                            torch.arange(rows, device="cuda")
                            + case["identity"]["start_pos"]
                            + step
                        )
                        graph.replay()
                        raw, _ = execute_case(
                            owner,
                            self.shards[case["layer"]],
                            self.official,
                            self.kernel,
                            attention.clone(),
                            owner.freqs_cis[positions],
                            self.shards[case["layer"]][0].freqs_cis[positions],
                        )
                        self.exact(raw["outputs"]["released"], result)
            finally:
                owner.freqs_cis = original
            self.assertIs(owner.freqs_cis, original)
        print(
            'DSV41_ROPE_COMPUTE_GRAPHS {"graphs": 2, "changed_replays": 6}',
            flush=True,
        )

    @torch.inference_mode()
    def test_fresh_capture_rejection_and_prepared_memo_reuse(self):
        args = parameters(self.owners[6])
        device = self.owners[6].freqs_cis.device
        with patch.object(rope, "_FREQS_CIS_CACHE", {}):
            source = torch.zeros(1, device=device)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                source.add_(1)
                with self.assertRaisesRegex(RuntimeError, "before graph capture"):
                    rope.precompute_freqs_cis(
                        *args, device=device, compute_device=device
                    )
            self.assertFalse(rope._FREQS_CIS_CACHE)
            prepared = rope.precompute_freqs_cis(
                *args, device=device, compute_device=device
            )
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                self.assertIs(
                    prepared,
                    rope.precompute_freqs_cis(
                        *args, device=device, compute_device="cuda"
                    ),
                )
                result = source + prepared.real[0, 0]
            graph.replay()
            self.exact(source + prepared.real[0, 0], result)
        print(
            'DSV41_ROPE_COMPUTE_CAPTURE {"fresh_rejected": 1, "prepared_reused": 1}',
            flush=True,
        )


if __name__ == "__main__":
    unittest.main()
