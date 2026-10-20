import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import torch

from rtp_llm.models_py.modules.qwen4_exp.qsa import fmha_inputs, fmha_validation
from rtp_llm.models_py.modules.qwen4_exp.qsa.target_metadata_triton import (
    try_target_metadata,
)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class QsaMainMetadataTest(unittest.TestCase):
    def fixture(self, batch=4, query=4):
        prefix = torch.tensor(
            [127 + i for i in range(batch)], dtype=torch.int32
        ).pin_memory()
        lengths = torch.full((batch,), query, dtype=torch.int32).pin_memory()
        table = torch.arange(1, batch * 2 + 1, device="cuda", dtype=torch.int32).view(
            batch, 2
        )
        inputs = SimpleNamespace(
            is_cuda_graph=True,
            prefix_lengths=prefix,
            input_lengths=lengths,
            kv_cache_kernel_block_id_device=table,
        )
        owner = SimpleNamespace(
            is_target_verify=True,
            is_prefill=True,
            attn_inputs=inputs,
            q_width=3072,
            kv_width=256,
            kv_head_num=1,
            head_dim=256,
            attn_configs=SimpleNamespace(kernel_tokens_per_block=128),
            _selected_indices=None,
            _is_draft_graph=lambda: False,
            _is_prefix_reuse_prefill=lambda: False,
            _paged_kv_scale_args=Mock(),
            _validate_main_paged_geometry=Mock(return_value=128),
        )
        source = torch.randn(batch * query, 3584, device="cuda", dtype=torch.bfloat16)
        cache = SimpleNamespace(
            kv_cache_base=torch.zeros(
                batch * 2 + 1, 65536, device="cuda", dtype=torch.float8_e4m3fn
            )
        )
        selected = torch.full(
            (batch * query, 2051), -1, device="cuda", dtype=torch.int32
        )
        return owner, source, cache, selected

    def test_preflight_metadata_is_layer_owned_consumed_once_and_not_stale(self):
        owner, source, cache, selected = self.fixture()
        visible = try_target_metadata(owner.attn_inputs.prefix_lengths.cuda(), 4, 4)[0]
        runtime = SimpleNamespace(
            main_cache=cache,
            main_inputs=owner.attn_inputs,
            _target_verify_plan=SimpleNamespace(visible_lengths=visible),
            validate_before_projection=Mock(),
        )
        fmha_validation.validate_qsa_before_side_write(owner, runtime, object(), source)
        plan = fmha_inputs._prepare_paged_inputs(owner, source, cache, selected)
        self.assertIs(plan.kv_lens, visible)
        self.assertIsNone(owner._qsa_target_visible_lengths)
        owner.attn_inputs.prefix_lengths.add_(1)
        fallback = fmha_inputs._prepare_paged_inputs(owner, source, cache, selected)
        self.assertIsNot(fallback.kv_lens, visible)
        torch.testing.assert_close(fallback.kv_lens, visible + 1, rtol=0, atol=0)

    def test_foreign_or_invalid_metadata_rejected_before_any_writer(self):
        owner, source, cache, selected = self.fixture()
        good = try_target_metadata(owner.attn_inputs.prefix_lengths.cuda(), 4, 4)[0]
        for foreign, tokens, visible in (
            (object(), 16, good),
            (owner.attn_inputs, 15, good),
            (owner.attn_inputs, 16, good.long()),
            (owner.attn_inputs, 16, good[:, :3]),
        ):
            owner._qsa_target_visible_lengths = (foreign, tokens, visible)
            with self.assertRaisesRegex(
                RuntimeError, "metadata owner/geometry changed"
            ):
                fmha_inputs._prepare_paged_inputs(owner, source, cache, selected)
            self.assertIsNone(owner._qsa_target_visible_lengths)
            self.assertEqual(
                int(cache.kv_cache_base.view(torch.uint8).count_nonzero()), 0
            )

    def test_two_graphs_use_live_producer_values_without_cross_layer_state(self):
        owner, source, cache, selected = self.fixture()
        prefixes = owner.attn_inputs.prefix_lengths.cuda()
        graphs = []
        outputs = []
        for _ in range(3):
            try_target_metadata(prefixes, 4, 4)
        for _ in range(2):
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                visible = try_target_metadata(prefixes, 4, 4)[0]
                owner._qsa_target_visible_lengths = (owner.attn_inputs, 16, visible)
                outputs.append(
                    fmha_inputs._prepare_paged_inputs(
                        owner, source, cache, selected
                    ).kv_lens
                )
            graphs.append(graph)
        for generation in range(8):
            owner.attn_inputs.prefix_lengths.add_(1)
            prefixes.copy_(owner.attn_inputs.prefix_lengths)
            graphs[generation % 2].replay()
            expected = (
                prefixes[:, None]
                + torch.arange(1, 5, device="cuda", dtype=torch.int32)[None, :]
            )
            torch.testing.assert_close(
                outputs[generation % 2], expected, rtol=0, atol=0
            )
        for graph in graphs:
            graph.reset()


if __name__ == "__main__":
    unittest.main()
