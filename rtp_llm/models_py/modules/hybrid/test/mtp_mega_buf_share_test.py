"""Clone ownership tests; real EP kernel correctness needs SM100 multi-rank runs."""

import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from rtp_llm.models_py.modules.glm5_mega_moe import (
    mega_moe,
    mega_moe_fp8,
    mega_moe_fp8_se,
    mega_moe_fused,
    mega_moe_se,
)
from rtp_llm.models_py.modules.glm5_mega_moe.mega_moe_fp8_se_wrapper import (
    MegaMoeFp8SEWrapper,
)
from rtp_llm.models_py.modules.glm5_mega_moe.mega_moe_fp8_wrapper import (
    MegaMoeFp8Wrapper,
)
from rtp_llm.models_py.modules.glm5_mega_moe.mega_moe_fused_wrapper import (
    MegaMoeFusedWrapper,
)
from rtp_llm.models_py.modules.glm5_mega_moe.mega_moe_se_wrapper import MegaMoeSEWrapper
from rtp_llm.models_py.modules.glm5_mega_moe.mega_moe_wrapper import MegaMoeWrapper

VARIANTS = (
    (mega_moe.GLM5MegaMoE, MegaMoeWrapper),
    (mega_moe_se.GLM5MegaMoESE, MegaMoeSEWrapper),
    (mega_moe_fp8.GLM5MegaMoEFP8, MegaMoeFp8Wrapper),
    (mega_moe_fp8_se.GLM5MegaMoEFP8SE, MegaMoeFp8SEWrapper),
    (mega_moe_fused.GLM5MegaMoEFused, MegaMoeFusedWrapper),
)
ALLOCATORS = (
    (mega_moe, "_get_or_create_cuda_graph_clone_buf"),
    (mega_moe_se, "get_or_create_mega_moe_se_clone_buf"),
    (mega_moe_fp8, "_get_or_create_cuda_graph_clone_buf_fp8"),
    (mega_moe_fused, "_get_or_create_cuda_graph_clone_buf_fused"),
)


def make_source(cls, wrapper_cls):
    model = cls.__new__(cls)
    nn.Module.__init__(model)
    model.cfg = SimpleNamespace()
    model._mega_group = object()
    model._mega_buf = SimpleNamespace(
        num_max_tokens_per_rank=8,
        x=torch.zeros(8, 4),
        x_sf=torch.zeros(8, 1),
        topk_idx=torch.zeros(8, 2, dtype=torch.int64),
        topk_weights=torch.zeros(8, 2),
    )
    model._mega_y = torch.zeros(8, 4)
    for name in (
        "_mega_l1_w",
        "_mega_l1_sf",
        "_mega_l2_w",
        "_mega_l2_sf",
        "_shared_l1_w",
        "_shared_l1_sf",
        "_shared_l2_w",
        "_shared_l2_sf",
        "_shared_mid_fp8",
        "_shared_mid_sf",
    ):
        setattr(model, name, torch.ones(2, 2))
    model._num_shared_experts = 1
    wrapper = wrapper_cls.__new__(wrapper_cls)
    nn.Module.__init__(wrapper)
    wrapper.mega_moe = model
    wrapper.expert_num = 4
    return wrapper


class MtpMegaBufShareTest(unittest.TestCase):
    def test_decode_shares_communication_but_not_outputs_all_variants(self):
        for cls, wrapper_cls in VARIANTS:
            with self.subTest(variant=cls.__name__), ExitStack() as stack:
                allocators = [
                    stack.enter_context(patch.object(m, n)) for m, n in ALLOCATORS
                ]
                source = make_source(cls, wrapper_cls)
                clone = source.clone_for_cuda_graph(share_mega_buf=True)
                self.assertIs(clone.mega_moe._mega_buf, source.mega_moe._mega_buf)
                self.assertIs(clone.mega_moe._mega_group, source.mega_moe._mega_group)
                for name in ("_mega_l1_w", "_mega_l1_sf", "_mega_l2_w", "_mega_l2_sf"):
                    self.assertIs(
                        getattr(clone.mega_moe, name), getattr(source.mega_moe, name)
                    )
                if cls in (
                    mega_moe_se.GLM5MegaMoESE,
                    mega_moe_fp8_se.GLM5MegaMoEFP8SE,
                    mega_moe_fused.GLM5MegaMoEFused,
                ):
                    for name in (
                        "_shared_l1_w",
                        "_shared_l1_sf",
                        "_shared_l2_w",
                        "_shared_l2_sf",
                    ):
                        self.assertIs(
                            getattr(clone.mega_moe, name),
                            getattr(source.mega_moe, name),
                        )
                clone.mega_moe._mega_y.fill_(7)
                self.assertEqual(source.mega_moe._mega_y.count_nonzero().item(), 0)
                for alloc in allocators:
                    alloc.assert_not_called()
                # CMP producers must resolve the same storage even at different batch sizes.
                for tokens in (1, 8, 3, 1):
                    clone_views = clone.prepacked_input_views(tokens)
                    source_views = source.prepacked_input_views(tokens)
                    self.assertEqual(len(clone_views), 4)
                    self.assertEqual(len(source_views), 4)
                    for dst, src in zip(clone_views, source_views):
                        self.assertEqual(dst.shape, src.shape)
                        self.assertEqual(dst.shape[0], tokens)
                        self.assertEqual(dst.data_ptr(), src.data_ptr())
                        dst.fill_(tokens)
                        torch.testing.assert_close(dst, src)
                if cls is mega_moe_fused.GLM5MegaMoEFused:
                    for name in ("_shared_mid_fp8", "_shared_mid_sf"):
                        src = getattr(source.mega_moe, name)
                        dst = getattr(clone.mega_moe, name)
                        self.assertEqual(dst.shape, src.shape)
                        self.assertEqual(dst.dtype, src.dtype)
                        dst.zero_()
                        torch.testing.assert_close(src, torch.ones_like(src))

    def test_default_prefill_clone_still_allocates_separate_buffer(self):
        for cls, wrapper_cls in VARIANTS:
            with self.subTest(variant=cls.__name__), ExitStack() as stack:
                # Distinct sentinels detect dispatch to the wrong backend allocator.
                buffers = [object() for _ in ALLOCATORS]
                allocators = [
                    stack.enter_context(patch.object(m, n, return_value=buf))
                    for (m, n), buf in zip(ALLOCATORS, buffers)
                ]
                expected_allocator = {
                    mega_moe.GLM5MegaMoE: 0,
                    mega_moe_se.GLM5MegaMoESE: 1,
                    mega_moe_fp8.GLM5MegaMoEFP8: 2,
                    mega_moe_fp8_se.GLM5MegaMoEFP8SE: 2,
                    mega_moe_fused.GLM5MegaMoEFused: 3,
                }[cls]
                source = make_source(cls, wrapper_cls)
                clone = source.clone_for_cuda_graph()
                self.assertIs(clone.mega_moe._mega_buf, buffers[expected_allocator])
                self.assertIsNot(clone.mega_moe._mega_buf, source.mega_moe._mega_buf)
                args = [
                    source.mega_moe._mega_buf,
                    source.mega_moe._mega_group,
                    source.mega_moe.cfg,
                ]
                if expected_allocator in (1, 2):
                    args.append(source.mega_moe._num_shared_experts)
                allocators[expected_allocator].assert_called_once_with(*args)

    def test_uninitialized_shared_buffer_fails_before_capture(self):
        for cls, wrapper_cls in VARIANTS:
            for missing in ("_mega_buf", "_mega_group"):
                with self.subTest(variant=cls.__name__, missing=missing):
                    source = make_source(cls, wrapper_cls)
                    setattr(source.mega_moe, missing, None)
                    with self.assertRaisesRegex(RuntimeError, "must be initialized"):
                        source.clone_for_cuda_graph(share_mega_buf=True)

    def test_model_clone_propagates_sharing_and_keeps_roles_independent(self):
        from rtp_llm.models_py.model_desc.generic_moe import (
            GenericMoeDecoderLayer,
            GenericMoeLayer,
        )
        from rtp_llm.models_py.model_desc.generic_moe_mtp import GenericMoeMTPModel

        moe = GenericMoeLayer.__new__(GenericMoeLayer)
        nn.Module.__init__(moe)
        for name in (
            "config",
            "parallelism_config",
            "hidden_dim",
            "ffn_dim",
            "num_experts",
            "top_k",
            "gate_chunk_rows",
            "gate",
            "select_topk",
            "fake_balance_expert",
            "w1",
            "w2",
            "num_local_experts",
            "add_shared_expert",
            "ffn_tp_size",
            "ep_size",
            "shared_expert",
            "shared_expert_gate",
            "sigmoid_gate_scale_add",
            "correction_bias",
            "_use_mega_moe_fused_shared",
        ):
            setattr(moe, name, None)
        moe.fused_moe = make_source(*VARIANTS[0])
        layer = GenericMoeDecoderLayer.__new__(GenericMoeDecoderLayer)
        nn.Module.__init__(layer)
        for name in (
            "layer_idx",
            "self_attn",
            "input_layernorm",
            "post_attention_layernorm",
            "_fuse_input_norm_quant",
            "_fuse_input_scale_ue8m0",
            "_fuse_post_norm_quant",
            "_fuse_post_norm_quant_moe",
            "cmp",
        ):
            setattr(layer, name, None)
        layer.mlp = moe
        from rtp_llm.models_py.modules.hybrid.glm5_cmp import Glm5Cmp

        layer.cmp = Glm5Cmp(
            layer_idx=0,
            config=SimpleNamespace(
                model_type="glm_5",
                moe_layer_index=(0,),
                attn_config=SimpleNamespace(use_mla=True, kernel_tokens_per_block=64),
            ),
            parallelism_config=SimpleNamespace(tp_size=1, get_attn_tp_size=lambda: 1),
            self_attn=SimpleNamespace(has_indexer=False),
            input_layernorm=object(),
            mlp=moe,
            post_attention_layernorm=object(),
        )
        layer.cmp._events = object()
        model = GenericMoeMTPModel.__new__(GenericMoeMTPModel)
        nn.Module.__init__(model)
        for name in (
            "config",
            "parallelism_config",
            "weight",
            "fmha_config",
            "py_hw_kernel_config",
            "micro_batch_size",
            "layer_num",
            "vocab_size",
            "pinned_mla_groups",
            "_pinned_mla_cache_key",
            "device_type",
            "moe_config",
            "max_generate_batch_size",
            "device_resource_config",
            "embed_tokens",
            "pre_fc_norm_embedding",
            "pre_fc_norm_hidden",
            "fc",
            "norm",
        ):
            setattr(model, name, None)
        model.layers = nn.ModuleList([layer])
        model._mtp_indexer_share_enabled = True
        model._mtp_indexer_role = 2
        model._mtp_shared_topk_indices = torch.zeros(8, 2, dtype=torch.int32)
        clone = model.clone_for_cuda_graph(share_mega_buf=True)
        clone.set_mtp_indexer_role(1)
        self.assertEqual(model._mtp_indexer_role, 2)
        self.assertEqual(clone._mtp_indexer_role, 1)
        self.assertIs(clone._mtp_shared_topk_indices, model._mtp_shared_topk_indices)
        self.assertIs(
            clone.layers[0].mlp.fused_moe.mega_moe._mega_buf,
            model.layers[0].mlp.fused_moe.mega_moe._mega_buf,
        )
        self.assertIsNot(clone.layers[0].mlp, model.layers[0].mlp)
        self.assertIs(clone.layers[0].cmp.mlp, clone.layers[0].mlp)
        self.assertIsNone(clone.layers[0].cmp._events)
        self.assertIsNotNone(layer.cmp._events)

        # Default model cloning must still forward no new keyword to legacy MoE.
        class LegacyMoe(nn.Module):
            def clone_for_cuda_graph(self):
                return LegacyMoe()

        moe.fused_moe = LegacyMoe()
        legacy_clone = model.clone_for_cuda_graph()
        self.assertIsInstance(legacy_clone.layers[0].mlp.fused_moe, LegacyMoe)
        self.assertIsNot(legacy_clone.layers[0].mlp.fused_moe, moe.fused_moe)
        with self.assertRaisesRegex(RuntimeError, "does not support"):
            model.clone_for_cuda_graph(share_mega_buf=True)

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA graph support")
    def test_alternating_graphs_share_staging_without_aliasing_outputs(self):
        # Real CUDA capture/replay of the ownership pattern, not an EP kernel test.
        source = make_source(*VARIANTS[0])
        buf = source.mega_moe._mega_buf
        for name in ("x", "x_sf", "topk_idx", "topk_weights"):
            setattr(buf, name, getattr(buf, name).cuda())
        source.mega_moe._mega_y = source.mega_moe._mega_y.cuda()
        clone = source.clone_for_cuda_graph(share_mega_buf=True)
        graphs = []
        for wrapper, tokens in ((source, 1), (clone, 8), (source, 3), (clone, 1)):
            inp = torch.zeros(tokens, 4, device="cuda")
            output = wrapper.mega_moe._mega_y[:tokens]
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                wrapper.prepacked_input_views(tokens)[0].copy_(inp)
                torch.mul(wrapper.prepacked_input_views(tokens)[0], 2, out=output)
            graphs.append((graph, inp, output))
        snapshots = []
        for step in range(32):
            graph, inp, output = graphs[step % len(graphs)]
            inp.fill_(step)
            graph.replay()
            snapshots.append((output.clone(), step))
        for output, step in snapshots:
            torch.testing.assert_close(output, torch.full_like(output, 2 * step))


if __name__ == "__main__":
    unittest.main()
