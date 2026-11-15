"""PPU vision dispatch, cached/CP image rows and visual router contracts.

These are CPU component tests of the production helpers, not device or P/D
acceptance. Device attention and graph cases have their own explicit target.
"""

import json
import os
import tempfile
import unittest
import weakref
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn.functional as F
from rtp_llm.models.deepseek_v41 import DeepSeekV41
from rtp_llm.models_py.modules.dsv41.ppu.layers import (
    PPUSharedEmbedding,
    PPUSharedGate,
    PPUSharedMoE,
)
from rtp_llm.models_py.modules.dsv41.ppu.model import DeepSeekV41PPUModel
from rtp_llm.models_py.modules.dsv41.ppu.model_runtime import SharedNativeRuntime
from rtp_llm.models_py.modules.dsv41.ppu.moe import sqrtsoftplus_gate
from rtp_llm.models_py.modules.dsv41.ppu.prefill_runtime import SharedForwardState
from rtp_llm.multimodal.multimodal_mixins.deepseek_v41 import (
    deepseek_v41_vision,
    packed_vision,
)
from rtp_llm.utils.model_weight import W


class _ImageOwner:
    _global_image_rows = DeepSeekV41PPUModel._global_image_rows
    _image_row_indices = DeepSeekV41PPUModel._image_row_indices
    _embed_image_rows = DeepSeekV41PPUModel._embed_image_rows
    _engram_input_ids = DeepSeekV41PPUModel._engram_input_ids

    def __init__(self, locations, lengths, cp=None):
        self.values = torch.arange(sum(lengths) * 4).view(-1, 4).to(torch.bfloat16)
        self._image_plan = (self.values, torch.tensor(locations), lengths, cp)
        self.v4 = SimpleNamespace(_image_token_mask=None)
        self.shared_weight_views = SimpleNamespace(
            geometry=SimpleNamespace(image_token_id=13, vocab_size=16)
        )


class ImageEmbeddingTest(unittest.TestCase):
    def test_negative_location_preserves_source_offset_after_prefix_reuse(self):
        owner = _ImageOwner([-2, 3], [4, 3])
        rows = owner._image_row_indices(6, "cpu")
        self.assertEqual(rows.tolist(), [2, 3, -1, 4, 5, 6])

    def test_hashed_and_negative_visual_ids_never_reach_embedding_gather(self):
        owner = _ImageOwner([-2, 3], [4, 3])
        weight = torch.arange(16 * 4).view(16, 4).to(torch.bfloat16)
        embedding = PPUSharedEmbedding(weight)
        embedding.bind_owner(owner)
        ids = torch.tensor([-1, 2147483647, 5, -7, 999, -1])
        saved_ids, saved_values = ids.clone(), owner.values.clone()
        actual = embedding(ids)
        torch.testing.assert_close(actual[2], weight[5], rtol=0, atol=0)
        torch.testing.assert_close(
            actual[[0, 1, 3, 4, 5]], owner.values[[2, 3, 4, 5, 6]], rtol=0, atol=0
        )
        self.assertEqual(
            owner.v4._image_token_mask.tolist(), [True, True, False, True, True, True]
        )
        self.assertTrue(torch.equal(ids, saved_ids))
        self.assertTrue(torch.equal(owner.values, saved_values))
        owner._image_plan = None
        text_ids = torch.tensor([1, 7, 2])
        torch.testing.assert_close(
            embedding(text_ids), F.embedding(text_ids, weight), rtol=0, atol=0
        )

    def test_cp_shuffle_and_padding_use_rank_local_image_rows(self):
        cp = SimpleNamespace(
            prefill_shuffle_indices=torch.tensor([0, 1, 6, 9]),
            prefill_actual_input_lengths_cpu=torch.tensor([8]),
            prefill_cp_chunk_lengths=torch.tensor([4]),
        )
        owner = _ImageOwner([1], [6], cp)
        self.assertEqual(owner._image_row_indices(4, "cpu").tolist(), [-1, 0, 5, -1])
        weight = torch.arange(16 * 4).view(16, 4).to(torch.bfloat16)
        actual = owner._embed_image_rows(torch.tensor([2, -1, -1, 0]), weight)
        torch.testing.assert_close(actual[[1, 2]], owner.values[[0, 5]], rtol=0, atol=0)
        torch.testing.assert_close(actual[[0, 3]], weight[[2, 0]], rtol=0, atol=0)

    def test_location_count_mismatch_fails_before_embedding(self):
        owner = _ImageOwner([1, 4], [3])
        with self.assertRaisesRegex(ValueError, "locations"):
            owner._image_row_indices(8, "cpu")


class EngramVisualIdsTest(unittest.TestCase):
    def make_runtime(self):
        owner = _ImageOwner([-1], [3])
        owner.v4.layers = [SimpleNamespace()]
        runtime = SharedNativeRuntime.__new__(SharedNativeRuntime)
        runtime._model = weakref.ref(owner)
        runtime.device, runtime.hash_params = torch.device("cpu"), object()
        runtime.geometry = SimpleNamespace(
            image_token_id=13, engram_max_ngram_size=3, engram_layer_ids=[0]
        )
        runtime.current = SimpleNamespace(
            prefill=True, attn=SimpleNamespace(context_parallel_info=object())
        )
        ids = torch.tensor([-7, 2147483647, 5, 0], dtype=torch.int32)
        windows = torch.tensor([[13, -1, -1], [13, 13, -1], [5, 13, 13], [-1, -1, -1]])
        inputs = SimpleNamespace(input_ids=ids, engram_token_windows=windows)
        return runtime, owner, inputs

    def test_semantic_windows_and_visual_sentinels_keep_the_same_history(self):
        runtime, owner, inputs = self.make_runtime()
        original = inputs.input_ids.clone()
        original_windows = inputs.engram_token_windows.clone()
        with patch(
            "rtp_llm.models_py.modules.dsv41.ppu.model_runtime.compute_engram_hash_ids",
            return_value=torch.zeros((4, 1, 2), dtype=torch.long),
        ) as hashing:
            runtime.prepare_hashes(inputs, idle=False)
        self.assertTrue(torch.equal(inputs.input_ids, original))
        self.assertTrue(torch.equal(inputs.engram_token_windows, original_windows))
        self.assertTrue(
            torch.equal(hashing.call_args.kwargs["dead_mask"], original_windows == 13)
        )
        self.assertEqual(
            owner.v4.layers[0].engram_token_mask.tolist(), [False, False, True, False]
        )
        owner._embed_image_rows(
            inputs.input_ids, torch.zeros(16, 4, dtype=torch.bfloat16)
        )
        routed_ids = torch.where(owner.v4._image_token_mask, 16, inputs.input_ids)
        state = SharedForwardState.__new__(SharedForwardState)
        state.runtime = runtime
        state.prefill, state.compacted = True, False
        state.executed_layers = []
        state.window_current_tokens = runtime.current.window_current_tokens
        state.context = SimpleNamespace(
            local_is_real=torch.tensor([True, True, True, False])
        )
        hidden, positions, cu = torch.zeros(4, 4), torch.arange(4), torch.tensor([0, 3])
        got = state.before_block(
            SimpleNamespace(layer_id=0), hidden, routed_ids, positions, cu
        )
        self.assertIs(got[1], routed_ids)
        self.assertEqual(routed_ids.tolist(), [16, 16, 5, 0])
        self.assertTrue(torch.equal(inputs.input_ids, original))
        with self.assertRaisesRegex(ValueError, "disagree"):
            state.before_block(
                SimpleNamespace(layer_id=0),
                hidden,
                torch.tensor([16, 16, 6, 0]),
                positions,
                cu,
            )

    def test_nonimage_text_and_noncanonical_history_still_fail(self):
        for changed in ("text", "history", "raw_hash_history"):
            with self.subTest(changed=changed):
                runtime, owner, inputs = self.make_runtime()
                if changed == "text":
                    inputs.input_ids[2] = 6
                elif changed == "history":
                    inputs.engram_token_windows[0, 0] = 12
                else:
                    inputs.engram_token_windows[0, 0] = 2147483647
                with patch(
                    "rtp_llm.models_py.modules.dsv41.ppu.model_runtime.compute_engram_hash_ids"
                ) as hashing:
                    with self.assertRaisesRegex(ValueError, "disagree"):
                        runtime.prepare_hashes(inputs, idle=False)
                    hashing.assert_not_called()

    def test_text_fast_path_and_explicit_rank_local_mask(self):
        owner = _ImageOwner([1], [1])
        ids = torch.tensor([2, 16, 5, -1], dtype=torch.int32)
        before = ids.clone()
        semantic = owner._engram_input_ids(
            ids, torch.tensor([False, True, False, False])
        )
        self.assertEqual(semantic.tolist(), [2, 13, 5, -1])
        self.assertEqual(semantic.dtype, ids.dtype)
        self.assertTrue(torch.equal(ids, before))
        owner._image_plan = None
        self.assertIs(owner._engram_input_ids(ids), ids)


class VisualGateTest(unittest.TestCase):
    def test_visual_bias_changes_ranking_without_changing_weight_scores(self):
        hidden = torch.tensor([[1, 2], [2, -1]], dtype=torch.bfloat16)
        weight = torch.tensor([[1, 0], [0, 1], [1, 1], [-1, 1]], dtype=torch.bfloat16)
        text_bias = torch.tensor([20.0, 10.0, 0.0, 0.0])
        visual_bias = torch.tensor([0.0, 0.0, 10.0, 20.0])
        image_rows = torch.tensor([False, True])
        ids, weights = sqrtsoftplus_gate(
            hidden,
            weight,
            text_bias,
            top_k=2,
            image_rows=image_rows,
            router_bias_vl=visual_bias,
        )
        original = F.softplus(hidden.float() @ weight.float().T).sqrt()
        expected_ids = (
            (original + torch.stack([text_bias, visual_bias])).topk(2, dim=-1).indices
        )
        expected_weights = original.gather(1, expected_ids)
        expected_weights = (
            expected_weights / (expected_weights.sum(1, keepdim=True) + 1e-12) * 1.5
        )
        self.assertTrue(torch.equal(ids, expected_ids))
        torch.testing.assert_close(weights, expected_weights)

    def test_invalid_visual_mask_rejected(self):
        with self.assertRaisesRegex(ValueError, "visual router"):
            sqrtsoftplus_gate(
                torch.zeros(2, 4),
                torch.zeros(4, 4),
                torch.zeros(4),
                top_k=2,
                image_rows=torch.ones(2, dtype=torch.int32),
                router_bias_vl=torch.ones(4),
            )

    def test_visual_direct_routes_keep_padding_inactive(self):
        image_rows = torch.tensor([True, False])
        valid = torch.tensor([True, False])
        count = torch.full((), -1, dtype=torch.int64)
        ids, weights = sqrtsoftplus_gate(
            torch.zeros(2, 4),
            torch.zeros(4, 4),
            torch.tensor([4.0, 3.0, 2.0, 1.0]),
            top_k=2,
            image_rows=image_rows,
            router_bias_vl=torch.tensor([1.0, 2.0, 3.0, 4.0]),
            graph_routes=(valid, count),
        )
        self.assertEqual(ids[0, :2].tolist(), [3, 2])
        self.assertTrue((ids[1] == -1).all())
        self.assertTrue((weights[1] == 0).all())
        self.assertEqual(int(count), 1)

    def test_moe_forwards_visual_mask_to_direct_route_consumer(self):
        tensors = {
            W.v4_router_w: torch.zeros(4, 4, dtype=torch.bfloat16),
            W.v4_router_bias: torch.tensor([4.0, 3.0, 2.0, 1.0]),
            W.v4_router_bias_vl: torch.tensor([1.0, 2.0, 3.0, 4.0]),
        }
        views = SimpleNamespace(
            tensor=lambda layer, key: tensors[key],
            geometry=SimpleNamespace(
                vocab_size=16, n_activated_experts=2, routed_scaling_factor=1.5
            ),
        )
        moe = PPUSharedMoE.__new__(PPUSharedMoE)
        torch.nn.Module.__init__(moe)
        moe.routed_enabled, moe.calls, moe.layer_id = True, 0, 0
        moe.gate, moe.shared_expert = PPUSharedGate(views, 0), None
        state = SimpleNamespace(prefill=False, routed_rows={})
        moe.runtime = SimpleNamespace(
            current=state, model=SimpleNamespace(_image_plan=object())
        )
        seen = []

        def consume(hidden, ids, weights, **kwargs):
            seen.append(ids.clone())
            return hidden

        moe.ops = SimpleNamespace(moe=consume)
        with patch(
            "rtp_llm.models_py.modules.dsv41.ppu.layers._local_prefill_direct_moe_real",
            return_value=torch.ones(2, dtype=torch.bool),
        ):
            moe(torch.zeros(2, 4, dtype=torch.bfloat16), torch.tensor([16, 5]))
        self.assertEqual(seen[0][:, :2].tolist(), [[3, 2], [0, 1]])


class AttentionDispatchTest(unittest.TestCase):
    def test_ppu_config_enables_complete_multimodal_contract(self):
        text = {
            "num_hidden_layers": 40,
            "hidden_size": 5120,
            "vocab_size": 129280,
            "num_attention_heads": 64,
            "num_key_value_heads": 1,
            "head_dim": 512,
            "qk_rope_head_dim": 64,
            "q_lora_rank": 1280,
            "compress_ratios": [0, 0] + [2] * 18 + [1] * 20 + [0] * 3,
            "o_groups": 8,
            "o_lora_rank": 1024,
            "sliding_window": 128,
            "index_head_dim": 128,
            "index_n_heads": 32,
            "index_topk": 512,
            "scoring_func": "sqrtsoftplus",
            "routed_scaling_factor": 1.5,
            "num_experts_per_tok": 6,
            "n_routed_experts": 384,
            "moe_intermediate_size": 2304,
            "n_shared_experts": 1,
            "max_position_embeddings": 131072,
            "kv_source_layer_ids": [0, 20],
        }
        raw = {
            "model_type": "deepseek_v41",
            "text_config": text,
            "vision_config": {"hidden_size": 1024, "num_hidden_layers": 32},
            "quantization_config": {
                "quant_method": "fp8",
                "activation_scheme": "dynamic",
                "weight_block_size": [32, 32],
                "scale_fmt": "ue8m0",
                "expert_dtype": "fp4",
            },
            "dtype": "bfloat16",
            "image_token_id": 129264,
        }
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "config.json").write_text(json.dumps(raw))
            with patch(
                "rtp_llm.models.deepseek_v41.v41_backend_for_config", return_value="ppu"
            ):
                config = DeepSeekV41._create_config(directory)
        self.assertTrue(config.mm_model_config.is_multimodal)
        self.assertEqual(config.mm_model_config.mm_sep_tokens, [[129264]])
        self.assertEqual(config.mm_model_config.semantic_mm_token_id, 129264)
        self.assertEqual(config.mm_related_params.config["v41_config"], raw)

    def test_ppu_default_and_explicit_fa4_rejection(self):
        config = {"vision_dim": 1024, "vision_n_heads": 16}
        with patch.object(deepseek_v41_vision, "is_ppu", return_value=True), patch.dict(
            os.environ, {}, clear=True
        ):
            self.assertEqual(
                deepseek_v41_vision.V41VisionAttention(config)._attention_backend, "fa2"
            )
            with patch.dict(os.environ, {"V41_VISION_ATTN_BACKEND": "fa4"}):
                with self.assertRaisesRegex(ValueError, "NVIDIA"):
                    deepseek_v41_vision.V41VisionAttention(config)

    def test_fa2_mixed_grids_preserve_boundaries(self):
        from rtp_llm.models_py.modules.dsv41 import vision_attention

        q, k, v = [torch.zeros(9, 16, 64) for _ in range(3)]
        expected = torch.ones_like(q)
        with patch.object(
            vision_attention, "apply_vision_fa2_varlen", return_value=expected
        ) as attention:
            actual = packed_vision._packed_attention(
                q, k, v, [(1, 3), (2, 3)], [3, 6], backend="fa2"
            )
        self.assertEqual(tuple(actual.shape), (9, 1024))
        self.assertEqual(attention.call_args.args[3].tolist(), [0, 3, 9])
        self.assertEqual(attention.call_args.args[4], 6)


@unittest.skipUnless(
    os.environ.get("RTP_V41_TEST_VISION_PPU_DEVICE") == "1", "explicit PPU device run"
)
class PPUAttentionDeviceTest(unittest.TestCase):
    def setUp(self):
        from rtp_llm.device.device_type import is_ppu

        self.assertTrue(is_ppu(), "explicit PPU device run requires a real PPU")

    @torch.inference_mode()
    def test_varlen_attention_isolates_images_and_matches_independent_math(self):
        from rtp_llm.models_py.modules.dsv41.vision_attention import (
            apply_vision_fa2_varlen,
        )

        torch.manual_seed(4114)
        qkv = torch.randn(26, 3072, device="cuda", dtype=torch.bfloat16)
        q, k, v = [part.view(26, 16, 64) for part in qkv.chunk(3, -1)]
        cu = torch.tensor([0, 9, 26], device="cuda", dtype=torch.int32)
        actual = apply_vision_fa2_varlen(q, k, v, cu, 17)
        self.assertIsNotNone(actual)
        expected = []
        for start, stop in ((0, 9), (9, 26)):
            parts = [
                part[start:stop].float().transpose(0, 1).unsqueeze(0)
                for part in (q, k, v)
            ]
            expected.append(
                F.scaled_dot_product_attention(*parts).squeeze(0).transpose(0, 1)
            )
        torch.testing.assert_close(
            actual.float(), torch.cat(expected), rtol=0.02, atol=0.02
        )
        changed = v.clone()
        changed[9:].add_(7)
        changed_output = apply_vision_fa2_varlen(q, k, changed, cu, 17)
        self.assertTrue(torch.equal(actual[:9], changed_output[:9]))
        self.assertFalse(torch.equal(actual[9:], changed_output[9:]))

    @torch.inference_mode()
    def test_graph_replay_consumes_changed_pixels_and_image_boundaries(self):
        from rtp_llm.models_py.modules.dsv41.vision_attention import (
            apply_vision_fa2_varlen,
        )

        qkv = torch.randn(26, 3072, device="cuda", dtype=torch.bfloat16)
        q, k, v = [part.view(26, 16, 64) for part in qkv.chunk(3, -1)]
        cu = torch.tensor([0, 9, 26], device="cuda", dtype=torch.int32)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            apply_vision_fa2_varlen(q, k, v, cu, 17)
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            output = apply_vision_fa2_varlen(q, k, v, cu, 17)
        for offsets in ([0, 17, 26], [0, 9, 26]):
            qkv.copy_(torch.randn_like(qkv))
            cu.copy_(torch.tensor(offsets, device="cuda", dtype=torch.int32))
            graph.replay()
            eager = apply_vision_fa2_varlen(q, k, v, cu, 17)
            torch.testing.assert_close(output, eager, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
