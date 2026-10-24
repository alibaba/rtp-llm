import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.config.quant_config import (
    Fp8BlockWiseQuantConfig,
    Fp8MxBlockWiseQuantConfig,
    QuantizationConfig,
)
from rtp_llm.model_loader.weight_module import AtomicWeight, WeightModule
from rtp_llm.models.minimax_m31 import MiniMaxM31
from rtp_llm.models.minimax_m31_config import configure_m31_model, finalize_m31_kv_cache
from rtp_llm.models.minimax_m31_kv_cache import (
    INDEXER_KV_TAG,
    MAIN_KV_TAG,
    build_m31_kv_cache_descs,
)
from rtp_llm.models.minimax_m31_vl import configure_m31_vl
from rtp_llm.models.minimax_m31_weights import MiniMaxM31Weight, stack_w13_global_scales
from rtp_llm.models_py.model_desc.generic_moe import DecodeLayerOutput
from rtp_llm.models_py.model_desc.hidden_state_capture import CaptureContext
from rtp_llm.models_py.model_desc.minimax_m31_activation import swiglu_oai
from rtp_llm.models_py.model_desc.minimax_m31_addressing import m31_decode_addressing
from rtp_llm.models_py.model_desc.minimax_m31_cache import m31_cache_views
from rtp_llm.models_py.model_desc.minimax_m31_cp_prefill import (
    _publish_local_pages,
    _working_pages,
    plan_m31_cp_prefill,
)
from rtp_llm.models_py.model_desc.minimax_m31_kv_write import write_m31_nvfp4_rows
from rtp_llm.models_py.model_desc.minimax_m31_model import MiniMaxM31Model
from rtp_llm.models_py.model_desc.minimax_m31_moe import (
    M31NVFP4MegaExecutor,
    M31PrefillChunkPlan,
    _M31Fp32Router,
    reorder_m31_nvfp4_checkpoint,
    run_m31_prefill_chunks,
)
from rtp_llm.models_py.model_desc.minimax_m31_moe_pack import pack_m31_nvfp4_inputs
from rtp_llm.models_py.model_desc.minimax_m31_prefill import m31_prefill_addressing
from rtp_llm.models_py.model_desc.minimax_m31_prefill_fmha import stage_m31_main_pages
from rtp_llm.ops import DataType, KVCacheSpecType
from rtp_llm.utils.model_weight import CkptWeightInfo, W, identity, transpose


class MiniMaxM31KVCacheTest(unittest.TestCase):
    def test_dspark_target_captures_configured_layer_boundaries(self):
        class Layer(torch.nn.Module):
            def forward(self, hidden, residual, attention_inputs, cache):
                return DecodeLayerOutput(hidden + 1, residual + 10)

        class Cache:
            def get_layer_cache_groups(self, layer_idx):
                return [
                    SimpleNamespace(tag=MAIN_KV_TAG),
                    SimpleNamespace(tag=INDEXER_KV_TAG),
                ]

        model = MiniMaxM31Model.__new__(MiniMaxM31Model)
        torch.nn.Module.__init__(model)
        model.kv_cache = Cache()
        model.layers = torch.nn.ModuleList([Layer() for _ in range(3)])
        model.layer_num = 3
        model.norm = lambda hidden, residual: (hidden + residual, residual)
        model._mtp_aux_capture_layer_ids = (0, 2)
        model._mtp_aux_capture_layer_id_set = frozenset((0, 2))
        self.assertTrue(model.has_mtp_hidden_buffer())
        model._mtp_target_hidden_states = None
        model._mtp_target_graph_buffer = None
        model._mtp_target_prompt_buffer = None
        model._capture_context = CaptureContext.configured(
            (), model._capture_canonical_layer, model._capture_canonical_final
        )
        inputs = SimpleNamespace(
            embedded=torch.ones((2, 3), dtype=torch.bfloat16),
            capture_hidden_states=False,
            attention_inputs={
                MAIN_KV_TAG: SimpleNamespace(is_prefill=False, is_target_verify=True),
                INDEXER_KV_TAG: SimpleNamespace(is_prefill=False),
            },
        )
        with patch.object(
            MiniMaxM31Model, "embedding", lambda self, _: inputs.embedded
        ), patch(
            "rtp_llm.models_py.model_desc.minimax_m31_model.prepare_m31_prefill_chunks",
            return_value=None,
        ) as prepare_chunks:
            output = model.forward(inputs)
        prepare_chunks.assert_not_called()
        self.assertEqual(output.mtp_target_hidden_states.shape, (2, 6))
        self.assertEqual(
            output.mtp_target_hidden_states[0].tolist(), [12, 12, 12, 34, 34, 34]
        )
        self.assertEqual(output.hidden_states[0].tolist(), [34, 34, 34])
        self.assertEqual(
            output.mtp_target_hidden_states.data_ptr(),
            model.get_mtp_target_hidden_states(2).data_ptr(),
        )
        inputs.attention_inputs[MAIN_KV_TAG].is_prefill = True
        inputs.attention_inputs[MAIN_KV_TAG].is_target_verify = False
        inputs.attention_inputs[INDEXER_KV_TAG].is_prefill = True
        with patch.object(
            MiniMaxM31Model, "embedding", lambda self, _: inputs.embedded
        ), patch(
            "rtp_llm.models_py.model_desc.minimax_m31_model.prepare_m31_prefill_chunks",
            return_value=None,
        ) as prepare_chunks:
            model.forward(inputs)
        prepare_chunks.assert_called_once()
        if torch.cuda.is_available():
            inputs.embedded = inputs.embedded.cuda()
            with patch.object(
                MiniMaxM31Model, "embedding", lambda self, _: inputs.embedded
            ), patch(
                "torch.cuda.is_current_stream_capturing", return_value=True
            ), patch(
                "rtp_llm.models_py.model_desc.minimax_m31_model.prepare_m31_prefill_chunks"
            ) as prepare_chunks:
                model.forward(inputs)
            prepare_chunks.assert_not_called()

    def test_native_page_contract(self):
        layers = build_m31_kv_cache_descs(60, kv_heads=4, head_dim=128, index_dim=128)
        self.assertEqual(len(layers), 60)
        self.assertEqual(
            [desc.tag for desc in layers[0]], [MAIN_KV_TAG, INDEXER_KV_TAG]
        )
        self.assertEqual([desc.entry_elems for desc in layers[0]], [576, 72])
        self.assertEqual([desc.entry_elems * 128 for desc in layers[0]], [73728, 9216])
        for desc in layers[0]:
            self.assertEqual(desc.cache_type, KVCacheSpecType.OPAQUE_KV)
            self.assertEqual(desc.entry_dtype, DataType.TYPE_UINT8)

    def test_separate_nvfp4_group_views(self):
        main = torch.zeros((2, 73728), dtype=torch.uint8)
        index = torch.zeros((3, 9216), dtype=torch.uint8)
        views = m31_cache_views(main, index, 4, 128, 128, 128)
        self.assertEqual(
            [
                view.shape[1]
                for view in (
                    views.main_k,
                    views.main_v,
                    views.main_k_scale,
                    views.main_v_scale,
                    views.index_k,
                    views.index_k_scale,
                )
            ],
            [32768, 32768, 4096, 4096, 8192, 1024],
        )
        views.main_v[1, 0] = 17
        views.index_k[2, 0] = 19
        self.assertEqual(int(main[1, 32768]), 17)
        self.assertEqual(int(index[2, 0]), 19)
        with self.assertRaisesRegex(ValueError, "128-token"):
            m31_cache_views(main, index, 4, 128, 128, 64)

    def test_native_prefill_stages_only_active_pages_in_request_order(self):
        main = torch.zeros((5, 73728), dtype=torch.uint8)
        index = torch.zeros((5, 9216), dtype=torch.uint8)
        views = m31_cache_views(main, index, 4, 128, 128, 128)
        for page in range(5):
            views.main_k[page].fill_(page + 1)
            views.main_v[page].fill_(page + 11)
            views.main_k_scale[page].view(torch.uint8).fill_(page + 21)
        table = torch.tensor([[3, 1], [4, 2]], dtype=torch.int32)
        k, v, ks, vs, compact = stage_m31_main_pages(views, table, [129, 1])
        self.assertEqual(k.shape, (3, 4, 128, 64))
        self.assertEqual(k[:, 0, 0, 0].tolist(), [4, 2, 5])
        self.assertEqual(v[:, 0, 0, 0].tolist(), [14, 12, 15])
        self.assertEqual(ks[:, 0].reshape(3, -1)[:, 0].tolist(), [24, 22, 25])
        self.assertEqual(compact.tolist(), [[0, 1], [2, 0]])
        self.assertTrue(k.is_contiguous())
        self.assertTrue(v.is_contiguous())

    def test_decode_addressing_uses_each_groups_own_page_table(self):
        current_lengths = torch.tensor([1, 129, 257, 1], dtype=torch.int32)
        input_lengths = torch.ones(4, dtype=torch.int32)  # stale graph tail
        main = SimpleNamespace(
            is_prefill=False,
            kv_cache_kernel_block_id_device=torch.tensor(
                [[3, -1, -1], [4, 5, -1], [6, 7, 8], [0, 0, 0]], dtype=torch.int32
            ),
            sequence_lengths_plus_1_device=current_lengths,
            sequence_lengths=torch.zeros_like(current_lengths),  # stale host view
            input_lengths_device=input_lengths,
        )
        index = SimpleNamespace(
            is_prefill=False,
            kv_cache_kernel_block_id_device=torch.tensor(
                [[9, -1, -1], [10, 11, -1], [12, 13, 14], [0, 0, 0]], dtype=torch.int32
            ),
            sequence_lengths_plus_1_device=current_lengths,
            sequence_lengths=torch.zeros_like(current_lengths),  # stale host view
            input_lengths_device=input_lengths,
        )
        address = m31_decode_addressing(main, index)
        self.assertEqual(address.main_slots.tolist(), [384, 640, 1024, -1])
        self.assertEqual(address.index_slots.tolist(), [1152, 1408, 1792, -1])
        self.assertEqual(address.seq_lens.tolist(), [1, 129, 257, 0])
        self.assertEqual(address.positions.tolist(), [0, 128, 256, 0])
        self.assertEqual(address.valid_rows.tolist(), [True, True, True, False])

    def test_prefill_addressing_keeps_request_and_cache_group_ownership(self):
        lengths = torch.tensor([2, 3], dtype=torch.int32)
        prefixes = torch.tensor([0, 128], dtype=torch.int32)
        main = SimpleNamespace(
            is_prefill=True,
            context_parallel_info=None,
            input_lengths=lengths,
            prefix_lengths=prefixes,
            kv_cache_kernel_block_id_device=torch.tensor(
                [[3, 0], [4, 7]], dtype=torch.int32
            ),
        )
        index = SimpleNamespace(
            is_prefill=True,
            context_parallel_info=None,
            input_lengths=lengths,
            prefix_lengths=prefixes,
            kv_cache_kernel_block_id_device=torch.tensor(
                [[5, 0], [6, 9]], dtype=torch.int32
            ),
        )
        address = m31_prefill_addressing(main, index, 5)
        self.assertEqual(address.request_ids.tolist(), [0, 0, 1, 1, 1])
        self.assertEqual(address.positions.tolist(), [0, 1, 128, 129, 130])
        self.assertEqual(address.main_slots.tolist(), [384, 385, 896, 897, 898])
        self.assertEqual(address.index_slots.tolist(), [640, 641, 1152, 1153, 1154])
        with self.assertRaisesRegex(ValueError, "lengths"):
            m31_prefill_addressing(main, index, 4)
        index.context_parallel_info = object()
        with self.assertRaisesRegex(ValueError, "CP"):
            m31_prefill_addressing(main, index, 5)

    def test_cp_prefill_uses_global_zigzag_positions_and_restore(self):
        info = SimpleNamespace(
            prefill_actual_input_lengths_cpu=torch.tensor([5], dtype=torch.int32),
            prefill_prefix_lengths_cpu=torch.tensor([128], dtype=torch.int32),
            prefill_shuffle_indices=torch.tensor([0, 1, 6, 7], dtype=torch.int32),
            prefill_qkv_padding_mask_cpu=torch.tensor(
                [1, 1, 1, 1, 1, 0, 0, 0], dtype=torch.int32
            ),
            prefill_qkv_restore_indice_cpu=torch.tensor(
                [0, 1, 4, 5, 6, 7, 2, 3], dtype=torch.int32
            ),
        )
        inputs = SimpleNamespace(is_prefill=True, context_parallel_info=info)
        plan = plan_m31_cp_prefill(inputs, inputs, 4, 2)
        self.assertEqual(plan.local_request_ids.tolist(), [0, 0, 0, 0])
        self.assertEqual(plan.local_positions.tolist(), [128, 129, 132, 132])
        self.assertEqual(plan.restore_indices.tolist(), [0, 1, 4, 5, 6])
        self.assertEqual(plan.full_lengths, (133,))
        with self.assertRaisesRegex(ValueError, "local rows"):
            plan_m31_cp_prefill(inputs, inputs, 3, 2)

    def test_cp_prefill_multiple_requests_and_padding(self):
        info = SimpleNamespace(
            prefill_actual_input_lengths_cpu=torch.tensor([5, 3], dtype=torch.int32),
            prefill_prefix_lengths_cpu=torch.tensor([0, 128], dtype=torch.int32),
            prefill_shuffle_indices=torch.tensor([0, 1, 6, 7, 0, 3], dtype=torch.int32),
            prefill_qkv_padding_mask_cpu=torch.tensor(
                [1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0], dtype=torch.int32
            ),
            prefill_qkv_restore_indice_cpu=torch.tensor(
                [0, 1, 6, 7, 8, 9, 2, 3, 4, 10, 11, 5], dtype=torch.int32
            ),
        )
        main = SimpleNamespace(is_prefill=True, context_parallel_info=info)
        index = SimpleNamespace(
            is_prefill=True,
            context_parallel_info=SimpleNamespace(
                prefill_shuffle_indices=info.prefill_shuffle_indices
            ),
        )
        plan = plan_m31_cp_prefill(main, index, 6, 2)
        self.assertEqual(plan.local_request_ids.tolist(), [0, 0, 0, 0, 1, 1])
        self.assertEqual(plan.local_positions.tolist(), [0, 1, 4, 4, 128, 130])
        self.assertEqual(plan.restore_indices.tolist(), [0, 1, 6, 7, 8, 4, 10, 11])
        self.assertEqual(plan.full_lengths, (5, 131))

    def test_cp_prefix_and_suffix_follow_tag_local_block_table(self):
        # Logical prefix pages 0/1 are owned by ranks 0/1, respectively.
        pool = torch.zeros((8, 4), dtype=torch.uint8)
        pool[3].fill_(11)
        table = torch.tensor([[3, 6]], dtype=torch.int32)
        remote = torch.full((1, 4), 22, dtype=torch.uint8)
        with patch(
            "rtp_llm.models_py.model_desc.minimax_m31_cp_prefill.all_gather",
            side_effect=lambda send, group: torch.cat((send, remote)),
        ):
            work, compact, starts = _working_pages(
                pool, table, (384,), (256,), 0, 2, True
            )
        self.assertEqual(compact.tolist(), [[1, 2, 3]])
        self.assertEqual(work[1].tolist(), [11] * 4)
        self.assertEqual(work[2].tolist(), [22] * 4)
        work[3].fill_(33)
        _publish_local_pages(pool, table, work, starts, (384,), (256,), 0, 2, True)
        self.assertEqual(pool[3].tolist(), [11] * 4)
        self.assertEqual(pool[6].tolist(), [33] * 4)

    def test_writer_requires_independent_slot_vectors(self):
        views = m31_cache_views(
            torch.zeros((2, 73728), dtype=torch.uint8),
            torch.zeros((3, 9216), dtype=torch.uint8),
            4,
            128,
            128,
            128,
        )
        k = torch.zeros((2, 4, 128), dtype=torch.bfloat16)
        idx = torch.zeros((2, 128), dtype=torch.bfloat16)
        slots = torch.zeros(2, dtype=torch.int64)
        with self.assertRaisesRegex(ValueError, "separate int64 slots"):
            write_m31_nvfp4_rows(k, k, idx, slots, slots[:1], views)
        with self.assertRaisesRegex(ValueError, "requires CUDA"):
            write_m31_nvfp4_rows(k, k, idx, slots, slots, views)

    def test_checkpoint_configuration_contract(self):
        config = ModelConfig()
        config.max_seq_len = 8192
        checkpoint = {
            "quantization_config": {
                "moe_quant_algo": "NVFP4",
                "moe_quant_format": "nvfp4-pack-quantized",
            },
            "text_config": {
                "hidden_size": 6144,
                "num_hidden_layers": 60,
                "num_attention_heads": 64,
                "num_key_value_heads": 4,
                "head_dim": 128,
                "vocab_size": 200064,
                "intermediate_size": 3072,
                "dense_intermediate_size": 12288,
                "shared_intermediate_size": 3072,
                "n_shared_experts": 1,
                "num_local_experts": 128,
                "num_experts_per_tok": 8,
                "moe_layer_freq": [0] * 3 + [1] * 57,
                "swiglu_alpha": 1.702,
                "swiglu_limit": 7.0,
                "sparse_attention_config": {
                    "use_sparse_attention": True,
                    "sparse_attention_freq": [1] * 60,
                    "sparse_disable_index_value": [1] * 60,
                    "sparse_index_dim": 128,
                    "sparse_num_index_heads": 4,
                    "sparse_topk_blocks": 16,
                    "sparse_block_size": 128,
                    "sparse_init_block": 1,
                    "sparse_local_block": 1,
                },
            },
        }
        configure_m31_model(config, checkpoint)
        self.assertEqual(config.moe_layer_index, list(range(3, 60)))
        self.assertEqual(config.msa_sparse_config["sparse_layer_ids"], list(range(60)))
        self.assertEqual(
            config.msa_sparse_config["disable_value_layer_ids"], list(range(60))
        )
        self.assertEqual(config.msa_sparse_config["topk_blocks"], 16)
        self.assertEqual(config.inter_size, 12288)
        self.assertTrue(config.prepacked_nvfp4_moe)
        config.attn_config.tokens_per_block = 64
        config.attn_config.kernel_tokens_per_block = 64
        finalize_m31_kv_cache(config)
        self.assertEqual(config.attn_config.tokens_per_block, 128)
        self.assertEqual(
            [desc.tag for desc in config.kv_cache_spec_descs[0]],
            [MAIN_KV_TAG, INDEXER_KV_TAG],
        )

        configured = ModelConfig()
        configure_m31_model(configured, checkpoint)
        configured.attn_config.tokens_per_block = 64
        configured.attn_config.kernel_tokens_per_block = 64
        MiniMaxM31._apply_kv_cache_config(configured, None)
        self.assertEqual(configured.attn_config.tokens_per_block, 128)
        self.assertEqual(
            [desc.tag for desc in configured.kv_cache_spec_descs[0]],
            [MAIN_KV_TAG, INDEXER_KV_TAG],
        )
        self.assertIs(MiniMaxM31.get_weight_cls(), MiniMaxM31Weight)
        self.assertTrue(MiniMaxM31.support_cuda_graph(None))

    def test_prefill_chunking_preserves_reused_outputs_and_dummy_calls(self):
        hidden = torch.arange(12, dtype=torch.float32).view(6, 2)
        weights = torch.ones((6, 2), dtype=torch.float32)
        ids = torch.zeros((6, 2), dtype=torch.int64)
        scratch = torch.zeros((3, 2), dtype=torch.float32)
        calls = []

        def forward(x, routed_weights, routed_ids):
            calls.append((x.shape[0], routed_weights.clone(), routed_ids.clone()))
            scratch[: x.shape[0]].copy_(x + 1)
            return scratch[: x.shape[0]]

        output = run_m31_prefill_chunks(
            hidden, weights, ids, forward, M31PrefillChunkPlan(3, 3)
        )
        torch.testing.assert_close(output, hidden + 1)
        self.assertEqual([count for count, _, _ in calls], [3, 3, 1])
        self.assertEqual(calls[-1][1].tolist(), [[0.0, 0.0]])
        self.assertEqual(calls[-1][2].tolist(), [[0, 1]])
        calls.clear()
        empty = run_m31_prefill_chunks(
            hidden[:0],
            weights[:0],
            ids[:0],
            forward,
            M31PrefillChunkPlan(3, 2),
        )
        self.assertEqual(empty.shape, (0, 2))
        self.assertEqual([count for count, _, _ in calls], [1, 1])
        with self.assertRaisesRegex(ValueError, "exceed"):
            run_m31_prefill_chunks(
                hidden, weights, ids, forward, M31PrefillChunkPlan(3, 1)
            )

    def test_vl_metadata_stays_on_m31_text_config(self):
        config = ModelConfig()
        checkpoint = {
            "vision_config": {"hidden_size": 1024},
            "image_token_index": 200025,
            "video_token_index": 200026,
            "projector_hidden_size": 6144,
        }
        configure_m31_vl(config, checkpoint, "/checkpoint")
        self.assertTrue(config.mm_model_config.is_multimodal)
        self.assertEqual(config.mm_model_config.mm_sep_tokens, [[200025], [200026]])
        self.assertEqual(config.mm_related_params.config["projector_hidden_size"], 6144)
        self.assertEqual(config.mm_related_params.config["ckpt_path"], "/checkpoint")

    def test_mxfp8_checkpoint_uses_micro_scale_loader(self):
        checkpoint = {
            "quantization_config": {
                "quant_method": "mxfp8",
                "weight_block_size": [1, 32],
                "moe_quant_algo": "NVFP4",
                "moe_quant_format": "nvfp4-pack-quantized",
            }
        }
        with tempfile.TemporaryDirectory() as path:
            Path(path, "config.json").write_text(json.dumps(checkpoint))
            quant = QuantizationConfig.load_from_ckpt(path)
        weight = AtomicWeight(
            W.attn_o_w,
            [CkptWeightInfo("model.layers.{i}.self_attn.o_proj.weight", identity)],
            transpose,
        )
        loaded = WeightModule.create(weight, quant)
        self.assertEqual(type(loaded).__name__, "Mxfp8Weight")
        scale = loaded.sub_weights[W.attn_o_s]
        self.assertEqual(
            scale.weights[0].name,
            "model.layers.{i}.self_attn.o_proj.weight_scale_inv",
        )
        legacy = WeightModule.create(weight, Fp8BlockWiseQuantConfig(is_quanted=True))
        self.assertEqual(type(legacy).__name__, "PerBlockFp8Weight")

    def test_custom_index_weights_are_replicated_under_tensor_parallelism(self):
        from rtp_llm.models.minimax_m31_weights import (
            IDX_K_NORM,
            IDX_K_SCALE,
            IDX_K_WEIGHT,
            IDX_Q_NORM,
            IDX_Q_SCALE,
            IDX_Q_WEIGHT,
            RAW_K_NORM,
            RAW_Q_NORM,
        )
        from rtp_llm.utils.model_weight import sp_id

        info = object.__new__(MiniMaxM31Weight)
        info._hidden_size = 6144
        info._size_per_head = 128
        info._head_num = 64
        info._head_num_kv = 4
        info._align_size = 0
        info._is_gated_activation = True
        info.expert_num_ = 128
        info.moe_layer_index_ = []
        weights = info._get_hf_layer_weight_info(0)
        by_name = {weight.name: weight for weight in weights}
        for name in (
            IDX_Q_WEIGHT,
            IDX_Q_SCALE,
            IDX_Q_NORM,
            IDX_K_WEIGHT,
            IDX_K_SCALE,
            IDX_K_NORM,
            RAW_Q_NORM,
            RAW_K_NORM,
        ):
            self.assertIs(by_name[name]._get_split_func(), sp_id)

    def test_prepacked_moe_weights_bypass_mxfp8_quantizer(self):
        weight_info = object.__new__(MiniMaxM31Weight)
        weight_info._align_size = 0
        weight_info._is_gated_activation = True
        weight_info.expert_num_ = 128
        weight_info.moe_layer_index_ = [3]
        quant = Fp8MxBlockWiseQuantConfig(is_quanted=True)
        modules = weight_info._get_hf_ffn_layer_weight_info(3)
        by_name = {module.name: module for module in modules}
        for name in (
            W.moe_w1,
            W.moe_w2,
            W.moe_s1,
            W.moe_s2,
        ):
            loaded = WeightModule.create(by_name[name], quant)
            self.assertEqual(type(loaded).__name__, "M31PackedMoeWeight")

    def test_nvfp4_global_scales_preserve_gate_up_order(self):
        scales = [torch.tensor(value) for value in (2.0, 3.0, 5.0, 7.0)]
        self.assertEqual(
            stack_w13_global_scales(scales).tolist(),
            [[2.0, 5.0], [3.0, 7.0]],
        )

    def test_nvfp4_mega_rejects_unbounded_prefill_capacity(self):
        executor = object.__new__(M31NVFP4MegaExecutor)
        executor.cfg = SimpleNamespace(max_tokens_per_rank=65536)
        with self.assertRaisesRegex(ValueError, "Prefill chunking"):
            executor.setup_weights({})

    def test_nvfp4_mega_packer_rejects_non_cuda_input(self):
        x = torch.zeros((1, 6144), dtype=torch.bfloat16)
        router = torch.zeros((1, 8), dtype=torch.float32)
        ids = torch.zeros((1, 8), dtype=torch.int64)
        with self.assertRaisesRegex(ValueError, "CUDA BF16"):
            pack_m31_nvfp4_inputs(x, router, ids, None)

    def test_nvfp4_deepgemm_weight_order_and_global_scale(self):
        w13 = torch.tensor([[[1], [2], [3], [4]]], dtype=torch.int8)
        s13 = torch.tensor([[[1], [2], [3], [4]]], dtype=torch.uint8).view(
            torch.float8_e4m3fn
        )
        g13 = torch.tensor([[2.0, 4.0]])
        g2 = torch.tensor([8.0])
        w, scale, l1_gsf, l2_gsf = reorder_m31_nvfp4_checkpoint(w13, s13, g13, g2)
        self.assertEqual(w.flatten().tolist(), [3, 4, 1, 2])
        self.assertEqual(scale.view(torch.uint8).flatten().tolist(), [3, 4, 1, 2])
        self.assertEqual(l1_gsf.tolist(), [[0.25, 0.5]])
        self.assertEqual(l2_gsf.tolist(), [0.125])
        with self.assertRaisesRegex(ValueError, "positive finite"):
            reorder_m31_nvfp4_checkpoint(w13, s13, g13, torch.tensor([0.0]))

    def test_fp32_router_accepts_bf16_hidden_states(self):
        linear = torch.nn.Linear(4, 3, bias=False).float()
        hidden = torch.tensor([[1, -2, 3, 4]], dtype=torch.bfloat16)
        router = _M31Fp32Router(linear)
        self.assertTrue(torch.equal(router(hidden), linear(hidden.float())))

    def test_oai_activation_clamp_and_gate_order(self):
        # The up branch has a +1 offset; ordinary SiLU would differ here.
        gate = torch.tensor([[-9.0, 9.0]])
        up = torch.tensor([[9.0, -9.0]])
        expected_gate = gate.clamp(max=7.0)
        expected_up = up.clamp(min=-7.0, max=7.0)
        expected = (
            expected_gate * torch.sigmoid(expected_gate * 1.702) * (expected_up + 1)
        )
        torch.testing.assert_close(
            swiglu_oai(torch.cat([gate, up], -1), 1.702, 7.0), expected
        )
        torch.testing.assert_close(
            swiglu_oai(torch.cat([up, gate], -1), 1.702, 7.0, False), expected
        )

    def test_rejects_incompatible_runtime_page_size(self):
        config = ModelConfig()
        config.num_layers = 1
        config.attn_config.kv_head_num = 4
        config.attn_config.size_per_head = 128
        config.attn_config.indexer_head_dim = 128
        config.attn_config.tokens_per_block = 256
        config.attn_config.kernel_tokens_per_block = 256
        with self.assertRaisesRegex(ValueError, "page size 128"):
            finalize_m31_kv_cache(config)

    def test_rejects_unpacked_geometry(self):
        with self.assertRaisesRegex(ValueError, "divisible by 16"):
            build_m31_kv_cache_descs(1, kv_heads=4, head_dim=120, index_dim=128)


if __name__ == "__main__":
    unittest.main()
