import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from rtp_llm.model_factory import ModelFactory
from rtp_llm.model_factory_register import ModelDict, ensure_model_registered
from rtp_llm.models.qwen_3_dspark import (
    Qwen3DSpark,
    Qwen3DSparkWeight,
    dspark_offset_d2t_to_absolute,
)
from rtp_llm.models.qwen_v3 import QWenV3Weight
from rtp_llm.ops import RoleType
from rtp_llm.models_py.model_desc.module_base import GptModelBase
from rtp_llm.models_py.model_desc.qwen3 import Qwen3Model
from rtp_llm.models_py.model_desc.qwen3_dspark_model import (
    Qwen3DSparkModel,
    _apply_non_interleaved_rope,
    _write_rocm_paged_kv_cache,
)
from rtp_llm.models_py.speculative.dspark_proposer_mixin import device_metadata_tensor
from rtp_llm.ops.compute_ops import PyAttentionInputs
from rtp_llm.utils.model_weight import W, WeightStyle, merge_qkv_hf


class Qwen3DSparkRegistrationTest(unittest.TestCase):
    def test_native_draft_reuses_target_tokenizer_and_eos(self) -> None:
        target = SimpleNamespace(eos_token_id=154820)
        config = SimpleNamespace(
            special_tokens=SimpleNamespace(eos_token_id=-1),
            enable_output_vocab_pruning=False,
            generate_env_config=None,
        )
        with patch.object(Qwen3DSpark, "load_tokenizer", side_effect=AssertionError):
            model = Qwen3DSpark(
                model_config=config,
                parallelism_config=None,
                hw_kernel_config=None,
                kv_cache_config=SimpleNamespace(
                    multi_task_prompt=False, multi_task_prompt_str=""
                ),
                fmha_config=None,
                moe_config=None,
                max_generate_batch_size=1,
                load_method=None,
                vit_config=None,
                merge_lora=False,
                shared_tokenizer=target,
            )
        self.assertIs(model.tokenizer, target)
        self.assertEqual(config.special_tokens.eos_token_id, target.eos_token_id)

    def test_torch_rope_matches_qwen_reference_and_preserves_tail(self) -> None:
        positions = torch.tensor([0, 1, 7], dtype=torch.int32)
        rope_dim = 4
        head_dim = 6
        inv_freq = 1.0 / (10_000.0 ** (torch.arange(0, rope_dim, 2).float() / rope_dim))
        freqs = torch.outer(torch.arange(8).float(), inv_freq)
        cache = torch.cat((freqs.cos(), freqs.sin()), dim=-1)
        query = torch.arange(3 * 2 * head_dim, dtype=torch.float32).view(3, 2, head_dim)
        key = (query[:, :1] + 0.5).clone()
        expected_query = query.clone()
        expected_key = key.clone()

        def reference(tensor: torch.Tensor) -> None:
            selected = cache[positions.long()]
            cos = selected[:, : rope_dim // 2].unsqueeze(1)
            sin = selected[:, rope_dim // 2 :].unsqueeze(1)
            first = tensor[..., : rope_dim // 2].clone()
            second = tensor[..., rope_dim // 2 : rope_dim].clone()
            tensor[..., :rope_dim] = torch.cat(
                (first * cos - second * sin, second * cos + first * sin), dim=-1
            )

        reference(expected_query)
        reference(expected_key)
        query_tail = query[..., rope_dim:].clone()
        key_tail = key[..., rope_dim:].clone()

        _apply_non_interleaved_rope(query, key, cache, positions, rope_dim)

        torch.testing.assert_close(query, expected_query)
        torch.testing.assert_close(key, expected_key)
        torch.testing.assert_close(query[..., rope_dim:], query_tail)
        torch.testing.assert_close(key[..., rope_dim:], key_tail)

    def test_rocm_nonasm_context_cache_uses_aiter_physical_layout(self) -> None:
        block_count, kv_heads, page_size, head_dim = 3, 2, 16, 16
        cache = torch.zeros(
            block_count,
            2,
            kv_heads,
            page_size,
            head_dim,
            dtype=torch.bfloat16,
        )
        pages = torch.tensor([2, 0, 1], dtype=torch.long)
        slots = torch.tensor([3, 9, 15], dtype=torch.long)
        key = torch.arange(3 * kv_heads * head_dim, dtype=torch.bfloat16).view(
            3, kv_heads, head_dim
        )
        value = key + 1000

        _write_rocm_paged_kv_cache(
            cache,
            pages,
            slots,
            key,
            value,
            vectorized_value=False,
        )

        vector_size = 16 // cache.element_size()
        key_physical = cache.view(
            block_count,
            2,
            kv_heads,
            head_dim // vector_size,
            page_size,
            vector_size,
        )[:, 0]
        value_physical = cache.view(block_count, 2, kv_heads, head_dim, page_size)[:, 1]
        actual_key = key_physical[pages, :, :, slots, :].reshape_as(key)
        actual_value = value_physical[pages, :, :, slots]
        torch.testing.assert_close(actual_key, key)
        torch.testing.assert_close(actual_value, value)

    def test_draft_forward_uses_scalar_positions_then_restores_target_mrope(
        self,
    ) -> None:
        model = Qwen3DSparkModel.__new__(Qwen3DSparkModel)
        torch.nn.Module.__init__(model)
        model.kv_cache = None

        query_positions = torch.arange(6196, 6204, dtype=torch.long).view(1, 8)
        target_mrope = query_positions.repeat_interleave(3).to(torch.int32)
        original_target_mrope = target_mrope.clone()
        attention = PyAttentionInputs()
        attention.combo_position_ids = target_mrope
        inputs = SimpleNamespace(attention_inputs=attention)
        observed = []

        def fake_qwen_forward(_model, forwarded_inputs, _fmha_impl):
            observed.append(
                forwarded_inputs.attention_inputs.combo_position_ids[:8].clone()
            )
            return SimpleNamespace(hidden_states=torch.ones(8, 4))

        with patch.object(Qwen3Model, "forward", new=fake_qwen_forward):
            hidden = model.forward_query_block(
                torch.zeros(1, 8, dtype=torch.int32),
                query_positions,
                torch.tensor([6196], dtype=torch.int32),
                torch.tensor([True]),
                inputs,
                fmha_impl=object(),
            )

        self.assertEqual(tuple(hidden.shape), (8, 4))
        self.assertEqual(len(observed), 1)
        torch.testing.assert_close(observed[0], query_positions.reshape(-1).int())
        torch.testing.assert_close(target_mrope, original_target_mrope)

    def test_config_wires_shared_dspark_contract(self) -> None:
        raw = {
            "architectures": ["Qwen3DSparkForCausalLM"],
            "hidden_size": 128,
            "intermediate_size": 256,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "num_hidden_layers": 2,
            "vocab_size": 1024,
            "aux_hidden_state_layer_ids": [0, 1],
            "mask_token_id": 1000,
            "block_size": 8,
            "markov_rank": 32,
        }
        with tempfile.TemporaryDirectory() as path:
            Path(path, "config.json").write_text(json.dumps(raw))
            config = Qwen3DSpark._create_config(path)

        self.assertFalse(config.attn_config.is_causal)
        self.assertFalse(config.dspark_sample_from_anchor)
        self.assertEqual(config.dspark_noise_token_id, 1000)
        self.assertEqual(config.dspark_target_layer_ids, [0, 1])
        self.assertEqual(config.dspark_markov_rank, 32)
        self.assertTrue(ensure_model_registered("qwen_3_dspark"))
        self.assertEqual(ModelDict.get_ft_model_type_by_config(raw), "qwen_3_dspark")

    def test_native_draft_graph_refreshes_only_its_own_cache_tag(self):
        draft, target = PyAttentionInputs(), PyAttentionInputs()
        tag = SimpleNamespace(tag="draft")
        model = SimpleNamespace(
            parallelism_config=SimpleNamespace(
                prefill_cp_config=SimpleNamespace(is_enabled=lambda: False)
            ),
            config=SimpleNamespace(dspark_layer_windows=(2048, None)),
            kv_cache=SimpleNamespace(get_layer_cache_groups=lambda i: [tag]),
            attn_configs=object(),
            layer_num=2,
        )
        package = "rtp_llm.models_py.model_desc.qwen3_dspark_model"
        with patch(package + ".is_ppu", return_value=True), patch(
            package + "._PpuDraftAttention", side_effect=lambda cfg, inputs, w: inputs
        ):
            plans = Qwen3DSparkModel.prepare_fmha_impl(
                model,
                SimpleNamespace(attention_inputs={"indexer": target, "draft": draft}),
            )
            self.assertEqual(plans, {"draft": draft})
            single = Qwen3DSparkModel.prepare_fmha_impl(
                model, SimpleNamespace(attention_inputs=draft)
            )
            self.assertIs(single, draft)

    def test_native_glm_draft_uses_post_layer_indices_and_target_vocab_weights(
        self,
    ):
        raw = {
            "architectures": ["Qwen3DSparkModel"],
            "hidden_size": 128,
            "intermediate_size": 256,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "num_hidden_layers": 5,
            "vocab_size": 1024,
            "dflash_config": {"target_layer_ids": [1, 20, 38, 56, 75]},
            "dspark_config": {"markov_rank": 32, "mask_token_id": 1000},
            "layer_types": ["sliding_attention"] * 4 + ["full_attention"],
            "sliding_window": 2048,
        }
        with tempfile.TemporaryDirectory() as path:
            Path(path, "config.json").write_text(json.dumps(raw))
            config = Qwen3DSpark._create_config(path)
            config.ckpt_path = path
            self.assertEqual(
                ModelDict.get_ft_model_type_by_config(raw), "qwen_3_dspark"
            )
            self.assertEqual(config.dspark_target_layer_ids, [1, 20, 38, 56, 75])
            self.assertEqual(
                config.dspark_layer_windows, (2048, 2048, 2048, 2048, None)
            )
            self.assertTrue(config.dspark_sample_from_anchor)
            self.assertEqual(config.dspark_noise_token_id, 1000)
            self.assertEqual(config.dspark_markov_rank, 32)
            owner = SimpleNamespace(model_config=config)
            self.assertEqual(
                Qwen3DSpark.speculative_weight_alias_names(owner, config),
                (W.embedding, W.lm_head),
            )
            # Native drafts always use the target, even if an exporter writes
            # its own tables or omits an LM head tied to its embedding.
            config.tie_word_embeddings = True
            Path(path, "model.safetensors.index.json").write_text(
                json.dumps(
                    {
                        "weight_map": {
                            "embed_tokens.weight": "shard",
                            "lm_head.weight": "shard",
                        }
                    }
                )
            )
            self.assertEqual(
                Qwen3DSpark.speculative_weight_alias_names(owner, config),
                (W.embedding, W.lm_head),
            )
            raw["is_causal"] = True
            Path(path, "config.json").write_text(json.dumps(raw))
            with self.assertRaisesRegex(NotImplementedError, "is_causal"):
                Qwen3DSpark._create_config(path)
            config.input_vocab_size -= 1
            with self.assertRaisesRegex(ValueError, "input vocabulary"):
                Qwen3DSpark.speculative_weight_alias_names(owner, config)

    def test_native_prefill_omits_proposal_only_weights_but_decode_keeps_them(self):
        layer = [W.attn_qkv_w, W.k_ln_gamma, W.attn_o_w, W.ffn_w1]
        tables = [
            W.embedding,
            W.lm_head,
            W.dspark_fc_w,
            W.dspark_hidden_norm_gamma,
            W.dspark_markov_w1,
        ]
        for role, tp, want_layer, want_tables in (
            (RoleType.PREFILL, 1, [W.k_ln_gamma], tables[:4] + [W.dspark_context_kv_w]),
            (RoleType.PREFILL, 2, layer[:2], tables[:4]),
            (RoleType.DECODE, 1, layer, tables),
        ):
            desc = Qwen3DSparkWeight.__new__(Qwen3DSparkWeight)
            desc.model_config = SimpleNamespace(dspark_layer_windows=(16, None))
            desc.role_type, desc.tp_size, desc.weight_style = role, tp, WeightStyle.NONE
            desc.transformer_prefix = "model."
            info = SimpleNamespace(
                layer_weights=[
                    [SimpleNamespace(name=n) for n in layer] for _ in range(2)
                ],
                weights=[SimpleNamespace(name=n) for n in tables],
            )
            with patch.object(QWenV3Weight, "get_weight_info", return_value=info):
                loaded = desc.get_weight_info()
            self.assertEqual([x.name for x in loaded.layer_weights[0]], want_layer)
            self.assertEqual([x.name for x in loaded.weights], want_tables)
            if W.dspark_context_kv_w in want_tables:
                packed = loaded.weights[-1]
                self.assertEqual(
                    [w.tensor_name(None) for w in packed.weights],
                    [
                        f"model.layers.{i}.self_attn.{kv}_proj.weight"
                        for i in range(2)
                        for kv in ("k", "v")
                    ],
                )
                # Packed projection must write the same per-layer K/V as the
                # proposal layout, including both the layer and K/V order.
                q = torch.arange(16).reshape(4, 4).float()
                kv = [torch.arange(8).reshape(2, 4).float() + 10 * i for i in range(4)]
                ref = torch.cat(
                    [
                        merge_qkv_hf([q, *kv[2 * i : 2 * i + 2]])[:, 4:]
                        for i in range(2)
                    ],
                    dim=1,
                )
                torch.testing.assert_close(packed.process_fun(kv), ref, rtol=0, atol=0)

    def test_batched_cp_commit_restores_real_rows_and_independent_chunk_offsets(self):
        info = SimpleNamespace(
            prefill_qkv_padding_mask_cpu=torch.tensor([1, 1, 1, 0, 1, 1, 0, 0]),
            prefill_qkv_restore_indice_cpu=torch.tensor([0, 4, 5, 1, 2, 6, 7, 3]),
            prefill_actual_input_lengths_cpu=torch.tensor([3, 2]),
        )
        model = SimpleNamespace(
            parallelism_config=SimpleNamespace(
                tp_size=2, prefill_cp_config=SimpleNamespace(is_enabled=lambda: True)
            ),
            dspark_attention_inputs=lambda _: SimpleNamespace(
                context_parallel_info=info
            ),
        )
        req, positions, order = Qwen3DSparkModel.map_commit_rows(
            model,
            torch.tensor([0, 2]),
            torch.tensor([2, 2]),
            torch.tensor([13, 23]),
            4,
            object(),
        )
        rank_major = torch.tensor([10, -1, 21, -1, 11, 12, 22, -1])
        self.assertEqual(req.tolist(), [0, 0, 0, 1, 1])
        self.assertEqual(positions.tolist(), [10, 11, 12, 21, 22])
        self.assertEqual(rank_major.index_select(0, order).tolist(), positions.tolist())

    def test_speculators_config_normalizes_vocab_and_layer_contract(self) -> None:
        raw = {
            "architectures": ["DSparkDraftModel"],
            "speculators_model_type": "dspark",
            "transformer_layer_config": {
                "hidden_size": 128,
                "intermediate_size": 256,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "num_hidden_layers": 2,
                "vocab_size": 1024,
                "rope_parameters": {"rope_theta": 10_000_000},
                "dtype": "bfloat16",
            },
            "draft_vocab_size": 1001,
            "aux_hidden_state_layer_ids": [1, 2],
            "mask_token_id": 1000,
            "markov_rank": 32,
        }
        with tempfile.TemporaryDirectory() as path:
            Path(path, "config.json").write_text(json.dumps(raw))
            config = Qwen3DSpark._create_config(path)

        self.assertEqual(config.input_vocab_size, 1024)
        self.assertEqual(config.vocab_size, 1001)
        self.assertEqual(config.dspark_target_layer_ids, [0, 1])
        self.assertFalse(config.dspark_sample_from_anchor)
        self.assertEqual(config.attn_config.rope_config.base, 10_000_000)
        self.assertEqual(config.config_dtype, "bfloat16")

    def test_speculators_noise_token_is_validated_in_input_vocab(self) -> None:
        sp_config = SimpleNamespace(
            gen_num_per_cycle=7,
            sp_dspark_mask_token_id=-1,
            sp_dspark_sample_from_anchor=False,
        )
        target_config = SimpleNamespace(
            num_layers=48,
            capture_aux_hidden_layer_ids=None,
        )
        draft_config = SimpleNamespace(
            dspark_noise_token_id=248127,
            dspark_target_layer_ids=[7, 15, 23, 31, 39],
            dspark_markov_rank=256,
            input_vocab_size=248320,
            vocab_size=20000,
            dspark_sample_from_anchor=False,
            capture_aux_hidden_layer_ids=None,
        )

        ModelFactory._setup_dspark_configs(sp_config, target_config, draft_config)

        self.assertEqual(sp_config.sp_dspark_mask_token_id, 248127)

    def test_target_capture_layers_must_be_unique_and_ordered(self) -> None:
        sp_config = SimpleNamespace(
            gen_num_per_cycle=7,
            sp_dspark_mask_token_id=-1,
            sp_dspark_sample_from_anchor=False,
        )
        target_config = SimpleNamespace(
            num_layers=48,
            capture_aux_hidden_layer_ids=None,
        )
        draft_config = SimpleNamespace(
            dspark_noise_token_id=248127,
            dspark_target_layer_ids=[15, 7, 15],
            dspark_markov_rank=256,
            input_vocab_size=248320,
            vocab_size=20000,
            dspark_sample_from_anchor=False,
            capture_aux_hidden_layer_ids=None,
        )

        with self.assertRaisesRegex(ValueError, "unique and ordered"):
            ModelFactory._setup_dspark_configs(sp_config, target_config, draft_config)

    @staticmethod
    def _capture_harness(layer_ids: tuple[int, ...]) -> SimpleNamespace:
        return SimpleNamespace(
            _mtp_aux_capture_layer_ids=layer_ids,
            _mtp_aux_capture_layer_id_set=frozenset(layer_ids),
            _mtp_target_hidden_states=None,
            _mtp_target_graph_buffer=None,
            _mtp_target_prompt_buffer=None,
            _mtp_aux_capture_buffer=None,
            _mtp_aux_capture_rows=0,
            _mtp_aux_capture_index=0,
        )

    def test_aux_capture_materializes_fused_residual_boundary(self) -> None:
        capture = self._capture_harness((1,))
        hidden = torch.tensor([[1.0, 2.0]])
        residual = torch.tensor([[10.0, 20.0]])

        GptModelBase.begin_aux_hidden_capture(capture, hidden, False)
        GptModelBase.capture_aux_hidden(capture, 0, hidden, residual)
        GptModelBase.capture_aux_hidden(capture, 1, hidden, residual)
        GptModelBase.finish_aux_hidden_capture(capture)

        torch.testing.assert_close(
            capture._mtp_target_hidden_states, torch.tensor([[11.0, 22.0]])
        )

    def test_target_graph_capture_reuses_fixed_address(self) -> None:
        capture = self._capture_harness((0, 1))
        hidden = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        residual = torch.tensor([[10.0, 20.0], [30.0, 40.0]])

        GptModelBase.begin_aux_hidden_capture(capture, hidden, True)
        GptModelBase.capture_aux_hidden(capture, 0, hidden)
        GptModelBase.capture_aux_hidden(capture, 1, hidden, residual)
        GptModelBase.finish_aux_hidden_capture(capture)
        graph_ptr = capture._mtp_target_graph_buffer.data_ptr()

        GptModelBase.begin_aux_hidden_capture(capture, hidden[:1], True)
        GptModelBase.capture_aux_hidden(capture, 0, hidden[:1])
        GptModelBase.capture_aux_hidden(capture, 1, hidden[:1], residual[:1])
        GptModelBase.finish_aux_hidden_capture(capture)
        self.assertEqual(capture._mtp_target_graph_buffer.data_ptr(), graph_ptr)
        self.assertEqual(tuple(capture._mtp_target_hidden_states.shape), (1, 4))

        prompt_hidden = hidden.repeat(4, 1)
        GptModelBase.begin_aux_hidden_capture(capture, prompt_hidden, False)
        GptModelBase.capture_aux_hidden(capture, 0, prompt_hidden)
        GptModelBase.capture_aux_hidden(capture, 1, prompt_hidden)
        GptModelBase.finish_aux_hidden_capture(capture)
        self.assertEqual(capture._mtp_target_graph_buffer.data_ptr(), graph_ptr)
        self.assertNotEqual(capture._mtp_target_prompt_buffer.data_ptr(), graph_ptr)

    def test_speculators_d2t_offsets_are_normalized_to_absolute_ids(self) -> None:
        offsets = torch.tensor([1, 1, 2, -1], dtype=torch.int64)
        absolute = dspark_offset_d2t_to_absolute([offsets])
        self.assertEqual(absolute.tolist(), [1, 2, 4, 2])

    def test_commit_uses_kernel_block_ids_not_physical_cache_ids(self) -> None:
        kernel_ids = torch.tensor([[3, 4]], dtype=torch.int32)
        physical_ids = torch.tensor([[3003, 3004]], dtype=torch.int32)
        attention = SimpleNamespace(
            kv_cache_kernel_block_id_device=kernel_ids,
            kv_cache_kernel_block_id=torch.tensor([[5, 6]], dtype=torch.int32),
            kv_cache_block_id_device=physical_ids,
            kv_cache_block_id=torch.tensor([[5005, 5006]], dtype=torch.int32),
        )
        model = SimpleNamespace(
            embed_tokens=SimpleNamespace(weight=torch.empty(0)), kv_cache=None
        )
        model.dspark_attention_inputs = lambda inputs: inputs.attention_inputs

        selected = Qwen3DSparkModel._block_table(
            model, SimpleNamespace(attention_inputs=attention)
        )

        self.assertIs(selected, kernel_ids)

    def test_hybrid_tagged_inputs_select_draft_layer_group(self) -> None:
        full = SimpleNamespace()
        linear = SimpleNamespace()
        layer_cache = SimpleNamespace(tag="full")
        model = SimpleNamespace(
            kv_cache=SimpleNamespace(
                get_layer_cache_groups=lambda layer_idx: [layer_cache]
            )
        )
        inputs = SimpleNamespace(attention_inputs={"linear0": linear, "full": full})

        selected = Qwen3DSparkModel.dspark_attention_inputs(model, inputs)

        self.assertIs(selected, full)

    def test_cuda_graph_metadata_prefers_device_mirror(self) -> None:
        host = torch.tensor([11, 12], dtype=torch.int32)
        device_mirror = torch.tensor([21, 22], dtype=torch.int32)
        attention = SimpleNamespace(
            prefix_lengths=host,
            prefix_lengths_device=device_mirror,
        )

        selected = device_metadata_tensor(attention, "prefix_lengths")

        self.assertIs(selected, device_mirror)

    def test_metadata_falls_back_for_eager_callers(self) -> None:
        host = torch.tensor([11, 12], dtype=torch.int32)
        attention = SimpleNamespace(
            input_lengths=host,
            input_lengths_device=torch.empty(0, dtype=torch.int32),
        )

        selected = device_metadata_tensor(attention, "input_lengths")

        self.assertIs(selected, host)


if __name__ == "__main__":
    unittest.main()
