"""Two checkpoint-backed service owners; CPU causal attention, not a GPU smoke."""

import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from safetensors.torch import save_file
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import PreTrainedTokenizerFast

from rtp_llm.model_factory import ModelFactory
from rtp_llm.models.base_model import BaseModel
from rtp_llm.models.qwen_v3_moe import Qwen3Moe, Qwen3MoeEagle3
from rtp_llm.models_py.new_models.qwen3_moe_eagle3.body import Eagle3DraftBody
from rtp_llm.models_py.new_models.qwen3_moe_eagle3.draft_body_test import (
    _CPUAttention,
    _oracle,
)
from rtp_llm.ops import FMHAConfig, HWKernelConfig, SpeculativeType


def topology(size=1, rank=0):
    return SimpleNamespace(
        tp_size=size,
        tp_rank=rank,
        dp_size=1,
        dp_rank=0,
        ep_size=1,
        ep_rank=0,
        pp_size=1,
        pp_rank=0,
        pp_stage_layer_counts=[],
        world_rank=rank,
        world_size=size,
        local_rank=rank,
        local_world_size=size,
        get_attn_tp_rank=lambda: rank,
        get_ffn_tp_rank=lambda: rank,
        get_attn_tp_size=lambda: size,
        get_ffn_tp_size=lambda: size,
        prefill_cp_config=SimpleNamespace(
            is_enabled=lambda: False, is_prefill_enabled=lambda: False
        ),
        ffn_disaggregate_config=SimpleNamespace(enable_ffn_disaggregate=False),
    )


class ServiceTest(unittest.TestCase):
    @torch.inference_mode()
    def exercise(self, method, size=1, rank=0):
        torch.manual_seed(4)
        with tempfile.TemporaryDirectory() as folder:
            t, d = Path(folder) / "target", Path(folder) / "draft"
            t.mkdir()
            d.mkdir()
            shared = dict(
                hidden_size=8,
                vocab_size=24,
                head_dim=4,
                num_attention_heads=2,
                num_key_value_heads=1,
                intermediate_size=12,
                hidden_act="silu",
                rms_norm_eps=1e-5,
            )
            (t / "config.json").write_text(
                json.dumps(
                    dict(
                        shared,
                        model_type="qwen3_moe",
                        num_hidden_layers=3,
                        num_experts=2,
                        num_experts_per_tok=1,
                        moe_intermediate_size=12,
                        max_position_embeddings=128,
                        decoder_sparse_step=1,
                        norm_topk_prob=True,
                    )
                )
            )
            (d / "config.json").write_text(
                json.dumps(
                    dict(
                        shared,
                        architectures=["LlamaForCausalLMEagle3"],
                        num_hidden_layers=1,
                        eagle_config=dict(
                            eagle_aux_hidden_state_layer_ids=[2, 0, 1],
                            use_aux_hidden_state=True,
                            use_input_layernorm_in_first_layer=True,
                            use_last_layernorm=True,
                        ),
                    )
                )
            )
            target, draft = (
                Qwen3Moe._create_config(str(t)),
                Qwen3MoeEagle3._create_config(str(d)),
            )
            pc = topology(size, rank)
            for cfg, kind in ((target, "qwen_3_moe"), (draft, "qwen_3_moe_eagle3")):
                cfg.model_type = kind
                cfg.enable_fp32_lm_head = False
                cfg.data_type = "fp32"
                cfg.quant_config = None
            ModelFactory._setup_eagle3_configs(target, draft, pc)
            g = (8, 12, 2, 1, 4, 1e-5, torch.float32)
            weights = {
                n: torch.randn_like(p) / p.shape[-1] ** 0.5
                for n, p in Eagle3DraftBody(*g).named_parameters()
            }
            save_file(weights, str(d / "model.safetensors"))
            target_w = {
                "model.embed_tokens.weight": torch.randn(24, 8),
                "lm_head.weight": torch.randn(24, 8),
                "model.norm.weight": torch.ones(8),
            }
            for i in range(3):
                base = f"model.layers.{i}."
                for name, shape in {
                    "input_layernorm.weight": (8,),
                    "post_attention_layernorm.weight": (8,),
                    "self_attn.q_proj.weight": (8, 8),
                    "self_attn.k_proj.weight": (4, 8),
                    "self_attn.v_proj.weight": (4, 8),
                    "self_attn.o_proj.weight": (8, 8),
                    "self_attn.q_norm.weight": (4,),
                    "self_attn.k_norm.weight": (4,),
                    "mlp.gate.weight": (2, 8),
                }.items():
                    target_w[base + name] = torch.randn(shape) / 4
                for e in range(2):
                    for name, shape in (
                        ("gate", (12, 8)),
                        ("up", (12, 8)),
                        ("down", (8, 12)),
                    ):
                        target_w[base + f"mlp.experts.{e}.{name}_proj.weight"] = (
                            torch.randn(shape) / 4
                        )
            save_file(target_w, str(t / "model.safetensors"))

            PreTrainedTokenizerFast(
                tokenizer_object=Tokenizer(
                    WordLevel({str(i): i for i in range(24)}, unk_token="0")
                ),
                unk_token="0",
            ).save_pretrained(str(t))
            target.tokenizer_path = draft.tokenizer_path = str(t)
            engine = SimpleNamespace(
                parallelism_config=pc,
                hw_kernel_config=HWKernelConfig(),
                fmha_config=FMHAConfig(),
                kv_cache_config=SimpleNamespace(
                    multi_task_prompt=False, multi_task_prompt_str=""
                ),
                moe_config=SimpleNamespace(
                    ll_num_max_token=1,
                    masked_max_token_num=1,
                    moe_strategy=0,
                    use_mori_ep=False,
                    use_deepep_moe=False,
                    use_deepep_low_latency=False,
                    use_all_gather=True,
                    fake_balance_expert=False,
                ),
                sp_config=SimpleNamespace(
                    type=SpeculativeType.EAGLE3, gen_num_per_cycle=3
                ),
                runtime_config=SimpleNamespace(
                    warm_up=False, model_warm_up=False, max_generate_batch_size=1
                ),
                load_config=SimpleNamespace(
                    load_method=method,
                    loader_recycle_handles=False,
                    moe_pure_tp_preshard=False,
                    # Independent per-rank FST range reads without claiming a
                    # process group. Full collective execution belongs to smoke.
                    force_cpu_load_weights=size > 1,
                ),
            )
            # BaseModel production resolves rank -> CUDA. Only device placement
            # is overridden here; factory, tokenizer, ownership and loads are real.
            with patch.object(BaseModel, "_get_device_str", return_value="cpu"):
                owner = Qwen3Moe.from_config(
                    model_config=target,
                    parallelism_config=pc,
                    hw_kernel_config=engine.hw_kernel_config,
                    kv_cache_config=engine.kv_cache_config,
                    fmha_config=engine.fmha_config,
                    moe_config=engine.moe_config,
                    load_method=method,
                    max_generate_batch_size=1,
                    vit_config=None,
                    merge_lora=False,
                    force_cpu_load_weights=engine.load_config.force_cpu_load_weights,
                )
                borrower = ModelFactory.get_sp_model(target, draft, engine, owner).model
            for role in (owner, borrower):
                self.assertEqual(
                    role.model_weights_loader._selected_load_method.value, method
                )
            for key in ("embedding", "lm_head"):
                self.assertIs(
                    owner.weight.get_global_weight(key),
                    borrower.weight.get_global_weight(key),
                )
            model = borrower.py_model
            if size > 1:
                # The release has fewer KV heads than TP ranks. Check that Q is
                # partitioned but K/V remain replicated; local CPU test only,
                # not distributed execution. Inspect projection output so the
                # check is independent of the parameter's internal layout.
                x = torch.randn(3, 16)
                projections = [
                    weights[f"midlayer.self_attn.{p}_proj.weight"] for p in "qkv"
                ]
                projections[0] = projections[0].chunk(size, 0)[rank]
                torch.testing.assert_close(
                    model.body.midlayer.self_attn.qkv_proj(x),
                    x @ torch.cat(projections).T,
                )
                for name, p in (
                    ("model.embed_tokens.weight", model.embed_tokens.weight),
                    ("lm_head.weight", model.lm_head.weight),
                ):
                    torch.testing.assert_close(p, target_w[name].chunk(size, 0)[rank])
                return
            captures = {}
            for i, layer in enumerate(owner.py_model.layers):
                # Serving has no CPU fused-MoE kernel. Execute just that kernel
                # with torch, consuming the actual loaded physical expert weights.
                class CpuExperts:
                    topk_ids_dtype = torch.int32

                    def __init__(self, experts):
                        self.w13, self.w2 = experts.w13, experts.w2

                    def __call__(self, *, hidden_states, topk_weights, topk_ids, **_):
                        out = torch.zeros_like(hidden_states)
                        for e, (w13, w2) in enumerate(zip(self.w13, self.w2)):
                            rows, slots = torch.where(topk_ids == e)
                            up, gate = (hidden_states[rows] @ w13.T).chunk(2, -1)
                            value = (torch.nn.functional.silu(gate) * up) @ w2.T
                            out.index_add_(
                                0, rows, value * topk_weights[rows, slots, None]
                            )
                        return out

                layer.mlp.experts.fused_moe = CpuExperts(layer.mlp.experts)
                layer.register_forward_hook(
                    lambda _m, _a, res, i=i: captures.update({i: res[0] + res[1]})
                )
            from rtp_llm.ops.compute_ops import PyAttentionInputs, PyModelInputs

            inp = PyModelInputs()
            inp.input_ids = torch.tensor([1, 2, 3])
            inp.attention_inputs = PyAttentionInputs()

            class CPU(_CPUAttention):
                def forward(self, packed, cache, layer):
                    return super().forward(packed, cache, 0)

            fmha = CPU(2, 1, 4, None)
            output = owner.py_model(inp, fmha)
            gathered = torch.cat([captures[i] for i in (2, 0, 1)], -1)
            torch.testing.assert_close(
                output.mtp_target_hidden_states, gathered @ weights["fc.weight"].T
            )
            state = gathered
            inp.input_hiddens = output.mtp_target_hidden_states
            for ids in ([1, 2, 3], [4, 5, 6]):
                inp.input_ids = torch.tensor(ids)
                ref, state = _oracle(
                    weights, target_w["model.embed_tokens.weight"][ids], state, g
                )
                out = model(inp, fmha)
                torch.testing.assert_close(out.mtp_target_hidden_states, state)
                torch.testing.assert_close(
                    model.lm_head(out.hidden_states), ref @ target_w["lm_head.weight"].T
                )
                inp.input_hiddens = out.mtp_target_hidden_states

    def test_scratch(self):
        self.exercise("scratch")

    def test_tp2_scratch_shards(self):
        for rank in range(2):
            self.exercise("scratch", 2, rank)

    @unittest.skipUnless(
        os.environ.get("NEWLOADER_TEST_NATIVE_FST") == "1", "native reader required"
    )
    def test_native_fst(self):
        self.exercise("fastsafetensors")

    @unittest.skipUnless(
        os.environ.get("NEWLOADER_TEST_NATIVE_FST") == "1", "native reader required"
    )
    def test_tp2_fst_shards(self):
        for rank in range(2):
            self.exercise("fastsafetensors", 2, rank)


if __name__ == "__main__":
    unittest.main()
