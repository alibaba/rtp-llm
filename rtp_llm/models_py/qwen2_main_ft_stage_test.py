"""Requires the matching main native bindings; no inference backend is stubbed."""

import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from rtp_llm.models_py.model_loader import NewLoaderConfig, NewModelLoader
from rtp_llm.models_py.quant_methods import QuantizationConfig
from rtp_llm.tools.convert.ft_export_metadata import generate_ft_index
from rtp_llm.tools.convert.ft_export_writer import write_ft_shards


def produce_records(rank):
    hidden, local_ffn, qkv = 4, 2, 6
    rows = torch.arange(32, dtype=torch.float32).reshape(8, hidden)
    yield None, "embedding", rows.chunk(2, dim=1)[rank]
    padded = torch.cat([rows + 91, torch.zeros_like(rows)], dim=0)
    yield None, "lm_head", padded.chunk(2)[rank]
    yield None, "final_layernorm.gamma", torch.full((hidden,), 7.0)
    for layer_id in range(3):
        base = layer_id * 1000 + rank * 100
        for key, shape in [
            ("pre_layernorm_weights.gamma", (hidden,)),
            ("post_layernorm_weights.gamma", (hidden,)),
            ("self_attention_weights.query_weight.kernel", (hidden, qkv)),
            ("self_attention_weights.query_weight.bias", (1, qkv)),
            ("self_attention_weights.attention_output_weight.kernel", (2, hidden)),
            ("ffn_weights.intermediate_weight13.kernel", (hidden, 2 * local_ffn)),
            ("ffn_weights.intermediate_weight2.kernel", (local_ffn, hidden)),
        ]:
            yield layer_id, key, torch.arange(
                torch.Size(shape).numel(), dtype=torch.float32
            ).reshape(shape) + base


class MainFTStageTest(unittest.TestCase):
    def test_qkv_operator_bias_rank_follows_attention_partition(self):
        from rtp_llm.models_py.new_models.qwen2.language import (
            Qwen2Attention,
            Qwen2ForCausalLM,
        )

        tensors = {}
        for name, rows in (("q_proj", 4), ("k_proj", 2), ("v_proj", 2)):
            tensors[f"{name}.weight"] = torch.eye(rows, 4)
            tensors[f"{name}.bias"] = torch.arange(rows, dtype=torch.float32)
        for tp, dp, ep in ((1, 1, 1), (2, 1, 1), (1, 2, 1), (1, 1, 2)):
            kwargs = Qwen2ForCausalLM._decoder_kwargs(
                SimpleNamespace(dp_size=dp, ep_size=ep)
            )
            for rank in range(tp):
                with self.subTest(tp=tp, rank=rank, dp=dp, ep=ep):
                    proj = Qwen2Attention(
                        4,
                        2,
                        1,
                        2,
                        tp_size=tp,
                        tp_rank=rank,
                        params_dtype=torch.float32,
                        **kwargs,
                    ).qkv_proj
                    proj.load_weights(tensors)
                    x = torch.eye(4)
                    with patch.object(
                        torch.nn.functional, "linear", wraps=torch.nn.functional.linear
                    ) as gemm:
                        proj(x)
                    bias = torch.cat(
                        [
                            tensors["q_proj.bias"].chunk(tp)[rank],
                            tensors["k_proj.bias"],
                            tensors["v_proj.bias"],
                        ]
                    )
                    if max(tp, dp, ep) > 1:
                        bias = bias.unsqueeze(0)
                    torch.testing.assert_close(gemm.call_args.args[2], bias)

    def test_main_writer_to_qwen2_stage_owners(self):
        self._check_stage_owners("qwen_2")

    def test_main_writer_to_qwen1_stage_owners(self):
        self._check_stage_owners("qwen_7b")

    def _check_stage_owners(self, kind):
        # Exercise main's actual serialization and indexing with deterministic
        # already-converted records, not a mock of FT or a full HF conversion.
        with tempfile.TemporaryDirectory() as directory:
            classifier = torch.arange(12, dtype=torch.float16).reshape(3, 4)
            for rank in (0, 1):
                records = list(produce_records(rank))
                if kind == "qwen_7b":
                    records.append((None, "__custom__.score", classifier))
                write_ft_shards(
                    records, directory, tp_rank=rank, ep_rank=rank
                )
            generate_ft_index(directory, {"TP_SIZE": 2, "DP_SIZE": 1, "EP_SIZE": 0})
            config = SimpleNamespace(
                model_type=kind,
                num_layers=3,
                is_mtp=False,
                vocab_size=8,
                hidden_size=4,
                inter_size=4,
                attn_config=SimpleNamespace(head_num=2, kv_head_num=1, size_per_head=2),
                layernorm_eps=1e-6,
                enable_fp32_lm_head=False,
                tie_word_embeddings=False,
                capture_aux_hidden_layer_ids=[],
                hidden_state_capture_layer_ids=[],
            )
            for method in ("scratch", "fastsafetensors"):
                for stage in (0, 1):
                    for rank in (0, 1):
                        pc = SimpleNamespace(
                            tp_size=2,
                            tp_rank=rank,
                            ep_size=2,
                            ep_rank=rank,
                            pp_size=2,
                            pp_rank=stage,
                            pp_stage_layer_counts=[1, 2],
                            dp_size=1,
                            dp_rank=0,
                            world_size=4,
                            world_rank=2 * stage + rank,
                            get_attn_tp_size=lambda: 2,
                            get_attn_tp_rank=lambda: rank,
                            get_ffn_tp_size=lambda: 2,
                            get_ffn_tp_rank=lambda: rank,
                        )
                        load = NewLoaderConfig(
                            tp_size=2,
                            tp_rank=rank,
                            ep_size=2,
                            ep_rank=rank,
                            compute_dtype=torch.float32,
                            device="cpu",
                            load_method=method,
                            quant_config=QuantizationConfig("none"),
                            parallelism_config=pc,
                            custom_weight_mappings=(
                                (("__custom__.score", "classifier.weight", torch.float16),)
                                if kind == "qwen_7b" else ()
                            ),
                        )
                        loader = NewModelLoader(config, load, model_path=directory)
                        model = loader.load()
                        self.assertEqual(loader.loaded_method.value, method)
                        self.assertEqual(
                            tuple(model.layers), ("1", "2") if stage else ("0",)
                        )
                        expected = (
                            torch.arange(32, dtype=torch.float32)
                            .reshape(8, 4)
                            .chunk(2)[rank]
                        )
                        if stage:
                            self.assertIsNone(model.embed_tokens)
                            torch.testing.assert_close(
                                model.lm_head.weight, expected + 91
                            )
                        else:
                            self.assertIsNone(model.lm_head)
                            torch.testing.assert_close(
                                model.embed_tokens.weight, expected
                            )
                        for lid, module in model.layers.items():
                            record = {
                                k: v
                                for i, k, v in produce_records(rank)
                                if i == int(lid)
                            }
                            torch.testing.assert_close(
                                module.input_layernorm.weight,
                                record["pre_layernorm_weights.gamma"],
                            )
                            torch.testing.assert_close(
                                module.self_attn.qkv_proj.weight,
                                record["self_attention_weights.query_weight.kernel"].T,
                            )
                            qkv_proj = module.self_attn.qkv_proj
                            x = torch.full((3, 4), 0.5)
                            reference_bias = record[
                                "self_attention_weights.query_weight.bias"
                            ]
                            with patch.object(
                                torch.nn.functional,
                                "linear",
                                wraps=torch.nn.functional.linear,
                            ) as gemm:
                                projection = qkv_proj(x)
                            # Check the actual operator argument, not a flag:
                            # the vector and matrix forms can agree on CPU.
                            actual_bias = gemm.call_args.args[2]
                            torch.testing.assert_close(actual_bias, reference_bias)
                            torch.testing.assert_close(
                                projection,
                                torch.nn.functional.linear(
                                    x, qkv_proj.weight, reference_bias
                                ),
                            )
                        if kind == "qwen_7b":
                            torch.testing.assert_close(
                                model.handler_heads.exports()["__custom__.score"], classifier
                            )
                            continue
                        # A non-generating consumer of this exact generation
                        # archive needs the final norm, but not its LM projection.
                        from rtp_llm.ops import TaskType

                        embedding_config = SimpleNamespace(**vars(config))
                        embedding_config.task_type = TaskType.DENSE_EMBEDDING
                        other_loader = NewModelLoader(
                            embedding_config, load, model_path=directory
                        )
                        backbone = other_loader.load()
                        self.assertIsNone(backbone.lm_head)
                        self.assertEqual(other_loader.loaded_method.value, method)
                        if stage:
                            torch.testing.assert_close(
                                backbone.norm.weight, torch.full((4,), 7.0)
                            )
                        else:
                            self.assertIsNone(backbone.norm)


if __name__ == "__main__":
    unittest.main()
