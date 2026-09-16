"""CPU TorchSpec Qwen3 DSpark checkpoint loading and TP contracts."""

import json
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from safetensors.torch import save_file

from rtp_llm.config.kv_cache_config import KVCacheConfig
from rtp_llm.model_loader.load_config import LoadMethod
from rtp_llm.model_loader.loader import ModelLoader
from rtp_llm.models.qwen_3_dspark import Qwen3DSpark, Qwen3DSparkWeight
from rtp_llm.ops import HWKernelConfig, ParallelismConfig
from rtp_llm.utils.database import CkptDatabase
from rtp_llm.utils.model_weight import W

_DEVICE = SimpleNamespace(
    maybe_rewrite_weight_by_key=lambda _name, tensor, **_kwargs: tensor,
)
_HIDDEN = 64
_VOCAB = 64
_LAYERS = 5
_TARGET_LAYERS = [39, 47, 55]
_HEADS = 32
_KV_HEADS = 8
_HEAD_DIM = 2
_INTERMEDIATE = 192
_MARKOV_RANK = 4
_CONFIDENCE_KEYS = {"confidence_head.proj.weight", "confidence_head.proj.bias"}


def _raw_config(*, lm_head_source="target"):
    config = {
        "architectures": ["Qwen3DSparkForCausalLM"],
        "model_type": "qwen_3_dspark",
        "hidden_size": _HIDDEN,
        "intermediate_size": _INTERMEDIATE,
        "num_attention_heads": _HEADS,
        "num_key_value_heads": _KV_HEADS,
        "head_dim": _HEAD_DIM,
        "num_hidden_layers": _LAYERS,
        "vocab_size": _VOCAB,
        "dtype": "bfloat16",
        "rms_norm_eps": 1e-6,
        "rope_parameters": {"rope_theta": 10_000_000, "rope_type": "default"},
        "layer_types": ["full_attention"] * _LAYERS,
        "block_size": 5,
        "mask_token_id": _VOCAB - 1,
        "aux_hidden_state_layer_ids": _TARGET_LAYERS,
        "markov_rank": _MARKOV_RANK,
        "sample_from_anchor": True,
        "enable_confidence_head": True,
        "confidence_head_with_markov": True,
    }
    if lm_head_source is not None:
        config["lm_head_source"] = lm_head_source
    return config


def _checkpoint(*, include_lm_head=False):
    tensors = {}

    def add(name, shape):
        offset = len(tensors) * 17
        values = (torch.arange(math.prod(shape)) + offset).remainder(251).float() / 8
        tensors[name] = values.reshape(shape).to(torch.bfloat16)

    add("fc.weight", (_HIDDEN, len(_TARGET_LAYERS) * _HIDDEN))
    add("hidden_norm.weight", (_HIDDEN,))
    add("model.embed_tokens.weight", (_VOCAB, _HIDDEN))
    add("model.norm.weight", (_HIDDEN,))
    for layer in range(_LAYERS):
        prefix = f"model.layers.{layer}."
        add(prefix + "input_layernorm.weight", (_HIDDEN,))
        add(prefix + "post_attention_layernorm.weight", (_HIDDEN,))
        add(prefix + "self_attn.q_norm.weight", (_HEAD_DIM,))
        add(prefix + "self_attn.k_norm.weight", (_HEAD_DIM,))
        add(prefix + "self_attn.q_proj.weight", (_HEADS * _HEAD_DIM, _HIDDEN))
        add(prefix + "self_attn.k_proj.weight", (_KV_HEADS * _HEAD_DIM, _HIDDEN))
        add(prefix + "self_attn.v_proj.weight", (_KV_HEADS * _HEAD_DIM, _HIDDEN))
        add(prefix + "self_attn.o_proj.weight", (_HIDDEN, _HEADS * _HEAD_DIM))
        add(prefix + "mlp.gate_proj.weight", (_INTERMEDIATE, _HIDDEN))
        add(prefix + "mlp.up_proj.weight", (_INTERMEDIATE, _HIDDEN))
        add(prefix + "mlp.down_proj.weight", (_HIDDEN, _INTERMEDIATE))
    add("markov_head.markov_w1.weight", (_VOCAB, _MARKOV_RANK))
    add("markov_head.markov_w2.weight", (_VOCAB, _MARKOV_RANK))
    # Converter preserves these training tensors although the current runtime
    # deliberately has no descriptor for either confidence-head parameter.
    add("confidence_head.proj.weight", (1, _HIDDEN + _MARKOV_RANK))
    add("confidence_head.proj.bias", (1,))
    if include_lm_head:
        add("lm_head.weight", (_VOCAB, _HIDDEN))
    return tensors


def _parallel(tp, rank):
    parallel = ParallelismConfig()
    parallel.tp_size = parallel.world_size = parallel.local_world_size = tp
    parallel.tp_rank = parallel.world_rank = parallel.local_rank = rank
    parallel.ffn_tp_size, parallel.ffn_tp_rank = tp, rank
    return parallel


class Qwen3DSparkWeightTest(unittest.TestCase):
    def _load(self, path, checkpoint, tp, rank, *, aliases=None):
        config = Qwen3DSpark._create_config(path)
        config.ckpt_path = path
        config.phy2log_path = ""
        config.data_type = "bf16"
        config.enable_fp32_lm_head = False
        database = CkptDatabase(path)
        try:
            builder = Qwen3DSparkWeight(
                model_config=config,
                parallelism_config=_parallel(tp, rank),
                hw_kernel_config=HWKernelConfig(),
                kv_cache_config=KVCacheConfig(),
            )
            builder.process_meta_from_ckpt(database.pretrain_file_list)
            self.assertEqual(builder.transformer_prefix, "model.")
            info = builder.get_weight_info()
            self.assertEqual(len(info.layer_weights), _LAYERS)
            with (
                patch("rtp_llm.device.get_current_device", return_value=_DEVICE),
                patch.object(ModelLoader, "force_clean_cuda_memory"),
                patch.object(
                    database, "load_tensor", wraps=database.load_tensor
                ) as reads,
            ):
                loader = ModelLoader(
                    config,
                    builder,
                    None,
                    database,
                    load_method=LoadMethod.SCRATCH,
                    force_cpu_load_weights=True,
                )
                weights = loader.load_weights("cpu", global_weight_aliases=aliases)
            return config, weights, {call.args[0] for call in reads.call_args_list}
        finally:
            for file_info in database.pretrain_file_list:
                file_info.close_safetensor_handle()

    def test_real_torchspec_converted_checkpoint_tp_contract(self):
        checkpoint = _checkpoint()
        with tempfile.TemporaryDirectory() as path:
            root = Path(path)
            root.joinpath("config.json").write_text(json.dumps(_raw_config()))
            save_file(checkpoint, str(root / "model.safetensors"))
            for tp in (1, 2, 8):
                for rank in range(tp):
                    with self.subTest(tp=tp, rank=rank):
                        target_head = torch.full(
                            (_VOCAB // tp, _HIDDEN), rank + 1, dtype=torch.bfloat16
                        )
                        config, weights, requested = self._load(
                            path, checkpoint, tp, rank, aliases={W.lm_head: target_head}
                        )
                        self.assertTrue(config.dspark_share_target_lm_head)
                        self.assertTrue(config.dspark_sample_from_anchor)
                        self.assertEqual(config.dspark_target_layer_ids, _TARGET_LAYERS)
                        globals_ = weights.global_weights
                        self.assertIs(globals_[W.lm_head], target_head)
                        self.assertEqual(
                            globals_[W.lm_head].data_ptr(), target_head.data_ptr()
                        )
                        torch.testing.assert_close(
                            globals_[W.embedding],
                            checkpoint["model.embed_tokens.weight"].chunk(tp, dim=1)[
                                rank
                            ],
                            atol=0,
                            rtol=0,
                        )
                        torch.testing.assert_close(
                            globals_[W.final_ln_gamma],
                            checkpoint["model.norm.weight"],
                            atol=0,
                            rtol=0,
                        )
                        torch.testing.assert_close(
                            globals_[W.dspark_fc_w],
                            checkpoint["fc.weight"].T,
                            atol=0,
                            rtol=0,
                        )
                        for runtime_name, source_name in (
                            (W.dspark_hidden_norm_gamma, "hidden_norm.weight"),
                            (W.dspark_markov_w1, "markov_head.markov_w1.weight"),
                            (W.dspark_markov_w2, "markov_head.markov_w2.weight"),
                        ):
                            torch.testing.assert_close(
                                globals_[runtime_name],
                                checkpoint[source_name],
                                atol=0,
                                rtol=0,
                            )
                        layer = weights.weights[0]
                        prefix = "model.layers.0."
                        expected_qkv = torch.cat(
                            [
                                checkpoint[prefix + f"self_attn.{part}_proj.weight"]
                                .chunk(tp, dim=0)[rank]
                                .T
                                for part in ("q", "k", "v")
                            ],
                            dim=1,
                        )
                        torch.testing.assert_close(
                            layer[W.attn_qkv_w], expected_qkv, atol=0, rtol=0
                        )
                        torch.testing.assert_close(
                            layer[W.attn_o_w],
                            checkpoint[prefix + "self_attn.o_proj.weight"].T.chunk(
                                tp, dim=0
                            )[rank],
                            atol=0,
                            rtol=0,
                        )
                        expected_w13 = torch.cat(
                            [
                                checkpoint[prefix + "mlp.gate_proj.weight"].T.chunk(
                                    tp, dim=1
                                )[rank],
                                checkpoint[prefix + "mlp.up_proj.weight"].T.chunk(
                                    tp, dim=1
                                )[rank],
                            ],
                            dim=1,
                        )
                        torch.testing.assert_close(
                            layer[W.ffn_w13], expected_w13, atol=0, rtol=0
                        )
                        torch.testing.assert_close(
                            layer[W.ffn_w2],
                            checkpoint[prefix + "mlp.down_proj.weight"].T.chunk(
                                tp, dim=0
                            )[rank],
                            atol=0,
                            rtol=0,
                        )
                        self.assertEqual(requested, set(checkpoint) - _CONFIDENCE_KEYS)
                        self.assertNotIn("lm_head.weight", requested)

    def test_checkpoint_lm_head_remains_owned_when_alias_is_not_declared(self):
        checkpoint = _checkpoint(include_lm_head=True)
        with tempfile.TemporaryDirectory() as path:
            root = Path(path)
            root.joinpath("config.json").write_text(
                json.dumps(_raw_config(lm_head_source="checkpoint"))
            )
            save_file(checkpoint, str(root / "model.safetensors"))
            config, weights, requested = self._load(path, checkpoint, 2, 1)
        self.assertFalse(config.dspark_share_target_lm_head)
        torch.testing.assert_close(
            weights.global_weights[W.lm_head],
            checkpoint["lm_head.weight"]
            .chunk(2, dim=0)[1]
            .to(weights.global_weights[W.lm_head].dtype),
            atol=0,
            rtol=0,
        )
        self.assertIn("lm_head.weight", requested)

    def test_missing_lm_head_fails_when_alias_is_not_declared(self):
        checkpoint = _checkpoint()
        with tempfile.TemporaryDirectory() as path:
            root = Path(path)
            root.joinpath("config.json").write_text(
                json.dumps(_raw_config(lm_head_source="checkpoint"))
            )
            save_file(checkpoint, str(root / "model.safetensors"))
            with self.assertRaisesRegex(Exception, "ts is empty"):
                self._load(path, checkpoint, 1, 0)


if __name__ == "__main__":
    unittest.main()
