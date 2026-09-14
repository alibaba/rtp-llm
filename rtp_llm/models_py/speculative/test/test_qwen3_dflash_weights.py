"""CPU checkpoint lookup/merge/TP contracts using synthetic DFlash weights."""

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
from rtp_llm.models.qwen_3_dflash import Qwen3DFlash, Qwen3DFlashWeight
from rtp_llm.ops import HWKernelConfig, ParallelismConfig
from rtp_llm.utils.database import CkptDatabase
from rtp_llm.utils.model_weight import W

# Preserve the three checkpoint families' layer/feature counts and H/Q ratio;
# reduce matrix dimensions only. No trained tensor or network access is needed.
_FAMILIES = (
    ("qwen36_27b", 80, 5, [1, 16, 31, 46, 61], 2048, 248070),
    ("qwen35_27b", 80, 6, [1, 10, 18, 27, 35, 44, 52, 61], 4096, 248077),
    ("qwen35_397b", 64, 6, [1, 9, 17, 25, 33, 41, 49, 57], 4096, 248077),
)
_DEVICE = SimpleNamespace(
    maybe_rewrite_weight_by_key=lambda _name, tensor, **_kwargs: tensor,
)


def _checkpoint(hidden, layers, feature_count):
    tensors = {}

    def add(name, shape):
        offset = len(tensors) * 17
        tensor = (torch.arange(math.prod(shape)) + offset).remainder(251).float() / 8
        tensors[name] = tensor.reshape(shape).to(torch.bfloat16)

    add("fc.weight", (hidden, feature_count * hidden))
    add("hidden_norm.weight", (hidden,))
    add("norm.weight", (hidden,))
    for index in range(layers):
        prefix = f"layers.{index}."
        add(prefix + "input_layernorm.weight", (hidden,))
        add(prefix + "post_attention_layernorm.weight", (hidden,))
        add(prefix + "self_attn.q_norm.weight", (2,))
        add(prefix + "self_attn.k_norm.weight", (2,))
        add(prefix + "self_attn.q_proj.weight", (64, hidden))
        add(prefix + "self_attn.k_proj.weight", (16, hidden))
        add(prefix + "self_attn.v_proj.weight", (16, hidden))
        add(prefix + "self_attn.o_proj.weight", (hidden, 64))
        add(prefix + "mlp.gate_proj.weight", (128, hidden))
        add(prefix + "mlp.up_proj.weight", (128, hidden))
        add(prefix + "mlp.down_proj.weight", (hidden, 128))
    return tensors


def _raw_config(hidden, layers, ids, window, mask):
    return {
        "architectures": ["DFlashDraftModel"],
        "model_type": "qwen3",
        "hidden_size": hidden,
        "intermediate_size": 128,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "head_dim": 2,
        "num_hidden_layers": layers,
        "vocab_size": 248320,
        "dtype": "bfloat16",
        "rms_norm_eps": 1e-6,
        "rope_parameters": {"rope_theta": 10_000_000, "rope_type": "default"},
        "layer_types": ["sliding_attention"] * (layers - 1) + ["full_attention"],
        "sliding_window": window,
        "dflash_config": {
            "block_size": 16,
            "mask_token_id": mask,
            "target_layer_ids": ids,
        },
    }


class Qwen3DFlashWeightTest(unittest.TestCase):
    def test_real_checkpoint_mapper_and_all_tp_ranks(self):
        for name, hidden, layers, ids, window, mask in _FAMILIES:
            checkpoint = _checkpoint(hidden, layers, len(ids))
            with tempfile.TemporaryDirectory() as path:
                Path(path, "config.json").write_text(
                    json.dumps(_raw_config(hidden, layers, ids, window, mask))
                )
                save_file(checkpoint, str(Path(path, "model.safetensors")))
                config = Qwen3DFlash._create_config(path)
                config.ckpt_path = path
                config.phy2log_path = ""
                config.data_type = "bf16"
                database = CkptDatabase(path)
                try:
                    for tp in (1, 2, 8):
                        for rank in range(tp):
                            with self.subTest(family=name, tp=tp, rank=rank):
                                self._check_rank(config, database, checkpoint, tp, rank)
                finally:
                    for file_info in database.pretrain_file_list:
                        file_info.close_safetensor_handle()

    def _check_rank(self, config, database, checkpoint, tp, rank):
        parallel = ParallelismConfig()
        parallel.tp_size = parallel.world_size = parallel.local_world_size = tp
        parallel.tp_rank = parallel.world_rank = parallel.local_rank = rank
        parallel.ffn_tp_size, parallel.ffn_tp_rank = tp, rank
        builder = Qwen3DFlashWeight(
            model_config=config,
            parallelism_config=parallel,
            hw_kernel_config=HWKernelConfig(),
            kv_cache_config=KVCacheConfig(),
        )
        builder.process_meta_from_ckpt(database.pretrain_file_list)
        self.assertEqual(builder.transformer_prefix, "")
        info = builder.get_weight_info()
        self.assertEqual(len(info.layer_weights), config.num_layers)
        # Exercise the production loader: it must recognize the declared aliases
        # and skip their absent checkpoint tensors before looking up weights.
        aliases = {
            name: torch.empty(
                config.vocab_size // tp, config.hidden_size, dtype=torch.bfloat16
            )
            for name in (W.embedding, W.lm_head)
        }
        with (
            patch("rtp_llm.device.get_current_device", return_value=_DEVICE),
            patch.object(ModelLoader, "force_clean_cuda_memory"),
            patch.object(database, "load_tensor", wraps=database.load_tensor) as reads,
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
            globals_ = weights.global_weights
            for name, tensor in aliases.items():
                self.assertIs(globals_[name], tensor)
            # FC and its output norm remain replicated, including TP8. A split
            # here would silently corrupt the target-feature combiner.
            torch.testing.assert_close(
                globals_[W.dspark_fc_w], checkpoint["fc.weight"].T, atol=0, rtol=0
            )
            for runtime_name, checkpoint_name in (
                (W.dspark_hidden_norm_gamma, "hidden_norm.weight"),
                (W.final_ln_gamma, "norm.weight"),
            ):
                torch.testing.assert_close(
                    globals_[runtime_name], checkpoint[checkpoint_name], atol=0, rtol=0
                )
            for index, loaded in enumerate(weights.weights):
                prefix = f"layers.{index}.self_attn."
                expected_qkv = torch.cat(
                    [
                        checkpoint[prefix + part + "_proj.weight"]
                        .chunk(tp, dim=0)[rank]
                        .T
                        for part in ("q", "k", "v")
                    ],
                    dim=1,
                )
                self.assertEqual(
                    tuple(loaded[W.attn_qkv_w].shape), (config.hidden_size, 96 // tp)
                )
                torch.testing.assert_close(
                    loaded[W.attn_qkv_w], expected_qkv, atol=0, rtol=0
                )
                torch.testing.assert_close(
                    loaded[W.attn_o_w],
                    checkpoint[prefix + "o_proj.weight"].T.chunk(tp, dim=0)[rank],
                    atol=0,
                    rtol=0,
                )
                for runtime_name, suffix in (
                    (W.q_ln_gamma, "q_norm.weight"),
                    (W.k_ln_gamma, "k_norm.weight"),
                ):
                    torch.testing.assert_close(
                        loaded[runtime_name],
                        checkpoint[prefix + suffix],
                        atol=0,
                        rtol=0,
                    )
            requested_names = {call.args[0] for call in reads.call_args_list}
            self.assertEqual(requested_names, set(checkpoint))
            self.assertFalse(any(name.startswith("model.") for name in requested_names))
            self.assertFalse(
                any(
                    "embed_tokens" in name or "lm_head" in name
                    for name in requested_names
                )
            )


if __name__ == "__main__":
    unittest.main()
