import os
import tempfile
import unittest
from types import SimpleNamespace

import torch
from safetensors.torch import save_file

from rtp_llm.config.kv_cache_config import KVCacheConfig
from rtp_llm.config.model_config import ModelConfig
from rtp_llm.models.moss_vl import MossVLWeight
from rtp_llm.model_loader.tensor_source import DatabaseTensorSource
from rtp_llm.ops import HWKernelConfig, ParallelismConfig
from rtp_llm.utils.database import CkptDatabase
from rtp_llm.utils.model_weight import W

HIDDEN = 64
HEADS = 2
KV_HEADS = 1
HEAD_DIM = 32
INTER = 96
NUM_LAYERS = 4
CROSS_LAYERS = [1, 3]
PREFIX = "model.language_model."


def _config() -> ModelConfig:
    config = ModelConfig()
    config.hidden_size = HIDDEN
    config.num_layers = NUM_LAYERS
    config.inter_size = INTER
    config.attn_config.head_num = HEADS
    config.attn_config.kv_head_num = KV_HEADS
    config.attn_config.size_per_head = HEAD_DIM
    config.qk_norm = True
    config.mm_related_params.config["cross_attention_layers"] = list(CROSS_LAYERS)
    return config


def _marker_checkpoint() -> dict:
    """Distinct, non-square marker values per tensor family."""
    ckpt = {}

    def put(name, tensor):
        ckpt[PREFIX + name] = tensor

    for layer in range(NUM_LAYERS):
        base = f"layers.{layer}."
        put(base + "input_layernorm.weight", torch.full((HIDDEN,), 1.0 + layer))
        put(base + "post_attention_layernorm.weight", torch.full((HIDDEN,), 2.0 + layer))
        is_cross = layer in CROSS_LAYERS
        attn = "cross_attn." if is_cross else "self_attn."
        q_out, kv_out = HEADS * HEAD_DIM, KV_HEADS * HEAD_DIM
        # non-square [out, in] HF layout, distinct values per projection
        put(base + attn + "q_proj.weight", torch.full((q_out, HIDDEN), 3.0))
        put(base + attn + "k_proj.weight", torch.full((kv_out, HIDDEN), 5.0))
        put(base + attn + "v_proj.weight", torch.full((kv_out, HIDDEN), 7.0))
        put(base + attn + "o_proj.weight", torch.full((HIDDEN, q_out), 9.0))
        put(base + attn + "q_norm.weight", torch.full((HEAD_DIM,), 11.0 + layer))
        put(base + attn + "k_norm.weight", torch.full((HEAD_DIM,), 13.0 + layer))
        put(base + "mlp.gate_proj.weight", torch.full((INTER, HIDDEN), 15.0))
        put(base + "mlp.up_proj.weight", torch.full((INTER, HIDDEN), 17.0))
        put(base + "mlp.down_proj.weight", torch.full((HIDDEN, INTER), 19.0))
        if is_cross:
            ckpt[PREFIX + base + "cross_attn_attn_gate"] = torch.full((1,), 21.0 + layer)
            ckpt[PREFIX + base + "cross_attn_mlp_gate"] = torch.full((1,), 23.0 + layer)
    ckpt[PREFIX + "embed_tokens.weight"] = torch.full((97, HIDDEN), 25.0)
    ckpt[PREFIX + "norm.weight"] = torch.full((HIDDEN,), 27.0)
    ckpt["lm_head.weight"] = torch.full((97, HIDDEN), 29.0)
    return ckpt


class MossVLWeightTest(unittest.TestCase):
    def _tensor_name_templates(self, modules):
        names = set()
        for module in modules:
            for component in module.get_components():
                for ckpt in getattr(component, "weights", []):
                    names.add(ckpt.tensor_name(0))
        return names

    def test_heterogeneous_layer_descriptor_layout(self):
        weight_builder = MossVLWeight(
            model_config=_config(),
            parallelism_config=ParallelismConfig(),
            hw_kernel_config=HWKernelConfig(),
            kv_cache_config=KVCacheConfig(),
        )
        weight_builder._process_meta([{}], set(_marker_checkpoint()))
        self.assertEqual(weight_builder.cross_attention_layers, CROSS_LAYERS)
        self.assertEqual(weight_builder.transformer_prefix, PREFIX)
        weight_info = weight_builder._get_hf_weight_info()
        self.assertEqual(len(weight_info.layer_weights), NUM_LAYERS)
        cross_names = self._tensor_name_templates(weight_info.layer_weights[1])
        self_names = self._tensor_name_templates(weight_info.layer_weights[0])
        self.assertTrue(any("cross_attn.q_proj.weight" in n for n in cross_names))
        self.assertFalse(any("cross_attn." in n for n in self_names))
        self.assertFalse(any("self_attn." in n for n in cross_names))
        self.assertTrue(any("self_attn.q_proj.weight" in n for n in self_names))
        # every layer (both types) declares its own gates-or-not correctly
        for layer, modules in enumerate(weight_info.layer_weights):
            names = self._tensor_name_templates(modules)
            has_gates = any("cross_attn_attn_gate" in n for n in names)
            self.assertEqual(has_gates, layer in CROSS_LAYERS)

    def test_marker_weight_values_through_real_loader(self):
        checkpoint = _marker_checkpoint()
        weight_builder = MossVLWeight(
            model_config=_config(),
            parallelism_config=ParallelismConfig(),
            hw_kernel_config=HWKernelConfig(),
            kv_cache_config=KVCacheConfig(),
        )
        weight_builder._process_meta([{}], set(checkpoint))
        weight_info = weight_builder._get_hf_weight_info()
        exported_device = SimpleNamespace(
            maybe_rewrite_weight_by_key=lambda _name, tensor, use_swizzle_a=None: tensor
        )
        with tempfile.TemporaryDirectory() as checkpoint_path:
            save_file(checkpoint, os.path.join(checkpoint_path, "model.safetensors"))
            database = CkptDatabase(checkpoint_path)
            source = DatabaseTensorSource(database)
            load_config = weight_builder.create_load_config(
                torch.float32, database, exported_device=exported_device
            )
            for layer, modules in enumerate(weight_info.layer_weights):
                loaded = {}
                for weight in modules:
                    loaded.update(weight.load(source, layer, "cpu", load_config))
                q_size = HEADS * HEAD_DIM
                kv_size = KV_HEADS * HEAD_DIM
                # merged QKV: [hidden, q|k|v] with transposed projections
                self.assertEqual(loaded[W.attn_qkv_w].shape, (HIDDEN, q_size + 2 * kv_size))
                self.assertTrue(torch.all(loaded[W.attn_qkv_w][:, :q_size] == 3.0))
                self.assertTrue(
                    torch.all(loaded[W.attn_qkv_w][:, q_size : q_size + kv_size] == 5.0)
                )
                self.assertTrue(torch.all(loaded[W.attn_qkv_w][:, q_size + kv_size :] == 7.0))
                # o_proj transposed to [q_out, hidden]
                self.assertEqual(loaded[W.attn_o_w].shape, (q_size, HIDDEN))
                self.assertTrue(torch.all(loaded[W.attn_o_w] == 9.0))
                self.assertTrue(torch.all(loaded[W.q_ln_gamma] == 11.0 + layer))
                self.assertTrue(torch.all(loaded[W.k_ln_gamma] == 13.0 + layer))
                # gated MLP: loader merges gate|up into w13 ([in, out] layout,
                # gate columns first), w2 separate; all transposed to [in, out]
                self.assertEqual(loaded[W.ffn_w13].shape, (HIDDEN, 2 * INTER))
                self.assertTrue(torch.all(loaded[W.ffn_w13][:, :INTER] == 15.0))
                self.assertTrue(torch.all(loaded[W.ffn_w13][:, INTER:] == 17.0))
                self.assertEqual(loaded[W.ffn_w2].shape, (INTER, HIDDEN))
                self.assertTrue(torch.all(loaded[W.ffn_w2] == 19.0))
                if layer in CROSS_LAYERS:
                    self.assertEqual(loaded[W.cross_attn_attn_gate].shape, (1,))
                    self.assertEqual(
                        loaded[W.cross_attn_attn_gate].item(), 21.0 + layer
                    )
                    self.assertEqual(
                        loaded[W.cross_attn_mlp_gate].item(), 23.0 + layer
                    )
            globals_loaded = {}
            for weight in weight_info.weights:
                globals_loaded.update(weight.load(source, 0, "cpu", load_config))
            self.assertEqual(globals_loaded[W.embedding].shape, (97, HIDDEN))
            self.assertTrue(torch.all(globals_loaded[W.embedding] == 25.0))
            self.assertTrue(torch.all(globals_loaded[W.lm_head] == 29.0))
            self.assertTrue(torch.all(globals_loaded[W.final_ln_gamma] == 27.0))


if __name__ == "__main__":
    unittest.main()
