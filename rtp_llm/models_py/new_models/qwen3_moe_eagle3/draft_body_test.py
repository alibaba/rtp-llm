"""Actual checkpoint loads, including release weights when explicitly enabled."""

import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from torch.nn import functional as F

from rtp_llm.models_py.model_loader import (
    NewLoaderConfig,
    NewLoaderLoadMethod,
    NewModelLoader,
)
from rtp_llm.models_py.new_models.qwen3_moe_eagle3.body import Eagle3DraftBody
from rtp_llm.models_py.registry import register_model


@register_model("_test_qwen3_eagle3_body")
class _Body(Eagle3DraftBody):
    fastsafetensors_load_isolated = True

    def __init__(self, model_config, load_config):
        super().__init__(*model_config.geometry)


class _CPUAttention:
    """CPU causal GQA oracle, not the production paged attention backend."""

    def __init__(self, q, kv, dim, cache):
        self.q, self.kv, self.dim, self.cache = q, kv, dim, cache

    def forward(self, packed, cache, layer):
        assert cache is self.cache and layer == 0
        q, k, v = packed.split(
            [self.q * self.dim, self.kv * self.dim, self.kv * self.dim], -1
        )

        def heads(x, n):
            return x.view(-1, n, self.dim).transpose(0, 1).float()

        g = self.q // self.kv
        y = F.scaled_dot_product_attention(
            heads(q, self.q),
            heads(k, self.kv).repeat_interleave(g, 0),
            heads(v, self.kv).repeat_interleave(g, 0),
            is_causal=True,
        )
        return y.transpose(0, 1).to(packed.dtype).contiguous()


def _load(folder, geometry, method):
    loader = NewModelLoader(
        SimpleNamespace(model_type="_test_qwen3_eagle3_body", geometry=geometry),
        NewLoaderConfig(device="cpu", load_method=method),
        model_path=str(folder),
    )
    model = loader.load()
    assert loader._selected_load_method == NewLoaderLoadMethod(method)
    return model


def _oracle(t, embeddings, initial, geometry):
    """Independent checkpoint-tensor math, including causal softmax."""
    h, _, q, kv, d, eps, _ = geometry

    def norm(x, key):
        return F.rms_norm(x, (h,), t[key], eps)

    def lin(x, key):
        return x @ t[key].T

    s = lin(initial, "fc.weight") if initial.shape[-1] == 3 * h else initial
    raw = torch.cat(
        [
            norm(embeddings, "midlayer.input_layernorm.weight"),
            norm(s, "midlayer.hidden_norm.weight"),
        ],
        -1,
    )

    def proj(name, n):
        x = lin(raw, "midlayer.self_attn." + name + "_proj.weight")
        return x.view(-1, n, d).transpose(0, 1).float()

    a = (
        proj("q", q)
        @ proj("k", kv).repeat_interleave(q // kv, 0).transpose(-1, -2)
        / d**0.5
    )
    mask = torch.ones(a.shape[-2:], dtype=torch.bool).triu(1)
    v = torch.softmax(a.masked_fill(mask, float("-inf")), -1) @ proj(
        "v", kv
    ).repeat_interleave(q // kv, 0)
    s = s + lin(
        v.transpose(0, 1).reshape(-1, q * d).to(s.dtype),
        "midlayer.self_attn.o_proj.weight",
    )
    z = norm(s, "midlayer.post_attention_layernorm.weight")
    s = s + lin(
        F.silu(lin(z, "midlayer.mlp.gate_proj.weight"))
        * lin(z, "midlayer.mlp.up_proj.weight"),
        "midlayer.mlp.down_proj.weight",
    )
    return norm(s, "norm.weight"), s


class DraftBodyTest(unittest.TestCase):
    def setUp(self):
        self.g = (8, 12, 4, 2, 4, 1e-5, torch.float32)
        torch.manual_seed(42)
        model = Eagle3DraftBody(*self.g)
        self.t = {
            n: torch.randn_like(p) / max(1, p.shape[-1] ** 0.5)
            for n, p in model.named_parameters()
        }
        self.assertEqual(len(self.t), 12)

    def _compare(self, model, state, g, n):
        h, _, q, kv, d, _, dtype = g
        inputs = torch.randn(n, h, dtype=dtype)
        aux = torch.randn(n, 3 * h, dtype=dtype)
        attention = _CPUAttention(q, kv, d, object())
        with torch.inference_mode():
            want, want_recurrent = _oracle(state, inputs, aux, g)
            out = model(inputs, aux, attention, attention.cache)
            torch.testing.assert_close(out.recurrent_hidden, want_recurrent)
            torch.testing.assert_close(out.logits_input, want)
            want2, want_recurrent2 = _oracle(state, inputs, want_recurrent, g)
            out2 = model(inputs, out.recurrent_hidden, attention, attention.cache)
            torch.testing.assert_close(out2.logits_input, want2)
            torch.testing.assert_close(out2.recurrent_hidden, want_recurrent2)
        with self.assertRaisesRegex(ValueError, "three ordered"):
            model(inputs, aux[:, : 2 * h], attention, attention.cache)

    def _exercise(self, method):
        with tempfile.TemporaryDirectory() as p:
            save_file(self.t, str(Path(p) / "model.safetensors"))
            model = _load(p, self.g, method)
        # FST buffers and checkpoint have gone out of scope here.
        for k, v in model.named_parameters():
            torch.testing.assert_close(v, self.t[k], rtol=0, atol=0)
        self._compare(model, self.t, self.g, 3)

    def test_scratch(self):
        self._exercise("scratch")

    @unittest.skipUnless(
        os.getenv("NEWLOADER_TEST_NATIVE_FST") == "1", "native FST required"
    )
    def test_native_fst(self):
        import fastsafetensors.cpp as cpp

        self.assertTrue(cpp.__file__.endswith(".so"))
        self._exercise("fastsafetensors")

    def test_incomplete_and_wrong_qkv_width_rejected_before_publish(self):
        for key in (
            "fc.weight",
            "norm.weight",
            "midlayer.hidden_norm.weight",
            "midlayer.self_attn.q_proj.weight",
        ):
            with self.subTest(key=key), tempfile.TemporaryDirectory() as p:
                t = dict(self.t)
                if key.endswith("q_proj.weight"):
                    t[key] = t[key][:, :8].contiguous()
                else:
                    del t[key]
                save_file(t, str(Path(p) / "model.safetensors"))
                with self.assertRaisesRegex(
                    (RuntimeError, ValueError), "missing|[Ss]hape"
                ):
                    _load(p, self.g, "scratch")

    @unittest.skipUnless(
        os.getenv("EAGLE3_RELEASE_PATH"), "requires original published draft shard"
    )
    def test_official_qwen3_235b_release_body(self):
        folder = Path(os.environ["EAGLE3_RELEASE_PATH"])
        import json

        cfg = json.loads((folder / "config.json").read_text())
        g = (
            cfg["hidden_size"],
            cfg["intermediate_size"],
            cfg["num_attention_heads"],
            cfg["num_key_value_heads"],
            cfg["head_dim"],
            cfg["rms_norm_eps"],
            torch.bfloat16,
        )
        self.assertEqual(
            cfg["eagle_config"]["eagle_aux_hidden_state_layer_ids"], [1, 46, 90]
        )
        method = os.getenv("EAGLE3_RELEASE_METHOD", "scratch")
        model = _load(folder, g, method)
        with safe_open(
            str(folder / "model.safetensors"), framework="pt", device="cpu"
        ) as f:

            class State:
                __getitem__ = staticmethod(f.get_tensor)

            for key, p in model.named_parameters():
                torch.testing.assert_close(p, f.get_tensor(key), rtol=0, atol=0)
            self._compare(model, State(), g, 1)


if __name__ == "__main__":
    unittest.main()
