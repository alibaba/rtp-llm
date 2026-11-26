"""Use a complete one-layer Qwen3.5 decoder and an independently packed oracle."""

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import torch
from safetensors.torch import save_file

from rtp_llm.config.quant_config import AWQConfig, GPTQConfig
from rtp_llm.models_py.checkpoint_reader_test_support import requires_native_fst
from rtp_llm.models_py.model_loader import NewModelLoader
from rtp_llm.models_py.new_models.qwen3_next.grouped_q_gate import (
    grouped_q_gate_read_spec,
)
from rtp_llm.models_py.new_models.qwen3_next.test.main_checkpoint_test import (
    _config,
    _load_config,
)
from rtp_llm.models_py.quant_methods.base import QuantizationConfig


def _pack_last(codes, bits):
    lanes = 32 // bits
    bits_at = torch.arange(lanes, dtype=torch.int64) * bits
    return (codes.long().reshape(codes.shape[0], -1, lanes) << bits_at).sum(-1).int()


def _encode(method, bits, fields):
    w, z, s = fields
    if method == "awq":

        def interleave(t):
            return t.reshape(t.shape[0], -1, 8)[
                :, :, [0, 2, 4, 6, 1, 3, 5, 7]
            ].reshape_as(t)

        w = _pack_last(interleave(w), bits)
        z = interleave(z)
    else:
        w = _pack_last(w.T, bits).T.contiguous()
    return {"qweight": w, "qzeros": _pack_last(z, bits), "scales": s.contiguous()}


def _fixture(method, bits):
    h, head, group = 128, 16, 32
    cfg = _config()
    cfg.model_type, cfg.hidden_size, cfg.inter_size = "qwen35_dense", h, h
    cfg.expert_num = cfg.moe_k = cfg.moe_style = cfg.moe_inter_size = 0
    (
        cfg.attn_config.head_num,
        cfg.attn_config.kv_head_num,
        cfg.attn_config.size_per_head,
    ) = 8, 1, head
    source = (AWQConfig if method == "awq" else GPTQConfig)(
        bits=bits, group_size=group, is_quanted=True
    )
    cfg.quant_config = source
    root, layer = "model.language_model.", "model.language_model.layers.0."
    raw = {
        root + "embed_tokens.weight": torch.ones(8, h).half(),
        root + "norm.weight": torch.ones(h),
    }
    for name, size in (
        ("input_layernorm", h),
        ("post_attention_layernorm", h),
        ("self_attn.q_norm", head),
        ("self_attn.k_norm", head),
    ):
        raw[layer + name + ".weight"] = torch.zeros(size)
    generator = torch.Generator().manual_seed(8365)

    def fields(n):
        return (
            torch.randint(1 << bits, (h, n), generator=generator),
            torch.randint(1 << bits, (h // group, n), generator=generator),
            (1 + torch.randint(32, (h // group, n), generator=generator)).half() / 8,
        )

    q, gate = fields(h), fields(h)
    # Encode the physical 2N-wide projection in true HF head order.
    fused = [
        torch.stack(
            (a.reshape(a.shape[0], 8, head), b.reshape(b.shape[0], 8, head)), dim=2
        ).reshape(a.shape[0], 2 * h)
        for a, b in zip(q, gate)
    ]
    reference = {"q": q, "gate": gate}
    for key, source_fields in (
        ("self_attn.q_proj", fused),
        ("self_attn.k_proj", fields(head)),
        ("self_attn.v_proj", fields(head)),
        ("self_attn.o_proj", fields(h)),
        ("mlp.gate_proj", fields(h)),
        ("mlp.up_proj", fields(h)),
        ("mlp.down_proj", fields(h)),
    ):
        reference[key] = source_fields
        for name, value in _encode(method, bits, source_fields).items():
            raw[layer + key + "." + name] = value
        if method == "gptq":
            raw[layer + key + ".g_idx"] = torch.arange(h, dtype=torch.int32) // group
            if key == "self_attn.q_proj":
                raw[layer + key + ".bias"] = torch.zeros(2 * h, dtype=torch.float16)
    return cfg, raw, reference


class GatedGroupwiseTest(unittest.TestCase):
    def check(self, method, bits, backend="scratch", tp=2):
        cfg, raw, ref = _fixture(method, bits)
        with tempfile.TemporaryDirectory() as folder:
            save_file(raw, str(Path(folder) / "model.safetensors"))
            for rank in range(tp):
                quant = QuantizationConfig(method, source_config=cfg.quant_config)
                options = replace(
                    _load_config(tp_size=tp, tp_rank=rank, quant_config=quant),
                    compute_dtype=torch.float16,
                    load_method=backend,
                )
                loader = NewModelLoader(cfg, options, model_path=folder)
                model = loader.load()
                self.assertEqual(loader.loaded_method.value, backend)
                attn = model.layers[0].self_attn
                cols = slice(rank * 128 // tp, (rank + 1) * 128 // tp)
                # Reference selects logical heads before bit-packing, opposite to the loader.
                q = _encode(method, bits, [x[:, cols] for x in ref["q"]])
                gate = _encode(method, bits, [x[:, cols] for x in ref["gate"]])
                k, v = (
                    _encode(method, bits, ref["self_attn." + p + "_proj"])
                    for p in ("k", "v")
                )
                for name in q:
                    torch.testing.assert_close(
                        getattr(attn.gate, name), gate[name], atol=0, rtol=0
                    )
                    torch.testing.assert_close(
                        getattr(attn.qkv_proj, name),
                        torch.cat((q[name], k[name], v[name]), 1),
                        atol=0,
                        rtol=0,
                    )
                for field in ("qweight", "qzeros", "scales"):
                    name = "model.language_model.layers.0.self_attn.q_proj." + field
                    target, spec = model.resolve_checkpoint_load_plan(
                        model.WEIGHTS_MAPPER.map_name(name), tuple(raw[name].shape)
                    )
                    self.assertIsNone(target)
                    self.assertEqual(spec.is_local_shard, tp > 1)
                    if tp > 1:
                        self.assertEqual(
                            (spec.shard_dim, spec.shard_count, spec.shard_rank),
                            (1, tp, rank),
                        )

    def test_scratch_both_tp_ranks_and_bit_widths(self):
        for method in ("awq", "gptq"):
            for bits in (4, 8):
                with self.subTest(method=method, bits=bits):
                    self.check(method, bits)

    def test_reject_invalid_source_layout(self):
        cfg, _, _ = _fixture("awq", 4)
        geometry = dict(
            hidden_size=128,
            head_dim=16,
            num_heads=8,
            attn_tp_size=2,
            attn_tp_rank=0,
            quant_config=QuantizationConfig("awq", cfg.quant_config),
        )
        with self.assertRaisesRegex(ValueError, "shape"):
            grouped_q_gate_read_spec(geometry, "q_proj.qweight", (128, 16))
        with self.assertRaisesRegex(ValueError, "align"):
            grouped_q_gate_read_spec(
                dict(geometry, head_dim=12), "q_proj.qzeros", (4, 24)
            )

    @requires_native_fst
    def test_native_fst(self):
        for method in ("awq", "gptq"):
            self.check(method, 4, backend="fastsafetensors", tp=1)

    def test_gptq_activation_order_and_hidden_bias_not_ignored(self):
        cfg, raw, _ = _fixture("gptq", 4)
        root = "model.language_model.layers.0.self_attn.q_proj."
        options = replace(
            _load_config(
                tp_size=2,
                tp_rank=0,
                quant_config=QuantizationConfig("gptq", cfg.quant_config),
            ),
            compute_dtype=torch.float16,
        )
        for key, message in (
            ("g_idx", "activation-order"),
            ("bias", "must be all zero"),
        ):
            with self.subTest(key=key), tempfile.TemporaryDirectory() as folder:
                field = raw[root + key].clone()
                field[-1] += 1
                save_file(
                    {**raw, root + key: field}, str(Path(folder) / "model.safetensors")
                )
                with self.assertRaisesRegex(ValueError, message):
                    NewModelLoader(cfg, options, model_path=folder).load()

    def test_excluding_qkv_keeps_the_gate_in_float_but_not_o_or_ffn(self):
        cfg, raw, _ = _fixture("awq", 4)
        root = "model.language_model.layers.0.self_attn."
        raw = {
            k: v
            for k, v in raw.items()
            if not any(
                k.startswith(root + p + ".") for p in ("q_proj", "k_proj", "v_proj")
            )
        }
        q = (
            torch.arange(256, dtype=torch.float16)[:, None]
            .expand(256, 128)
            .contiguous()
        )
        raw[root + "q_proj.weight"] = q
        for p in ("k_proj", "v_proj"):
            raw[root + p + ".weight"] = torch.ones(16, 128).half()
        ignored = ["layers.0.self_attn." + p for p in ("q_proj", "k_proj", "v_proj")]
        with tempfile.TemporaryDirectory() as folder:
            save_file(raw, str(Path(folder) / "model.safetensors"))

            def loader(patterns):
                quant = QuantizationConfig(
                    "awq", cfg.quant_config, ignored_layers=patterns
                )
                opts = replace(
                    _load_config(tp_size=2, tp_rank=1, quant_config=quant),
                    compute_dtype=torch.float16,
                )
                return NewModelLoader(cfg, opts, model_path=folder)

            attn = loader(ignored).load().layers[0].self_attn
            source = q.reshape(8, 2, 16, 128)[4:]
            torch.testing.assert_close(attn.gate.weight, source[:, 1].reshape(64, 128))
            torch.testing.assert_close(
                attn.qkv_proj.weight[:64], source[:, 0].reshape(64, 128)
            )
            self.assertEqual(attn.o_proj.qweight.dtype, torch.int32)
            with self.assertRaisesRegex(ValueError, "partially match fused layer"):
                loader(ignored[:1]).load()


if __name__ == "__main__":
    unittest.main()
