"""Real draft loader/commit/query calculation on small CPU checkpoint tensors.

These component contracts do not substitute for native PPU execution.
"""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS

import torch

from rtp_llm.models_py.modules.dsv41.ppu.dspark_model import DeepSeekV41PPUDSparkModel
from rtp_llm.models_py.modules.dsv41.ppu.dspark_weight import (
    draft_config,
    load_draft_weights,
)
from rtp_llm.models_py.modules.dsv41.ppu.kv_codec import encode_kv
from rtp_llm.models_py.modules.dsv41.ppu.native_cache import NativeCacheBridge
from rtp_llm.models_py.modules.dsv41.ppu.native_protocol import (
    native_cache_key_hash_seed,
)
from rtp_llm.models_py.modules.dsv41.ppu.prefill import _rms_norm, _rope_tail
from rtp_llm.models_py.modules.dsv41.ppu.test.test_native_cache import NativePools
from rtp_llm.models_py.modules.dsv41.ppu.weight import (
    read_safetensors_file,
    synthetic_config,
    write_synthetic_checkpoint,
)


class DSparkTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(119)
        cls.config = synthetic_config(engram_layers=(), engram_num_embeddings=())
        cls.config.update(
            num_nextn_predict_layers=3,
            dspark_n_routed_experts=4,
            dspark_num_experts_per_tok=3,
            dspark_target_layer_ids=[5, 6, 7],
            dspark_noise_token_id=96,
            dspark_markov_rank=4,
        )
        with tempfile.TemporaryDirectory() as root:
            path = write_synthetic_checkpoint(
                draft_config(cls.config), Path(root), seed=129
            )
            tensors = read_safetensors_file(path / "weights.safetensors")
        cls.checkpoint = {}
        for name, tensor in tensors.items():
            if name.startswith("layers."):
                name = "mtp." + name.removeprefix("layers.")
            elif name == "norm.weight":
                name = "mtp.2.norm.weight"
            cls.checkpoint[name] = tensor
        cls.checkpoint["mtp.0.main_proj.weight"] = (
            torch.randn(64, 192) * 0.03
        ).bfloat16()
        cls.checkpoint["mtp.0.main_norm.weight"] = torch.ones(64, dtype=torch.bfloat16)
        cls.checkpoint["mtp.2.confidence_head.proj.weight"] = torch.randn(1, 68) * 0.03
        cls.config["checkpoint_path"] = cls.checkpoint

    def model(self, gamma=3, commit_only=False, confidence=False):
        model = DeepSeekV41PPUDSparkModel(
            NS(
                deepseek_v41_config=self.config,
                gen_num_per_cycle=gamma,
                dspark_noise_token_id=96,
                dspark_target_layer_ids=[5, 6, 7],
                dspark_confidence_enabled=confidence,
            ),
            NS(
                role_type="PREFILL" if commit_only else "DECODE",
                ep_rank=0,
                ep_size=1,
                tp_rank=0,
                tp_size=1,
                dp_rank=0,
                dp_size=1,
            ),
            None,
            None,
            2,
        )
        native = NativePools(model.config, batch=1)
        model.initialize(NS(kv_cache=native, is_decode_role=not commit_only))
        return model, native

    def commit(self, model, native, rows=13):
        features = (torch.arange(rows * 192).reshape(rows, 192) % 31).bfloat16() / 31
        for primary in native.inputs.values():
            primary.input_lengths = torch.tensor([rows])
            primary.prefix_lengths = torch.tensor([0])
        inputs = NS(
            input_hiddens=features,
            input_ids=torch.arange(rows),
            attention_inputs=native.inputs,
            warmup=False,
        )
        model.forward_commit(inputs)
        return features

    def test_real_three_layer_geometry_and_commit_only_loader(self):
        minimal = {
            name: value
            for name, value in self.checkpoint.items()
            if name.startswith("mtp.0.main_")
            or ".attn.wkv." in name
            or ".attn.kv_norm." in name
        }
        weights = load_draft_weights(self.config, minimal, commit_only=True)
        self.assertEqual(weights.bundle.geometry.n_routed_experts, 4)
        self.assertEqual(weights.bundle.geometry.n_activated_experts, 3)
        self.assertEqual(tuple(weights.bundle.layers), (0, 1, 2))
        self.assertEqual(weights.bundle.embedding.numel(), 0)
        self.assertEqual(weights.bundle.geometry.original_seq_len, 0)

    def test_commit_matches_independent_per_layer_projections(self):
        model, native = self.model(commit_only=True)
        features = self.commit(model, native)
        # Compute each projection directly from checkpoint tensors. In
        # particular, never chain layer0 WKV into layer1's feature input.
        main = _rms_norm(
            torch.nn.functional.linear(
                features, self.checkpoint["mtp.0.main_proj.weight"]
            ),
            self.checkpoint["mtp.0.main_norm.weight"],
            model.geometry.norm_eps,
        )
        bridge = NativeCacheBridge(model.config, native, native.inputs)
        positions = torch.arange(5, 13)
        payloads = []
        for layer in range(3):
            expected = model.bundle.layers[layer].wkv(main)
            expected = _rms_norm(
                expected,
                self.checkpoint[f"mtp.{layer}.attn.kv_norm.weight"],
                model.geometry.norm_eps,
            )
            expected = _rope_tail(
                expected, model.bundle.freqs_for(layer)[torch.arange(13)]
            )
            actual = bridge._copy_rows((layer, "swa_kv"), 0, positions)
            torch.testing.assert_close(
                actual, encode_kv(expected[5:], pool_tag="swa_kv"), rtol=0, atol=0
            )
            payloads.append(actual)
        self.assertFalse(torch.equal(payloads[0], payloads[1]))
        self.assertEqual(len(native.calls), 3)

    def test_gamma_proposal_shapes_and_query_never_changes_committed_bytes(self):
        for gamma in (1, 3, 5):
            with self.subTest(gamma=gamma):
                model, native = self.model(gamma=gamma)
                self.commit(model, native)
                before = {
                    key: handle.kv_cache_base.clone()
                    for key, handle in native.handles.items()
                }
                for primary in native.inputs.values():
                    primary.prefix_lengths = torch.tensor([13])
                inputs = NS(
                    input_ids=torch.zeros(gamma, dtype=torch.int32),
                    attention_inputs=native.inputs,
                    warmup=False,
                    is_fake_stream=False,
                )
                output = model.forward_propose(inputs)
                self.assertEqual(tuple(output.hidden_states.shape), (gamma, 64))
                self.assertTrue(bool(output.hidden_states.isfinite().all()))
                for key, saved in before.items():
                    torch.testing.assert_close(
                        native.handles[key].kv_cache_base, saved, rtol=0, atol=0
                    )

    def test_query_block_is_noncausal(self):
        model, native = self.model(gamma=3)
        self.commit(model, native)
        bridge = model._bridge(NS(attention_inputs=native.inputs))
        layer = model.bundle.layers[0]
        x = torch.randn(3, 64, dtype=torch.bfloat16)
        args = (
            torch.tensor([[13, 14, 15]]),
            torch.tensor([13]),
            torch.tensor([True]),
            bridge,
        )
        before = model._attention(0, layer, x, *args)
        changed = x.clone()
        changed[-1] *= -3
        after = model._attention(0, layer, changed, *args)
        self.assertFalse(
            torch.equal(before[0], after[0]), "anchor must see future noise KV"
        )

    def test_confidence_uses_pre_norm_hidden(self):
        model, _ = self.model(confidence=True)
        hidden = torch.randn(3, 64, dtype=torch.bfloat16)
        result = model.compute_draft_outputs(hidden)
        expected = (
            hidden.float()
            @ self.checkpoint["mtp.2.confidence_head.proj.weight"][:, :64].T
        )
        torch.testing.assert_close(result.dspark_confidence_hidden_logits, expected)

    def test_target_draft_and_feature_cache_keys_are_distinct(self):
        target = dict(self.config, ppu_model_identity="target")
        captured = dict(target, ppu_target_capture_layer_ids=[5, 6, 7])
        draft = draft_config(self.config)
        self.assertEqual(
            len({native_cache_key_hash_seed(c) for c in (target, captured, draft)}), 3
        )
