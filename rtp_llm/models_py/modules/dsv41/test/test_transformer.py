import os
import unittest
from types import SimpleNamespace

import torch
from torch import nn

from rtp_llm.models_py.modules.dsv41.block import V41Block
from rtp_llm.models_py.modules.dsv41.engram import Engram
from rtp_llm.models_py.modules.dsv41.inputs import V41ModelRows
from rtp_llm.models_py.modules.dsv41.math import hc_pre, rms_norm
from rtp_llm.models_py.modules.dsv41.transformer import V41ImageFeatures, V41TargetModel
from rtp_llm.utils.model_weight import W


class RecordSublayer(nn.Module):
    def __init__(self, layer, part, calls):
        super().__init__()
        self.layer, self.part, self.calls = layer, part, calls

    def forward(self, hidden, metadata):
        self.calls.append((self.layer, self.part, metadata, hidden.clone()))
        return hidden * 0.125


class RecordHasher(nn.Module):
    def forward(self, ids, history, history_valid, token_mask):
        self.received = tuple(
            tensor.clone() for tensor in (ids, history, history_valid, token_mask)
        )
        base = torch.arange(48, dtype=torch.int64, device=ids.device).view(1, 1, 2, 24)
        return base.expand(ids.shape[0], ids.shape[1], 2, 24).contiguous()


class RecordLookup:
    def __init__(self):
        self.calls = []

    def lookup(self, layer, ids, valid_mask=None, out=None):
        self.calls.append((layer, ids.clone(), valid_mask.clone()))
        result = torch.ones((*ids.shape, 256), device=ids.device, dtype=torch.bfloat16)
        if out is not None:
            out.copy_(result)
            return out
        return result


class FixedProjection(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size

    def forward(self, values):
        return values.new_full((*values.shape[:-1], self.hidden_size * 5), 0.25)


class TargetComposerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.device = torch.device(os.environ.get("DSV41_TEST_DEVICE", "cuda"))
        if not torch.cuda.is_available() or cls.device.type != "cuda":
            raise RuntimeError(
                "V4.1 target integration fixtures require a real CUDA GPU"
            )
        torch.backends.cuda.matmul.allow_tf32 = False

    def config(self):
        return SimpleNamespace(
            pad_token_id=2,
            text={
                "num_hidden_layers": 40,
                "hidden_size": 32,
                "vocab_size": 129280,
                "hc_mult": 4,
                "dspark_target_layer_ids": (37, 38, 39),
            },
        )

    def weights(self, layer):
        result = {
            "attn_norm.weight": torch.ones(32, device=self.device).bfloat16(),
            "ffn_norm.weight": torch.ones(32, device=self.device).bfloat16(),
        }
        for part in ("attn", "ffn"):
            result[f"hc_{part}_fn"] = torch.zeros(24, 128, device=self.device)
            result[f"hc_{part}_scale"] = torch.ones(3, device=self.device)
            result[f"hc_{part}_base"] = (
                torch.arange(24, device=self.device).float() % (layer % 3 + 2)
            ) * 0.125
        return result

    def make_model(self):
        calls, blocks, lookup = [], [], RecordLookup()
        for layer in range(40):
            blocks.append(
                V41Block(
                    RecordSublayer(layer, "attn", calls),
                    RecordSublayer(layer, "moe", calls),
                    self.weights(layer),
                )
            )
        gate = torch.ones(4, 32, dtype=torch.bfloat16, device=self.device)
        engrams = {
            layer: Engram(layer, lookup, FixedProjection(32), gate, gate)
            for layer in (1, 14)
        }
        embedding = torch.ones(129280, 32, dtype=torch.bfloat16, device=self.device)
        embedding[3] *= 2
        hasher = RecordHasher()
        model = V41TargetModel(
            self.config(),
            embedding,
            torch.ones(32, dtype=torch.bfloat16, device=self.device),
            torch.ones(129280, 32, device=self.device),
            blocks,
            engrams,
            hasher,
        )
        return model, calls, lookup

    def rows(self, image=False):
        if image:
            ids, types = [3, 129264, 129264, 129264, 129264, -1], [-1, 0, 1, 2, 3, -1]
        else:
            ids, types = [3, 4, 5, -1], [-1] * 4
        n = len(ids)
        valid = torch.ones(n, dtype=torch.bool, device=self.device)
        valid[-1] = False
        return V41ModelRows(
            torch.tensor(ids, dtype=torch.int32, device=self.device),
            torch.tensor(types, dtype=torch.int32, device=self.device),
            valid,
            torch.zeros(n, 3, dtype=torch.int32, device=self.device),
            torch.zeros(n, 3, dtype=torch.bool, device=self.device),
        )

    def test_complete_shifted_pre_chain_engram_order_and_selected_aux_inputs(self):
        model, calls, lookup = self.make_model()
        rows = self.rows(image=True)
        image_rows = torch.arange(1, 5, device=self.device, dtype=torch.int64)
        features = V41ImageFeatures(
            image_rows,
            rows.token_types[image_rows].contiguous(),
            torch.arange(1, 129, device=self.device).reshape(4, 32).bfloat16(),
        )
        inputs, outputs, injected = {}, {}, {}
        hooks = []
        for index, block in enumerate(model.blocks):
            hooks.append(
                block.register_forward_pre_hook(
                    lambda _, args, index=index: inputs.__setitem__(
                        index, (args[0].clone(), args[1].clone())
                    )
                )
            )
            hooks.append(
                block.register_forward_hook(
                    lambda _, args, result, index=index: outputs.__setitem__(
                        index, (result[0].clone(), result[1].clone())
                    )
                )
            )
        for name, engram in model.engrams.items():
            hooks.append(
                engram.register_forward_hook(
                    lambda _, args, result, name=name: injected.__setitem__(
                        int(name), result.clone()
                    )
                )
            )
        context = object()
        aux_rows = torch.tensor([0, 3], device=self.device, dtype=torch.int64)
        try:
            result = model(
                rows,
                context,
                execution_mode="full",
                image_features=features,
                aux_row_indices=aux_rows,
            )
        finally:
            for hook in hooks:
                hook.remove()
        self.assertEqual(
            [(a, b) for a, b, _, _ in calls],
            [(i, p) for i in range(40) for p in ("attn", "moe")],
        )
        self.assertTrue(
            all(metadata is context for _, part, metadata, _ in calls if part == "attn")
        )
        for _, part, metadata, _ in calls:
            if part == "moe":
                self.assertTrue(torch.equal(metadata, rows.image_mask))
        expected_pre = torch.zeros_like(inputs[0][1])
        expected_pre[:, 0] = 1
        self.assertTrue(torch.equal(inputs[0][1], expected_pre))
        for layer in range(1, 40):
            self.assertTrue(torch.equal(inputs[layer][1], outputs[layer - 1][1]))
        for layer in (1, 14):
            self.assertTrue(torch.equal(inputs[layer][0], injected[layer]))
            self.assertTrue(
                torch.equal(injected[layer][1:5], outputs[layer - 1][0][1:5])
            )
        self.assertTrue(
            torch.equal(inputs[0][0][1:5], features.values[:, None].expand(-1, 4, -1))
        )
        self.assertEqual([call[0] for call in lookup.calls], [1, 14])
        for _, _, valid in lookup.calls:
            self.assertTrue(torch.equal(valid, rows.text_mask[:, None].expand(-1, 24)))
        expected_aux = torch.cat(
            [inputs[layer][0][aux_rows].mean(1) for layer in (37, 38, 39)], -1
        )
        self.assertTrue(torch.equal(result.aux_hidden_states, expected_aux))
        self.assertTrue(torch.equal(result.aux_row_indices, aux_rows))
        self.assertEqual(result.aux_layer_ids, (37, 38, 39))
        self.assertTrue(torch.equal(result.final_pre_mix, outputs[39][1]))
        expected_hidden = rms_norm(hc_pre(outputs[39][0], outputs[39][1]), model.norm)
        self.assertTrue(torch.equal(result.hidden_states, expected_hidden))
        self.assertTrue(
            torch.equal(
                result.hidden_states[-1],
                torch.zeros(32, device=self.device).bfloat16(),
            )
        )
        self.assertEqual(model.logits(result.hidden_states[:1]).dtype, torch.float32)

    def test_factories_bind_installed_weights_and_host_tables_stay_external(self):
        model, _, lookup = self.make_model()
        layers = []
        for layer in range(40):
            installed = {
                "v41." + name: value for name, value in self.weights(layer).items()
            }
            if layer in (1, 14):
                for name in ("q_weight", "k_weight"):
                    installed["v41.engram." + name] = torch.ones(
                        4, 32, device=self.device
                    ).bfloat16()
                installed["v41.engram.wkv.weight"] = torch.zeros(
                    160, 6144, device=self.device
                ).to(torch.float8_e4m3fn)
                installed["v41.engram.wkv.scale"] = torch.ones(
                    5, 192, device=self.device
                ).to(torch.float8_e8m0fnu)
            layers.append(installed)
        weights = SimpleNamespace(
            weights=layers,
            global_weights={
                W.embedding: model.embedding,
                W.final_ln_gamma: model.norm,
                W.lm_head: model.head,
            },
        )
        calls, initialized = [], []

        def factory(layer, local):
            initialized.append(layer)
            self.assertIs(
                local["attn_norm.weight"], layers[layer]["v41.attn_norm.weight"]
            )
            return RecordSublayer(layer, "component", calls)

        bound = V41TargetModel.from_model_weights(
            self.config(),
            weights,
            attention_factory=factory,
            moe_factory=factory,
            shared_lookup=lookup,
            token_hasher=RecordHasher(),
            projection_factory=lambda weight, scale: FixedProjection(32),
        )
        self.assertEqual(initialized, [layer for layer in range(40) for _ in range(2)])
        self.assertEqual(bound.embedding.data_ptr(), model.embedding.data_ptr())
        self.assertEqual(bound.head.data_ptr(), model.head.data_ptr())
        self.assertIs(bound.engrams["1"].shared_lookup, lookup)
        self.assertFalse(any("engram.embed" in name for name in bound.state_dict()))
        self.assertEqual(
            bound(self.rows(), object(), execution_mode="full").hidden_states.shape,
            (4, 32),
        )

    def test_requires_explicit_full_execution_and_keeps_aux_opt_in(self):
        model, _, _ = self.make_model()
        with self.assertRaisesRegex(ValueError, "explicit full"):
            model(self.rows(), object(), execution_mode="bounded")
        result = model(self.rows(), object(), execution_mode="full")
        self.assertIsNone(result.aux_hidden_states)
        self.assertIsNone(result.aux_row_indices)
        self.assertEqual(result.aux_layer_ids, ())


if __name__ == "__main__":
    unittest.main()
