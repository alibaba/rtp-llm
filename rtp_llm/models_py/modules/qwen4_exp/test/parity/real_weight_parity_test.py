import os
import unittest
from types import SimpleNamespace

import torch

from rtp_llm.models_py.modules.base.common.zero_centered_rms_norm import (
    ZeroCenteredFusedQKRMSNorm,
)
from rtp_llm.models_py.modules.hybrid.gated_residual import GatedResidual
from rtp_llm.models_py.modules.qwen4_exp.test.parity.upstream_fixture import (
    checkpoint_path,
    uncovered_fixture,
    upstream_source_path,
)
from rtp_llm.models_py.modules.qwen4_exp.test.support.upstream_parity import (
    _gamma,
    _load_upstream,
)

_UPSTREAM_SRC = upstream_source_path()
_CKPT_DIR = checkpoint_path()


class RealWeightGatedResidualParityTest(unittest.TestCase):
    """Same parity, driven by the released checkpoint's own tensors."""

    HIDDEN, HC, LOWRANK, EPS = 2560, 4, 320, 1e-6
    PREFIX = "model.language_model.layers.0.attn_hyper_connection."
    NAMES = (
        "hc_norm.weight",
        "input_mix_weight_down.weight",
        "input_mix_weight_up.weight",
        "block_inject_weight.weight",
    )

    @classmethod
    def setUpClass(cls):
        import json

        from safetensors.torch import load_file

        cls.up = _load_upstream(_UPSTREAM_SRC)
        index = os.path.join(_CKPT_DIR, "model.safetensors.index.json")
        weight_map = json.load(open(index))["weight_map"]
        keys = [cls.PREFIX + name for name in cls.NAMES]
        shards = {weight_map[key] for key in keys}
        absent = [s for s in shards if not os.path.exists(os.path.join(_CKPT_DIR, s))]
        if absent:
            uncovered_fixture(f"shards not downloaded yet: {sorted(absent)}")
        cls.tensors = {}
        for shard in shards:
            loaded = load_file(os.path.join(_CKPT_DIR, shard))
            cls.tensors.update({k: loaded[k] for k in keys if k in loaded})

    def _get(self, name):
        return self.tensors[self.PREFIX + name].double()

    def test_real_layer0_unit_is_bit_exact(self):
        config = SimpleNamespace(
            hidden_size=self.HIDDEN,
            hc_count=self.HC,
            hc_lowrank=self.LOWRANK,
            rms_norm_eps=self.EPS,
        )
        ref = self.up.Qwen4ExpTextGatedResidual(config).double()
        with torch.no_grad():
            ref.hc_norm.weight.copy_(self._get("hc_norm.weight"))
            ref.input_mix_weight_down.weight.copy_(
                self._get("input_mix_weight_down.weight")
            )
            ref.input_mix_weight_up.weight.copy_(
                self._get("input_mix_weight_up.weight")
            )
            ref.block_inject_weight.weight.copy_(
                self._get("block_inject_weight.weight")
            )
        ours = GatedResidual(
            _gamma(self._get("hc_norm.weight")),
            self._get("input_mix_weight_down.weight"),
            self._get("input_mix_weight_up.weight"),
            self._get("block_inject_weight.weight"),
            hc_mult=self.HC,
            norm_eps=self.EPS,
        )
        torch.manual_seed(0)
        x = torch.randn(1, 4, self.HC * self.HIDDEN, dtype=torch.float64)
        ref_mixed, _, ref_inject = ref(x)
        mixed, _, inject = ours(x)
        self.assertEqual((ref_mixed - mixed).abs().max().item(), 0.0)
        self.assertEqual((ref_inject - inject).abs().max().item(), 0.0)

    def test_real_qk_norm_is_bit_exact(self):
        """The 26 q/k norm tensors are the ones the Qwen3.5 base would fold."""
        import json

        from safetensors.torch import load_file

        index = os.path.join(_CKPT_DIR, "model.safetensors.index.json")
        weight_map = json.load(open(index))["weight_map"]
        pairs = {}
        for key, shard in weight_map.items():
            if not key.endswith(("self_attn.q_norm.weight", "self_attn.k_norm.weight")):
                continue
            if not os.path.exists(os.path.join(_CKPT_DIR, shard)):
                continue
            layer = key.rsplit(".self_attn.", 1)[0]
            pairs.setdefault(layer, {})[key.rsplit(".", 2)[-2]] = (key, shard)
        complete = {l: v for l, v in pairs.items() if {"q_norm", "k_norm"} <= v.keys()}
        if not complete:
            uncovered_fixture("no layer has both q_norm and k_norm downloaded")

        layer, entry = sorted(complete.items())[0]
        gammas = {}
        for role, (key, shard) in entry.items():
            gammas[role] = load_file(os.path.join(_CKPT_DIR, shard))[key].double()
        head_dim = gammas["q_norm"].shape[0]
        heads, kv_heads = 3, 2
        torch.manual_seed(0)
        qkv = torch.randn(4, (heads + 2 * kv_heads) * head_dim, dtype=torch.float64)

        def upstream(x, gamma):
            norm = self.up.Qwen4ExpTextRMSNorm(head_dim, eps=self.EPS).double()
            with torch.no_grad():
                norm.weight.copy_(gamma)
            return norm(x)

        packed = qkv.reshape(4, heads + 2 * kv_heads, head_dim)
        expected = packed.clone()
        expected[:, :heads] = upstream(packed[:, :heads], gammas["q_norm"])
        expected[:, heads : heads + kv_heads] = upstream(
            packed[:, heads : heads + kv_heads], gammas["k_norm"]
        )
        got = ZeroCenteredFusedQKRMSNorm(
            gammas["q_norm"], gammas["k_norm"], heads, kv_heads, head_dim, self.EPS
        )(qkv.clone())
        self.assertEqual(
            (expected.reshape(4, -1) - got).abs().max().item(), 0.0, msg=layer
        )

    def test_checkpoint_shapes_match_config(self):
        self.assertEqual(
            tuple(self._get("input_mix_weight_down.weight").shape),
            (self.LOWRANK, self.HC * self.HIDDEN),
        )
        self.assertEqual(
            tuple(self._get("block_inject_weight.weight").shape),
            (self.HC, self.HC * self.HIDDEN),
        )


if __name__ == "__main__":
    unittest.main()
