"""CPU contract: module-owned parameters produce the original Mega kernel ABI."""

import json
import os
import unittest
from dataclasses import fields
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

import torch
from safetensors import safe_open

from rtp_llm.models_py.modules.dsv4.fp8.decode.mega_csa_weights import (
    FLASH_GEOMETRY,
    CSAGeometry,
    MegaCSAWeights,
)
from rtp_llm.models_py.modules.dsv4.fp8.decode.mega_hca_weights import MegaHCAWeights
from rtp_llm.models_py.modules.dsv4.fp8.decode.mega_owned import (
    initialize_owned_mega_decode,
)
from rtp_llm.models_py.modules.dsv4.fp8.decode.mega_policy import mega_decode_enabled
from rtp_llm.utils.model_weight import W


def check_pack(test, ratio, g, read):
    """Build views from the same HF tensors as the legacy and post-load consumers."""
    weights = {}

    def tensor(tag, path, dims, dtype):
        x = read(path, dims, dtype)
        test.assertEqual(tuple(x.shape), dims)
        weights[tag] = x
        return x

    def fp8(name, wtag, stag, n, k):
        return NS(
            weight=tensor(wtag, name + ".weight", (n, k), torch.float8_e4m3fn),
            scale=tensor(
                stag, name + ".scale", (n // 128, k // 128), torch.float8_e8m0fnu
            ),
        )

    def comp(prefix, shape, head, ape, kv, gate, norm):
        return NS(
            ape=tensor(ape, prefix + ".ape", (ratio, shape), torch.float32),
            wkv=NS(
                weight=tensor(
                    kv, prefix + ".wkv.weight", (shape, g.dim), torch.bfloat16
                )
            ),
            wgate=NS(
                weight=tensor(
                    gate, prefix + ".wgate.weight", (shape, g.dim), torch.bfloat16
                )
            ),
            norm=NS(
                weight=tensor(norm, prefix + ".norm.weight", (head,), torch.bfloat16)
            ),
        )

    attn = NS(
        wq_a=fp8("attn.wq_a", W.v4_attn_wq_a_w, W.v4_attn_wq_a_s, g.q_lora_rank, g.dim),
        wq_b=fp8(
            "attn.wq_b", W.v4_attn_wq_b_w, W.v4_attn_wq_b_s, g.n_main, g.q_lora_rank
        ),
        wkv=fp8("attn.wkv", W.v4_attn_wkv_w, W.v4_attn_wkv_s, 512, g.dim),
        q_norm=tensor(
            W.v4_attn_q_norm, "attn.q_norm.weight", (g.q_lora_rank,), torch.bfloat16
        ),
        kv_norm=tensor(
            W.v4_attn_kv_norm, "attn.kv_norm.weight", (512,), torch.bfloat16
        ),
        compressor=comp(
            "attn.compressor",
            1024 if ratio == 4 else 512,
            512,
            W.v4_compressor_ape,
            W.v4_compressor_wkv,
            W.v4_compressor_wgate,
            W.v4_compressor_norm,
        ),
    )
    block = NS(
        attn=attn,
        attn_norm=NS(
            weight=tensor(W.v4_attn_norm, "attn_norm.weight", (g.dim,), torch.bfloat16)
        ),
        attn_hc=NS(
            fn=tensor(W.v4_hc_attn_fn, "hc_attn_fn", (24, 4 * g.dim), torch.float32),
            base=tensor(W.v4_hc_attn_base, "hc_attn_base", (24,), torch.float32),
            scale=tensor(W.v4_hc_attn_scale, "hc_attn_scale", (3,), torch.float32),
        ),
    )
    if ratio == 4:
        raw = tensor(
            W.v4_indexer_weights_proj_w,
            "attn.indexer.weights_proj.weight",
            (64, g.dim),
            torch.bfloat16,
        )
        attn.indexer = NS(
            wq_b=fp8(
                "attn.indexer.wq_b",
                W.v4_indexer_wq_b_w,
                W.v4_indexer_wq_b_s,
                g.n_index,
                g.q_lora_rank,
            ),
            compressor=comp(
                "attn.indexer.compressor",
                256,
                128,
                W.v4_indexer_compressor_ape,
                W.v4_indexer_compressor_wkv,
                W.v4_indexer_compressor_wgate,
                W.v4_indexer_compressor_norm,
            ),
            weights_proj=raw * (128**-0.5 * 64**-0.5),
            _weights_proj_folded=False,
        )
        with test.assertRaisesRegex(RuntimeError, "post-loaded"):
            MegaCSAWeights.from_owned_block(block, g)
        attn.indexer._weights_proj_folded = True
    pack = MegaCSAWeights if ratio == 4 else MegaHCAWeights
    reference = pack.from_layer_weights(weights, g)
    result = pack.from_owned_block(block, g)
    for f in fields(reference):
        a, b = getattr(result, f.name), getattr(reference, f.name)
        if torch.is_tensor(a):
            test.assertEqual(a.dtype, b.dtype, f.name)
            test.assertTrue(
                torch.equal(a.view(torch.uint8), b.view(torch.uint8)), f.name
            )
        else:
            test.assertEqual(a, b)


class MegaOwnedWeightsTest(unittest.TestCase):
    def test_csa_and_hca_packing(self):
        geometry = CSAGeometry(
            dim=128, q_lora_rank=128, main_heads=1, o_groups=1, index_topk=2
        )
        gen = torch.Generator(device="cpu").manual_seed(71)

        def read(path, shape, dtype):
            if dtype in (torch.float8_e4m3fn, torch.float8_e8m0fnu):
                return torch.randint(
                    0, 255, shape, generator=gen, dtype=torch.uint8
                ).view(dtype)
            return torch.randn(shape, generator=gen).to(dtype)

        for ratio in (4, 128):
            with self.subTest(ratio=ratio):
                check_pack(self, ratio, geometry, read)

    def test_policy(self):
        args = NS(n_layers=1, compress_ratios=[4], tp_size=2, fp8_kv_cache=True)
        with patch.dict(os.environ, {"DSV4_MEGA": "1"}):
            with self.assertRaisesRegex(RuntimeError, "TP1 is required"):
                mega_decode_enabled(args, torch.device("cpu"))
            args.compress_ratios = [0]
            self.assertFalse(mega_decode_enabled(args, torch.device("cpu")))
            initialize_owned_mega_decode(NS(commit_only=True), args)
        args.compress_ratios = [4]
        with patch.dict(os.environ, {"DSV4_MEGA": "0"}):
            self.assertFalse(mega_decode_enabled(args, torch.device("cpu")))

    @unittest.skipUnless(
        os.environ.get("V4_MEGA_RELEASE_CHECKPOINT"), "optional real Flash weights"
    )
    def test_release_csa_and_hca(self):
        root = Path(os.environ["V4_MEGA_RELEASE_CHECKPOINT"])
        index = json.loads((root / "model.safetensors.index.json").read_text())[
            "weight_map"
        ]
        ratios = json.loads((root / "config.json").read_text())["compress_ratios"]
        for ratio in (4, 128):
            pos = ratios.index(ratio)

            def read(path, shape, dtype):
                name = f"layers.{pos}.{path}"
                with safe_open(
                    root / index[name], framework="pt", device="cpu"
                ) as shard:
                    x = shard.get_tensor(name)
                # The old loader supplied BF16 norms to Mega; HC/APE remain FP32.
                if dtype in (torch.float8_e8m0fnu, torch.float8_e4m3fn):
                    self.assertEqual(x.dtype, dtype, name)
                    return x
                return x.to(dtype)

            with self.subTest(ratio=ratio):
                check_pack(self, ratio, FLASH_GEOMETRY, read)


if __name__ == "__main__":
    unittest.main()
