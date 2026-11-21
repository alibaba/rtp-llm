"""M890P parameter tests; CPU checks neither construct vendor kernels nor claim GPU support."""

import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS

import torch
from safetensors.torch import save_file

from rtp_llm.models_py.model_loader import NewLoaderConfig, NewModelLoader
from rtp_llm.models_py.new_models.deepseek_v4.ppu_attention import create_ppu_attention
from rtp_llm.models_py.read_spec import ReadSpec
from rtp_llm.models_py.weight_mapper import get_all_weights


def args(ratio, tp, rank):
    return NS(
        dim=256,
        n_heads=2,
        q_lora_rank=128,
        head_dim=512,
        o_groups=2,
        o_lora_rank=128,
        index_n_heads=2,
        index_head_dim=128,
        tp_size=tp,
        tp_rank=rank,
        compress_ratios=[ratio],
        commit_only=False,
    )


class PpuAttentionTest(unittest.TestCase):
    def file_roundtrip(self, ratio, fst=False):
        reference = create_ppu_attention(args(ratio, 1, 0), 0, object())
        generator = torch.Generator().manual_seed(17)
        data = {}
        for name, original in reference.named_parameters():
            shape = original.shape
            if original.dtype in (torch.float8_e4m3fn, torch.float8_e8m0fnu):
                data[name] = torch.randint(
                    1, 242, shape, generator=generator, dtype=torch.uint8
                ).view(original.dtype)
            else:
                data[name] = torch.rand(
                    shape, generator=generator, dtype=torch.float32
                ).to(original.dtype)
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "model.safetensors")
            save_file(data, path)
            for rank in (0, 1):
                with self.subTest(ratio=ratio, rank=rank, fst=fst):
                    owner = create_ppu_attention(args(ratio, 2, rank), 0, object())
                    loader = NewModelLoader(
                        NS(model_type="deepseek_v4"),
                        NewLoaderConfig(
                            device="cpu",
                            load_method="fastsafetensors" if fst else "scratch",
                        ),
                        model_path=tmp,
                    )
                    resolver = loader._checkpoint_read_spec_resolver(owner, None, None)
                    reader = (
                        loader._fastsafetensors_weights_iterator(
                            [path], "cpu", None, None, resolver
                        )
                        if fst
                        else get_all_weights([path], read_spec_resolver=resolver)
                    )
                    owner.load_weights(reader)
                    owner.validate_weights_loaded()
                    for key, value in owner.named_parameters():
                        raw = data[key]
                        if key == "attn_sink" or key.startswith(("wq_b.", "wo_a.")):
                            raw = raw.chunk(2, 0)[rank]
                        elif key.startswith("wo_b."):
                            raw = raw.chunk(2, 1)[rank]
                        self.assertEqual(raw.dtype, value.dtype, key)
                        self.assertTrue(
                            torch.equal(
                                raw.contiguous().view(torch.uint8),
                                value.view(torch.uint8),
                            ),
                            key,
                        )
                    if ratio == 4:
                        self.assertEqual(
                            owner.indexer.compressor.norm.weight.dtype, torch.float32
                        )
                    with self.assertRaisesRegex(RuntimeError, "M890P"):
                        owner.bind_runtime()

    def test_scratch_tp2_swa_csa_hca(self):
        for ratio in (0, 4, 128):
            self.file_roundtrip(ratio)

    @unittest.skipUnless(
        os.environ.get("PPU_ATTN_NATIVE_FST") == "1",
        "requires native FST with E8M0 support",
    )
    def test_native_fst_cpu_csa(self):
        self.file_roundtrip(4, True)

    def test_presliced_wrong_rank_missing_or_duplicate(self):
        left = create_ppu_attention(args(4, 2, 0), 0, object())
        right = create_ppu_attention(args(4, 2, 1), 0, object())
        value = torch.tensor([1.0, 3.0])
        left_spec = left.checkpoint_read_spec("attn_sink", (2,))
        with self.assertRaisesRegex(ValueError, "another TP rank"):
            right.load_weights({"attn_sink": left_spec.materialize(value)})
        right.load_weights({"attn_sink": value})
        self.assertEqual(right.attn_sink.item(), 3.0)
        with self.assertRaisesRegex(RuntimeError, "Duplicate"):
            right.load_weights({"attn_sink": value})
        with self.assertRaisesRegex(RuntimeError, "missing"):
            right.bind_runtime()

    @unittest.skipUnless(
        os.environ.get("PPU_FLASH_RELEASE_PATH"), "optional real Flash checkpoint"
    )
    def test_release_three_layers(self):
        root = Path(os.environ["PPU_FLASH_RELEASE_PATH"])
        cfg = json.loads((root / "config.json").read_text())
        for ratio in (0, 4, 128):
            layer = cfg["compress_ratios"].index(ratio)
            p = NS(
                dim=cfg["hidden_size"],
                n_heads=cfg["num_attention_heads"],
                q_lora_rank=cfg["q_lora_rank"],
                head_dim=cfg["head_dim"],
                o_groups=cfg["o_groups"],
                o_lora_rank=cfg["o_lora_rank"],
                index_n_heads=cfg["index_n_heads"],
                index_head_dim=cfg["index_head_dim"],
                tp_size=4,
                tp_rank=3,
                compress_ratios=cfg["compress_ratios"],
                commit_only=False,
            )
            owner = create_ppu_attention(p, layer, object())
            prefix = f"layers.{layer}.attn."

            def resolver(k, shape):
                spec = owner.resolve_checkpoint_read_spec(k[len(prefix) :], shape)
                return (
                    spec.for_checkpoint_name(k)
                    if spec is not None
                    else ReadSpec.full(k, shape)
                )

            index_file = root / "model.safetensors.index.json"
            if index_file.is_file():
                index = json.loads(index_file.read_text())["weight_map"]
                files = sorted(
                    {
                        root / path
                        for name, path in index.items()
                        if name.startswith(prefix)
                    }
                )
            else:
                index = {}
                files = [root / "model.safetensors"]
            owner.load_weights(
                (k[len(prefix) :], v)
                for k, v in get_all_weights(
                    list(map(str, files)),
                    name_filter=lambda k: k.startswith(prefix),
                    read_spec_resolver=resolver,
                )
            )
            owner.validate_weights_loaded()
            from safetensors import safe_open

            def read(name):
                key = prefix + name
                with safe_open(
                    root / index.get(key, "model.safetensors"),
                    framework="pt",
                    device="cpu",
                ) as reader:
                    return reader.get_tensor(key)

            expected = read("attn_sink").chunk(4, 0)[3].float()
            torch.testing.assert_close(owner.attn_sink, expected, atol=0, rtol=0)
            if ratio == 4:
                torch.testing.assert_close(
                    owner.indexer.weights_proj.weight,
                    read("indexer.weights_proj.weight").bfloat16(),
                    atol=0,
                    rtol=0,
                )


if __name__ == "__main__":
    unittest.main()
