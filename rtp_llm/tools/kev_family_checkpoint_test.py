"""CPU-only Kev family export tests: synthetic backbones, no downloads or GPU."""

import json
import os
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from tokenizers import AddedToken, Tokenizer, models

from rtp_llm.tools import kev_family_checkpoint as family


def _write_tokenizer(path: Path) -> list[int]:
    tokenizer = Tokenizer(
        models.WordLevel({"<unk>": 0, "hello": 1, "document": 2}, unk_token="<unk>")
    )
    for text in family.SPECIAL_TOKENS:
        tokenizer.add_tokens([AddedToken(text, special=True)])
    tokenizer.save(str(path / "tokenizer.json"))
    ids = [tokenizer.token_to_id(text) for text in family.SPECIAL_TOKENS]
    config = {
        "tokenizer_class": "Qwen2Tokenizer",
        "added_tokens_decoder": {
            str(token_id): {
                "content": text,
                "single_word": False,
                "lstrip": False,
                "rstrip": False,
                "normalized": False,
                "special": True,
            }
            for text, token_id in zip(family.SPECIAL_TOKENS, ids)
        },
    }
    (path / "tokenizer_config.json").write_text(json.dumps(config), encoding="utf-8")
    return ids


def _head(hidden: int, head_dim: int) -> dict:
    generator = torch.Generator().manual_seed(7)
    return {
        "q.weight": torch.randn(head_dim, hidden, generator=generator),
        "q.bias": torch.randn(head_dim, generator=generator),
        "k.weight": torch.randn(head_dim, hidden, generator=generator),
        "k.bias": torch.randn(head_dim, generator=generator),
    }


def _meta(base: str, hidden: int, head_dim: int, **overrides) -> dict:
    meta = {
        "base": base,
        "base_revision": None,
        "lora": 16,
        "head_dim": head_dim,
        "option_isolation": False,
        "special_embeddings": False,
        "weights_dtype": "fp32",
        "lora_placement": "full",
        "weights": "lora",
        "temperature": 1.5,
        "head": _head(hidden, head_dim),
    }
    meta.update(overrides)
    return meta


def _lora_source(path: Path, meta: dict) -> None:
    path.mkdir(parents=True)
    torch.save(meta, path / "head.pt")
    (path / "adapter_config.json").write_text(
        json.dumps(
            {
                "peft_type": "LORA",
                "base_model_name_or_path": meta["base"],
                "r": meta["lora"],
                "lora_alpha": 2 * meta["lora"],
            }
        ),
        encoding="utf-8",
    )
    generator = torch.Generator().manual_seed(11)
    save_file(
        {
            "base_model.model.layers.0.self_attn.q_proj.lora_A.weight": torch.randn(
                meta["lora"], 8, generator=generator
            ),
            "base_model.model.layers.0.self_attn.q_proj.lora_B.weight": torch.randn(
                8, meta["lora"], generator=generator
            ),
            "base_model.model.layers.0.mlp.gate_proj.lora_A.weight": torch.randn(
                meta["lora"], 8, generator=generator
            ),
            "base_model.model.layers.0.mlp.gate_proj.lora_B.weight": torch.randn(
                16, meta["lora"], generator=generator
            ),
        },
        str(path / "adapter_model.safetensors"),
    )


def _lora_base(path: Path) -> dict:
    path.mkdir(parents=True)
    (path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen3",
                "architectures": ["Qwen3ForCausalLM"],
                "hidden_size": 8,
                "intermediate_size": 16,
                "num_hidden_layers": 1,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "head_dim": 4,
                "vocab_size": 16,
                "max_position_embeddings": 40960,
            }
        ),
        encoding="utf-8",
    )
    _write_tokenizer(path)
    generator = torch.Generator().manual_seed(13)
    weights = {
        "model.embed_tokens.weight": torch.randn(16, 8, generator=generator),
        "model.layers.0.input_layernorm.weight": torch.ones(8),
        "model.layers.0.self_attn.q_proj.weight": torch.randn(
            8, 8, generator=generator
        ),
        "model.layers.0.mlp.gate_proj.weight": torch.randn(16, 8, generator=generator),
        "model.norm.weight": torch.ones(8),
        "lm_head.weight": torch.randn(16, 8, generator=generator),
    }
    save_file(weights, str(path / "model.safetensors"))
    return weights


class LoraExportTest(unittest.TestCase):
    def test_merges_once_in_fp32_and_drops_lm_head(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / "base"
            base_weights = _lora_base(base)
            source = root / "source"
            _lora_source(source, _meta("synthetic/base", 8, 4))
            output = root / "output"
            manifest = family.export(source, output, base=base)

            config = json.loads((output / "config.json").read_text())
            self.assertEqual(config["architectures"], ["Qwen3ForCausalLM"])
            self.assertEqual(config["dtype"], "bfloat16")
            self.assertFalse(config["use_cache"])
            index = json.loads(
                (output / "model.safetensors.index.json").read_text()
            )
            shard = index["weight_map"]["model.embed_tokens.weight"]
            exported = load_file(str(output / shard))
            self.assertNotIn("lm_head.weight", exported)
            self.assertEqual(set(exported), set(base_weights) - {"lm_head.weight"})
            self.assertEqual(exported["model.embed_tokens.weight"].dtype, torch.bfloat16)

            adapter = load_file(str(source / "adapter_model.safetensors"))
            expected = (
                base_weights["model.layers.0.self_attn.q_proj.weight"].float()
                + adapter[
                    "base_model.model.layers.0.self_attn.q_proj.lora_B.weight"
                ].float()
                @ adapter[
                    "base_model.model.layers.0.self_attn.q_proj.lora_A.weight"
                ].float()
                * 2.0
            ).to(torch.bfloat16)
            self.assertTrue(
                torch.equal(
                    exported["model.layers.0.self_attn.q_proj.weight"], expected
                )
            )
            untouched = base_weights["model.layers.0.mlp.gate_proj.weight"]
            self.assertTrue(torch.equal(
                exported["model.layers.0.mlp.gate_proj.weight"],
                (
                    untouched.float()
                    + adapter["base_model.model.layers.0.mlp.gate_proj.lora_B.weight"].float()
                    @ adapter["base_model.model.layers.0.mlp.gate_proj.lora_A.weight"].float()
                    * 2.0
                ).to(torch.bfloat16),
            ))

            self.assertEqual(manifest["hidden_size"], 8)
            self.assertEqual(manifest["head_dim"], 4)
            self.assertEqual(manifest["max_state_tokens"], 32768)
            self.assertEqual(manifest["max_row_tokens"], 40960)
            head = load_file(str(output / "kev_head.safetensors"))
            for name, tensor in head.items():
                self.assertTrue(torch.equal(tensor, _head(8, 4)[name]))

    def test_rejects_unsupported_head_modes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / "base"
            _lora_base(base)
            for field, value in (
                ("option_isolation", True),
                ("special_embeddings", True),
                ("lora_placement", "question"),
                ("weights", "unknown"),
            ):
                with self.subTest(field=field):
                    source = root / field
                    _lora_source(source, _meta("synthetic/base", 8, 4, **{field: value}))
                    with self.assertRaises(ValueError):
                        family.export(source, root / (field + "-out"), base=base)

    def test_rejects_missing_adapter_pair_and_wrong_rank(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / "base"
            _lora_base(base)
            source = root / "source"
            _lora_source(source, _meta("synthetic/base", 8, 4))
            adapter = load_file(str(source / "adapter_model.safetensors"))
            adapter.pop("base_model.model.layers.0.mlp.gate_proj.lora_B.weight")
            save_file(adapter, str(source / "adapter_model.safetensors"))
            with self.assertRaises(ValueError):
                family.export(source, root / "out", base=base)


def _full_source(path: Path) -> None:
    path.mkdir(parents=True)
    torch.save(
        _meta("Qwen/Qwen3.8-27B", 8, 4, lora=0, weights="full", weights_dtype="bf16"),
        path / "head.pt",
    )
    _write_tokenizer(path)
    (path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen3_5_text",
                "architectures": ["Qwen3_5TextModel"],
                "hidden_size": 8,
                "num_hidden_layers": 1,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "head_dim": 4,
                "vocab_size": 16,
                "max_position_embeddings": 262144,
            }
        ),
        encoding="utf-8",
    )
    generator = torch.Generator().manual_seed(17)
    first = {
        "model.embed_tokens.weight": torch.randn(
            16, 8, generator=generator, dtype=torch.bfloat16
        ),
        "model.norm.weight": torch.ones(8, dtype=torch.bfloat16),
    }
    second = {
        "model.layers.0.input_layernorm.weight": torch.ones(8, dtype=torch.bfloat16),
        "model.layers.0.mlp.gate_proj.weight": torch.randn(
            16, 8, generator=generator, dtype=torch.bfloat16
        ),
    }
    save_file(first, str(path / "model-00001-of-00002.safetensors"))
    save_file(second, str(path / "model-00002-of-00002.safetensors"))
    weight_map = {
        name: "model-00001-of-00002.safetensors" for name in first
    }
    weight_map.update({name: "model-00002-of-00002.safetensors" for name in second})
    (path / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": 1}, "weight_map": weight_map}),
        encoding="utf-8",
    )


class FullExportTest(unittest.TestCase):
    def test_relinks_text_only_checkpoint_and_rewrites_architecture(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            _full_source(source)
            output = root / "output"
            manifest = family.export(source, output)
            config = json.loads((output / "config.json").read_text())
            self.assertEqual(config["architectures"], ["Qwen3_5ForCausalLM"])
            self.assertEqual(manifest["max_state_tokens"], 65536)
            self.assertEqual(manifest["max_row_tokens"], 73728)
            self.assertEqual(manifest["base_model"], "Qwen/Qwen3.8-27B")
            for name in (
                "model-00001-of-00002.safetensors",
                "model-00002-of-00002.safetensors",
            ):
                self.assertEqual(
                    os.stat(output / name).st_ino, os.stat(source / name).st_ino
                )

    def test_rejects_composite_or_lm_head_full_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            _full_source(source)
            config = json.loads((source / "config.json").read_text())
            config["text_config"] = config.copy()
            (source / "config.json").write_text(json.dumps(config))
            with self.assertRaises(ValueError):
                family.export(source, root / "out")

    def test_prepare_cached_reuses_the_same_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            _full_source(source)
            cache = root / "cache"
            first = family.prepare_cached(source, cache_root=cache)
            second = family.prepare_cached(source, cache_root=cache)
            self.assertEqual(first, second)
            self.assertTrue((Path(first) / "kev_config.json").is_file())
            self.assertEqual(json.loads((Path(first) / "kev_config.json").read_text())[
                "hidden_size"
            ], 8)


if __name__ == "__main__":
    unittest.main()
