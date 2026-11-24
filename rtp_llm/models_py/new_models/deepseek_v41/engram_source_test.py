"""Actual layer-local Engram checkpoint loads (no GPU/UVA emulation)."""

import gc
import json
import mmap
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from safetensors.torch import save_file

from rtp_llm.models_py.model_loader import NewModelLoader
from rtp_llm.models_py.modules.dsv41 import engram as runtime_engram
from rtp_llm.models_py.modules.dsv41.engram import Engram
from rtp_llm.models_py.new_models.deepseek_v41.engram import OwnedEngram
from rtp_llm.models_py.new_models.deepseek_v41.engram_source import EngramSource


class EngramSourceTest(unittest.TestCase):
    cfg = dict(
        engram_layer_ids=[1, 14],
        engram_num_embeddings=[8, 12],
        engram_max_ngram_size=2,
        engram_n_heads=1,
        engram_head_dim=32,
        engram_compressed_vocab_size=8,
        engram_pad_token_id=0,
        engram_vocab_size=7,
        hidden_size=32,
        hc_mult=2,
        rms_norm_eps=1e-6,
    )

    def fixture(self, directory, *, bad=None):
        index = {}
        for layer_id, rows in ((1, 8), (14, 12)):
            prefix = f"layers.{layer_id}.engram."
            name = f"layer-{layer_id}.safetensors"
            tensors = {
                "embed.weight": (
                    torch.arange(rows * 32).reshape(rows, 32) % 7 + layer_id
                ).to(torch.float8_e4m3fn),
                "embed.scale": torch.full((rows, 1), 126, dtype=torch.uint8).view(
                    torch.float8_e8m0fnu
                ),
                "wkv.weight": ((torch.arange(96 * 32).reshape(96, 32) % 9) - 4).to(
                    torch.float8_e4m3fn
                ),
                "wkv.scale": torch.full((3, 1), 125, dtype=torch.uint8).view(
                    torch.float8_e8m0fnu
                ),
                "q_weight": torch.full((2, 32), 0.5, dtype=torch.bfloat16),
                "k_weight": torch.full((2, 32), 1.5, dtype=torch.bfloat16),
            }
            if bad is not None:
                bad(tensors)
            save_file({prefix + k: v for k, v in tensors.items()}, directory / name)
            index.update({prefix + k: name for k in tensors})
        (directory / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": index})
        )

    def test_ft_export_streams_only_engram_and_serves_both_readers(self):
        from safetensors import safe_open
        from rtp_llm.tools.convert.ft_export_engram import export_v41_engram

        with tempfile.TemporaryDirectory() as work:
            source, output = Path(work) / "hf", Path(work) / "ft"
            source.mkdir()
            output.mkdir()
            self.fixture(source)
            (source / "config.json").write_text(json.dumps({"text_config": self.cfg}))
            (output / "model.safetensors.index.json").write_text(
                json.dumps({"is_ft_style_weight": True, "weight_map": {}})
            )
            # One shard is exclusively Engram, the other shares a decoder tensor.
            mixed = source / "layer-1.safetensors"
            with safe_open(str(mixed), framework="pt") as shard:
                before = {key: shard.get_tensor(key).clone() for key in shard.keys()}
            save_file({**before, "decoder.unrelated": torch.ones(10)}, mixed)
            specs = {"embed.weight": ((8, 32), "F8_E4M3", torch.float8_e4m3fn)}
            with self.assertRaisesRegex(ValueError, "original indexed HF"):
                EngramSource(output, 1, specs, "scratch")
            export_v41_engram(source, output)
            side = output / "v41_engram"
            index = json.loads((side / "model.safetensors.index.json").read_text())[
                "weight_map"
            ]
            self.assertEqual(len(index), 12)
            for layer in (1, 14):
                with safe_open(
                    str(source / f"layer-{layer}.safetensors"), framework="pt"
                ) as orig, safe_open(
                    str(side / index[f"layers.{layer}.engram.embed.weight"]),
                    framework="pt",
                ) as got:
                    self.assertNotIn("decoder.unrelated", got.keys())
                    for key in got.keys():
                        self.assertTrue(
                            torch.equal(
                                orig.get_tensor(key).view(torch.uint8),
                                got.get_tensor(key).view(torch.uint8),
                            ),
                            key,
                        )
            for method in ("scratch", "fastsafetensors"):
                eng = EngramSource(output, 1, specs, method)
                self.assertTrue(
                    torch.equal(
                        eng.load("embed.weight").view(torch.uint8),
                        before["layers.1.engram.embed.weight"].view(torch.uint8),
                    )
                )

    def test_ready_is_reported_only_after_local_engram_tables_bind(self):
        from tokenizers import Tokenizer, models
        from transformers import PreTrainedTokenizerFast

        from rtp_llm.models_py.new_models.deepseek_v41.language import (
            DeepSeekV41ForCausalLM,
        )

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.fixture(root)
            vocab = {f"t{i}": i for i in range(8)}
            PreTrainedTokenizerFast(
                tokenizer_object=Tokenizer(models.WordLevel(vocab, unk_token="t0"))
            ).save_pretrained(folder)

            def owner():
                model = DeepSeekV41ForCausalLM.__new__(DeepSeekV41ForCausalLM)
                torch.nn.Module.__init__(model)
                model.v4 = torch.nn.Module()
                model.v4.embed = torch.nn.Embedding(8, 32)
                model.v4.layers = torch.nn.ModuleList(
                    torch.nn.Module() for _ in range(15)
                )
                model._v4_args = SimpleNamespace(v41_config=self.cfg)
                model._owned_source_method = "fastsafetensors"
                model._owned_checkpoint_path = folder
                return model

            name = "rtp_llm.models_py.new_models.deepseek_v41.language"
            model = owner()
            with self.assertLogs(name, level="INFO") as logs:
                model._load_owned_extra_weights()
            self.assertTrue(
                any(
                    line.endswith("DeepSeek V4.1 Engram initialized on layers [1, 14]")
                    for line in logs.output
                )
            )
            self.assertEqual(model._engram_layers, ((1, 0), (14, 1)))
            model._prepare_engram(
                SimpleNamespace(engram_token_windows=torch.tensor([[2, 1]]))
            )
            for layer in (model.v4.layers[1], model.v4.layers[14]):
                self.assertIsInstance(layer.engram, OwnedEngram)
                self.assertIsNotNone(layer.engram.embed_tokens)
                self.assertEqual(layer.engram_hashes.shape, (1, 1))
                self.assertTrue(layer.engram_token_mask.item())
            (root / "layer-14.safetensors").unlink()
            with self.assertNoLogs(name, level="INFO"), self.assertRaises(
                FileNotFoundError
            ):
                owner()._load_owned_extra_weights()

    def load(self, folder, layer, method):
        return OwnedEngram.from_checkpoint(
            self.cfg, layer, folder, "cpu", load_method=method
        )

    def test_native_fills_final_shared_mapping_without_an_owned_table(self):
        specs = {
            "embed.weight": ((8, 32), "F8_E4M3", torch.float8_e4m3fn),
            "embed.scale": ((8, 1), "F8_E8M0", torch.float8_e8m0fnu),
        }
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.fixture(root)
            scratch = EngramSource(root, 1, specs, "scratch")
            native = EngramSource(root, 1, specs, "fastsafetensors")
            expected = b"".join(
                scratch.load(key).view(torch.uint8).numpy().tobytes() for key in specs
            )
            with (root / "final_shared_pages").open("w+b") as file:
                file.truncate(len(expected))
                with mmap.mmap(file.fileno(), len(expected)) as mapping:
                    with patch.object(
                        torch,
                        "empty",
                        side_effect=AssertionError("extra owned staging buffer"),
                    ):
                        native.fill_shared_table(mapping)
                    self.assertEqual(mapping[:], expected)
                    with self.assertRaisesRegex(ValueError, "requires native FST"):
                        scratch.fill_shared_table(mapping)
                with mmap.mmap(file.fileno(), len(expected) - 1) as short:
                    with self.assertRaisesRegex(ValueError, "staging size"):
                        native.fill_shared_table(short)
                with (root / "layer-1.safetensors").open("ab") as shard:
                    shard.write(b"!")
                with mmap.mmap(file.fileno(), len(expected)) as mapping:
                    with self.assertRaisesRegex(RuntimeError, "changed during loading"):
                        native.fill_shared_table(mapping)

    def test_shared_cache_publishes_only_complete_data_and_skips_warm_writer(self):
        # CPU cache protocol test: stop at attachment (before CUDA registration).
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            def local(path):
                return root if str(path) in ("/dev/shm", "/tmp") else Path(path)

            def fail(mapping):
                mapping[0] = 17
                raise OSError("incomplete native read")

            def write(mapping):
                mapping[:] = bytes(range(len(mapping)))

            def attach(writer):
                runtime_engram._PinnedSharedTable(
                    None,
                    None,
                    "tiny-cache",
                    "cuda:0",
                    table_shape=(2, 32),
                    load_into=writer,
                )

            class ReachedAttach(Exception):
                pass

            with patch.object(
                runtime_engram, "_prune_expired_engram_tables"
            ), patch.object(runtime_engram, "Path", side_effect=local), patch.object(
                torch, "frombuffer", side_effect=ReachedAttach
            ):
                with self.assertRaisesRegex(OSError, "incomplete native read"):
                    attach(fail)
                self.assertFalse(list(root.glob("*.partial.*")))
                self.assertFalse([p for p in root.iterdir() if "." not in p.name])
                with self.assertRaises(ReachedAttach):
                    attach(write)
                with self.assertRaises(ReachedAttach):
                    attach(lambda mapping: self.fail("warm cache re-read"))
                (data,) = [p for p in root.iterdir() if "." not in p.name]
                self.assertEqual(data.read_bytes(), bytes(range(66)))

    def test_native_owned_values_equal_main_after_close_and_allocator_churn(self):
        import fastsafetensors.cpp as native

        self.assertTrue(native.__file__.endswith(".so"))
        print("Engram FST:", native.__file__)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            self.fixture(path)
            for layer in (1, 14):
                expected = Engram.from_checkpoint(self.cfg, layer, folder, "cpu")
                scratch = self.load(folder, layer, "scratch")
                native_layer = self.load(folder, layer, "fastsafetensors")
                NewModelLoader._validate_loaded_weights(native_layer)
                self.assertIsNone(native_layer.wkv.v41_runtime_linear)
                self.assertEqual(len(list(native_layer.parameters())), 6)
                self.assertEqual(
                    native_layer.embed.weight.data_ptr(),
                    native_layer.embed_tokens.weight.data_ptr(),
                )
                for (key, value), (key2, reference) in zip(
                    native_layer.named_parameters(), scratch.named_parameters()
                ):
                    self.assertEqual(key, key2)
                    self.assertEqual(value.dtype, reference.dtype)
                    self.assertTrue(
                        torch.equal(
                            value.view(torch.uint8), reference.view(torch.uint8)
                        )
                    )
        gc.collect()
        churn = [torch.full((96, 32), i, dtype=torch.uint8) for i in range(32)]
        self.assertEqual(len(churn), 32)
        hidden = (torch.arange(3 * 2 * 32).reshape(3, 2, 32) % 13).bfloat16()
        ids = torch.tensor([[0], [7], [-1]])
        mask = torch.tensor([True, True, False])
        torch.testing.assert_close(
            native_layer(hidden, ids, mask), expected(hidden, ids, mask), atol=0, rtol=0
        )

    def test_nonlocal_stage_does_not_open_or_read_other_shard(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            self.fixture(path)
            (path / "layer-14.safetensors").unlink()
            self.load(folder, 1, "fastsafetensors")
            with self.assertRaises(FileNotFoundError):
                self.load(folder, 14, "fastsafetensors")

    def test_missing_scale_or_incorrect_scale_format_not_silently_used(self):
        for modify in (
            lambda t: t.pop("embed.scale"),
            lambda t: t.update({"embed.scale": t["embed.scale"].view(torch.uint8)}),
            lambda t: t.update({"wkv.scale": t["wkv.scale"][:2]}),
        ):
            with tempfile.TemporaryDirectory() as folder:
                self.fixture(Path(folder), bad=modify)
                with self.assertRaises((ValueError, KeyError)):
                    self.load(folder, 1, "fastsafetensors")

    def test_must_receive_resolved_loader_no_implicit_fallback(self):
        with self.assertRaisesRegex(ValueError, "resolved"):
            self.load("unused", 1, "auto")


if __name__ == "__main__":
    unittest.main()
