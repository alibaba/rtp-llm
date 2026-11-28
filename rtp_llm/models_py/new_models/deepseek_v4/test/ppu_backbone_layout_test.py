"""Real file loading of PPU backbone globals; no vendor kernel is faked."""

import json
import os
import tempfile
import unittest
from itertools import product
from types import SimpleNamespace
from unittest.mock import patch

import torch
from safetensors.torch import save_file

from rtp_llm.models_py.checkpoint_reader_test_support import checkpoint_readers
from rtp_llm.models_py.fst_range_reader import iter_fst_range_tensors
from rtp_llm.models_py.ft_checkpoint import CheckpointTopology, FTCheckpoint
from rtp_llm.models_py.layers.embedding import ParallelLMHead, VocabParallelEmbedding
from rtp_llm.models_py.model_loader import NewLoaderConfig, NewModelLoader
from rtp_llm.models_py.module_base import RtpModule
from rtp_llm.models_py.modules.dsv4.ppu.ppu_module_provider import PpuModuleProvider
from rtp_llm.models_py.new_models.deepseek_v4.backbone import OwnedV4Args
from rtp_llm.models_py.new_models.deepseek_v4.language import OwnedV4ModelMixin, _BackboneMapper
from rtp_llm.models_py.new_models.deepseek_v4.ppu_backbone import PpuOwnedV4Transformer
from rtp_llm.models_py.new_models.deepseek_v4.ppu_globals import PpuHiddenShardEmbedding
from rtp_llm.models_py.new_weight_manager import NewLoaderWeightManager
from rtp_llm.models_py.weight_mapper import WeightsMapper, get_all_weights


class _GlobalsOnly(PpuOwnedV4Transformer):
    def create_block(self, args, layer_id, *, params_dtype):
        return torch.nn.Identity()


class _VocabTarget(OwnedV4ModelMixin, RtpModule):
    def __init__(self, rank, *, ppu=True):
        RtpModule.__init__(self)
        self.config = SimpleNamespace(model_type="deepseek_v4", tie_word_embeddings=True)
        self._v4_args = SimpleNamespace(vocab_size=8, dim=4)
        self.v4 = RtpModule()
        self.v4.layers = torch.nn.ModuleList()
        self.v4.embed = (
            PpuHiddenShardEmbedding(8, 4, 2, rank)
            if ppu else VocabParallelEmbedding(8, 4, params_dtype=torch.bfloat16)
        )
        self.v4.head = ParallelLMHead(8, 4, 2, rank, torch.float32)
        self.v4.norm = RtpModule()
        self.v4.norm.weight = torch.nn.Parameter(torch.ones(4), requires_grad=False)
        self.v4.norm.load_weights({"weight": torch.ones(4)})
        self.v4.WEIGHTS_MAPPER = WeightsMapper()
        self.WEIGHTS_MAPPER = _BackboneMapper(self.v4)

    def checkpoint_weight_name_filter(self):
        return {"embed.weight", "head.weight"}.__contains__


class PpuBackboneLayoutTest(unittest.TestCase):
    def test_tied_target_uses_global_embedding_rows_for_the_output_head(self):
        with tempfile.TemporaryDirectory() as root:
            matrix = torch.arange(32).reshape(8, 4).bfloat16()
            embed, head = [os.path.join(root, n + ".safetensors") for n in ("embed", "head")]
            save_file({"embed.weight": matrix}, embed)
            save_file({"head.weight": matrix + 50}, head)
            for rank, paths in product(range(2), ([embed], [embed, head], [head, embed])):
                with self.subTest(rank=rank, paths=paths):
                    model = _VocabTarget(rank)
                    loader = NewModelLoader(model.config, NewLoaderConfig(device="cpu"), model_path=root)
                    resolve = loader._checkpoint_read_spec_resolver(model, None, None)
                    self.assertFalse(resolve("embed.weight", (8, 4)).is_local_shard)
                    for path in paths:
                        model.load_weights(get_all_weights([path], read_spec_resolver=resolve))
                    NewModelLoader._validate_loaded_weights(model)
                    torch.testing.assert_close(model.v4.embed.weight, matrix.chunk(2, dim=1)[rank])
                    expected = matrix if len(paths) == 1 else matrix + 50
                    torch.testing.assert_close(model.v4.head.weight, expected.chunk(2)[rank].float())

    def test_online_vocab_updates_respect_both_layouts_and_head_ownership(self):
        matrix = torch.arange(32).reshape(8, 4).bfloat16()
        for ppu, rank, explicit in product((False, True), range(2), (False, True)):
            with self.subTest(ppu=ppu, rank=rank, explicit=explicit):
                model = _VocabTarget(rank, ppu=ppu)
                model.load_weights({"embed.weight": matrix})
                if explicit:
                    model.load_weights({"head.weight": (matrix + 17).float()})
                manager = NewLoaderWeightManager(model, NewModelLoader(
                    model.config, NewLoaderConfig(device="cpu"), model_path="."
                ))
                embed, head = model.v4.embed.weight, model.v4.head.weight
                ptrs = (embed.data_ptr(), head.data_ptr())
                for i, (embedding_name, head_name) in enumerate((
                    ("embed.weight", "head.weight"), ("embedding", "lm_head"),
                    ("v4.embed.weight", "v4.head.weight"),
                )):
                    value = matrix + i + 30
                    before = head.clone()
                    manager.update_tensors({embedding_name: value})
                    local = value.chunk(2, dim=1)[rank] if ppu else value
                    torch.testing.assert_close(embed, local)
                    expected = before if explicit else value.chunk(2)[rank].float()
                    torch.testing.assert_close(head, expected)
                    new_head = (value + 100).float()
                    if explicit:
                        manager.update_tensors({head_name: new_head})
                        torch.testing.assert_close(head, new_head.chunk(2)[rank])
                    else:
                        version = manager.version
                        with self.assertRaisesRegex(ValueError, "non-owning"):
                            manager.update_tensors({embedding_name: value + 1, head_name: new_head})
                        self.assertEqual(manager.version, version)
                        torch.testing.assert_close(head, expected)
                    torch.testing.assert_close(embed, local)
                    self.assertEqual((embed.data_ptr(), head.data_ptr()), ptrs)

    def test_ft_serialized_head_keeps_its_recorded_update_owner(self):
        matrix = torch.arange(32).reshape(8, 4).bfloat16()
        values = {
            "rank_00_00_00.global.embedding": matrix,
            "rank_00_00_00.global.lm_head": (matrix + 10).float(),
        }
        with tempfile.TemporaryDirectory() as root:
            shard = "ft.safetensors"
            save_file(values, os.path.join(root, shard))
            for source in ("embedding", "lm_head", None):
                with self.subTest(source=source):
                    index = {"is_ft_style_weight": True, "weight_map": dict.fromkeys(values, shard)}
                    if source is not None:
                        index["ft_lm_head_source"] = source
                    with open(os.path.join(root, "model.safetensors.index.json"), "w") as f:
                        json.dump(index, f)
                    model = _VocabTarget(1)
                    # The archive has one TP shard. The runtime head selects TP rank 1.
                    checkpoint = FTCheckpoint.open(root, CheckpointTopology())
                    model.load_ft_checkpoint(checkpoint)
                    manager = NewLoaderWeightManager(model, NewModelLoader(
                        model.config, NewLoaderConfig(device="cpu"), model_path=root
                    ))
                    head = model.v4.head.weight
                    original = (matrix + 10).float().chunk(2)[1]
                    torch.testing.assert_close(head, original)
                    replacement = matrix + 99
                    if source is None:
                        for name in ("embed.weight", "lm_head"):
                            with self.assertRaisesRegex(ValueError, "provenance"):
                                manager.update_tensors({name: replacement})
                        self.assertEqual(manager.version, 0)
                        torch.testing.assert_close(model.v4.embed.weight, matrix.chunk(2, 1)[1])
                    else:
                        manager.update_tensors({"embedding": replacement})
                    torch.testing.assert_close(
                        head, replacement.chunk(2)[1].float() if source == "embedding" else original
                    )

    def test_platform_selection_does_not_replace_pro_or_dspark(self):
        from rtp_llm.models_py.new_models.deepseek_v4.language import (
            DeepSeekV4ForCausalLM,
        )
        from rtp_llm.models_py.new_models.deepseek_v4.ppu_language import (
            PpuDeepSeekV4ForCausalLM,
        )

        # Predicate isolation only. PPU execution is a separate hardware target.
        for ppu in (False, True):
            with patch("rtp_llm.device.device_type.is_ppu", return_value=ppu):
                for name, dim, count in (
                    ("deepseek_v4", 4096, 43),
                    ("deepseek_v4", 7168, 61),
                    ("deepseek_v4_dspark", 4096, 3),
                ):
                    config = SimpleNamespace(
                        model_type=name, hidden_size=dim, num_layers=count
                    )
                    actual = DeepSeekV4ForCausalLM.resolve_runtime_class(config)
                    expected = (
                        PpuDeepSeekV4ForCausalLM
                        if ppu and name == "deepseek_v4" and dim == 4096
                        else DeepSeekV4ForCausalLM
                    )
                    self.assertIs(actual, expected)

    def test_native_head_mixing_and_global_checkpoint_layout(self):
        vocab_size, dim, mult = 18, 8, 2
        data = {
            "embed.weight": torch.arange(vocab_size * dim)
            .reshape(vocab_size, dim)
            .bfloat16(),
            "head.weight": torch.arange(vocab_size * dim)
            .reshape(vocab_size, dim)
            .flip((0,))
            .bfloat16(),
            "norm.weight": torch.arange(dim).bfloat16(),
            "hc_head_fn": torch.arange(mult * mult * dim)
            .reshape(mult, mult * dim)
            .float()
            / 37,
            "hc_head_base": torch.tensor([0.125, -0.125]),
            "hc_head_scale": torch.tensor([0.25]),
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "globals.safetensors")
            save_file(data, path)
            for rank, reader in product(
                range(2), checkpoint_readers(get_all_weights, iter_fst_range_tensors)
            ):
                with self.subTest(rank=rank, reader=reader.__name__):
                    args = OwnedV4Args(
                        dim=dim,
                        vocab_size=vocab_size,
                        hc_mult=mult,
                        n_layers=1,
                        n_mtp_layers=0,
                        compress_ratios=[0],
                        tp_size=2,
                        tp_rank=rank,
                        lm_head_tp_size=2,
                        lm_head_tp_rank=rank,
                    )
                    owner = _GlobalsOnly(args, platform_provider=PpuModuleProvider({}))
                    loader = NewModelLoader(
                        SimpleNamespace(model_type="deepseek_v4"),
                        NewLoaderConfig(device="cpu", tp_size=2, tp_rank=rank),
                        model_path=tmp,
                    )
                    resolver = loader._checkpoint_read_spec_resolver(
                        owner, owner.checkpoint_weight_name_filter(), None
                    )
                    owner.load_standalone_checkpoint(
                        reader([path], read_spec_resolver=resolver)
                    )
                    NewModelLoader._validate_loaded_weights(owner)
                    torch.testing.assert_close(
                        owner.embed.weight, data["embed.weight"].chunk(2, 1)[rank]
                    )
                    expected = torch.nn.functional.pad(
                        data["head.weight"], (0, 0, 0, 14)
                    )
                    torch.testing.assert_close(
                        owner.head_weight, expected.chunk(2)[rank]
                    )
                    torch.testing.assert_close(owner.norm.weight, data["norm.weight"])
                    x = (
                        torch.arange(4 * mult * dim).reshape(4, mult, dim).bfloat16()
                        / 8
                    )
                    flat = x.flatten(1).float()
                    mixes = (flat @ data["hc_head_fn"].T) * torch.rsqrt(
                        flat.square().mean(1, keepdim=True) + args.norm_eps
                    )
                    coeff = torch.sigmoid(
                        mixes * data["hc_head_scale"] + data["hc_head_base"]
                    )
                    expected = (
                        ((coeff + args.hc_eps)[:, :, None] * x.float())
                        .sum(1)
                        .bfloat16()
                    )
                    torch.testing.assert_close(
                        owner._hc_head_reduce(x), expected, atol=0, rtol=0
                    )


if __name__ == "__main__":
    unittest.main()
