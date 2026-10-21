"""Fast-loader weights must outlive borrowed checkpoint shard buffers."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import torch

from rtp_llm.model_loader.loader import ModelLoader
from rtp_llm.model_loader.weight_module import AtomicWeight
from rtp_llm.ops import TaskType
from rtp_llm.utils.model_weight import CkptWeightInfo, W


class RecyclingShardDatabase:
    """Each yielded tensor is valid only until iteration resumes.

    Reusing the same allocation deterministically models the no-GDS loader
    releasing each shard buffer without dereferencing actual freed memory.
    """

    def __init__(self, tensors):
        self.tensors = tensors
        self.released = []
        self.load_tensor = Mock(
            side_effect=AssertionError("complete fast weights must not fall back")
        )

    def fastsafetensors_weights_iterator(self, device, use_tqdm_on_load, **kwargs):
        buffer = torch.empty_like(self.tensors[0][1], device=device)
        for name, value in self.tensors:
            buffer.copy_(value)
            yield name, buffer
            buffer.zero_()
            self.released.append(name)


class FastSafetensorOwnershipTest(unittest.TestCase):
    def make_loader(self, database, globals=(), layers=(), dtype=torch.bfloat16):
        loader = object.__new__(ModelLoader)
        loader.model_config = SimpleNamespace(has_lm_head=False)
        loader._task_type = TaskType.LANGUAGE_MODEL
        loader._global_weight_aliases = {}
        loader._misc_weights_info = []
        loader._weights_info = SimpleNamespace(_quant_config=None)
        loader._model_weights_info = SimpleNamespace(
            weights=list(globals), layer_weights=[list(layers)] if layers else []
        )
        loader._load_config = SimpleNamespace(
            database=database,
            num_layers=1 if layers else 0,
            compute_dtype=dtype,
            tp_size=1,
            dp_size=1,
            ep_size=1,
            merge_lora=False,
            use_swizzleA=False,
            exported_device=SimpleNamespace(
                maybe_rewrite_weight_by_key=lambda name, tensor, **kwargs: tensor
            ),
        )
        return loader

    def test_identity_embedding_survives_shard_release(self):
        expected = torch.arange(24, dtype=torch.bfloat16).reshape(6, 4)
        database = RecyclingShardDatabase([("embed_tokens.weight", expected)])
        weight = AtomicWeight(W.embedding, [CkptWeightInfo("embed_tokens.weight")])
        loaded = self.make_loader(database, globals=[weight])._load_from_fastsafetensor(
            "cpu"
        )
        self.assertEqual(database.released, ["embed_tokens.weight"])
        torch.testing.assert_close(loaded.get_global_weight(W.embedding), expected)
        database.load_tensor.assert_not_called()

    def test_qkv_combination_preserves_inputs_from_previous_shards(self):
        parts = [
            torch.full((2, 4), value, dtype=torch.bfloat16) for value in (1, 2, 3)
        ]
        names = [f"layers.0.{projection}.weight" for projection in ("q", "k", "v")]
        database = RecyclingShardDatabase(list(zip(names, parts)))
        weight = AtomicWeight(
            W.attn_qkv_w,
            [CkptWeightInfo(name) for name in names],
            process_fun=lambda tensors: torch.cat(tensors, dim=0),
        )
        loaded = self.make_loader(database, layers=[weight])._load_from_fastsafetensor(
            "cpu"
        )
        self.assertEqual(database.released, names)
        torch.testing.assert_close(loaded.weights[0][W.attn_qkv_w], torch.cat(parts))
        database.load_tensor.assert_not_called()

    def test_inline_fp8_mode_preserves_non_quantized_weights(self):
        # Eligible FP8 weights that are not BF16 matrices take a different
        # collector branch from ordinary non-quantized global weights.
        for dtype, inline_candidate in (
            (torch.bfloat16, False),
            (torch.bfloat16, True),
            (torch.float8_e4m3fn, True),
        ):
            with self.subTest(dtype=dtype, inline_candidate=inline_candidate):
                expected = torch.arange(1, 5, dtype=torch.float32).to(dtype)
                if dtype == torch.float8_e4m3fn:
                    expected = expected.reshape(2, 2)
                database = RecyclingShardDatabase([("saved.weight", expected)])
                weight = AtomicWeight(
                    "saved.weight", [CkptWeightInfo("saved.weight")], data_type=dtype
                )
                loader = self.make_loader(database, globals=[weight], dtype=dtype)
                loader._is_online_ptpc = lambda: True
                loader._should_inline_fp8_quantize = lambda _: inline_candidate
                loaded = loader._load_from_fastsafetensor("cpu")
                self.assertEqual(database.released, ["saved.weight"])
                torch.testing.assert_close(
                    loaded.get_global_weight("saved.weight").float(), expected.float()
                )
                database.load_tensor.assert_not_called()


if __name__ == "__main__":
    unittest.main()
