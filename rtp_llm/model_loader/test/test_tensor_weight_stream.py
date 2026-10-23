"""CPU tests for tensor grouping and HF transforms; not engine parity tests."""

import functools
import unittest
from types import SimpleNamespace

import torch

from rtp_llm.model_loader.loader import ModelLoader
from rtp_llm.model_loader.model_weight_info import ModelWeightInfo
from rtp_llm.model_loader.weight_module import AtomicWeight
from rtp_llm.ops import TaskType
from rtp_llm.utils.database import BaseDatabase
from rtp_llm.utils.model_weight import CkptWeightInfo, W, identity, merge_qkv_hf, zeros


class IdentityLayout:
    """Keep CPU test outputs in the logical layout before device-specific packing."""

    def maybe_rewrite_weight_by_key(self, name, tensor, **kwargs):
        return tensor


def make_loader(tp=1, rank=0, generated=False):
    qkv = AtomicWeight(
        W.attn_qkv_w,
        [
            CkptWeightInfo(
                "model.layers.{i}.self_attn." + key + "_proj.weight", identity
            )
            for key in ("q", "k", "v")
        ],
        merge_qkv_hf,
    )
    q_norm = AtomicWeight(
        W.q_ln_gamma,
        [CkptWeightInfo("model.layers.{i}.self_attn.q_norm.weight", identity)],
    )
    k_norm = AtomicWeight(
        W.k_ln_gamma,
        [CkptWeightInfo("model.layers.{i}.self_attn.k_norm.weight", identity)],
    )
    final_norm = AtomicWeight(
        W.final_ln_gamma, [CkptWeightInfo("model.norm.weight", identity)]
    )
    loader = ModelLoader.__new__(ModelLoader)
    loader._load_config = SimpleNamespace(
        database=BaseDatabase(),
        num_layers=1,
        is_ft_style_weight=False,
        merge_lora=False,
        compute_dtype=torch.bfloat16,
        tp_size=tp,
        tp_rank=rank,
        ep_size=1,
        ep_rank=0,
        dp_size=1,
        dp_rank=0,
        lm_head_tp_size=tp,
        lm_head_tp_rank=rank,
        ffn_tp_size=tp,
        ffn_tp_rank=rank,
        hidden_size=4,
        head_num=2,
        head_num_kv=1,
        size_per_head=2,
        moe_pure_tp_mode=False,
        bit=16,
        exported_device=IdentityLayout(),
        use_swizzleA=False,
    )
    loader._model_weights_info = ModelWeightInfo([final_norm], [[qkv, q_norm, k_norm]])
    loader._misc_weights_info = []
    loader._global_weight_aliases = {}
    loader._task_type = TaskType.LANGUAGE_MODEL
    if generated:
        loader._model_weights_info.weights.append(
            AtomicWeight(W.final_ln_beta, [], functools.partial(zeros, shape=[4]))
        )
    return loader


def inputs(offset=0):
    values = [
        torch.arange(16).reshape(4, 4),
        torch.arange(8).reshape(2, 4) + 100,
        torch.arange(8).reshape(2, 4) + 200,
        torch.tensor([17, 18]),
        torch.tensor([21, 22]),
        torch.tensor([1, 2, 3, 4]),
    ]
    names = [
        "model.layers.0.self_attn." + key + "_proj.weight" for key in ("q", "k", "v")
    ]
    names += [
        "model.layers.0.self_attn.q_norm.weight",
        "model.layers.0.self_attn.k_norm.weight",
        "model.norm.weight",
    ]
    return [(name, value.float() + offset) for name, value in zip(names, values)]


def convert(loader, source, limit=4096):
    return {
        (layer, name): tensor
        for layer, name, tensor in loader.prepare_weights_from_tensors(
            source, "cpu", max_buffer_bytes=limit
        )
    }


class TensorWeightStreamTest(unittest.TestCase):
    def test_qkv_transpose_gqa_sharding_and_norms(self):
        for tp, rank in ((1, 0), (2, 0), (2, 1)):
            with self.subTest(tp=tp, rank=rank):
                source = inputs()
                result = convert(make_loader(tp, rank), reversed(source))
                q, k, v = [x[1] for x in source[:3]]
                q = q[rank * (4 // tp) : (rank + 1) * (4 // tp)]
                expected = torch.cat((q, k, v), dim=0).T.to(torch.bfloat16)
                torch.testing.assert_close(
                    result[(0, W.attn_qkv_w)], expected, rtol=0, atol=0
                )
                for output, entry in (
                    (W.q_ln_gamma, source[3]),
                    (W.k_ln_gamma, source[4]),
                ):
                    torch.testing.assert_close(
                        result[(0, output)], entry[1].to(torch.bfloat16), rtol=0, atol=0
                    )
                torch.testing.assert_close(
                    result[(None, W.final_ln_gamma)],
                    source[5][1].to(torch.bfloat16),
                    rtol=0,
                    atol=0,
                )

    def test_generated_weights_use_normal_load_path(self):
        loader = make_loader(generated=True)
        for offset in (0, 3):
            result = convert(loader, inputs(offset))
            torch.testing.assert_close(
                result[(None, W.final_ln_beta)],
                torch.zeros(4, dtype=torch.bfloat16),
                rtol=0,
                atol=0,
            )
            self.assertEqual(len(result), 5)

    def test_generated_weights_wait_for_source_coverage(self):
        emitted = []
        with self.assertRaisesRegex(ValueError, "missing HF tensors"):
            for layer, name, _ in make_loader(
                generated=True
            ).prepare_weights_from_tensors(inputs()[:-1], "cpu", max_buffer_bytes=4096):
                emitted.append((layer, name))
        self.assertNotIn((None, W.final_ln_beta), emitted)

    def test_input_buffer_can_be_reused(self):
        source = inputs()
        expected = convert(make_loader(), source)

        def reused():
            for name, value in source:
                yield name, value
                value.fill_(-100)

        result = convert(make_loader(), reused())
        for key in expected:
            torch.testing.assert_close(result[key], expected[key], rtol=0, atol=0)

    def test_repeated_updates_have_fresh_collectors(self):
        loader = make_loader(2, 1)
        first = convert(loader, inputs())
        second = convert(loader, inputs(3))
        restored = convert(loader, inputs())
        self.assertFalse(
            torch.equal(first[(0, W.attn_qkv_w)], second[(0, W.attn_qkv_w)])
        )
        for key in first:
            torch.testing.assert_close(first[key], restored[key], rtol=0, atol=0)

    def test_rejects_duplicate_even_after_group_is_emitted(self):
        source = inputs()
        with self.assertRaisesRegex(ValueError, "duplicate HF tensor"):
            convert(make_loader(), source + [source[0]])

    def test_rejects_unknown_tensor(self):
        with self.assertRaisesRegex(KeyError, "unexpected HF tensor"):
            convert(make_loader(), inputs() + [("unknown", torch.ones(1))])

    def test_rejects_incomplete_update_and_next_call_starts_fresh(self):
        loader = make_loader()
        with self.assertRaisesRegex(ValueError, "missing HF tensors"):
            convert(loader, inputs()[:-1])
        self.assertEqual(len(convert(loader, inputs())), 4)

    def test_rejects_non_tensor(self):
        with self.assertRaisesRegex(TypeError, "not a torch.Tensor"):
            convert(make_loader(), [(inputs()[0][0], [1, 2])])

    def test_buffer_budget_is_enforced_and_released_between_groups(self):
        source = inputs()
        group_bytes = sum(
            value.numel() * value.element_size() for _, value in source[:3]
        )
        with self.assertRaisesRegex(ValueError, "buffer exceeds"):
            convert(make_loader(), source, group_bytes - 1)
        self.assertEqual(len(convert(make_loader(), source, group_bytes)), 4)
        with self.assertRaisesRegex(ValueError, "must be positive"):
            convert(make_loader(), source, 0)

    def test_rejects_ambiguous_shared_input_before_conversion(self):
        loader = make_loader()
        loader._model_weights_info.weights.append(
            AtomicWeight(W.lm_head, [CkptWeightInfo("model.norm.weight", identity)])
        )
        with self.assertRaisesRegex(NotImplementedError, "shared HF inputs"):
            convert(loader, inputs())

    def test_rejects_non_hf_or_merged_lora(self):
        for name in ("is_ft_style_weight", "merge_lora"):
            loader = make_loader()
            setattr(loader._load_config, name, True)
            with self.assertRaises(NotImplementedError):
                convert(loader, inputs())


if __name__ == "__main__":
    unittest.main()
