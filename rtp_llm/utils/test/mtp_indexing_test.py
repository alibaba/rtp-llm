import unittest
from types import SimpleNamespace

import torch
from torch import nn

from rtp_llm.models_py.model_desc.generic_moe_mtp import GenericMoeMTPModel


def _fake_model(share: bool = True, layer_num: int = 1):
    model = object.__new__(GenericMoeMTPModel)
    nn.Module.__init__(model)
    model.config = None
    model.parallelism_config = None
    model.weight = None
    model.fmha_config = None
    model.py_hw_kernel_config = None
    model.micro_batch_size = 0
    model.layer_num = layer_num
    model.vocab_size = 0
    model.kv_cache = None
    model.device_type = None
    model.params_dict = {}
    model.moe_config = None
    model.max_generate_batch_size = 0
    model.device_resource_config = None
    model.embed_tokens = None
    model.pre_fc_norm_embedding = None
    model.pre_fc_norm_hidden = None
    model.fc = None
    model.layers = nn.ModuleList()
    model.norm = None
    model._share_mtp_topk_indices = share
    model._mtp_iteration_topk_buffers = [None for _ in range(layer_num)]
    model._mtp_iteration_topk_valid_tokens = [0 for _ in range(layer_num)]
    model._mtp_iteration_topk_indices = [None for _ in range(layer_num)]
    model._mtp_debug_enabled_flag = False
    return model


def _inputs(is_prefill: bool, mtp_iteration_step: int = -1):
    return SimpleNamespace(
        attention_inputs=SimpleNamespace(
            is_prefill=is_prefill, mtp_iteration_step=mtp_iteration_step
        )
    )


class MtpIndexingTest(unittest.TestCase):
    def test_step0_resets_cached_topk(self):
        model = _fake_model(layer_num=2)
        model._mtp_iteration_topk_indices = ["cached0", "cached1"]
        model._mtp_iteration_topk_valid_tokens = [1, 1]

        model._reset_mtp_iteration_topk_if_needed(
            _inputs(is_prefill=False, mtp_iteration_step=0)
        )

        self.assertEqual(model._mtp_iteration_topk_indices, [None, None])
        self.assertEqual(model._mtp_iteration_topk_valid_tokens, [0, 0])

    def test_unknown_prefill_resets_cached_topk(self):
        model = _fake_model(layer_num=1)
        model._mtp_iteration_topk_indices = ["cached"]
        model._mtp_iteration_topk_valid_tokens = [1]

        model._reset_mtp_iteration_topk_if_needed(_inputs(is_prefill=True))

        self.assertEqual(model._mtp_iteration_topk_indices, [None])
        self.assertEqual(model._mtp_iteration_topk_valid_tokens, [0])

    def test_later_step_reuses_cached_topk(self):
        model = _fake_model(layer_num=1)
        model._mtp_iteration_topk_indices = ["cached"]

        inputs = _inputs(is_prefill=False, mtp_iteration_step=1)
        model._reset_mtp_iteration_topk_if_needed(inputs)

        self.assertTrue(model._should_reuse_mtp_iteration_topk(inputs))
        self.assertEqual(model._get_mtp_iteration_topk(0), "cached")

    def test_unknown_decode_does_not_force_reuse(self):
        model = _fake_model(layer_num=1)

        self.assertFalse(
            model._should_reuse_mtp_iteration_topk(_inputs(is_prefill=False))
        )

    def test_disabled_share_ignores_cache(self):
        model = _fake_model(share=False, layer_num=1)
        model._set_mtp_iteration_topk(0, "cached")

        self.assertIsNone(model._get_mtp_iteration_topk(0))

    def test_set_topk_extends_layer_cache(self):
        model = _fake_model(layer_num=1)

        model._set_mtp_iteration_topk(2, "cached2")

        self.assertEqual(model._get_mtp_iteration_topk(2), "cached2")

    def test_clear_mutates_shared_cache_in_place(self):
        model = _fake_model(layer_num=2)
        shared_cache = model._mtp_iteration_topk_indices
        shared_cache[:] = ["cached0", "cached1"]
        valid_tokens = model._mtp_iteration_topk_valid_tokens
        valid_tokens[:] = [1, 1]

        model._clear_mtp_iteration_topk()

        self.assertIs(model._mtp_iteration_topk_indices, shared_cache)
        self.assertIs(model._mtp_iteration_topk_valid_tokens, valid_tokens)
        self.assertEqual(shared_cache, [None, None])
        self.assertEqual(valid_tokens, [0, 0])

    def test_cuda_graph_clone_shares_topk_cache_object(self):
        model = _fake_model(layer_num=1)

        clone = model.clone_for_cuda_graph()

        self.assertIs(
            clone._mtp_iteration_topk_indices, model._mtp_iteration_topk_indices
        )
        self.assertIs(
            clone._mtp_iteration_topk_buffers, model._mtp_iteration_topk_buffers
        )
        self.assertIs(
            clone._mtp_iteration_topk_valid_tokens,
            model._mtp_iteration_topk_valid_tokens,
        )

    def test_cached_topk_shape_must_match_current_tokens(self):
        model = _fake_model(layer_num=1)
        model._set_mtp_iteration_topk(0, torch.zeros((2, 4), dtype=torch.int32))

        self.assertIsNone(model._get_mtp_iteration_topk(0, expected_tokens=3))
        self.assertIsNotNone(model._get_mtp_iteration_topk(0, expected_tokens=2))

    def test_select_topk_cache_keeps_only_lm_output_rows(self):
        model = _fake_model(layer_num=1)
        topk = torch.arange(12, dtype=torch.int32).view(3, 4)
        model._set_mtp_iteration_topk(0, topk)

        model.select_mtp_iteration_topk_cache(torch.tensor([2], dtype=torch.int32), 3)

        cached = model._get_mtp_iteration_topk(0, expected_tokens=1)
        self.assertTrue(torch.equal(cached, topk[2:3]))

    def test_copy_topk_cache_from_other_model(self):
        source = _fake_model(layer_num=1)
        target = _fake_model(layer_num=1)
        topk = torch.arange(8, dtype=torch.int32).view(2, 4)
        source._set_mtp_iteration_topk(0, topk)
        source.select_mtp_iteration_topk_cache(torch.tensor([1], dtype=torch.int32), 2)

        target.copy_mtp_iteration_topk_cache_from(source)

        cached = target._get_mtp_iteration_topk(0, expected_tokens=1)
        self.assertTrue(torch.equal(cached, topk[1:2]))


if __name__ == "__main__":
    unittest.main()
