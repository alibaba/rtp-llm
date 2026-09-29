"""Metadata capacity should be reserved only when a graph will replay."""

import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import torch


SOURCE = Path(__file__).resolve().parents[1] / "mla_verify.py"


class FakeMlaImplBase:
    def __init__(self, attention, inputs, weights, _cache, _fmha, **kwargs):
        self.max_seq_len = kwargs["max_seq_len"]
        self.weights = weights


class FakeNativeMlaDecode:
    def __init__(self, **kwargs):
        self.page_size = kwargs["page_size"]


class FakeParams:
    def __init__(self):
        self.calls = []

    def fill_params(self, prefix, _sequence, lengths, _table, _page_size, _forbid_realloc):
        self.calls.append((prefix.clone(), lengths.clone()))


def module_with(**kwargs):
    module = types.ModuleType("stub")
    module.__dict__.update(kwargs)
    return module


stubs = {
    "rtp_llm.models_py.modules.factory.attention": module_with(
        common=types.SimpleNamespace(create_write_cache_store_impl=lambda _inputs: None)
    ),
    "rtp_llm.models_py.modules.factory.attention.fmha_impl_base": module_with(
        MlaImplBase=FakeMlaImplBase
    ),
    "rtp_llm.models_py.modules.kimi_k3.native_mla_decode": module_with(
        NativeMlaDecode=FakeNativeMlaDecode
    ),
    "rtp_llm.ops.compute_ops": module_with(
        rtp_llm_ops=types.SimpleNamespace(FlashInferMlaAttnParams=FakeParams)
    ),
    "rtp_llm.utils.model_weight": module_with(W=types.SimpleNamespace(mla_vc="mla_vc")),
}
with patch.dict(sys.modules, stubs):
    spec = importlib.util.spec_from_file_location("k3_mla_verify_reserve_under_test", SOURCE)
    mla_verify = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mla_verify)


class MlaVerifyReserveTest(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cpu")
        mla_verify._workspaces[self.device] = torch.empty(1, dtype=torch.uint8)
        self.config = types.SimpleNamespace(
            max_seq_len=1024,
            getAttentionConfigs=lambda _tp: types.SimpleNamespace(
                head_num=1,
                kv_lora_rank=512,
                nope_head_dim=128,
                rope_head_dim=64,
                kernel_tokens_per_block=128,
                softmax_extra_scale=1.0,
                mla_fp8_compute=True,
                mla_fp8_q_scale=1.0,
                mla_fp8_kv_scale=1.0,
            ),
        )
        self.parallelism = types.SimpleNamespace(get_attn_tp_size=lambda: 1)
        self.weights = types.SimpleNamespace(weights=[{"mla_vc": torch.empty(1)}])
        self.inputs = types.SimpleNamespace(
            input_lengths=torch.tensor([1], dtype=torch.int32),
            prefix_lengths=torch.tensor([0], dtype=torch.int32),
            physical_token_count=1,
            is_target_verify=False,
            is_mtp_draft_update=False,
            kv_cache_kernel_block_id=torch.zeros((1, 16), dtype=torch.int32),
        )

    def create(self, is_cuda_graph):
        with patch.object(mla_verify.KimiK3MlaVerifyImpl, "prepare"):
            return mla_verify.KimiK3MlaVerifyImpl(
                self.config, self.parallelism, self.weights, self.inputs, None,
                is_cuda_graph,
            )

    def test_eager_does_not_allocate_unused_replay_capacity(self):
        impl = self.create(False)
        self.assertEqual(impl.fmha_params.calls, [])

    def test_cuda_graph_reserves_full_future_sequence(self):
        impl = self.create(True)
        self.assertEqual(len(impl.fmha_params.calls), 1)
        prefix, lengths = impl.fmha_params.calls[0]
        self.assertEqual(prefix.tolist(), [1023])
        self.assertEqual(lengths.tolist(), [1])


if __name__ == "__main__":
    unittest.main()
