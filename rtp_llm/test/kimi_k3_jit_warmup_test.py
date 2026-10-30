"""The directed MLA compilation must leave live request cache untouched."""

import importlib.util
import sys
import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import torch

def load_warmup():
    from rtp_llm.models_py.modules.kimi_k3 import mla_jit_warmup

    spec = importlib.util.spec_from_file_location(
        "k3_mla_warmup_under_test", mla_jit_warmup.__file__
    )
    module = importlib.util.module_from_spec(spec)
    missing = object()
    previous = sys.modules.get(spec.name, missing)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        if previous is missing:
            sys.modules.pop(spec.name, None)
        else:
            sys.modules[spec.name] = previous
    return module


def model_fixture(dtype=torch.bfloat16):
    attention = type("KimiK3MLA", (), {})()
    attention.heads, attention.kv_rank = 3, 16
    attention.suffix_dim, attention.q_dim, attention.v_dim = 8, 192, 128
    storage = torch.ones((7, 64, 48), dtype=dtype)
    live = storage[:, :, :24]
    cache = SimpleNamespace(kv_cache_base=live, seq_size_per_block=64, tag="mla")
    model = SimpleNamespace(
        layers=[SimpleNamespace(index=i, attention=attention) for i in range(2)],
        kv_cache=SimpleNamespace(
            get_layer_cache_groups=lambda _index: [cache],
            get_seq_size_per_block=lambda _tag: 256,
        ),
        parallelism_config=SimpleNamespace(
            tp_size=2,
            tp_rank=1,
            prefill_cp_config=SimpleNamespace(kv_cache_sharded=True),
        ),
        config=SimpleNamespace(attn_config=SimpleNamespace(softmax_extra_scale=1.25)),
    )
    return model, live


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class MlaJitWarmupCudaTest(unittest.TestCase):
    def test_scratch_prefix_preparation_preserves_live_cache(self):
        warmup = load_warmup()
        for dtype in (torch.bfloat16, torch.float8_e4m3fn):
            with self.subTest(dtype=dtype):
                model, live = model_fixture(dtype)
                live_cuda = torch.empty_strided(
                    live.shape, live.stride(), device="cuda", dtype=dtype
                )
                live_cuda.copy_(live)
                model.kv_cache = SimpleNamespace(
                    get_layer_cache_groups=lambda _index: [
                        SimpleNamespace(
                            kv_cache_base=live_cuda,
                            seq_size_per_block=64,
                            tag="mla",
                        )
                    ],
                    get_seq_size_per_block=lambda _tag: 256,
                )
                before = live_cuda.to(torch.float32).clone()
                spec = next(warmup._mla_warmup_specs(model))
                # Exercise private movement/merge independently of Prefill.
                with patch.object(warmup, "_warmup_flashinfer"):
                    warmup._warmup_spec(spec)
                torch.cuda.synchronize()
                torch.testing.assert_close(
                    live_cuda.to(torch.float32), before, rtol=0, atol=0
                )

    def test_page_rr_metadata_changes_preserve_prefix_output(self):
        from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.mla_page_rr_cache_gather import (
            MlaPageRRCacheGather,
            _restore_mla_page_rr_prefix,
        )

        def launch(requests, blocks, local_pages):
            page, features = 64, 6
            cache = (
                torch.arange(blocks * page * features, device="cuda")
                .remainder(113)
                .reshape(blocks, page, features)
                .to(torch.bfloat16)
            )
            table = (
                torch.arange(requests * local_pages, device="cuda", dtype=torch.int32)
                .reshape(requests, local_pages)
                .remainder(blocks - 1)
                + 1
            )
            adapter = MlaPageRRCacheGather(page, 1, 0)
            payload = adapter._pack_prefix(
                cache, table, (page * local_pages,) * requests
            )
            output = _restore_mla_page_rr_prefix(
                payload.unsqueeze(0), adapter._prefix_descriptor
            )
            expected = torch.cat(
                [torch.cat([cache[int(block)] for block in row]) for row in table.cpu()]
            )
            torch.testing.assert_close(output, expected, rtol=0, atol=0)

        launch(1, 2, 1)
        launch(3, 7, 3)

    def test_flashinfer_partial_modes_preserve_zero_attention_and_lse(self):
        if torch.cuda.get_device_capability()[0] != 10:
            self.skipTest("FlashInfer TRTLLM Prefill requires Blackwell")
        from rtp_llm.models_py.modules.kimi_k3 import flashinfer_prefill

        warmup = load_warmup()
        model, _ = model_fixture(torch.float8_e4m3fn)
        spec = replace(
            next(warmup._mla_warmup_specs(model)),
            device=torch.device("cuda:0"),
            heads=8,
        )
        warmup._warmup_flashinfer(spec)
        wrapper = flashinfer_prefill.FlashinferMlaPrefill(
            spec.heads, spec.softmax_scale
        )
        workspace_pointer = None
        for causal, kv_tokens in ((True, 7), (False, 129)):
            q = torch.zeros((7, 8, 192), device="cuda", dtype=torch.float8_e4m3fn)
            k = torch.zeros((kv_tokens, 8, 192), device="cuda", dtype=q.dtype)
            v = torch.zeros((kv_tokens, 8, 128), device="cuda", dtype=q.dtype)
            output, lse = wrapper.run_partial(
                q,
                k,
                v,
                qo_indptr=torch.tensor((0, 7), device="cuda", dtype=torch.int32),
                kv_indptr=torch.tensor(
                    (0, kv_tokens), device="cuda", dtype=torch.int32
                ),
                max_q=7,
                max_k=kv_tokens,
                causal=causal,
            )
            torch.testing.assert_close(output, torch.zeros_like(output), rtol=0, atol=0)
            counts = (
                torch.arange(1, 8, device="cuda", dtype=torch.float32)
                if causal
                else torch.full((7,), kv_tokens, device="cuda")
            )
            expected_lse = counts.log()[:, None].expand(7, 8)
            torch.testing.assert_close(lse, expected_lse, rtol=1e-3, atol=1e-3)
            workspace = flashinfer_prefill._get_workspace(spec.device, 7, spec.heads)
            if workspace_pointer is not None:
                self.assertEqual(workspace.data_ptr(), workspace_pointer)
            workspace_pointer = workspace.data_ptr()


if __name__ == "__main__":
    unittest.main()
