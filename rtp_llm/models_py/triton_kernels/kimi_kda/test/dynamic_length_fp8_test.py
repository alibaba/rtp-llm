"""Changing request sizes must reuse compiled FP8 kernels, including tail rows."""

import unittest
import importlib

import torch

from rtp_llm.models_py.triton_kernels.kimi_kda import attn_res_fp8, fp8_producers, fp8_quant
from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl import (
    mla_fp8_kernels, mla_prefix_fp8_producer,
)


def cache_size(kernel):
    return len(kernel.device_caches[torch.cuda.current_device()][0])


def tensors(value):
    if isinstance(value, torch.Tensor):
        return [value]
    if isinstance(value, (tuple, list)):
        return [t for v in value for t in tensors(v)]
    return [value.values, value.scale_wire] + ([] if value.bf16 is None else [value.bf16])


def make_case(kind, m):
    """Same precision/layout configuration, different runtime row counts."""
    def rand(*shape):
        return torch.randn(*shape, device="cuda", dtype=torch.bfloat16)

    if kind == "beta":
        module = importlib.import_module("rtp_llm.models_py.triton_kernels.kimi_kda.chunk")
        x = rand(m, 7)
        return module._beta_sigmoid_fwd_kernel, lambda: module._fused_beta_sigmoid(x)
    if kind == "situ":
        module = importlib.import_module("rtp_llm.models_py.triton_kernels.common.activation")
        x, y = rand(m, 129), rand(m, 129)
        return module._situ_and_mul_kernel, lambda: module.situ_and_mul(x, y, 1.7, 1.3)
    if kind == "l2norm":
        module = importlib.import_module("rtp_llm.models_py.triton_kernels.fla.l2norm")
        x = rand(m, 128)
        return module.l2norm_fwd_kernel, lambda: module.l2norm_fwd(x)
    if kind.startswith("rms"):
        x, w = rand(m, 576)[:, :512], rand(512)
        retain = kind == "rms_bf16"
        kernel = fp8_producers._rmsnorm_bf16_fp8 if retain else fp8_producers._rmsnorm_fp8
        return kernel, lambda: fp8_producers.rmsnorm_fp8(x, w, 1e-5, retain_bf16=retain)
    if kind == "gate":
        x, g = rand(m, 1536), rand(m, 1600)[:, :1536]
        return fp8_producers._sigmoid_gate_fp8, lambda: fp8_producers.sigmoid_gate_fp8(x, g)
    if kind.startswith("kda"):
        mode = kind.split("_")[1]
        # B=2 exercises changing SEQ and batch strides; gate is also strided.
        x, g, w = rand(2, m, 4, 128), rand(2, m, 4, 256)[..., :128], rand(128)
        kernel = getattr(fp8_producers, "_kda_output_" + mode + "_fp8")
        return kernel, lambda: fp8_producers.kda_output_fp8(x, g, w, 1e-5, mode=mode)
    if kind.startswith("attn"):
        n = int(kind.split("_")[1])
        x, bank = rand(m, 7168), rand(m, 9, 7168)
        w, p, ow, delta = rand(7168), rand(7168), rand(7168), rand(m, 7168)
        def run():
            # The producer updates prefix and residual bank in place.
            xx, bb = x.clone(), bank.clone()
            y = attn_res_fp8.kimi_k3_attn_res_fp8(
                xx, bb, w, p, 1e-5, output_norm_weight=ow,
                output_norm_eps=1e-5, delta=delta, num_blocks=n, block_write_idx=n,
            )
            return [*tensors(y), xx, bb]
        return attn_res_fp8._multi_block_attn_res_fp8_kernel, run
    if kind == "forget":
        x = rand(m, 640)[:, 128:256]
        return fp8_quant._quantize_forget_latent_fp8, lambda: fp8_quant.quantize_forget_latent_fp8(x)
    if kind.startswith("mla"):
        if kind == "mla_contiguous":
            x = rand(m, 4, 128)
        elif kind == "mla_transpose":
            x = rand(4, m, 128).transpose(0, 1)
        else:
            x = rand(m, 576)[:, :512]
        return mla_fp8_kernels._quantize, lambda: mla_fp8_kernels.quantize_fp8(x, 0.5)
    if kind.startswith("prefix"):
        batch = m
        page = 16
        cache = rand(batch, page, 576).to(torch.float8_e4m3fn)
        pages = torch.arange(batch, device="cuda", dtype=torch.int32)
        info = torch.zeros(batch, 4, device="cuda", dtype=torch.int32)
        info[:, 1] = 3
        info[:, 2] = torch.arange(batch, device="cuda")
        qi = torch.arange(batch + 1, device="cuda", dtype=torch.int32) * 2
        current = rand(batch * 2, 576)
        c, r = current[:, :512], current[:, 512:]
        oc, ore = rand(batch * 5, 512), rand(batch * 5, 64)
        if kind == "prefix_fp8":
            producer = mla_prefix_fp8_producer.Fp8MlaPrefixGather(0.5)
            def run():
                y = producer(oc, ore, c, r, cache, pages, info, qi, page)
                return [*tensors(y), ore]
            return mla_prefix_fp8_producer._gather_quantized, run
        def run():
            mla_fp8_kernels.gather_fp8_prefix(oc, ore, c, r, cache, pages, info, qi, page, scale=0.5)
            return oc, ore
        return mla_fp8_kernels._gather_prefix, run
    raise ValueError(kind)


KINDS = ("beta", "situ", "l2norm", "rms", "rms_bf16", "gate", "kda_prefill", "kda_decode", "attn_0", "attn_2", "attn_8",
         "forget", "mla_contiguous", "mla_transpose", "mla_slice", "prefix_fp8", "prefix_bf16")


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class DynamicLengthFp8Test(unittest.TestCase):
    def test_new_lengths_reuse_compiled_kernel(self):
        torch.manual_seed(129)
        for kind in KINDS:
            # AttnRes changes tile policy below 256 for >1 residual blocks.
            # Test within the prefill policy, including unaligned row tails.
            lengths = (272, 273, 274, 275, 400, 500, 800) if kind.startswith("attn") else (16, 17, 18, 19, 32, 65)
            if kind.startswith("prefix"):
                lengths = (2, 3, 5, 7)
            expected = None
            if kind in ("mla_transpose", "mla_slice"):
                # Triton 3.6 recursively specializes tuple alignment despite
                # do_not_specialize. Warm both bounded alignment classes, then
                # require reuse across unseen exact sizes within both classes.
                for initial in (16, 17):
                    kernel, run = make_case(kind, initial)
                    run()
                expected = cache_size(kernel)
            for m in lengths:
                with self.subTest(kind=kind, m=m):
                    kernel, run = make_case(kind, m)
                    actual = run()
                    torch.cuda.synchronize()
                    if expected is None:
                        expected = cache_size(kernel)
                    self.assertEqual(cache_size(kernel), expected, "new length compiled another kernel")
                    # Repeat calls and graph replay must preserve values and scales.
                    saved = [x.clone() for x in tensors(actual)]
                    for _ in range(10):
                        run()
                    torch.cuda.synchronize()
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        captured = run()
                    graph.replay()
                    torch.cuda.synchronize()
                    for a, b in zip(tensors(captured), saved):
                        torch.testing.assert_close(a.reshape(-1).contiguous().view(torch.uint8), b.reshape(-1).contiguous().view(torch.uint8), atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()
