"""Exercise the RTP loader, scale packing and actual FP8 Linear together."""

import json
from types import SimpleNamespace
from unittest.mock import patch

import torch

from rtp_llm.config.quant_config import Fp8BlockWiseQuantConfig
from rtp_llm.model_loader.attn_weight import MlaAttnAtomicWeight, MlaConfig
from rtp_llm.models.kimi_k3.fp8_weight import KimiK3LoadFp8Weight
from rtp_llm.models_py.modules.factory import LinearFactory
from rtp_llm.utils.model_weight import W


def main():
    torch.manual_seed(20260905)
    config = Fp8BlockWiseQuantConfig()
    for name, n, k in ((W.mla_fusedqkrope_w, 2112, 7168), (W.mla_kv_b_w, 3072, 512)):
        source = MlaAttnAtomicWeight(
            name,
            [],
            config=MlaConfig(
                head_num=12, nope_head_dim=128, v_head_dim=128, kv_lora_rank=512
            ),
        )
        loader = KimiK3LoadFp8Weight(source, config, derive_mla=name == W.mla_kv_b_w)
        source_w = torch.randn(k, n, dtype=torch.bfloat16, device="cuda")
        load = SimpleNamespace(tp_size=1, tp_rank=0, merge_lora=False)
        with patch.object(source, "_load_raw_tensor", return_value={name: source_w}):
            raw = loader._load_raw_tensor(None, 0, "cuda", load)
        from rtp_llm.model_loader.per_block_fp8_quant_weight import (
            per_block_cast_to_fp8,
        )

        reference_weight, reference_scale = per_block_cast_to_fp8(
            source_w.T.contiguous(), 128, use_ue8m0=loader.use_ue8m0
        )
        assert torch.equal(
            raw[name].view(torch.uint8), reference_weight.view(torch.uint8)
        )
        torch.testing.assert_close(
            raw[loader.scale.name], reference_scale, rtol=0, atol=0
        )
        assert raw[name].untyped_storage().nbytes() == n * k
        del reference_weight, reference_scale
        loaded = loader._postprocess(loader._split(raw, load), "cuda", load)
        linear = LinearFactory.create_linear_from_weights(
            loaded, name, loader.scale.name, None, quant_config=config
        )
        if linear.__class__.__name__ != "CudaFp8DeepGEMMLinear":
            raise AssertionError(type(linear))
        for m in (1, 16, 256):
            x = torch.randn(m, k, dtype=torch.bfloat16, device="cuda")
            # Independent activation quantization and explicit dequant reference.
            xg = x.float().reshape(m, k // 128, 128)
            xs = xg.abs().amax(-1).clamp_min(1e-4) / 448.0
            xs = torch.pow(2.0, torch.ceil(torch.log2(xs)))
            xq = (xg / xs[..., None]).to(torch.float8_e4m3fn).float()
            xd = (xq * xs[..., None]).reshape(m, k)
            wd = (
                raw[name].float()
                * raw[loader.scale.name]
                .repeat_interleave(128, 0)
                .repeat_interleave(128, 1)[:n, :k]
            )
            expected = xd @ wd.T
            out = linear(x)
            torch.cuda.synchronize()
            error = (out.float() - expected).norm() / expected.norm()
            assert torch.isfinite(out).all() and float(error) < 0.01, float(error)
            if name == W.mla_kv_b_w:
                splits = (128, 64, 128)
                packed = x.new_full((m, n // 256 * 320), 17.0)
                assert linear.supports_skip_head_mid(x, splits)
                linear.forward_skip_head_mid(x, splits, output=packed)
                pv, ov = packed.view(m, -1, 320), out.view(m, -1, 256)
                torch.testing.assert_close(pv[..., :128], ov[..., :128], rtol=0, atol=0)
                torch.testing.assert_close(pv[..., 192:], ov[..., 128:], rtol=0, atol=0)
                assert torch.all(pv[..., 128:192] == 17.0)
                packed_graph = torch.cuda.CUDAGraph()
                packed_stream = torch.cuda.Stream()
                packed_stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(packed_stream):
                    linear.forward_skip_head_mid(x, splits, output=packed)
                packed_stream.synchronize()
                with torch.cuda.graph(packed_graph, stream=packed_stream):
                    linear.forward_skip_head_mid(x, splits, output=packed)
                packed_graph.replay()
                torch.cuda.synchronize()
                torch.testing.assert_close(pv[..., :128], ov[..., :128], rtol=0, atol=0)
                torch.testing.assert_close(pv[..., 192:], ov[..., 128:], rtol=0, atol=0)
                assert torch.all(pv[..., 128:192] == 17.0)
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                linear(x)
            stream.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                graph_out = linear(x)
            graph.replay()
            torch.cuda.synchronize()
            torch.testing.assert_close(graph_out, out, rtol=0, atol=0)
            print(
                json.dumps(
                    {
                        "name": name,
                        "m": m,
                        "n": n,
                        "k": k,
                        "relative_l2": float(error),
                        "graph_pass": True,
                    }
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
