"""Exact C4 Q3 scores directly from packed pages."""

import importlib
from pathlib import Path

HEADER = Path(__file__).resolve().with_name("c4_paged_score.cuh")

TEMPLATE = """
using namespace deep_gemm;
launch_paged_mqa_logits_metadata<64, {NUM_BLOCKS}>(batch, 3,
    reinterpret_cast<const uint32_t*>(ends), reinterpret_cast<uint32_t*>(metadata), stream);
using kernel_t=C4PagedScore<uint8_t, float, __nv_bfloat16, float, 3, 64, 64,
                            64, 32, 2, 3, 64, false, false>;
kernel_t::run(q, reinterpret_cast<const uint32_t*>(scales), cache,
              reinterpret_cast<const uint32_t*>(cache+4096),
              weights, batch, 12288, 4352, table_stride, reinterpret_cast<const uint32_t*>(ends),
              out, reinterpret_cast<const uint32_t*>(table), reinterpret_cast<const uint32_t*>(metadata),
              stream, num_sms, {NUM_BLOCKS}, num_pages, row_ends, nullptr);
"""


def can_paged_score(pool, table, q, scales, weights, lengths, width):
    import torch

    return (
        q.is_cuda
        and torch.cuda.get_device_name(q.device) == "ZW-M890P"
        and width == 12288
        and table.ndim == 2
        and table.shape[0] == 8
        and table.shape[1] >= 192
        and q.shape == (24, 64, 64)
        and q.dtype == torch.int8
        and scales.shape == (24, 64, 1)
        and scales.dtype == torch.int32
        and pool.ndim == 3
        and pool.shape[1:] == (64, 68)
        and pool.dtype == torch.uint8
        and weights.shape == (24, 64)
        and weights.dtype in (torch.float32, torch.bfloat16)
        and lengths.shape == (24,)
        and lengths.dtype == table.dtype == torch.int32
        and all(
            x.device == q.device and x.is_contiguous()
            for x in (pool, table, q, scales, weights, lengths)
        )
    )


def paged_score_exact(pool, table, q, scales, weights, lengths, out):
    import torch

    assert can_paged_score(pool, table, q, scales, weights, lengths, 12288)
    assert out.shape == (24, 12288) and out.dtype == torch.bfloat16
    assert out.device == q.device and out.is_contiguous()
    assert all(
        x.untyped_storage().data_ptr() != out.untyped_storage().data_ptr()
        for x in (pool, table, q, scales, weights, lengths)
    )
    impl = importlib.import_module("deep_gemm.jit_kernels.gemm_fp4")
    ends = (
        lengths.view(8, 3).amax(1).clamp(0, 12288).view(8, 1).expand(8, 3).contiguous()
    )
    weight_f32 = weights.to(torch.float32)
    sms = impl.get_num_sms()
    num_blocks = 5 * sms
    metadata = torch.empty((num_blocks + 1, 2), device=q.device, dtype=torch.int32)
    out.fill_(float("-inf"))
    args = (
        q.view(torch.uint8),
        scales,
        pool,
        weight_f32,
        ends,
        out,
        table,
        metadata,
        lengths,
        8,
        pool.shape[0],
        torch.cuda.current_stream(),
        sms,
        table.shape[1],
    )
    definitions = (
        ("q", torch.uint8),
        ("scales", torch.int32),
        ("cache", torch.uint8),
        ("weights", torch.float32),
        ("ends", torch.int32),
        ("out", torch.bfloat16),
        ("table", torch.int32),
        ("metadata", torch.int32),
        ("row_ends", torch.int32),
        ("batch", int),
        ("num_pages", int),
        ("stream", torch.cuda.Stream),
        ("num_sms", int),
        ("table_stride", int),
    )
    runtime = impl.jit_tuner.compile_and_tune(
        name="rtp_c4_paged_score_q3",
        keys={"NUM_BLOCKS": num_blocks},
        space=(),
        includes=(f'"{HEADER}"',),
        arg_defs=definitions,
        template=TEMPLATE,
        args=args,
        jit_include_dir="actlize_v1.0.0",
    )
    runtime(*args)
    return out
