"""SGLang FP32 Prefill TopK, adapted only at the PPU launch boundary.

The selection algorithm and tie handling are vendored in dsv4_ppu; see NOTICE.
The FP8 indexer emits FP32 scores. Do not narrow these to BF16: SGLang's BF16
TopK is paired with a different FP4 score producer. Returns request-local
compressed KV indices, with -1 after short valid ranges, on the current stream.
"""

from functools import lru_cache
from pathlib import Path

import torch

_TEMPLATE = """
const TopKPrefillParams params = {
    scores, row_starts, row_ends, page_table, out, nullptr,
    score_stride, 0, 31
};
topk_prefill_transform_kernel<<<rows, 512, 2048, stream>>>(params);
__return_code = 0;
"""


@lru_cache(maxsize=None)
def _page_zero(device):
    # A single identity page spans all supported columns (T < 2**31).
    # SG's unchanged page mapping therefore emits the raw local indices.
    return torch.zeros((1,), device=device, dtype=torch.int32)


def is_supported(scores, starts, ends, out, topk):
    return (
        topk == 512
        and scores.ndim == 2
        and scores.dtype == torch.float32
        and scores.is_cuda
        and scores.stride(1) == 1
        and scores.shape[1] < 2**31
        and torch.cuda.get_device_name(scores.device) == "ZW-M890P"
        and all(t.device == scores.device for t in (starts, ends, out))
        and all(
            t.dtype == torch.int32 and t.is_contiguous() for t in (starts, ends, out)
        )
        and starts.shape == ends.shape == (scores.shape[0],)
        and out.shape == (scores.shape[0], topk)
    )


def topk_prefill(scores, starts, ends, out, topk=512):
    if not is_supported(scores, starts, ends, out, topk):
        raise ValueError(
            "SG PPU TopK requires FP32 [M,T], int32 ranges and contiguous int32 [M,512] output"
        )
    if scores.shape[0] == 0:
        return out
    from deep_gemm.jit_kernels.tuner import jit_tuner
    from rtp_llm.platforms.ppu.runtime import install_deep_gemm_build_lock

    install_deep_gemm_build_lock()
    args = (
        scores,
        starts,
        ends,
        _page_zero(scores.device),
        out,
        scores.shape[0],
        scores.stride(0),
        torch.cuda.current_stream(scores.device),
    )
    header = Path(__file__).with_name("dsv4_ppu") / "sglang_topk_prefill.cuh"
    runtime = jit_tuner.compile_and_tune(
        name="rtp_sglang_prefill_topk_fp32_v1",
        keys={},
        space=(),
        includes=('"' + str(header) + '"',),
        arg_defs=(
            ("scores", torch.float32),
            ("row_starts", torch.int32),
            ("row_ends", torch.int32),
            ("page_table", torch.int32),
            ("out", torch.int32),
            ("rows", int),
            ("score_stride", int),
            ("stream", torch.cuda.Stream),
        ),
        template=_TEMPLATE,
        args=args,
        jit_include_dir="cutlass3",
    )
    runtime(*args)
    return out
