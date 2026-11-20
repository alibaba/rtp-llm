"""Native RTP TopK for SM90 prefill, followed by QSA selection packing.

Selection follows topk_v3's unordered output and unspecified equal-score ties.
The fused packer filters invalid blocks and appends tail tokens and counts.
"""

import hashlib
import logging
import threading

import torch

BLOCK_SELECTION_WIDTH = 514  # 512 selected blocks, one tail block, token count

_MODULE = None
_BUILD_FAILED = False
_LOCK = threading.Lock()
_LOGGER = logging.getLogger(__name__)

_BINDING = r"""
#include <torch/extension.h>
void qsa_pack_selected(torch::Tensor, torch::Tensor, torch::Tensor,
                      torch::Tensor, torch::Tensor, int64_t);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("expand", &qsa_pack_selected); }
"""
_CUDA = r"""
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <cmath>
#include <cstdint>
#include <cub/block/block_scan.cuh>

namespace {
constexpr int kBlockSelectionWidth = 514;
template<bool PACKED>
__global__ __launch_bounds__(128) void qsaPackSelected(const float* scores,
                                                       const int*   selected,
                                                       const int*   visible,
                                                       const int*   positions,
                                                       int*         output,
                                                       int          columns,
                                                       int          offset) {
    using Scan = cub::BlockScan<int, 128>;
    __shared__ union {
        typename Scan::TempStorage scan;
        int                        selected[512];
    } scratch;
    const int row    = blockIdx.x;
    const int length = min(columns, max(0, visible[offset + row]));
    int       blocks[4];
    int       valid = 0;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        int chosen = selected[int64_t(row) * 512 + threadIdx.x * 4 + i];
        // Keep the selector's unspecified order and equal-score membership.
        // Preserve value > -inf: exclude NaN and -inf, but retain +inf.
        blocks[i] = chosen >= 0 && chosen < length && scores[int64_t(row) * columns + chosen] > -INFINITY ? chosen : -1;
        valid += blocks[i] >= 0;
    }
    const int tail_start = length * 4;
    const int tail_count = min(3, max(0, positions[offset + row] + 1 - tail_start));
    if constexpr (PACKED) {
        int start, block_count;
        Scan(scratch.scan).ExclusiveSum(valid, start, block_count);
        int* dest = output + int64_t(offset + row) * kBlockSelectionWidth;
#pragma unroll
        for (int i = 0; i < 4; ++i)
            if (blocks[i] >= 0)
                dest[start++] = blocks[i];
        if (threadIdx.x == 0) {
            if (tail_count > 0)
                dest[block_count] = length;
            dest[kBlockSelectionWidth - 1] = block_count * 4 + tail_count;
        }
        return;
    }
#pragma unroll
    for (int i = 0; i < 4; ++i)
        scratch.selected[threadIdx.x * 4 + i] = blocks[i];
    __syncthreads();
    int* dest = output + int64_t(offset + row) * 2051;
    for (int col = threadIdx.x; col < 2051; col += 128) {
        int token;
        if (col < 2048) {
            int chosen = scratch.selected[col / 4];
            token      = chosen >= 0 ? chosen * 4 + col % 4 : -1;
        } else {
            int tail_offset = col - 2048;
            token           = tail_offset < tail_count ? tail_start + tail_offset : -1;
        }
        dest[col] = token;
    }
}
}  // namespace

void qsa_pack_selected(torch::Tensor scores,
                       torch::Tensor selected,
                       torch::Tensor visible,
                       torch::Tensor positions,
                       torch::Tensor output,
                       int64_t       offset) {
    TORCH_CHECK(scores.is_cuda() && scores.scalar_type() == at::kFloat && scores.dim() == 2 && scores.is_contiguous(),
                "scores must be contiguous CUDA FP32 [N,C]");
    TORCH_CHECK(scores.size(1) > 0 && scores.size(1) <= INT32_MAX / 4,
                "score width exceeds the int32 token-index range");
    TORCH_CHECK(visible.device() == scores.device() && positions.device() == scores.device()
                    && output.device() == scores.device(),
                "QSA tensors must share a device");
    TORCH_CHECK(visible.scalar_type() == at::kInt && positions.scalar_type() == at::kInt
                    && output.scalar_type() == at::kInt && visible.is_contiguous() && positions.is_contiguous()
                    && output.is_contiguous(),
                "QSA indices must be contiguous int32");
    TORCH_CHECK(offset >= 0 && visible.dim() == 1 && positions.dim() == 1 && visible.numel() >= offset + scores.size(0)
                    && positions.numel() >= offset + scores.size(0) && output.dim() == 2
                    && (output.size(1) == 2051 || output.size(1) == kBlockSelectionWidth)
                    && output.size(0) >= offset + scores.size(0),
                "invalid QSA packed shape or offset");
    TORCH_CHECK(selected.device() == scores.device() && selected.scalar_type() == at::kInt && selected.is_contiguous()
                    && selected.dim() == 2 && selected.size(0) == scores.size(0) && selected.size(1) == 512,
                "selected must be CUDA int32 [N,512] on the scores device");
    if (scores.size(0) == 0)
        return;
    c10::cuda::CUDAGuard guard(scores.device());
    auto                 stream = at::cuda::getCurrentCUDAStream();
    auto expand = output.size(1) == kBlockSelectionWidth ? qsaPackSelected<true> : qsaPackSelected<false>;
    expand<<<scores.size(0), 128, 0, stream>>>(scores.data_ptr<float>(),
                                               selected.data_ptr<int>(),
                                               visible.data_ptr<int>(),
                                               positions.data_ptr<int>(),
                                               output.data_ptr<int>(),
                                               scores.size(1),
                                               offset);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
"""


def _load_module():
    global _MODULE, _BUILD_FAILED
    if _MODULE is None and not _BUILD_FAILED:
        with _LOCK:
            if _MODULE is None and not _BUILD_FAILED:
                try:
                    return _build_module()
                except Exception:
                    _BUILD_FAILED = True
                    _LOGGER.warning(
                        "Qwen4 QSA prefill TopK unavailable; using fallback selection",
                        exc_info=True,
                    )
    return _MODULE


class _Selector:
    def __init__(self, module, topk):
        self._expand = module.expand
        self._topk = topk

    def select(self, scores, visible, positions, output, offset):
        rows, columns = scores.shape
        selected = torch.empty((rows, 512), device=scores.device, dtype=torch.int32)
        # topk_v3 retains this unused buffer only for ABI compatibility.
        workspace = torch.empty(0, device=scores.device, dtype=torch.uint8)
        self._topk(
            scores,
            visible.narrow(0, offset, rows),
            selected,
            workspace,
            512,
            columns,
        )
        self._expand(scores, selected, visible, positions, output, offset)


def _build_module():
    global _MODULE
    from torch.utils.cpp_extension import load_inline

    digest = hashlib.sha256((_BINDING + _CUDA).encode()).hexdigest()[:16]
    module = load_inline(
        name="qsa_prefill_pack_" + digest,
        cpp_sources=_BINDING,
        cuda_sources=_CUDA,
        extra_cflags=["-O3", "-std=c++17"],
        extra_cuda_cflags=["-O3", "-std=c++17"],
        verbose=False,
    )
    from rtp_llm.ops.compute_ops import rtp_llm_ops

    _MODULE = _Selector(module, rtp_llm_ops.topk_v3)
    return _MODULE
