"""Opt-in SM120 QSA TopK with native CUDA ordering and token expansion.

The model-local kernel uses NVIDIA CUB's installed stable block sort APIs. It does
not vendor vLLM/CUB source. Stable uint32 keys retain the pinned Torch TopK
selection and raw radix order, including signed zero and both NaN signs.
Compilation occurs during eager warmup; an unprimed CUDA graph falls back.
"""

import hashlib
import logging
import os
import threading

import torch

_TORCH_GIT = "70d99e998b4955e0049d13a98d77ae1b14db1f45"
_MODULE = None
_BUILD_FAILED = False
_BUILD_LOCK = threading.Lock()
_LOGGER = logging.getLogger(__name__)

_CUDA_SOURCE = r"""

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cub/block/block_radix_sort.cuh>
#include <cub/block/block_merge_sort.cuh>
#include <type_traits>

namespace rtp_llm {
struct QsaGreater {
    __device__ bool operator()(uint32_t a, uint32_t b) const {
        return a > b;
    }
};
template<int N, int Threads, bool UseMerge = false, bool DraftPacked = false>
__global__ void qsaCubStableTopK(const float*   scores,
                                 const int32_t* lengths,
                                 int32_t*       output,
                                 int            columns,
                                 bool           finite_only,
                                 const int32_t* draft_lengths = nullptr,
                                 const int32_t* draft_cu      = nullptr,
                                 int            draft_batch   = 0) {
    constexpr int Items = N / Threads;
    using FullSort      = std::conditional_t<UseMerge,
                                             cub::BlockMergeSort<uint32_t, Threads, Items, uint32_t>,
                                             cub::BlockRadixSort<uint32_t, Threads, Items, uint32_t>>;
    using SelectedSort  = cub::BlockRadixSort<uint32_t, Threads, 512 / Threads, uint32_t>;
    __shared__ union SharedSort {
        typename FullSort::TempStorage     full;
        typename SelectedSort::TempStorage selected;
    } storage;
    __shared__ uint32_t chosen[512];
    __shared__ uint32_t cutoff;
    const int           row        = blockIdx.x;
    int                 output_row = row;
    if constexpr (DraftPacked) {
        const int request = row / 4, lane = row % 4, count = draft_lengths[request];
        if (lane < count) {
            output_row = draft_cu[request] + lane;
        } else {
            output_row = draft_cu[draft_batch] + request * 4 - draft_cu[request] + lane - count;
            for (int j = threadIdx.x; j < 2051; j += Threads)
                output[output_row * 2051 + j] = -1;
            return;
        }
    }

    const int64_t length   = lengths[row];
    const int64_t complete = (length - (length < 0 ? 3 : 0)) / 4;
    uint32_t      key[Items], index[Items];
    bool          special = false;
#pragma unroll
    for (int i = 0; i < Items; ++i) {
        uint32_t col = threadIdx.x * Items + i;
        uint32_t bits =
            (col < columns && col < complete) ?
                (col < columns && col < complete ? __float_as_uint(scores[row * columns + col]) : 0xff800000u) :
                0xff800000u;
        uint32_t ordered = (bits & 0x80000000u) ? ~bits : bits ^ 0x80000000u;
        bool     isnan   = (bits & 0x7f800000u) == 0x7f800000u && (bits & 0x007fffffu);
        special |= isnan || bits == 0x80000000u;
        key[i]   = isnan ? 0xffffffffu : ordered;
        index[i] = col;
    }
    bool any_special = __syncthreads_or(special);
    if constexpr (UseMerge) {
        FullSort(storage.full).StableSort(key, index, QsaGreater{});
    } else {
        FullSort(storage.full).SortDescending(key, index);
    }
    if (any_special) {
        // Torch first gathers values strictly above the selected radix, then
        // its equal boundary, in column order. Its float radix sort treats
        // signed zeros as equal. Recreate that order only for NaNs/-0.

#pragma unroll
        for (int i = 0; i < Items; ++i) {
            int rank = threadIdx.x * Items + i;
            if (rank < 512)
                chosen[rank] = index[i];
            if (rank == 511)
                cutoff = key[i];
        }
        __syncthreads();
        uint32_t raw_key[512 / Threads], raw_index[512 / Threads];
#pragma unroll
        for (int i = 0; i < 512 / Threads; ++i) {
            uint32_t rank = threadIdx.x * (512 / Threads) + i;
            uint32_t col  = chosen[rank];
            uint32_t bits =
                (col < columns && col < complete ? __float_as_uint(scores[row * columns + col]) : 0xff800000u);
            uint32_t ordered   = (bits & 0x80000000u) ? ~bits : bits ^ 0x80000000u;
            bool     isnan     = (bits & 0x7f800000u) == 0x7f800000u && (bits & 0x007fffffu);
            uint32_t selection = isnan ? 0xffffffffu : ordered;
            raw_key[i]         = 2 * N - ((selection == cutoff ? N : 0) + col);
            raw_index[i]       = col;
        }
        SelectedSort(storage.selected).SortDescending(raw_key, raw_index);
        __syncthreads();
#pragma unroll
        for (int i = 0; i < 512 / Threads; ++i) {
            uint32_t col = raw_index[i];
            uint32_t bits =
                (col < columns && col < complete ? __float_as_uint(scores[row * columns + col]) : 0xff800000u);
            // CUB float keys canonicalize -0/+0; payloads retain original IDs.
            if ((bits & 0x7fffffffu) == 0)
                bits = 0;
            raw_key[i] = (bits & 0x80000000u) ? ~bits : bits ^ 0x80000000u;
        }
        SelectedSort(storage.selected).SortDescending(raw_key, raw_index);
#pragma unroll
        for (int i = 0; i < 512 / Threads; ++i) {
            uint32_t rank = threadIdx.x * (512 / Threads) + i;
            uint32_t col  = raw_index[i];
            uint32_t bits =
                (col < columns && col < complete ? __float_as_uint(scores[row * columns + col]) : 0xff800000u);
            bool valid = ((bits & 0x7f800000u) != 0x7f800000u || (!finite_only && bits == 0x7f800000u)) && col < columns
                         && col < complete;
#pragma unroll
            for (int j = 0; j < 4; ++j)
                output[output_row * 2051 + rank * 4 + j] = valid ? static_cast<int32_t>(col * 4 + j) : -1;
        }
    } else {
#pragma unroll
        for (int i = 0; i < Items; ++i) {
            uint32_t rank = threadIdx.x * Items + i;
            if (rank < 512) {
                uint32_t col = index[i];
                uint32_t bits =
                    (col < columns && col < complete ? __float_as_uint(scores[row * columns + col]) : 0xff800000u);
                bool valid = ((bits & 0x7f800000u) != 0x7f800000u || (!finite_only && bits == 0x7f800000u))
                             && col < columns && col < complete;
#pragma unroll
                for (int j = 0; j < 4; ++j)
                    output[output_row * 2051 + rank * 4 + j] = valid ? static_cast<int32_t>(col * 4 + j) : -1;
            }
        }
    }
    if (threadIdx.x == 0) {
#pragma unroll
        for (int j = 0; j < 3; ++j)
            output[output_row * 2051 + 2048 + j] = j < length - complete * 4 ? complete * 4 + j : -1;
    }
}
}  // namespace rtp_llm

void qsa_cub_topk(
    torch::Tensor scores, torch::Tensor lengths, torch::Tensor output, int64_t threads, bool finite_only, bool merge) {
    TORCH_CHECK(scores.is_cuda() && lengths.is_cuda() && output.is_cuda());
    TORCH_CHECK(scores.scalar_type() == at::kFloat && lengths.scalar_type() == at::kInt
                && output.scalar_type() == at::kInt);
    TORCH_CHECK(scores.dim() == 2 && lengths.dim() == 1 && output.dim() == 2);
    TORCH_CHECK(scores.is_contiguous() && lengths.is_contiguous() && output.is_contiguous());
    TORCH_CHECK(scores.device() == lengths.device() && scores.device() == output.device());
    TORCH_CHECK(scores.size(0) == lengths.numel() && output.size(0) == scores.size(0) && output.size(1) == 2051);
    TORCH_CHECK(scores.size(1) == 1023 || scores.size(1) == 1024 || scores.size(1) == 2047 || scores.size(1) == 2048
                || scores.size(1) == 2144);
    TORCH_CHECK(threads == 128 || threads == 256 || threads == 512);
    c10::cuda::CUDAGuard guard(scores.device());
    auto                 stream = at::cuda::getCurrentCUDAStream().stream();
    const int            rows = scores.size(0), columns = scores.size(1);
    if (!rows)
        return;
    // Measured 8K domains only. StableSort preserves equal-key column order;
    // the special-value second pass below remains the pinned Torch radix order.
    if (merge && (columns == 2047 || columns == 2048 || columns == 2144)) {
        if (columns == 2144) {
            rtp_llm::qsaCubStableTopK<2304, 256, true><<<rows, 256, 0, stream>>>(scores.data_ptr<float>(),
                                                                                 lengths.data_ptr<int32_t>(),
                                                                                 output.data_ptr<int32_t>(),
                                                                                 columns,
                                                                                 finite_only);
        } else {
            rtp_llm::qsaCubStableTopK<2048, 512, true><<<rows, 512, 0, stream>>>(scores.data_ptr<float>(),
                                                                                 lengths.data_ptr<int32_t>(),
                                                                                 output.data_ptr<int32_t>(),
                                                                                 columns,
                                                                                 finite_only);
        }
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        return;
    }
#define LAUNCH(N, T)                                                                                                   \
    rtp_llm::qsaCubStableTopK<N, T><<<rows, T, 0, stream>>>(                                                           \
        scores.data_ptr<float>(), lengths.data_ptr<int32_t>(), output.data_ptr<int32_t>(), columns, finite_only)
    if (columns <= 1024) {
        if (threads == 128) {
            LAUNCH(1024, 128);
        } else if (threads == 256) {
            LAUNCH(1024, 256);
        } else {
            LAUNCH(1024, 512);
        }
    } else if (columns == 2144) {
        if (threads == 128) {
            LAUNCH(2176, 128);
        } else if (threads == 256) {
            LAUNCH(2304, 256);
        } else {
            LAUNCH(2560, 512);
        }
    } else {
        if (threads == 128) {
            LAUNCH(2048, 128);
        } else if (threads == 256) {
            LAUNCH(2048, 256);
        } else {
            LAUNCH(2048, 512);
        }
    }
#undef LAUNCH
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
void qsa_cub_topk_draft(
    torch::Tensor scores, torch::Tensor lengths, torch::Tensor output, torch::Tensor row_lengths, torch::Tensor cu) {
    TORCH_CHECK(scores.is_cuda() && lengths.is_cuda() && output.is_cuda() && row_lengths.is_cuda() && cu.is_cuda());
    TORCH_CHECK(scores.scalar_type() == at::kFloat && lengths.scalar_type() == at::kInt
                && output.scalar_type() == at::kInt && row_lengths.scalar_type() == at::kInt
                && cu.scalar_type() == at::kInt);
    TORCH_CHECK(scores.is_contiguous() && lengths.is_contiguous() && output.is_contiguous()
                && row_lengths.is_contiguous() && cu.is_contiguous());
    TORCH_CHECK(scores.device() == lengths.device() && scores.device() == output.device()
                && scores.device() == row_lengths.device() && scores.device() == cu.device());
    TORCH_CHECK(scores.dim() == 2 && lengths.dim() == 1 && row_lengths.dim() == 1 && cu.dim() == 1
                && output.dim() == 2);
    int rows = scores.size(0), cols = scores.size(1), batch = row_lengths.numel();
    TORCH_CHECK(rows == batch * 4 && batch >= 1 && batch <= 8 && lengths.numel() == rows && output.size(0) == rows
                && output.size(1) == 2051 && cu.numel() == batch + 1 && cols == 2144);
    c10::cuda::CUDAGuard guard(scores.device());
    auto                 stream = at::cuda::getCurrentCUDAStream().stream();
    rtp_llm::qsaCubStableTopK<2304, 256, true, true><<<rows, 256, 0, stream>>>(scores.data_ptr<float>(),
                                                                               lengths.data_ptr<int32_t>(),
                                                                               output.data_ptr<int32_t>(),
                                                                               cols,
                                                                               true,
                                                                               row_lengths.data_ptr<int32_t>(),
                                                                               cu.data_ptr<int32_t>(),
                                                                               batch);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
"""


def is_supported(scores, lengths):
    return (
        os.environ.get("QWEN4_QSA_CUB_TOPK", "0") == "1"
        and scores.is_cuda
        and torch.version.hip is None
        and torch.version.git_version == _TORCH_GIT
        and scores.dtype == torch.float32
        and scores.ndim == 2
        and 1 <= scores.shape[0] <= 32
        # The 8K MTP draft graph reserves three additional 128-token pages:
        # (64 + 3) * 32 compressed columns. Preserve the full-width Torch order.
        and scores.shape[1] in (2047, 2048, 2144)
        and scores.is_contiguous()
        and lengths.dtype == torch.int32
        and lengths.shape == (scores.shape[0],)
        and lengths.device == scores.device
        and lengths.is_contiguous()
        and not (scores.requires_grad or lengths.requires_grad)
        and torch.cuda.get_device_capability(scores.device) == (12, 0)
    )


def _load_module(device):
    global _MODULE, _BUILD_FAILED
    if _MODULE is not None:
        return _MODULE
    if _BUILD_FAILED:
        return None
    with torch.cuda.device(device):
        if torch.cuda.is_current_stream_capturing():
            return None
        with _BUILD_LOCK:
            if _MODULE is not None:
                return _MODULE
            if _BUILD_FAILED:
                return None
            try:
                from torch.utils.cpp_extension import load_inline

                flags = [
                    "-O3",
                    "--expt-relaxed-constexpr",
                    "-gencode=arch=compute_120,code=sm_120",
                ]
                identity = hashlib.sha256(
                    (_CUDA_SOURCE + _TORCH_GIT + repr(flags)).encode()
                ).hexdigest()[:16]
                _MODULE = load_inline(
                    name="qwen4_qsa_topk_" + identity,
                    cpp_sources="void qsa_cub_topk(torch::Tensor scores, torch::Tensor lengths, torch::Tensor output, int64_t threads, bool finite_only, bool merge); void qsa_cub_topk_draft(torch::Tensor scores, torch::Tensor lengths, torch::Tensor output, torch::Tensor row_lengths, torch::Tensor cu);",
                    cuda_sources=_CUDA_SOURCE,
                    functions=["qsa_cub_topk", "qsa_cub_topk_draft"],
                    extra_cflags=["-O2"],
                    extra_cuda_cflags=flags,
                    with_cuda=True,
                    verbose=False,
                )
            except Exception:
                _BUILD_FAILED = True
                _LOGGER.warning(
                    "Qwen4 QSA CUB compilation unavailable; using native selection",
                    exc_info=True,
                )
            return _MODULE


def try_cub_qsa_tokens(scores, lengths, *, finite_only=True):
    if not is_supported(scores, lengths):
        return None
    module = _load_module(scores.device)
    if module is None:
        return None
    output = torch.empty(
        (scores.shape[0], 2051), device=scores.device, dtype=torch.int32
    )
    module.qsa_cub_topk(
        scores,
        lengths,
        output,
        256,
        finite_only,
        os.environ.get("QWEN4_QSA_MERGE_TOPK", "0") == "1",
    )
    return output


def try_cub_draft_qsa_tokens(scores, visible, row_lengths, cu):
    """Select directly into packed draft order; metadata is owner-validated.

    Fixed four-lane input rows and packed output rows have a bijection even
    for empty requests. Inactive rows fill the output padding with -1.
    """
    if not (
        is_supported(scores, visible)
        and os.environ.get("QWEN4_QSA_MERGE_TOPK", "0") == "1"
        and os.environ.get("QWEN4_DRAFT_QSA_DIRECT", "0") == "1"
        and scores.shape[1] == 2144
        and row_lengths.ndim == cu.ndim == 1
        and 1 <= row_lengths.numel() <= 8
        and scores.shape[0] == 4 * row_lengths.numel()
        and cu.numel() == row_lengths.numel() + 1
        and all(
            t.dtype == torch.int32
            and t.device == scores.device
            and t.is_contiguous()
            and not t.requires_grad
            for t in (row_lengths, cu)
        )
    ):
        return None
    module = _load_module(scores.device)
    if module is None:
        return None
    output = torch.empty(
        (scores.shape[0], 2051), device=scores.device, dtype=torch.int32
    )
    module.qsa_cub_topk_draft(scores, visible, output, row_lengths, cu)
    return output
