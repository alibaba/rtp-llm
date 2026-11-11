"""vLLM histogram TopK for SM90 prefill, with stable ordering and expansion.

Only the <=8192-column register path and its exact overflow recovery are
ported. Multi-CTA/decode/long-row dispatch remains in the existing kernels.
"""

import hashlib
import logging
import threading
from pathlib import Path

import torch

_MODULE = None
_BUILD_FAILED = False
_LOCK = threading.Lock()
_LOGGER = logging.getLogger(__name__)

_BINDING = r"""

#include <torch/extension.h>
void qsa_filtered_topk(torch::Tensor scores, torch::Tensor visible,
                       torch::Tensor positions, torch::Tensor output,
                       int64_t offset);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("select", &qsa_filtered_topk);
}
"""
_CUDA = r"""


#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <cub/block/block_merge_sort.cuh>
#include <cub/block/block_reduce.cuh>
#include <cub/block/block_scan.cuh>

#include "topk_histogram_4096_overflow.cuh"

namespace {
__device__ uint32_t ordered_key(float value) {
  uint32_t bits;
  asm("mov.b32 %0, %1;" : "=r"(bits) : "f"(value));
  return (bits & 0x80000000u) ? ~bits : bits ^ 0x80000000u;
}
struct Minimum {
  __device__ uint32_t operator()(uint32_t a, uint32_t b) const {
    return a < b ? a : b;
  }
};
__global__ __launch_bounds__(1024, 2) void qsaFilteredTopK(const float* scores,
                                                           const int* visible,
                                                           int* selected_output,
                                                           int columns,
                                                           int offset) {
  extern __shared__ __align__(128) unsigned char scratch[];
  __shared__ int selected[512];
  const int row = blockIdx.x;
  const int length = visible[offset + row];
  const int lane = threadIdx.x;
  // Preserve the original score-descending order and ascending column ties.
  // Unordered but set-equivalent selections change BF16 attention rounding.
  using Reduce = cub::BlockReduce<uint32_t, 1024>;
  using Scan = cub::BlockScan<uint64_t, 1024>;
  __shared__ union {
    typename Reduce::TempStorage reduce;
    typename Scan::TempStorage scan;
  } ops;
  __shared__ uint32_t threshold;
  const float* row_scores = scores + int64_t(row) * columns;
  // The small sort generates the complete prefix directly for these rows.
  if (length <= 512) return;
  {
    namespace H = vllm::topk_histogram_4096;
    H::histogram_4096_topk<512, 12, 2, false,
                           H::OverflowRecovery::kRegisterValues>(
        row_scores, selected, length, scratch);
    __syncthreads();
    uint32_t value = lane < 512
                         ? H::convert_to_uint32_v2(row_scores[selected[lane]])
                         : 0xffffffffu;
    uint32_t boundary = Reduce(ops.reduce).Reduce(value, Minimum());
    if (lane == 0) threshold = boundary;
    __syncthreads();
    // The upstream histogram uses atomic compaction. Restore the original
    // lowest-column boundary membership before sorting, including dense ties.
    const int items = (length + 1023) / 1024;
    const int first = lane * items;
    unsigned above = 0, equal = 0;
    uint32_t values[8];
    for (int i = 0; i < items; ++i) {
      values[i] = first + i < length
                      ? H::convert_to_uint32_v2(row_scores[first + i])
                      : 0;
    }
    if (threshold == 0x7fffffffu || threshold == 0x80000000u) {
      unsigned positive = 0;
      for (int i = 0; i < items; ++i)
        positive += first + i < length && values[i] >= 0x80000000u;
      unsigned total_positive = Reduce(ops.reduce).Sum(positive);
      if (lane == 0)
        threshold = total_positive >= 512 ? 0x80000000u : 0x7fffffffu;
      __syncthreads();
    }
    for (int i = 0; i < items; ++i) {
      above += first + i < length && values[i] > threshold;
      equal += first + i < length && values[i] == threshold;
    }
    uint64_t prefix, total;
    Scan(ops.scan).ExclusiveSum((uint64_t(above) << 32) | equal, prefix, total);
    __syncthreads();
    unsigned above_rank = prefix >> 32;
    unsigned equal_rank = unsigned(prefix) + unsigned(total >> 32);
    for (int i = 0; i < items; ++i) {
      if (first + i >= length) break;
      if (values[i] > threshold)
        selected[above_rank++] = first + i;
      else if (values[i] == threshold && equal_rank < 512)
        selected[equal_rank++] = first + i;
    }
  }
  __syncthreads();
  if (lane < 512) selected_output[int64_t(row) * 512 + lane] = selected[lane];
}

struct Greater {
  __device__ bool operator()(uint32_t a, uint32_t b) const { return a > b; }
};
__global__ __launch_bounds__(128) void qsaStableOrderExpand(
    const float* scores, const int* selected, const int* visible,
    const int* positions, int* output, int columns, int offset) {
  using Sort = cub::BlockMergeSort<uint32_t, 128, 4, uint32_t>;
  __shared__ union {
    typename Sort::TempStorage sort;
    int selected[512];
  } scratch;
  const int row = blockIdx.x;
  const int length = visible[offset + row];
  uint32_t key[4], column[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    int slot = threadIdx.x * 4 + i;
    int chosen = length <= 512 ? (slot < length ? slot : -1)
                               : selected[int64_t(row) * 512 + slot];
    column[i] = uint32_t(chosen);
    key[i] = 0;
    if (chosen >= 0) {
      float value = scores[int64_t(row) * columns + chosen];
      key[i] = ordered_key(value == 0.0f ? 0.0f : value);
    }
  }
  Sort(scratch.sort).StableSort(key, column, Greater{});
  __syncthreads();
#pragma unroll
  for (int i = 0; i < 4; ++i)
    // Preserve value > -inf: exclude NaN and -inf, but retain +inf.
    scratch.selected[threadIdx.x * 4 + i] =
        key[i] > 0x007fffffu && key[i] <= 0xff800000u ? int(column[i]) : -1;
  __syncthreads();
  int* dest = output + int64_t(offset + row) * 2051;
  // Coalesce the 2051-stride token output; four-token vector stores would
  // be misaligned on three out of four rows.
  for (int col = threadIdx.x; col < 2051; col += 128) {
    int token;
    if (col < 2048) {
      int chosen = scratch.selected[col / 4];
      token = chosen >= 0 ? chosen * 4 + col % 4 : -1;
    } else {
      int tail = length * 4 + col - 2048;
      token = tail <= positions[offset + row] ? tail : -1;
    }
    dest[col] = token;
  }
}
}  // namespace

void qsa_filtered_topk(torch::Tensor scores, torch::Tensor visible,
                       torch::Tensor positions, torch::Tensor output,
                       int64_t offset) {
  TORCH_CHECK(scores.is_cuda() && scores.scalar_type() == at::kFloat &&
                  scores.dim() == 2 && scores.is_contiguous(),
              "scores must be contiguous CUDA FP32 [N,C]");
  TORCH_CHECK(
      scores.size(1) > 0 && scores.size(1) <= 8192 && scores.size(1) % 4 == 0,
      "score width must be aligned and <=8192");
  TORCH_CHECK(visible.device() == scores.device() &&
                  positions.device() == scores.device() &&
                  output.device() == scores.device(),
              "QSA tensors must share a device");
  TORCH_CHECK(visible.scalar_type() == at::kInt &&
                  positions.scalar_type() == at::kInt &&
                  output.scalar_type() == at::kInt && visible.is_contiguous() &&
                  positions.is_contiguous() && output.is_contiguous(),
              "QSA indices must be contiguous int32");
  TORCH_CHECK(offset >= 0 && visible.dim() == 1 && positions.dim() == 1 &&
                  visible.numel() >= offset + scores.size(0) &&
                  positions.numel() >= offset + scores.size(0) &&
                  output.dim() == 2 && output.size(1) == 2051 &&
                  output.size(0) >= offset + scores.size(0),
              "invalid QSA packed shape or offset");
  if (scores.size(0) == 0) return;
  c10::cuda::CUDAGuard guard(scores.device());
  namespace H = vllm::topk_histogram_4096;
  constexpr size_t shared_bytes = sizeof(H::Histogram4096Smem<512, 12>);
  static_assert(shared_bytes >= sizeof(H::CoarseRefineSmem));
  // The zero-capacity rescan fallback and tie refinement fit below 16 KiB.
  static_assert(shared_bytes >= 16 * 1024);
  auto selected = torch::empty({scores.size(0), 512}, output.options());
  auto stream = at::cuda::getCurrentCUDAStream();
  qsaFilteredTopK<<<scores.size(0), 1024, shared_bytes, stream>>>(
      scores.data_ptr<float>(), visible.data_ptr<int>(),
      selected.data_ptr<int>(), scores.size(1), offset);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  qsaStableOrderExpand<<<scores.size(0), 128, 0, stream>>>(
      scores.data_ptr<float>(), selected.data_ptr<int>(),
      visible.data_ptr<int>(), positions.data_ptr<int>(),
      output.data_ptr<int>(), scores.size(1), offset);
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
                        "Qwen4 QSA prefill TopK unavailable; using native selection",
                        exc_info=True,
                    )
    return _MODULE


def _build_module():
    global _MODULE
    from torch.utils.cpp_extension import load_inline

    headers = Path(__file__).with_name("vllm_topk")
    digest = hashlib.sha256((_BINDING + _CUDA).encode())
    for path in sorted(headers.glob("*.cuh")):
        digest.update(path.read_bytes())
    _MODULE = load_inline(
        name="qsa_prefill_filtered_" + digest.hexdigest()[:16],
        cpp_sources=_BINDING,
        cuda_sources=_CUDA,
        extra_include_paths=[str(headers)],
        extra_cflags=["-O3", "-std=c++17"],
        extra_cuda_cflags=["-O3", "-std=c++17"],
        verbose=False,
    )
    return _MODULE
