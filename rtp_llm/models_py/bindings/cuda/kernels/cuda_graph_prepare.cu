#include "rtp_llm/models_py/bindings/cuda/kernels/cuda_graph_prepare.h"
#include "rtp_llm/models_py/bindings/common/kernels/fuse_copy_util.h"

#include <algorithm>
#include <limits>
#include <vector>
#include <c10/util/Exception.h>
#if USING_CUDA
#include <c10/cuda/CUDAException.h>
#endif
#if USING_ROCM
#include <hip/hip_runtime.h>
#else
#include <cuda_runtime.h>
#endif

namespace rtp_llm {

namespace {

struct PrepareTask {
    CudaGraphPrepareFillRegion fill;
    const char* src = nullptr;
    size_t rows = 0;
    size_t row_bytes = 0;
    size_t src_stride = 0;
    size_t dst_stride = 0;
};

struct PrepareTasks {
    int count = 0;
    PrepareTask tasks[kMaxCudaGraphPrepareFillRegions];
};
static_assert(sizeof(PrepareTasks) < 4096, "keep compatibility with the legacy kernel parameter limit");

template<typename T>
__device__ void prepareCopyRow(const char* src, char* dst, size_t bytes, size_t lane, size_t lanes) {
    const size_t tid = lane * blockDim.x + threadIdx.x;
    const size_t stride = lanes * blockDim.x;
    for (size_t i = tid; i < bytes / sizeof(T); i += stride) {
        reinterpret_cast<T*>(dst)[i] = reinterpret_cast<const T*>(src)[i];
    }
    for (size_t i = bytes / sizeof(T) * sizeof(T) + tid; i < bytes; i += stride) {
        dst[i] = src[i];
    }
}

__global__ void cudaGraphPrepareKernel(PrepareTasks params) {
    if (params.count == 0) {
        return;
    }
    const auto& task = params.tasks[blockIdx.y];
    const auto& fill = task.fill;
    if (task.src == nullptr) {
        const uint32_t value = (fill.value_ptr ? static_cast<uint32_t>(*fill.value_ptr)
                                               : static_cast<uint32_t>(fill.value))
                               + static_cast<uint32_t>(fill.value_offset);
        for (size_t i = blockIdx.x * blockDim.x + threadIdx.x;
             i < static_cast<size_t>(fill.count); i += gridDim.x * blockDim.x) {
            fill.ptr[i] = static_cast<int32_t>(value + static_cast<uint32_t>(fill.step) * static_cast<uint32_t>(i));
        }
        return;
    }

    // A page-table task owns the entire destination: copy its live rectangle,
    // then clear only the uncovered columns and rows. No cross-block barrier or
    // clear-before-copy race is needed.
    const size_t rows = fill.count ? static_cast<size_t>(fill.count) * sizeof(int32_t) / task.dst_stride : task.rows;
    const size_t row_lanes = min(rows, static_cast<size_t>(gridDim.x));
    const size_t first_row = blockIdx.x % row_lanes;
    const size_t lane = blockIdx.x / row_lanes;
    const size_t lanes = (gridDim.x - 1 - first_row) / row_lanes + 1;
    for (size_t row = first_row; row < rows; row += row_lanes) {
        char* dst = reinterpret_cast<char*>(fill.ptr) + row * task.dst_stride;
        if (row < task.rows) {
            const char* src = task.src + row * task.src_stride;
            const uintptr_t alignment = reinterpret_cast<uintptr_t>(src) | reinterpret_cast<uintptr_t>(dst);
            if (alignment % alignof(int4) == 0) {
                prepareCopyRow<int4>(src, dst, task.row_bytes, lane, lanes);
            } else if (alignment % alignof(int32_t) == 0) {
                prepareCopyRow<int32_t>(src, dst, task.row_bytes, lane, lanes);
            } else {
                prepareCopyRow<char>(src, dst, task.row_bytes, lane, lanes);
            }
        }
        if (fill.count) {
            const size_t start = row < task.rows ? task.row_bytes / sizeof(int32_t) : 0;
            for (size_t i = start + lane * blockDim.x + threadIdx.x;
                 i < task.dst_stride / sizeof(int32_t); i += lanes * blockDim.x) {
                reinterpret_cast<int32_t*>(dst)[i] = 0;
            }
        }
    }
}

struct PrepareRange {
    uintptr_t begin = 0;
    uintptr_t end = 0;
    bool overlaps(PrepareRange other) const {
        return begin < other.end && other.begin < end;
    }
};

bool prepareRange(const void* ptr, size_t rows, size_t stride, size_t bytes, PrepareRange& range) {
    if (!ptr || !rows || !bytes || (rows > 1 && stride < bytes)) {
        return false;
    }
    range.begin = reinterpret_cast<uintptr_t>(ptr);
    if (bytes > std::numeric_limits<uintptr_t>::max() - range.begin
        || (rows - 1) > (std::numeric_limits<uintptr_t>::max() - range.begin - bytes) / std::max<size_t>(stride, 1)) {
        return false;
    }
    range.end = range.begin + (rows - 1) * stride + bytes;
    return true;
}

__global__ void cudaGraphPrepareFillKernel(CudaGraphPrepareFillParams params) {
    const int64_t tid    = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;

    for (int32_t region_idx = 0; region_idx < params.region_count; ++region_idx) {
        const auto region = params.regions[region_idx];
        if (region.ptr == nullptr || region.count <= 0) {
            continue;
        }
        const int32_t fill_value = (region.value_ptr != nullptr ? *region.value_ptr : region.value)
                                 + region.value_offset;
        for (int64_t i = tid; i < region.count; i += stride) {
            region.ptr[i] = fill_value + region.step * i;
        }
    }
}

__global__ void prepareFlashInferDecodeParamsKernel(const int32_t* sequence_lengths_plus_1,
                                                    const int32_t* block_ids,
                                                    int32_t*       batch_indice,
                                                    int32_t*       page_indice,
                                                    int32_t*       decode_page_indptr,
                                                    int32_t*       paged_kv_last_page_len,
                                                    int32_t*       qo_indptr,
                                                    int32_t*       kvlen,
                                                    int32_t*       positions,
                                                    int32_t        batch_size,
                                                    int32_t        max_blocks_per_batch,
                                                    int32_t        seq_size_per_block) {
    // Replay path is small-batch metadata; one CUDA block avoids any host prefix-sum.
    if (threadIdx.x != 0 || blockIdx.x != 0) {
        return;
    }

    int32_t page_offset        = 0;
    decode_page_indptr[0]      = 0;
    qo_indptr[0]               = 0;
    const int32_t safe_page_sz = seq_size_per_block > 0 ? seq_size_per_block : 1;

    for (int32_t batch = 0; batch < batch_size; ++batch) {
        const int32_t seq_len = sequence_lengths_plus_1[batch] > 1 ? sequence_lengths_plus_1[batch] : 1;
        const int32_t pages   = (seq_len + safe_page_sz - 1) / safe_page_sz;

        batch_indice[batch]           = batch;
        positions[batch]              = seq_len - 1;
        kvlen[batch]                  = seq_len;
        paged_kv_last_page_len[batch] = (seq_len - 1) % safe_page_sz + 1;

        const int32_t pages_to_copy = pages < max_blocks_per_batch ? pages : max_blocks_per_batch;
        for (int32_t page = 0; page < pages_to_copy; ++page) {
            page_indice[page_offset + page] = block_ids[batch * max_blocks_per_batch + page];
        }
        page_offset += pages_to_copy;
        decode_page_indptr[batch + 1] = page_offset;
        qo_indptr[batch + 1]          = batch + 1;
    }
}

}  // namespace

#if USING_CUDA
cudaGraphNode_t addCudaGraphPrepareNode(cudaGraph_t graph) {
    size_t count = 0;
    C10_CUDA_CHECK(cudaGraphGetRootNodes(graph, nullptr, &count));
    std::vector<cudaGraphNode_t> roots(count);
    C10_CUDA_CHECK(cudaGraphGetRootNodes(graph, roots.data(), &count));
    PrepareTasks tasks;
    void* arguments[] = {&tasks};
    cudaKernelNodeParams params{};
    params.func = reinterpret_cast<void*>(cudaGraphPrepareKernel);
    params.gridDim = dim3(1);
    params.blockDim = dim3(256);
    params.kernelParams = arguments;
    cudaGraphNode_t node = nullptr;
    C10_CUDA_CHECK(cudaGraphAddKernelNode(&node, graph, nullptr, 0, &params));
    for (auto root : roots) {
#if CUDART_VERSION >= 13000
        C10_CUDA_CHECK(cudaGraphAddDependencies(graph, &node, &root, nullptr, 1));
#else
        C10_CUDA_CHECK(cudaGraphAddDependencies(graph, &node, &root, 1));
#endif
    }
    return node;
}

void disableCudaGraphPrepareReplay(const CudaGraphPrepareReplay& replay) {
    if (replay.exec && replay.node) {
        C10_CUDA_CHECK(cudaGraphNodeSetEnabled(replay.exec, replay.node, 0));
    }
}
#endif

bool tryInvokeCudaGraphPrepare(const CudaGraphPrepareFillParams& fills,
                               const FusedD2DCopyParams& copies,
                               const FusedStridedCopyParams& strided_copies,
                               CudaGraphPrepareStream stream,
                               const CudaGraphPrepareReplay& replay) {
    if (fills.region_count < 0 || fills.region_count > kMaxCudaGraphPrepareFillRegions
        || copies.num_copies < 0 || copies.num_copies > MAX_FUSED_D2D_COPIES
        || strided_copies.num_copies < 0 || strided_copies.num_copies > MAX_FUSED_STRIDED_COPIES) {
        return false;
    }
    PrepareTasks params;
    PrepareRange reads[kMaxCudaGraphPrepareFillRegions], writes[kMaxCudaGraphPrepareFillRegions];
    bool used_fill[kMaxCudaGraphPrepareFillRegions] = {};
    auto addCopy = [&](const void* src, void* dst, size_t rows, size_t bytes, size_t src_stride, size_t dst_stride) {
        if (!rows || !bytes) {
            return true;
        }
        if (params.count == kMaxCudaGraphPrepareFillRegions) {
            return false;
        }
        const int index = params.count++;
        auto& task = params.tasks[index];
        task.fill.ptr = static_cast<int32_t*>(dst);
        task.src = static_cast<const char*>(src);
        task.rows = rows;
        task.row_bytes = bytes;
        task.src_stride = src_stride;
        task.dst_stride = dst_stride;
        return prepareRange(src, rows, src_stride, bytes, reads[index])
               && prepareRange(dst, rows, dst_stride, bytes, writes[index]);
    };
    for (int i = 0; i < copies.num_copies; ++i) {
        if (!addCopy(copies.src[i], copies.dst[i], 1, copies.size[i], copies.size[i], copies.size[i])) {
            return false;
        }
    }
    for (int i = 0; i < strided_copies.num_copies; ++i) {
        if (!addCopy(strided_copies.src[i], strided_copies.dst[i], strided_copies.num_rows[i],
                     strided_copies.row_bytes[i], strided_copies.src_row_stride[i], strided_copies.dst_row_stride[i])) {
            return false;
        }
    }
    for (int i = 0; i < fills.region_count; ++i) {
        const auto& fill = fills.regions[i];
        if (fill.count <= 0) {
            continue;
        }
        PrepareRange range;
        if (static_cast<uint64_t>(fill.count) > std::numeric_limits<size_t>::max() / sizeof(int32_t)
            || !prepareRange(fill.ptr, 1, 0, fill.count * sizeof(int32_t), range)) {
            return false;
        }
        for (int j = 0; j < params.count; ++j) {
            auto& task = params.tasks[j];
            if (task.src && task.fill.ptr == fill.ptr && task.fill.count == 0
                && !fill.value_ptr && fill.value == 0 && fill.value_offset == 0 && fill.step == 0
                && task.dst_stride && task.dst_stride % sizeof(int32_t) == 0
                && task.row_bytes % sizeof(int32_t) == 0 && task.row_bytes <= task.dst_stride
                && (range.end - range.begin) % task.dst_stride == 0
                && (range.end - range.begin) / task.dst_stride >= task.rows) {
                task.fill = fill;
                writes[j] = range;
                used_fill[i] = true;
                break;
            }
        }
        if (!used_fill[i]) {
            if (params.count == kMaxCudaGraphPrepareFillRegions) {
                return false;
            }
            writes[params.count] = range;
            params.tasks[params.count++].fill = fill;
        }
    }
    // Independent destinations are required. Conservatively retain the old
    // ordered launches for aliases, including device-sourced fill values.
    size_t max_bytes = 0;
    for (int i = 0; i < params.count; ++i) {
        max_bytes = std::max(max_bytes, writes[i].end - writes[i].begin);
        PrepareRange value;
        const auto* value_ptr = params.tasks[i].fill.value_ptr;
        if (value_ptr && !prepareRange(value_ptr, 1, 0, sizeof(int32_t), value)) {
            return false;
        }
        for (int j = 0; j < params.count; ++j) {
            if ((i != j && writes[i].overlaps(writes[j])) || reads[i].overlaps(writes[j])
                || (value_ptr && value.overlaps(writes[j]))) {
                return false;
            }
        }
    }
    if (params.count) {
        const auto blocks = std::min<size_t>(64, std::max<size_t>(8, 1 + (max_bytes - 1) / 4096));
#if USING_CUDA
        if (replay.exec && replay.node) {
            void* arguments[] = {&params};
            cudaKernelNodeParams node_params{};
            node_params.func = reinterpret_cast<void*>(cudaGraphPrepareKernel);
            node_params.gridDim = dim3(blocks, params.count);
            node_params.blockDim = dim3(256);
            node_params.kernelParams = arguments;
            C10_CUDA_CHECK(cudaGraphExecKernelNodeSetParams(replay.exec, replay.node, &node_params));
            C10_CUDA_CHECK(cudaGraphNodeSetEnabled(replay.exec, replay.node, 1));
            return true;
        }
#else
        (void)replay;
#endif
        cudaGraphPrepareKernel<<<dim3(blocks, params.count), 256, 0, stream>>>(params);
#if USING_ROCM
        const auto result = hipGetLastError();
        TORCH_CHECK(result == hipSuccess, "cuda graph preparation failed: ", hipGetErrorString(result));
#else
        const auto result = cudaGetLastError();
        TORCH_CHECK(result == cudaSuccess, "cuda graph preparation failed: ", cudaGetErrorString(result));
#endif
    }
    return true;
}

void invokeCudaGraphPrepareFill(CudaGraphPrepareFillParams params, CudaGraphPrepareStream stream) {
    TORCH_CHECK(params.region_count >= 0 && params.region_count <= kMaxCudaGraphPrepareFillRegions,
                "invalid cuda graph prepare fill region count: ",
                params.region_count);

    int64_t total_count = 0;
    for (int32_t i = 0; i < params.region_count; ++i) {
        total_count += params.regions[i].count > 0 ? params.regions[i].count : 0;
    }
    if (total_count <= 0) {
        return;
    }

    constexpr int block_size = 256;
    const int     blocks     = static_cast<int>(std::min<int64_t>((total_count + block_size - 1) / block_size, 1024));
    cudaGraphPrepareFillKernel<<<blocks, block_size, 0, stream>>>(params);
#if USING_ROCM
    const auto result = hipGetLastError();
    TORCH_CHECK(result == hipSuccess, "cuda graph prepare fill kernel failed: ", hipGetErrorString(result));
#else
    const auto result = cudaGetLastError();
    TORCH_CHECK(result == cudaSuccess, "cuda graph prepare fill kernel failed: ", cudaGetErrorString(result));
#endif
}

void invokePrepareFlashInferDecodeParams(const int32_t* sequence_lengths_plus_1,
                                         const int32_t* block_ids,
                                         int32_t*       batch_indice,
                                         int32_t*       page_indice,
                                         int32_t*       decode_page_indptr,
                                         int32_t*       paged_kv_last_page_len,
                                         int32_t*       qo_indptr,
                                         int32_t*       kvlen,
                                         int32_t*       positions,
                                         int32_t        batch_size,
                                         int32_t        max_blocks_per_batch,
                                         int32_t        seq_size_per_block,
                                         CudaGraphPrepareStream stream) {
    TORCH_CHECK(sequence_lengths_plus_1 != nullptr, "sequence_lengths_plus_1 is null");
    TORCH_CHECK(block_ids != nullptr, "block_ids is null");
    TORCH_CHECK(batch_indice != nullptr && page_indice != nullptr && decode_page_indptr != nullptr
                    && paged_kv_last_page_len != nullptr && qo_indptr != nullptr && kvlen != nullptr
                    && positions != nullptr,
                "FlashInfer decode metadata output buffer is null");
    if (batch_size <= 0 || max_blocks_per_batch <= 0) {
        return;
    }
    prepareFlashInferDecodeParamsKernel<<<1, 1, 0, stream>>>(sequence_lengths_plus_1,
                                                             block_ids,
                                                             batch_indice,
                                                             page_indice,
                                                             decode_page_indptr,
                                                             paged_kv_last_page_len,
                                                             qo_indptr,
                                                             kvlen,
                                                             positions,
                                                             batch_size,
                                                             max_blocks_per_batch,
                                                             seq_size_per_block);
#if USING_ROCM
    const auto result = hipGetLastError();
    TORCH_CHECK(result == hipSuccess,
                "FlashInfer decode CUDA graph prepare kernel failed: ",
                hipGetErrorString(result));
#else
    const auto result = cudaGetLastError();
    TORCH_CHECK(
        result == cudaSuccess, "FlashInfer decode CUDA graph prepare kernel failed: ", cudaGetErrorString(result));
#endif
}

}  // namespace rtp_llm
