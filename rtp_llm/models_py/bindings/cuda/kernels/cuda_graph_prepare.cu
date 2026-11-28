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
    const char*                src        = nullptr;
    size_t                     rows       = 0;
    size_t                     row_bytes  = 0;
    size_t                     src_stride = 0;
    size_t                     dst_stride = 0;
};

struct PrepareTasks {
    int         count = 0;
    PrepareTask tasks[kMaxCudaGraphPrepareFillRegions];
};
static_assert(sizeof(PrepareTasks) < 4096, "keep compatibility with the legacy kernel parameter limit");

template<typename T>
__device__ void prepareCopyRow(const char* src, char* dst, size_t bytes, size_t lane, size_t lanes) {
    const size_t tid    = lane * blockDim.x + threadIdx.x;
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
        const uint32_t value =
            (fill.value_ptr ? static_cast<uint32_t>(*fill.value_ptr) : static_cast<uint32_t>(fill.value))
            + static_cast<uint32_t>(fill.value_offset);
        for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < static_cast<size_t>(fill.count);
             i += gridDim.x * blockDim.x) {
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
    const size_t lane      = blockIdx.x / row_lanes;
    const size_t lanes     = (gridDim.x - 1 - first_row) / row_lanes + 1;
    for (size_t row = first_row; row < rows; row += row_lanes) {
        char* dst = reinterpret_cast<char*>(fill.ptr) + row * task.dst_stride;
        if (row < task.rows) {
            const char*     src       = task.src + row * task.src_stride;
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
            for (size_t i = start + lane * blockDim.x + threadIdx.x; i < task.dst_stride / sizeof(int32_t);
                 i += lanes * blockDim.x) {
                reinterpret_cast<int32_t*>(dst)[i] = 0;
            }
        }
    }
}

struct PrepareRange {
    uintptr_t begin = 0;
    uintptr_t end   = 0;
    bool      overlaps(PrepareRange other) const {
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

#if USING_ROCM
__global__ void cudaGraphPrepareFillKernel(CudaGraphPrepareFillParams params) {
    const int64_t tid    = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;

    for (int32_t region_idx = 0; region_idx < params.region_count; ++region_idx) {
        const auto region = params.regions[region_idx];
        if (region.ptr == nullptr || region.count <= 0) {
            continue;
        }
        const int32_t fill_value =
            (region.value_ptr != nullptr ? *region.value_ptr : region.value) + region.value_offset;
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

#endif

dim3 prepareGrid(const PrepareTasks& params, size_t max_bytes) {
    const auto blocks = std::min<size_t>(64, std::max<size_t>(8, 1 + (max_bytes - 1) / 4096));
    return dim3(blocks, params.count);
}

#if USING_CUDA
void updatePrepareNode(cudaGraphExec_t exec, cudaGraphNode_t node, PrepareTasks& params, size_t max_bytes) {
    if (params.count) {
        void*                arguments[] = {&params};
        cudaKernelNodeParams node_params{};
        node_params.func         = reinterpret_cast<void*>(cudaGraphPrepareKernel);
        node_params.gridDim      = prepareGrid(params, max_bytes);
        node_params.blockDim     = dim3(256);
        node_params.kernelParams = arguments;
        C10_CUDA_CHECK(cudaGraphExecKernelNodeSetParams(exec, node, &node_params));
    }
    C10_CUDA_CHECK(cudaGraphNodeSetEnabled(exec, node, params.count != 0));
}
#endif

}  // namespace

#if USING_CUDA
cudaGraphNode_t addIndependentCudaGraphPrepareNode(cudaGraph_t graph) {
    PrepareTasks         tasks;
    void*                arguments[] = {&tasks};
    cudaKernelNodeParams params{};
    params.func          = reinterpret_cast<void*>(cudaGraphPrepareKernel);
    params.gridDim       = dim3(1);
    params.blockDim      = dim3(256);
    params.kernelParams  = arguments;
    cudaGraphNode_t node = nullptr;
    C10_CUDA_CHECK(cudaGraphAddKernelNode(&node, graph, nullptr, 0, &params));
    return node;
}

cudaGraphNode_t addCudaGraphPrepareNode(cudaGraph_t graph, cudaGraphNode_t independent_root) {
    size_t count = 0;
    C10_CUDA_CHECK(cudaGraphGetRootNodes(graph, nullptr, &count));
    std::vector<cudaGraphNode_t> roots(count);
    C10_CUDA_CHECK(cudaGraphGetRootNodes(graph, roots.data(), &count));
    const auto node = addIndependentCudaGraphPrepareNode(graph);
    for (auto root : roots) {
        if (root == independent_root) {
            continue;
        }
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
        if (replay.windows_node) {
            C10_CUDA_CHECK(cudaGraphNodeSetEnabled(replay.exec, replay.windows_node, 0));
        }
    }
}
#endif

bool tryInvokeCudaGraphPrepare(const CudaGraphPrepareFillParams& fills,
                               const FusedD2DCopyParams&         copies,
                               const FusedStridedCopyParams&     strided_copies,
                               CudaGraphPrepareStream            stream,
                               const CudaGraphPrepareReplay&     replay) {
    if (fills.region_count < 0 || fills.region_count > kMaxCudaGraphPrepareFillRegions || copies.num_copies < 0
        || copies.num_copies > MAX_FUSED_D2D_COPIES || strided_copies.num_copies < 0
        || strided_copies.num_copies > MAX_FUSED_STRIDED_COPIES) {
        return false;
    }
    PrepareTasks params;
    PrepareRange reads[kMaxCudaGraphPrepareFillRegions], writes[kMaxCudaGraphPrepareFillRegions];
    bool         used_fill[kMaxCudaGraphPrepareFillRegions] = {};
    auto addCopy = [&](const void* src, void* dst, size_t rows, size_t bytes, size_t src_stride, size_t dst_stride) {
        if (!rows || !bytes) {
            return true;
        }
        if (params.count == kMaxCudaGraphPrepareFillRegions) {
            return false;
        }
        const int index = params.count++;
        auto&     task  = params.tasks[index];
        task.fill.ptr   = static_cast<int32_t*>(dst);
        task.src        = static_cast<const char*>(src);
        task.rows       = rows;
        task.row_bytes  = bytes;
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
        if (!addCopy(strided_copies.src[i],
                     strided_copies.dst[i],
                     strided_copies.num_rows[i],
                     strided_copies.row_bytes[i],
                     strided_copies.src_row_stride[i],
                     strided_copies.dst_row_stride[i])) {
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
            if (task.src && task.fill.ptr == fill.ptr && task.fill.count == 0 && !fill.value_ptr && fill.value == 0
                && fill.value_offset == 0 && fill.step == 0 && task.dst_stride && task.dst_stride % sizeof(int32_t) == 0
                && task.row_bytes % sizeof(int32_t) == 0 && task.row_bytes <= task.dst_stride
                && (range.end - range.begin) % task.dst_stride == 0
                && (range.end - range.begin) / task.dst_stride >= task.rows) {
                task.fill    = fill;
                writes[j]    = range;
                used_fill[i] = true;
                break;
            }
        }
        if (!used_fill[i]) {
            if (params.count == kMaxCudaGraphPrepareFillRegions) {
                return false;
            }
            writes[params.count]              = range;
            params.tasks[params.count++].fill = fill;
        }
    }
    // Independent destinations are required. Conservatively retain the old
    // ordered launches for aliases, including device-sourced fill values.
    size_t max_bytes = 0;
    for (int i = 0; i < params.count; ++i) {
        max_bytes = std::max(max_bytes, writes[i].end - writes[i].begin);
        PrepareRange value;
        const auto*  value_ptr = params.tasks[i].fill.value_ptr;
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
#if USING_CUDA
    if (replay.exec && replay.node) {
        if (replay.windows_node) {
            // Only split after validating all cross-task aliases. Both the
            // live copy and the -1 tail fill belong before the window hash.
            PrepareRange window_range;
            if (!prepareRange(replay.windows_dst, 1, 0, replay.windows_bytes, window_range)) {
                return false;
            }
            PrepareTasks window_tasks;
            size_t       window_bytes = 0;
            int          other_count  = 0;
            for (int i = 0; i < params.count; ++i) {
                if (writes[i].overlaps(window_range)) {
                    if (writes[i].begin < window_range.begin || writes[i].end > window_range.end) {
                        return false;
                    }
                    window_tasks.tasks[window_tasks.count++] = params.tasks[i];
                    window_bytes                             = std::max(window_bytes, writes[i].end - writes[i].begin);
                } else {
                    params.tasks[other_count++] = params.tasks[i];
                }
            }
            params.count = other_count;
            updatePrepareNode(replay.exec, replay.windows_node, window_tasks, window_bytes);
        }
        updatePrepareNode(replay.exec, replay.node, params, max_bytes);
        return true;
    }
#else
    (void)replay;
#endif
    if (params.count) {
        cudaGraphPrepareKernel<<<prepareGrid(params, max_bytes), 256, 0, stream>>>(params);
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

#if USING_ROCM
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

void invokePrepareFlashInferDecodeParams(const int32_t*         sequence_lengths_plus_1,
                                         const int32_t*         block_ids,
                                         int32_t*               batch_indice,
                                         int32_t*               page_indice,
                                         int32_t*               decode_page_indptr,
                                         int32_t*               paged_kv_last_page_len,
                                         int32_t*               qo_indptr,
                                         int32_t*               kvlen,
                                         int32_t*               positions,
                                         int32_t                batch_size,
                                         int32_t                max_blocks_per_batch,
                                         int32_t                seq_size_per_block,
                                         CudaGraphPrepareStream stream) {
    if (batch_size <= 0 || max_blocks_per_batch <= 0) {
        return;
    }
    TORCH_CHECK(sequence_lengths_plus_1 != nullptr, "sequence_lengths_plus_1 is null");
    TORCH_CHECK(block_ids != nullptr, "block_ids is null");
    TORCH_CHECK(batch_indice != nullptr && page_indice != nullptr && decode_page_indptr != nullptr
                    && paged_kv_last_page_len != nullptr && qo_indptr != nullptr && kvlen != nullptr
                    && positions != nullptr,
                "FlashInfer decode metadata output buffer is null");
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
    TORCH_CHECK(
        result == hipSuccess, "FlashInfer decode CUDA graph prepare kernel failed: ", hipGetErrorString(result));
#else
    const auto result = cudaGetLastError();
    TORCH_CHECK(
        result == cudaSuccess, "FlashInfer decode CUDA graph prepare kernel failed: ", cudaGetErrorString(result));
#endif
}

#endif

}  // namespace rtp_llm

#if !USING_ROCM
#include "rtp_llm/models_py/bindings/cuda/kernels/cuda_graph_prepare.h"

#include <algorithm>
#include <c10/util/Exception.h>
#include <cub/cub.cuh>
#include <cuda_runtime.h>

namespace rtp_llm {

namespace {

__global__ void cudaGraphPrepareFillKernel(CudaGraphPrepareFillParams params) {
    const int64_t tid    = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;

    for (int32_t region_idx = 0; region_idx < params.region_count; ++region_idx) {
        const auto region = params.regions[region_idx];
        if (region.ptr == nullptr || region.count <= 0) {
            continue;
        }
        for (int64_t i = tid; i < region.count; i += stride) {
            const int32_t fill_value =
                (region.value_ptr != nullptr ? *region.value_ptr : region.value) + region.value_offset;
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
                                                    int64_t*       slot_mapping,
                                                    int32_t        batch_size,
                                                    int32_t        max_blocks_per_batch,
                                                    int32_t        seq_size_per_block,
                                                    int32_t        captured_batch_capacity) {
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
        const int32_t block_index     = (seq_len - 1) / safe_page_sz;
        const int32_t block_offset    = (seq_len - 1) % safe_page_sz;
        const int32_t block_number =
            block_index < max_blocks_per_batch ? block_ids[batch * max_blocks_per_batch + block_index] : 0;
        if (slot_mapping != nullptr)
            slot_mapping[batch] =
                static_cast<int64_t>(block_number) * safe_page_sz + static_cast<int64_t>(block_offset);

        const int32_t pages_to_copy = pages < max_blocks_per_batch ? pages : max_blocks_per_batch;
        for (int32_t page = 0; page < pages_to_copy; ++page) {
            page_indice[page_offset + page] = block_ids[batch * max_blocks_per_batch + page];
        }
        page_offset += pages_to_copy;
        decode_page_indptr[batch + 1] = page_offset;
        qo_indptr[batch + 1]          = batch + 1;
    }

    // Decode CUDA graph replay can use a graph captured for a larger batch
    // than the current live batch. Clear stale entries so the captured kernels
    // do not process phantom rows with old kvlen/page metadata and block_id=0.
    for (int32_t batch = batch_size; batch < captured_batch_capacity; ++batch) {
        batch_indice[batch]           = 0;
        positions[batch]              = 0;
        kvlen[batch]                  = 0;
        paged_kv_last_page_len[batch] = 0;
        if (slot_mapping != nullptr)
            slot_mapping[batch] = -1;
        decode_page_indptr[batch + 1] = page_offset;
        qo_indptr[batch + 1]          = batch_size;
    }
}

// =====================================================================
// Optimized V1 decode kernel — single block + CUB BlockScan + warp-stride
// Used when batch_size <= 256.
// =====================================================================

constexpr int kDecodeV1Threads = 256;
constexpr int kWarpSize        = 32;

template<int BLOCK_THREADS>
__global__ void prepareFlashInferDecodeParamsKernelV1(const int32_t* __restrict__ sequence_lengths_plus_1,
                                                      const int32_t* __restrict__ block_ids,
                                                      int32_t* __restrict__ batch_indice,
                                                      int32_t* __restrict__ page_indice,
                                                      int32_t* __restrict__ decode_page_indptr,
                                                      int32_t* __restrict__ paged_kv_last_page_len,
                                                      int32_t* __restrict__ qo_indptr,
                                                      int32_t* __restrict__ kvlen,
                                                      int32_t* __restrict__ positions,
                                                      int64_t* __restrict__ slot_mapping,
                                                      int32_t batch_size,
                                                      int32_t max_blocks_per_batch,
                                                      int32_t seq_size_per_block,
                                                      int32_t captured_batch_capacity) {
    using BlockScan = cub::BlockScan<int, BLOCK_THREADS>;

    __shared__ typename BlockScan::TempStorage scan_storage;
    __shared__ int32_t                         smem_seq_len[BLOCK_THREADS];
    __shared__ int32_t                         smem_pages[BLOCK_THREADS];
    __shared__ int32_t                         smem_page_prefix[BLOCK_THREADS];
    __shared__ int32_t                         smem_total_pages;

    const int     tid          = static_cast<int>(threadIdx.x);
    const int32_t safe_page_sz = seq_size_per_block > 0 ? seq_size_per_block : 1;

    int32_t seq_len = 0;
    int32_t pages   = 0;
    if (tid < batch_size) {
        const int32_t raw_seq   = sequence_lengths_plus_1[tid];
        seq_len                 = raw_seq > 1 ? raw_seq : 1;
        const int32_t raw_pages = (seq_len + safe_page_sz - 1) / safe_page_sz;
        pages                   = raw_pages < max_blocks_per_batch ? raw_pages : max_blocks_per_batch;
        smem_seq_len[tid]       = seq_len;
        smem_pages[tid]         = pages;
    } else {
        smem_seq_len[tid] = 0;
        smem_pages[tid]   = 0;
    }
    __syncthreads();

    int32_t page_prefix = 0;
    int32_t page_total  = 0;
    BlockScan(scan_storage).ExclusiveSum(pages, page_prefix, page_total);
    smem_page_prefix[tid] = page_prefix;
    if (tid == 0) {
        smem_total_pages = page_total;
    }
    __syncthreads();

    if (tid == 0) {
        decode_page_indptr[0] = 0;
        qo_indptr[0]          = 0;
    }
    if (tid < batch_size) {
        const int32_t cur_seq      = smem_seq_len[tid];
        const int32_t cur_pages    = smem_pages[tid];
        const int32_t pos          = cur_seq - 1;
        const int32_t block_index  = pos / safe_page_sz;
        const int32_t block_offset = pos % safe_page_sz;
        const int32_t block_number =
            block_index < max_blocks_per_batch ? block_ids[tid * max_blocks_per_batch + block_index] : 0;

        batch_indice[tid]           = tid;
        positions[tid]              = pos;
        kvlen[tid]                  = cur_seq;
        paged_kv_last_page_len[tid] = (cur_seq - 1) % safe_page_sz + 1;
        if (slot_mapping != nullptr)
            slot_mapping[tid] = static_cast<int64_t>(block_number) * safe_page_sz + static_cast<int64_t>(block_offset);
        decode_page_indptr[tid + 1] = page_prefix + cur_pages;
        qo_indptr[tid + 1]          = tid + 1;
    }

    const int warp_id = tid / kWarpSize;
    const int lane_id = tid % kWarpSize;
    const int n_warps = BLOCK_THREADS / kWarpSize;
    for (int b = warp_id; b < batch_size; b += n_warps) {
        const int32_t prefix = smem_page_prefix[b];
        const int32_t cnt    = smem_pages[b];
        for (int32_t p = lane_id; p < cnt; p += kWarpSize) {
            page_indice[prefix + p] = block_ids[b * max_blocks_per_batch + p];
        }
    }

    const int32_t total_pages = smem_total_pages;
    for (int b = batch_size + tid; b < captured_batch_capacity; b += BLOCK_THREADS) {
        batch_indice[b]           = 0;
        positions[b]              = 0;
        kvlen[b]                  = 0;
        paged_kv_last_page_len[b] = 0;
        if (slot_mapping != nullptr)
            slot_mapping[b] = -1;
        decode_page_indptr[b + 1] = total_pages;
        qo_indptr[b + 1]          = batch_size;
    }
}

// =====================================================================
// Optimized V3 target-verify / prefill kernels — warp-shuffle cumsum
// + warp-per-batch writes.
// No batch_size limitation, no CUB dependency.
// =====================================================================

constexpr int kV3WarpSize       = 32;
constexpr int kV3WarpsPerBlock  = 8;
constexpr int kV3BlockThreads   = kV3WarpSize * kV3WarpsPerBlock;
constexpr int kV3CumsumWarpSize = 32;

__device__ __forceinline__ int32_t warpInclusiveScan(int32_t v) {
    const unsigned mask = 0xffffffffu;
#pragma unroll
    for (int offset = 1; offset < kV3CumsumWarpSize; offset <<= 1) {
        int32_t n = __shfl_up_sync(mask, v, offset);
        if (static_cast<int>(threadIdx.x) >= offset) {
            v += n;
        }
    }
    return v;
}

__global__ void prepareTargetVerifyCumsumKernel(const int32_t* __restrict__ input_lengths,
                                                const int32_t* __restrict__ prefix_lengths,
                                                int32_t* __restrict__ decode_page_indptr,
                                                int32_t* __restrict__ paged_kv_last_page_len,
                                                int32_t* __restrict__ qo_indptr,
                                                int32_t* __restrict__ prefill_ragged_kv_len_indptr,
                                                int32_t* __restrict__ kvlen,
                                                int32_t batch_size,
                                                int32_t safe_page_sz,
                                                int32_t max_blocks_per_batch,
                                                int32_t captured_batch_capacity) {
    if (blockIdx.x != 0 || threadIdx.x >= kV3CumsumWarpSize) {
        return;
    }

    if (threadIdx.x == 0) {
        decode_page_indptr[0]           = 0;
        qo_indptr[0]                    = 0;
        prefill_ragged_kv_len_indptr[0] = 0;
    }

    int32_t        running_token_offset = 0;
    int32_t        running_page_offset  = 0;
    int32_t        running_kv_offset    = 0;
    const unsigned full_mask            = 0xffffffffu;

    for (int32_t chunk_start = 0; chunk_start < batch_size; chunk_start += kV3CumsumWarpSize) {
        const int32_t lane = static_cast<int32_t>(threadIdx.x);
        const int32_t bid  = chunk_start + lane;
        const bool    live = bid < batch_size;

        int32_t input_len   = 0;
        int32_t kv_len      = 0;
        int32_t pages       = 0;
        int32_t last_pg_len = 0;
        if (live) {
            input_len               = input_lengths[bid];
            const int32_t pfx       = prefix_lengths[bid];
            kv_len                  = input_len + pfx;
            const int32_t pages_raw = (kv_len + safe_page_sz - 1) / safe_page_sz;
            pages                   = pages_raw < max_blocks_per_batch ? pages_raw : max_blocks_per_batch;
            last_pg_len             = (kv_len - 1) % safe_page_sz + 1;
        }

        const int32_t tok_inclusive = warpInclusiveScan(input_len);
        const int32_t pg_inclusive  = warpInclusiveScan(pages);
        const int32_t kv_inclusive  = warpInclusiveScan(kv_len);

        const int32_t tok_chunk_total = __shfl_sync(full_mask, tok_inclusive, kV3CumsumWarpSize - 1);
        const int32_t pg_chunk_total  = __shfl_sync(full_mask, pg_inclusive, kV3CumsumWarpSize - 1);
        const int32_t kv_chunk_total  = __shfl_sync(full_mask, kv_inclusive, kV3CumsumWarpSize - 1);

        if (live) {
            qo_indptr[bid + 1]                    = running_token_offset + tok_inclusive;
            decode_page_indptr[bid + 1]           = running_page_offset + pg_inclusive;
            prefill_ragged_kv_len_indptr[bid + 1] = running_kv_offset + kv_inclusive;
            kvlen[bid]                            = kv_len;
            paged_kv_last_page_len[bid]           = last_pg_len;
        }
        running_token_offset += tok_chunk_total;
        running_page_offset += pg_chunk_total;
        running_kv_offset += kv_chunk_total;
    }

    const int32_t tail_tokens = running_token_offset;
    const int32_t tail_pages  = running_page_offset;
    const int32_t tail_kv     = running_kv_offset;
    for (int32_t b = batch_size + static_cast<int32_t>(threadIdx.x); b < captured_batch_capacity;
         b += kV3CumsumWarpSize) {
        kvlen[b]                            = 0;
        paged_kv_last_page_len[b]           = 0;
        decode_page_indptr[b + 1]           = tail_pages;
        qo_indptr[b + 1]                    = tail_tokens;
        prefill_ragged_kv_len_indptr[b + 1] = tail_kv;
    }
}

__global__ void prepareTargetVerifyWriteKernel(const int32_t* __restrict__ input_lengths,
                                               const int32_t* __restrict__ prefix_lengths,
                                               const int32_t* __restrict__ block_ids,
                                               const int32_t* __restrict__ decode_page_indptr,
                                               const int32_t* __restrict__ qo_indptr,
                                               const int32_t* __restrict__ prefill_ragged_kv_len_indptr,
                                               int32_t* __restrict__ batch_indice,
                                               int32_t* __restrict__ page_indice,
                                               int32_t* __restrict__ positions,
                                               int64_t* __restrict__ slot_mapping,
                                               int32_t* __restrict__ expanded_seq_lens,
                                               int32_t* __restrict__ topk_indices_offset,
                                               int32_t* __restrict__ ks,
                                               int32_t* __restrict__ ke,
                                               int32_t batch_size,
                                               int32_t max_blocks_per_batch,
                                               int32_t safe_page_sz,
                                               int32_t captured_total_tokens) {
    const int32_t warp_id_in_block = static_cast<int32_t>(threadIdx.x) / kV3WarpSize;
    const int32_t lane             = static_cast<int32_t>(threadIdx.x) % kV3WarpSize;
    const int32_t global_warp_id   = static_cast<int32_t>(blockIdx.x) * kV3WarpsPerBlock + warp_id_in_block;
    const int32_t total_warps      = static_cast<int32_t>(gridDim.x) * kV3WarpsPerBlock;

    const bool has_sparse = (expanded_seq_lens != nullptr);

    for (int32_t bid = global_warp_id; bid < batch_size; bid += total_warps) {
        const int32_t input_len   = input_lengths[bid];
        const int32_t prefix_len  = prefix_lengths[bid];
        const int32_t kv_len      = input_len + prefix_len;
        const int32_t token_start = qo_indptr[bid];
        const int32_t page_start  = decode_page_indptr[bid];
        const int32_t k_off_base  = has_sparse ? prefill_ragged_kv_len_indptr[bid] : 0;

        const int32_t pages_raw     = (kv_len + safe_page_sz - 1) / safe_page_sz;
        const int32_t pages_to_copy = pages_raw < max_blocks_per_batch ? pages_raw : max_blocks_per_batch;

        for (int32_t j = lane; j < input_len; j += kV3WarpSize) {
            const int32_t t        = token_start + j;
            const int32_t position = j + prefix_len;
            batch_indice[t]        = bid;
            positions[t]           = position;

            const int32_t block_index  = position / safe_page_sz;
            const int32_t block_offset = position % safe_page_sz;
            const int32_t block_number =
                block_index < max_blocks_per_batch ? block_ids[bid * max_blocks_per_batch + block_index] : 0;
            slot_mapping[t] = static_cast<int64_t>(block_number) * static_cast<int64_t>(safe_page_sz)
                              + static_cast<int64_t>(block_offset);

            if (has_sparse) {
                const int32_t seq_len_value = kv_len - input_len + 1 + j;
                expanded_seq_lens[t]        = seq_len_value;
                topk_indices_offset[t]      = 0;
                ks[t]                       = k_off_base;
                ke[t]                       = k_off_base + seq_len_value;
            }
        }

        for (int32_t p = lane; p < pages_to_copy; p += kV3WarpSize) {
            page_indice[page_start + p] = block_ids[bid * max_blocks_per_batch + p];
        }
    }

    // Tail-fill stale token entries. Read live_total_tokens from qo_indptr[batch_size]
    // (written by cumsum kernel on the same stream — ordering guaranteed).
    const int32_t live_total_tokens = qo_indptr[batch_size];
    const int32_t global_tid = static_cast<int32_t>(blockIdx.x) * kV3BlockThreads + static_cast<int32_t>(threadIdx.x);
    const int32_t total_tids = static_cast<int32_t>(gridDim.x) * kV3BlockThreads;
    for (int32_t t = live_total_tokens + global_tid; t < captured_total_tokens; t += total_tids) {
        batch_indice[t] = 0;
        positions[t]    = 0;
        if (slot_mapping != nullptr) {
            if (slot_mapping != nullptr)
                slot_mapping[t] = -1;
        }
        if (has_sparse) {
            expanded_seq_lens[t]   = 0;
            topk_indices_offset[t] = 0;
            ks[t]                  = 0;
            ke[t]                  = 0;
        }
    }
}

}  // namespace

void invokeCudaGraphPrepareFill(CudaGraphPrepareFillParams params, cudaStream_t stream) {
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
    const auto result = cudaGetLastError();
    TORCH_CHECK(result == cudaSuccess, "cuda graph prepare fill kernel failed: ", cudaGetErrorString(result));
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
                                         int64_t*       slot_mapping,
                                         int32_t        batch_size,
                                         int32_t        max_blocks_per_batch,
                                         int32_t        seq_size_per_block,
                                         int32_t        captured_batch_capacity,
                                         cudaStream_t   stream) {
    if (batch_size <= 0 || max_blocks_per_batch <= 0) {
        return;
    }
    TORCH_CHECK(sequence_lengths_plus_1 != nullptr, "sequence_lengths_plus_1 is null");
    TORCH_CHECK(block_ids != nullptr, "block_ids is null");
    TORCH_CHECK(batch_indice != nullptr && page_indice != nullptr && decode_page_indptr != nullptr
                    && paged_kv_last_page_len != nullptr && qo_indptr != nullptr && kvlen != nullptr
                    && positions != nullptr,
                "FlashInfer decode metadata output buffer is null");
    if (batch_size <= kDecodeV1Threads) {
        prepareFlashInferDecodeParamsKernelV1<kDecodeV1Threads>
            <<<1, kDecodeV1Threads, 0, stream>>>(sequence_lengths_plus_1,
                                                 block_ids,
                                                 batch_indice,
                                                 page_indice,
                                                 decode_page_indptr,
                                                 paged_kv_last_page_len,
                                                 qo_indptr,
                                                 kvlen,
                                                 positions,
                                                 slot_mapping,
                                                 batch_size,
                                                 max_blocks_per_batch,
                                                 seq_size_per_block,
                                                 captured_batch_capacity);
    } else {
        prepareFlashInferDecodeParamsKernel<<<1, 1, 0, stream>>>(sequence_lengths_plus_1,
                                                                 block_ids,
                                                                 batch_indice,
                                                                 page_indice,
                                                                 decode_page_indptr,
                                                                 paged_kv_last_page_len,
                                                                 qo_indptr,
                                                                 kvlen,
                                                                 positions,
                                                                 slot_mapping,
                                                                 batch_size,
                                                                 max_blocks_per_batch,
                                                                 seq_size_per_block,
                                                                 captured_batch_capacity);
    }
    const auto result = cudaGetLastError();
    TORCH_CHECK(
        result == cudaSuccess, "FlashInfer decode CUDA graph prepare kernel failed: ", cudaGetErrorString(result));
}

void invokePrepareSparseMlaTargetVerifyParams(const int32_t* input_lengths,
                                              const int32_t* prefix_lengths,
                                              const int32_t* block_ids,
                                              int32_t*       batch_indice,
                                              int32_t*       page_indice,
                                              int32_t*       decode_page_indptr,
                                              int32_t*       paged_kv_last_page_len,
                                              int32_t*       qo_indptr,
                                              int32_t*       prefill_ragged_kv_len_indptr,
                                              int32_t*       kvlen,
                                              int32_t*       positions,
                                              int64_t*       slot_mapping,
                                              int32_t*       expanded_seq_lens,
                                              int32_t*       topk_indices_offset,
                                              int32_t*       ks,
                                              int32_t*       ke,
                                              int32_t        batch_size,
                                              int32_t        max_blocks_per_batch,
                                              int32_t        seq_size_per_block,
                                              int32_t        captured_batch_capacity,
                                              int32_t        captured_total_tokens,
                                              cudaStream_t   stream) {
    TORCH_CHECK(input_lengths != nullptr, "input_lengths is null");
    TORCH_CHECK(prefix_lengths != nullptr, "prefix_lengths is null");
    TORCH_CHECK(block_ids != nullptr, "block_ids is null");
    TORCH_CHECK(slot_mapping != nullptr, "slot_mapping is null");
    if (batch_size <= 0 || max_blocks_per_batch <= 0) {
        return;
    }
    const int32_t safe_page_sz = seq_size_per_block > 0 ? seq_size_per_block : 1;
    prepareTargetVerifyCumsumKernel<<<1, kV3CumsumWarpSize, 0, stream>>>(input_lengths,
                                                                         prefix_lengths,
                                                                         decode_page_indptr,
                                                                         paged_kv_last_page_len,
                                                                         qo_indptr,
                                                                         prefill_ragged_kv_len_indptr,
                                                                         kvlen,
                                                                         batch_size,
                                                                         safe_page_sz,
                                                                         max_blocks_per_batch,
                                                                         captured_batch_capacity);
    const int32_t write_blocks = std::max((batch_size + kV3WarpsPerBlock - 1) / kV3WarpsPerBlock, 1);
    prepareTargetVerifyWriteKernel<<<write_blocks, kV3BlockThreads, 0, stream>>>(input_lengths,
                                                                                 prefix_lengths,
                                                                                 block_ids,
                                                                                 decode_page_indptr,
                                                                                 qo_indptr,
                                                                                 prefill_ragged_kv_len_indptr,
                                                                                 batch_indice,
                                                                                 page_indice,
                                                                                 positions,
                                                                                 slot_mapping,
                                                                                 expanded_seq_lens,
                                                                                 topk_indices_offset,
                                                                                 ks,
                                                                                 ke,
                                                                                 batch_size,
                                                                                 max_blocks_per_batch,
                                                                                 safe_page_sz,
                                                                                 captured_total_tokens);
    const auto result = cudaGetLastError();
    TORCH_CHECK(result == cudaSuccess,
                "SparseMLA target verify CUDA graph prepare kernel failed: ",
                cudaGetErrorString(result));
}

void invokePrepareFlashInferDecodeParams(const int32_t*         sequence_lengths_plus_1,
                                         const int32_t*         block_ids,
                                         int32_t*               batch_indice,
                                         int32_t*               page_indice,
                                         int32_t*               decode_page_indptr,
                                         int32_t*               paged_kv_last_page_len,
                                         int32_t*               qo_indptr,
                                         int32_t*               kvlen,
                                         int32_t*               positions,
                                         int32_t                batch_size,
                                         int32_t                max_blocks_per_batch,
                                         int32_t                seq_size_per_block,
                                         CudaGraphPrepareStream stream) {
    invokePrepareFlashInferDecodeParams(sequence_lengths_plus_1,
                                        block_ids,
                                        batch_indice,
                                        page_indice,
                                        decode_page_indptr,
                                        paged_kv_last_page_len,
                                        qo_indptr,
                                        kvlen,
                                        positions,
                                        nullptr,
                                        batch_size,
                                        max_blocks_per_batch,
                                        seq_size_per_block,
                                        batch_size,
                                        stream);
}

}  // namespace rtp_llm

#endif
