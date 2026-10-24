#include "rtp_llm/models_py/bindings/cuda/kernels/attention_input_metadata.h"

#include <c10/util/Exception.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cub/block/block_reduce.cuh>
#include <cub/block/block_scan.cuh>

namespace rtp_llm {

namespace {

constexpr int kMetadataThreads = 256;

struct LengthMax {
    __device__ int32_t operator()(int32_t a, int32_t b) const {
        return max(a, b);
    }
};

struct LengthPairSum {
    __device__ int2 operator()(const int2& a, const int2& b) const {
        return make_int2(a.x + b.x, a.y + b.y);
    }
};

__global__ void buildAttentionInputMetadataKernel(AttentionInputMetadataParams params) {
    if (params.batch_size == 0) {
        return;
    }
    const auto* input_lengths = params.input_lengths;
    const auto* prefix_lengths = params.prefix_lengths;
    auto* cu_seqlens = params.cu_seqlens;
    auto* cu_kv_seqlens = params.cu_kv_seqlens;
    auto* padding_offset = params.padding_offset;
    auto* sequence_lengths_plus_1 = params.sequence_lengths_plus_1;
    const auto batch_size = params.batch_size;
    const auto total_tokens = params.total_tokens;
    using Reduce = cub::BlockReduce<int32_t, kMetadataThreads>;
    using Scan   = cub::BlockScan<int2, kMetadataThreads>;
    __shared__ typename Reduce::TempStorage reduce_storage;
    __shared__ typename Scan::TempStorage   scan_storage;
    __shared__ int32_t                      max_input_len;
    __shared__ int2                         carry;

    int32_t local_max = 0;
    for (int32_t b = threadIdx.x; b < batch_size; b += blockDim.x) {
        local_max = max(local_max, input_lengths[b]);
    }
    const int32_t block_max = Reduce(reduce_storage).Reduce(local_max, LengthMax());
    if (threadIdx.x == 0) {
        max_input_len = block_max;
        carry         = make_int2(0, 0);
        cu_seqlens[0] = cu_kv_seqlens[0] = 0;
    }
    __syncthreads();

    for (int32_t base = 0; base < batch_size; base += blockDim.x) {
        const int32_t b      = base + threadIdx.x;
        const int32_t len    = b < batch_size ? input_lengths[b] : 0;
        const int32_t prefix = b < batch_size && prefix_lengths ? prefix_lengths[b] : 0;
        int2          inclusive;
        Scan(scan_storage).InclusiveScan(make_int2(len, len + prefix), inclusive, LengthPairSum());
        const int2 accumulated = LengthPairSum()(carry, inclusive);
        if (b < batch_size) {
            if (sequence_lengths_plus_1) {
                sequence_lengths_plus_1[b] = static_cast<int32_t>(static_cast<uint32_t>(prefix) + 1u);
            }
            cu_seqlens[b + 1]    = accumulated.x;
            cu_kv_seqlens[b + 1] = accumulated.y;
            const int32_t start  = accumulated.x - len;
            const int32_t offset = b * max_input_len - start;
            if (padding_offset) {
                for (int32_t j = 0; j < len && start + j < total_tokens; ++j) {
                    padding_offset[start + j] = offset;
                }
            }
        }
        // All threads consume the previous tile's carry before it is updated.
        __syncthreads();
        if (threadIdx.x == blockDim.x - 1) {
            carry = accumulated;
        }
        __syncthreads();
    }
}

}  // namespace

AttentionInputMetadataParams prepareAttentionInputMetadata(const at::Tensor& input_lengths,
                                                          const at::Tensor& prefix_lengths,
                                                          const at::Tensor& cu_seqlens,
                                                          const at::Tensor& cu_kv_seqlens,
                                                          const at::Tensor& padding_offset,
                                                          const at::Tensor& sequence_lengths_plus_1) {
    TORCH_CHECK(input_lengths.defined(), "input_lengths must be defined");
    TORCH_CHECK(input_lengths.is_cuda(), "input_lengths must be a CUDA tensor");
    TORCH_CHECK(input_lengths.scalar_type() == at::kInt, "input_lengths must be int32");
    TORCH_CHECK(input_lengths.is_contiguous(), "input_lengths must be contiguous");
    TORCH_CHECK(!prefix_lengths.defined() || prefix_lengths.numel() == 0 || prefix_lengths.is_cuda(),
                "prefix_lengths must be CUDA or empty");
    TORCH_CHECK(!prefix_lengths.defined() || prefix_lengths.numel() == 0 || prefix_lengths.scalar_type() == at::kInt,
                "prefix_lengths must be int32");
    TORCH_CHECK(cu_seqlens.is_cuda() && cu_seqlens.scalar_type() == at::kInt, "cu_seqlens must be CUDA int32");
    TORCH_CHECK(cu_kv_seqlens.is_cuda() && cu_kv_seqlens.scalar_type() == at::kInt, "cu_kv_seqlens must be CUDA int32");
    TORCH_CHECK(!padding_offset.defined() || padding_offset.is_cuda(), "padding_offset must be CUDA");

    const auto batch_size   = static_cast<int32_t>(input_lengths.size(0));
    const auto total_tokens = padding_offset.defined() ? static_cast<int32_t>(padding_offset.numel()) : 0;
    TORCH_CHECK(!sequence_lengths_plus_1.defined()
                    || (sequence_lengths_plus_1.device() == input_lengths.device()
                        && sequence_lengths_plus_1.scalar_type() == at::kInt
                        && sequence_lengths_plus_1.is_contiguous() && sequence_lengths_plus_1.numel() == batch_size),
                "sequence_lengths_plus_1 must be contiguous CUDA int32 [batch]");
    const int32_t* prefix_ptr = nullptr;
    if (prefix_lengths.defined() && prefix_lengths.numel() > 0) {
        TORCH_CHECK(prefix_lengths.is_contiguous(), "prefix_lengths must be contiguous");
        TORCH_CHECK(prefix_lengths.size(0) >= batch_size, "prefix_lengths size must cover input_lengths");
        prefix_ptr = prefix_lengths.data_ptr<int32_t>();
    }

    return {input_lengths.data_ptr<int32_t>(),
            prefix_ptr,
            cu_seqlens.data_ptr<int32_t>(),
            cu_kv_seqlens.data_ptr<int32_t>(),
            padding_offset.defined() && padding_offset.numel() > 0 ? padding_offset.data_ptr<int32_t>() : nullptr,
            sequence_lengths_plus_1.defined() ? sequence_lengths_plus_1.data_ptr<int32_t>() : nullptr,
            batch_size,
            total_tokens,
            input_lengths.get_device()};
}

cudaGraphNode_t addAttentionInputMetadataNode(cudaGraph_t graph, cudaGraphNode_t before) {
    AttentionInputMetadataParams metadata;
    void* arguments[] = {&metadata};
    cudaKernelNodeParams params{};
    params.func = reinterpret_cast<void*>(buildAttentionInputMetadataKernel);
    params.gridDim = dim3(1);
    params.blockDim = dim3(kMetadataThreads);
    params.kernelParams = arguments;
    cudaGraphNode_t node = nullptr;
    C10_CUDA_CHECK(cudaGraphAddKernelNode(&node, graph, nullptr, 0, &params));
#if CUDART_VERSION >= 13000
    C10_CUDA_CHECK(cudaGraphAddDependencies(graph, &node, &before, nullptr, 1));
#else
    C10_CUDA_CHECK(cudaGraphAddDependencies(graph, &node, &before, 1));
#endif
    return node;
}

void invokeBuildAttentionInputMetadata(AttentionInputMetadataParams params,
                                       cudaStream_t stream,
                                       const CudaGraphPrepareReplay& replay) {
    if (params.batch_size == 0) {
        return;
    }
    const c10::cuda::CUDAGuard guard(params.device);
    if (replay.exec && replay.node) {
        void* arguments[] = {&params};
        cudaKernelNodeParams node_params{};
        node_params.func = reinterpret_cast<void*>(buildAttentionInputMetadataKernel);
        node_params.gridDim = dim3(1);
        node_params.blockDim = dim3(kMetadataThreads);
        node_params.kernelParams = arguments;
        C10_CUDA_CHECK(cudaGraphExecKernelNodeSetParams(replay.exec, replay.node, &node_params));
        C10_CUDA_CHECK(cudaGraphNodeSetEnabled(replay.exec, replay.node, 1));
    } else {
        buildAttentionInputMetadataKernel<<<1, kMetadataThreads, 0, stream>>>(params);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
}

void invokeBuildAttentionInputMetadata(const at::Tensor& input_lengths,
                                       const at::Tensor& prefix_lengths,
                                       at::Tensor& cu_seqlens,
                                       at::Tensor& cu_kv_seqlens,
                                       at::Tensor& padding_offset,
                                       cudaStream_t stream,
                                       const at::Tensor& sequence_lengths_plus_1) {
    auto params = prepareAttentionInputMetadata(input_lengths, prefix_lengths, cu_seqlens, cu_kv_seqlens,
                                                padding_offset, sequence_lengths_plus_1);
    const c10::cuda::CUDAGuard guard(input_lengths.device());
    if (params.batch_size == 0) {
        if (cu_seqlens.numel()) cu_seqlens.zero_();
        if (cu_kv_seqlens.numel()) cu_kv_seqlens.zero_();
        if (padding_offset.defined() && padding_offset.numel()) padding_offset.zero_();
        return;
    }
    invokeBuildAttentionInputMetadata(params, stream);
}

}  // namespace rtp_llm
