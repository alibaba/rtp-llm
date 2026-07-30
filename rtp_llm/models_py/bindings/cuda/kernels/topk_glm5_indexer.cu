#include "topk_glm5_indexer.h"
#include "topk_glm5_indexer.cuh"

#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <torch/all.h>

#include <algorithm>
#include <cstdint>

namespace torch_ext {
namespace {

namespace impl = device::topk;
using impl::TopKProblem;
using Register1 = impl::TopKRegister<1>;
using Register2 = impl::TopKRegister<2>;
using Register4 = impl::TopKRegister<4>;
using Streaming = impl::TopKStreaming;
using Cluster2 = impl::TopKCluster<2>;
using Cluster4 = impl::TopKCluster<4>;
using Cluster8 = impl::TopKCluster<8>;

constexpr uint32_t kBlockSize = impl::TopKConfig::kBlockSize;
constexpr uint32_t kOccupancy = impl::TopKConfig::kOccupancy;
constexpr uint32_t kMaxTopK = impl::TopKConfig::kMaxTopK;
constexpr uint32_t kReg1MaxSeqLen = Register1::kMaxSeqLen;
constexpr uint32_t kReg2MaxSeqLen = Register2::kMaxSeqLen;
constexpr uint32_t kReg4MaxSeqLen = Register4::kMaxSeqLen;
constexpr uint32_t kCluster32KMaxSeqLen = 32768;
constexpr uint32_t kCluster64KMaxSeqLen = 65536;
constexpr uint32_t kCluster4MaxSeqLen = 524288;
constexpr uint32_t kCluster2MaxSeqLen = 262144;
constexpr uint32_t kPersistentClusters = 15 * kOccupancy;
constexpr uint32_t kCluster8SmallBatchLimit = 8;
constexpr uint32_t kCluster8LongBatchLimit = 16;
constexpr uint32_t kCluster4SmallBatchLimit = 16;
constexpr uint32_t kCluster4BatchLimit = 36;
constexpr uint32_t kCluster2BatchLimit = 72;
constexpr uint32_t kCluster2LongBatchLimit = 144;

bool uniform_plan_uses_cluster(uint32_t batch, uint32_t seq_len) {
    struct Candidate {
        uint32_t threshold;
        uint32_t capacity;
    };
    constexpr Candidate candidates[] = {
        {65536, 30},
        {98304, 48},
        {131072, 60},
        {196608, 80},
        {262144, 112},
        {393216, 128},
    };
    for (const auto candidate : candidates) {
        const uint32_t routed = seq_len > candidate.threshold ? batch : 0;
        if (routed <= candidate.capacity) {
            return seq_len > candidate.threshold;
        }
    }
    return false;
}

struct Params {
    const float* __restrict__ scores;
    const int32_t* __restrict__ lengths;
    int32_t* __restrict__ output;
    int64_t stride;
    uint32_t batch;
    uint32_t topk;

    __device__ __forceinline__ TopKProblem problem(uint32_t row) const {
        return TopKProblem{
            .in = scores + static_cast<int64_t>(row) * stride,
            .out = output + static_cast<int64_t>(row) * topk,
            .raw_out = nullptr,
            .page_table = nullptr,
            .topk = topk,
            .seq_len = static_cast<uint32_t>(lengths[row]),
            .page_bits = 0,
        };
    }
};

__device__ __forceinline__ void trivial(const TopKProblem& problem) {
    for (uint32_t i = threadIdx.x; i < problem.topk; i += blockDim.x) {
        problem.out[i] = i < problem.seq_len ? static_cast<int32_t>(i) : -1;
    }
}

__device__ __forceinline__ void copy_output(
    const TopKProblem& problem, int32_t* destination) {
    for (uint32_t i = threadIdx.x; i < problem.topk; i += blockDim.x) {
        destination[i] = problem.out[i];
    }
}

template <int Level>
__global__ __launch_bounds__(kBlockSize, kOccupancy)
void main_kernel(const __grid_constant__ Params params) {
    auto problem = params.problem(blockIdx.x);
    if (problem.seq_len <= problem.topk) {
        trivial(problem);
        return;
    }
    __shared__ impl::MaxSmem<
        Register1::Smem, Register2::Smem, Register4::Smem, Streaming::Smem> smem;
    if constexpr (Level == -1) {
        Register1::forward<false>(problem, &smem);
    } else if constexpr (Level == 0) {
        Register2::forward<false>(problem, &smem);
    } else if constexpr (Level == 1) {
        Register4::forward<false>(problem, &smem);
    } else {
        if (problem.seq_len <= kReg4MaxSeqLen) {
            Register4::forward<false>(problem, &smem);
        } else {
            Streaming::forward<false>(problem, &smem);
        }
    }
}

template <typename ClusterImpl>
__global__ __launch_bounds__(kBlockSize, kOccupancy)
void direct_cluster_kernel(const __grid_constant__ Params params) {
    auto problem = params.problem(blockIdx.x);
    constexpr uint32_t cluster_size = ClusterImpl::kClusterSize;
    const uint32_t worker_rank = blockIdx.x % cluster_size;
    if (problem.seq_len <= problem.topk) {
        if (blockIdx.y == worker_rank) trivial(problem);
        return;
    }
    __shared__ impl::MaxSmem<typename ClusterImpl::Smem> smem;
    __shared__ int32_t indices[kMaxTopK];
    auto* destination = problem.out;
    const auto cluster = cooperative_groups::this_cluster();
    problem.out = cluster.map_shared_rank(indices, worker_rank);
    ClusterImpl::template forward<false>(problem, &smem);
    cluster.sync();
    if (blockIdx.y == worker_rank) copy_output(problem, destination);
}

template <typename ClusterImpl>
__global__ __launch_bounds__(kBlockSize, kOccupancy)
void persistent_cluster_kernel(const __grid_constant__ Params params) {
    constexpr uint32_t cluster_size = ClusterImpl::kClusterSize;
    __shared__ impl::MaxSmem<typename ClusterImpl::Smem> smem;
    __shared__ int32_t indices[kMaxTopK];
    const auto cluster = cooperative_groups::this_cluster();
    const uint32_t worker_rank = blockIdx.x % cluster_size;
    for (uint32_t row = blockIdx.x; row < params.batch; row += kPersistentClusters) {
        auto problem = params.problem(row);
        auto* destination = problem.out;
        if (problem.seq_len <= problem.topk) {
            if (blockIdx.y == worker_rank) trivial(problem);
        } else {
            problem.out = cluster.map_shared_rank(indices, worker_rank);
            ClusterImpl::template forward<false>(problem, &smem);
            cluster.sync();
            if (blockIdx.y == worker_rank) copy_output(problem, destination);
        }
        cluster.sync();
    }
}

template <typename Kernel>
cudaError_t launch_cluster(Kernel kernel,
                           const Params& params,
                           uint32_t clusters,
                           uint32_t cluster_size,
                           cudaStream_t stream) {
    cudaLaunchAttribute attr{};
    attr.id = cudaLaunchAttributeClusterDimension;
    attr.val.clusterDim.x = 1;
    attr.val.clusterDim.y = cluster_size;
    attr.val.clusterDim.z = 1;
    cudaLaunchConfig_t config{};
    config.gridDim = dim3(clusters, cluster_size, 1);
    config.blockDim = dim3(kBlockSize, 1, 1);
    config.stream = stream;
    config.attrs = &attr;
    config.numAttrs = 1;
    return cudaLaunchKernelEx(&config, kernel, params);
}

}  // namespace

void topk_glm5_indexer(const torch::Tensor& scores,
                          const torch::Tensor& lengths,
                          torch::Tensor& output,
                          torch::Tensor& workspace,
                          int64_t k,
                          int64_t max_seq_len) {
    TORCH_CHECK(scores.is_cuda() && lengths.is_cuda() && output.is_cuda() &&
                    workspace.is_cuda(),
                "scores, lengths, output, and workspace must be CUDA tensors");
    TORCH_CHECK(scores.dtype() == torch::kFloat32 &&
                    lengths.dtype() == torch::kInt32 &&
                    output.dtype() == torch::kInt32,
                "expected float32 scores and int32 lengths/output");
    TORCH_CHECK(scores.dim() == 2 && scores.stride(1) == 1,
                "scores must be a dense-last-dimension 2D tensor");
    TORCH_CHECK(lengths.is_contiguous() && lengths.numel() == scores.size(0),
                "lengths shape mismatch");
    TORCH_CHECK(output.dim() == 2 && output.is_contiguous() &&
                    output.size(0) == scores.size(0) && output.size(1) == k,
                "output shape mismatch");
    TORCH_CHECK(k == 512 || k == 1024 || k == 2048,
                "topk_glm5_indexer supports K=512, 1024, or 2048");
    TORCH_CHECK(max_seq_len > 0 && max_seq_len <= scores.size(1),
                "invalid max_seq_len");
    TORCH_CHECK(workspace.dtype() == torch::kUInt8 &&
                    workspace.is_contiguous(),
                "workspace must be a contiguous CUDA uint8 tensor");

    const Params params{
        .scores = scores.data_ptr<float>(),
        .lengths = lengths.data_ptr<int32_t>(),
        .output = output.data_ptr<int32_t>(),
        .stride = scores.stride(0),
        .batch = static_cast<uint32_t>(scores.size(0)),
        .topk = static_cast<uint32_t>(k),
    };
    const uint32_t batch = params.batch;
    const auto stream = at::cuda::getCurrentCUDAStream();
    cudaError_t error = cudaSuccess;
    if (max_seq_len <= kReg1MaxSeqLen && (batch <= 32 || k <= 1024)) {
        main_kernel<-1><<<batch, kBlockSize, 0, stream>>>(params);
    } else if (max_seq_len <= kReg2MaxSeqLen) {
        main_kernel<0><<<batch, kBlockSize, 0, stream>>>(params);
    } else if (max_seq_len <= kReg4MaxSeqLen) {
        main_kernel<1><<<batch, kBlockSize, 0, stream>>>(params);
    } else if (max_seq_len <= kCluster32KMaxSeqLen &&
               batch <= kCluster8SmallBatchLimit) {
        error = launch_cluster(
            direct_cluster_kernel<Cluster8>,
            params,
            batch,
            Cluster8::kClusterSize,
            stream);
    } else if (max_seq_len <= kCluster32KMaxSeqLen &&
               batch <= kCluster4SmallBatchLimit) {
        error = launch_cluster(
            direct_cluster_kernel<Cluster4>,
            params,
            batch,
            Cluster4::kClusterSize,
            stream);
    } else if (max_seq_len <= kCluster32KMaxSeqLen) {
        main_kernel<2><<<batch, kBlockSize, 0, stream>>>(params);
    } else if (max_seq_len <= kCluster64KMaxSeqLen &&
               batch <= kCluster8SmallBatchLimit) {
        error = launch_cluster(
            direct_cluster_kernel<Cluster8>,
            params,
            batch,
            Cluster8::kClusterSize,
            stream);
    } else if (max_seq_len <= kCluster64KMaxSeqLen &&
               batch <= kCluster4BatchLimit) {
        error = launch_cluster(
            direct_cluster_kernel<Cluster4>,
            params,
            batch,
            Cluster4::kClusterSize,
            stream);
    } else if (max_seq_len <= kCluster64KMaxSeqLen &&
               batch <= kCluster2BatchLimit) {
        error = launch_cluster(
            direct_cluster_kernel<Cluster2>,
            params,
            batch,
            Cluster2::kClusterSize,
            stream);
    } else if (max_seq_len <= kCluster64KMaxSeqLen) {
        main_kernel<2><<<batch, kBlockSize, 0, stream>>>(params);
    } else if (batch <= kCluster8LongBatchLimit) {
        error = launch_cluster(
            direct_cluster_kernel<Cluster8>,
            params,
            batch,
            Cluster8::kClusterSize,
            stream);
    } else if (batch <= kCluster4BatchLimit &&
               max_seq_len <= kCluster4MaxSeqLen) {
        error = launch_cluster(
            direct_cluster_kernel<Cluster4>,
            params,
            batch,
            Cluster4::kClusterSize,
            stream);
    } else if (batch <= kCluster4BatchLimit) {
        error = launch_cluster(
            direct_cluster_kernel<Cluster8>,
            params,
            batch,
            Cluster8::kClusterSize,
            stream);
    } else if (batch <= kCluster2BatchLimit &&
               max_seq_len <= kCluster2MaxSeqLen) {
        error = launch_cluster(
            direct_cluster_kernel<Cluster2>,
            params,
            batch,
            Cluster2::kClusterSize,
            stream);
    } else if (batch <= kCluster2BatchLimit) {
        error = launch_cluster(
            direct_cluster_kernel<Cluster4>,
            params,
            batch,
            Cluster4::kClusterSize,
            stream);
    } else if (batch <= kCluster2LongBatchLimit &&
               max_seq_len > 131072) {
        error = launch_cluster(
            direct_cluster_kernel<Cluster2>,
            params,
            batch,
            Cluster2::kClusterSize,
            stream);
    } else if (!uniform_plan_uses_cluster(
                   batch, static_cast<uint32_t>(max_seq_len))) {
        main_kernel<2><<<batch, kBlockSize, 0, stream>>>(params);
    } else {
        error = launch_cluster(
            persistent_cluster_kernel<Cluster8>,
            params,
            kPersistentClusters,
            Cluster8::kClusterSize,
            stream);
    }
    if (error == cudaSuccess) error = cudaGetLastError();
    TORCH_CHECK(error == cudaSuccess,
                "GLM5 SGLang TopK launch failed: ", cudaGetErrorString(error));
}

}  // namespace torch_ext
