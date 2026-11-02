#include "topk_v3.h"
#include "topk_v3.cuh"
#include "dsa_range_topk.cuh"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <torch/all.h>
#include <cub/block/block_reduce.cuh>
#include <c10/cuda/CUDAException.h>

#include <algorithm>
#include <cstdint>
#include <climits>
#include <limits>

namespace torch_ext {
namespace {

struct DsaFloatMinimum {
    __device__ float operator()(float a, float b) const { return fminf(a, b); }
};

// Float radix selection is reused unchanged. Only rows whose boundary score
// has more candidates than remaining slots need the global-position radix
// refinement below. Even an all-equal row remains exact without a history-sized
// key array, a stable argsort, or a model-level query loop.
__global__ void repair_dsa_boundary(const float* scores, const int32_t* positions,
                                    const int32_t* raw, float* output_scores,
                                    int32_t* output_positions, int64_t score_stride,
                                    int64_t position_stride, int cols, int k) {
    constexpr int B = 256;
    using FloatReduce = cub::BlockReduce<float, B>;
    using IntReduce = cub::BlockReduce<int, B>;
    __shared__ union { FloatReduce::TempStorage f; IntReduce::TempStorage i; } temp;
    __shared__ float pivot;
    __shared__ int valid_count, selected_ties, total_ties, emit_count, remaining;
    __shared__ uint32_t prefix, histogram[256];
    const int tx = threadIdx.x;
    const int64_t row = blockIdx.x;
    scores += row * score_stride;
    positions += row * position_stride;
    raw += row * k;
    output_scores += row * k;
    output_positions += row * k;
    float minimum = INFINITY;
    int valid = 0;
    for (int i = tx; i < k; i += B) {
        const int index = raw[i];
        const float value = index >= 0 && index < cols ? scores[index] : -INFINITY;
        const int position = index >= 0 && index < cols ? positions[index] : -1;
        const bool live = isfinite(value) && position >= 0;
        output_scores[i] = live ? value : -INFINITY;
        output_positions[i] = live ? position : -1;
        if (live) { minimum = fminf(minimum, value); ++valid; }
    }
    const float reduced_minimum = FloatReduce(temp.f).Reduce(minimum, DsaFloatMinimum{});
    if (tx == 0) pivot = reduced_minimum;
    __syncthreads();
    const int reduced_valid = IntReduce(temp.i).Sum(valid);
    if (tx == 0) valid_count = reduced_valid;
    __syncthreads();
    if (valid_count < k) return;
    int selected_equal = 0;
    for (int i = tx; i < k; i += B) selected_equal += output_scores[i] == pivot;
    const int reduced_selected = IntReduce(temp.i).Sum(selected_equal);
    if (tx == 0) selected_ties = reduced_selected;
    __syncthreads();
    int equal = 0;
    for (int i = tx; i < cols; i += B) equal += scores[i] == pivot && positions[i] >= 0;
    const int reduced_equal = IntReduce(temp.i).Sum(equal);
    if (tx == 0) total_ties = reduced_equal;
    __syncthreads();
    if (total_ties == selected_ties) return;
    if (tx == 0) { prefix = 0; remaining = selected_ties; }
    __syncthreads();
    // Select the smallest remaining global positions amongst equal scores.
    for (int round = 0; round < 4; ++round) {
        histogram[tx] = 0;
        __syncthreads();
        const uint32_t mask = round == 0 ? 0u : (0xffffffffu << (32 - round * 8));
        const int shift = 24 - round * 8;
        for (int i = tx; i < cols; i += B) {
            const int position = positions[i];
            const uint32_t p = static_cast<uint32_t>(position);
            if (scores[i] == pivot && position >= 0 && (p & mask) == prefix)
                atomicAdd(&histogram[(p >> shift) & 255u], 1u);
        }
        __syncthreads();
        if (tx == 0) {
            int before = 0;
            for (uint32_t bin = 0; bin < 256; ++bin) {
                if (before + histogram[bin] >= remaining) {
                    prefix |= bin << shift;
                    remaining -= before;
                    break;
                }
                before += histogram[bin];
            }
        }
        __syncthreads();
    }
    if (tx == 0) emit_count = 0;
    __syncthreads();
    // Read raw indices, not the output being compacted in place.
    for (int i = tx; i < k; i += B) {
        const int index = raw[i];
        if (index >= 0 && index < cols && scores[index] > pivot) {
            const int slot = atomicAdd(&emit_count, 1);
            output_scores[slot] = scores[index];
            output_positions[slot] = positions[index];
        }
    }
    __syncthreads();
    for (int i = tx; i < cols; i += B) {
        const int position = positions[i];
        if (scores[i] == pivot && position >= 0 && static_cast<uint32_t>(position) <= prefix) {
            const int slot = atomicAdd(&emit_count, 1);
            if (slot < k) { output_scores[slot] = pivot; output_positions[slot] = position; }
        }
    }
}

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
    uint32_t max_seq_len;

    __device__ __forceinline__ TopKProblem problem(uint32_t row) const {
        const int32_t requested_seq_len = lengths[row];
        const uint32_t seq_len = requested_seq_len <= 0
                                     ? 0u
                                     : min(static_cast<uint32_t>(requested_seq_len),
                                           max_seq_len);
        return TopKProblem{
            .in = scores + static_cast<int64_t>(row) * stride,
            .out = output + static_cast<int64_t>(row) * topk,
            .raw_out = nullptr,
            .page_table = nullptr,
            .topk = topk,
            .seq_len = seq_len,
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

template <int Level, bool kAlignedInput>
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
        Register1::forward<false, kAlignedInput>(problem, &smem);
    } else if constexpr (Level == 0) {
        Register2::forward<false, kAlignedInput>(problem, &smem);
    } else if constexpr (Level == 1) {
        Register4::forward<false, kAlignedInput>(problem, &smem);
    } else {
        if (problem.seq_len <= kReg4MaxSeqLen) {
            Register4::forward<false, kAlignedInput>(problem, &smem);
        } else {
            Streaming::forward<false, kAlignedInput>(problem, &smem);
        }
    }
}

template <typename ClusterImpl, bool kAlignedInput>
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
    ClusterImpl::template forward<false, kAlignedInput>(problem, &smem);
    cluster.sync();
    if (blockIdx.y == worker_rank) copy_output(problem, destination);
}

template <typename ClusterImpl, bool kAlignedInput>
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
            ClusterImpl::template forward<false, kAlignedInput>(problem, &smem);
            cluster.sync();
            if (blockIdx.y == worker_rank) copy_output(problem, destination);
        }
        cluster.sync();
    }
}

template <typename Kernel, typename KernelParams>
cudaError_t launch_cluster(Kernel kernel,
                           const KernelParams& params,
                           uint32_t clusters,
                           uint32_t cluster_size,
                           cudaStream_t stream) {
    cudaLaunchAttribute attrs[1]{};
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim.x = 1;
    attrs[0].val.clusterDim.y = cluster_size;
    attrs[0].val.clusterDim.z = 1;
    cudaLaunchConfig_t config{};
    config.gridDim = dim3(clusters, cluster_size, 1);
    config.blockDim = dim3(kBlockSize, 1, 1);
    config.stream = stream;
    config.attrs = attrs;
    config.numAttrs = 1;
    return cudaLaunchKernelEx(&config, kernel, params);
}

template <bool kAlignedInput>
cudaError_t launch_topk(const Params& params,
                        int64_t k,
                        int64_t max_seq_len,
                        cudaStream_t stream) {
    const uint32_t batch = params.batch;
    cudaError_t error = cudaSuccess;
    if (max_seq_len <= kReg1MaxSeqLen && (batch <= 32 || k <= 1024)) {
        main_kernel<-1, kAlignedInput>
            <<<batch, kBlockSize, 0, stream>>>(params);
    } else if (max_seq_len <= kReg2MaxSeqLen) {
        main_kernel<0, kAlignedInput>
            <<<batch, kBlockSize, 0, stream>>>(params);
    } else if (max_seq_len <= kReg4MaxSeqLen) {
        main_kernel<1, kAlignedInput>
            <<<batch, kBlockSize, 0, stream>>>(params);
    } else if (max_seq_len <= kCluster32KMaxSeqLen &&
               batch <= kCluster8SmallBatchLimit) {
        error = launch_cluster(
            direct_cluster_kernel<Cluster8, kAlignedInput>,
            params,
            batch,
            Cluster8::kClusterSize,
            stream);
    } else if (max_seq_len <= kCluster32KMaxSeqLen &&
               batch <= kCluster4SmallBatchLimit) {
        error = launch_cluster(
            direct_cluster_kernel<Cluster4, kAlignedInput>,
            params,
            batch,
            Cluster4::kClusterSize,
            stream);
    } else if (max_seq_len <= kCluster32KMaxSeqLen) {
        main_kernel<2, kAlignedInput>
            <<<batch, kBlockSize, 0, stream>>>(params);
    } else if (max_seq_len <= kCluster64KMaxSeqLen &&
               batch <= kCluster8SmallBatchLimit) {
        error = launch_cluster(
            direct_cluster_kernel<Cluster8, kAlignedInput>,
            params,
            batch,
            Cluster8::kClusterSize,
            stream);
    } else if (max_seq_len <= kCluster64KMaxSeqLen &&
               batch <= kCluster4BatchLimit) {
        error = launch_cluster(
            direct_cluster_kernel<Cluster4, kAlignedInput>,
            params,
            batch,
            Cluster4::kClusterSize,
            stream);
    } else if (max_seq_len <= kCluster64KMaxSeqLen &&
               batch <= kCluster2BatchLimit) {
        error = launch_cluster(
            direct_cluster_kernel<Cluster2, kAlignedInput>,
            params,
            batch,
            Cluster2::kClusterSize,
            stream);
    } else if (max_seq_len <= kCluster64KMaxSeqLen) {
        main_kernel<2, kAlignedInput>
            <<<batch, kBlockSize, 0, stream>>>(params);
    } else if (batch <= kCluster8LongBatchLimit) {
        error = launch_cluster(
            direct_cluster_kernel<Cluster8, kAlignedInput>,
            params,
            batch,
            Cluster8::kClusterSize,
            stream);
    } else if (batch <= kCluster4BatchLimit &&
               max_seq_len <= kCluster4MaxSeqLen) {
        error = launch_cluster(
            direct_cluster_kernel<Cluster4, kAlignedInput>,
            params,
            batch,
            Cluster4::kClusterSize,
            stream);
    } else if (batch <= kCluster4BatchLimit) {
        error = launch_cluster(
            direct_cluster_kernel<Cluster8, kAlignedInput>,
            params,
            batch,
            Cluster8::kClusterSize,
            stream);
    } else if (batch <= kCluster2BatchLimit &&
               max_seq_len <= kCluster2MaxSeqLen) {
        error = launch_cluster(
            direct_cluster_kernel<Cluster2, kAlignedInput>,
            params,
            batch,
            Cluster2::kClusterSize,
            stream);
    } else if (batch <= kCluster2BatchLimit) {
        error = launch_cluster(
            direct_cluster_kernel<Cluster4, kAlignedInput>,
            params,
            batch,
            Cluster4::kClusterSize,
            stream);
    } else if (batch <= kCluster2LongBatchLimit && max_seq_len > 131072) {
        error = launch_cluster(
            direct_cluster_kernel<Cluster2, kAlignedInput>,
            params,
            batch,
            Cluster2::kClusterSize,
            stream);
    } else if (!uniform_plan_uses_cluster(
                   batch, static_cast<uint32_t>(max_seq_len))) {
        main_kernel<2, kAlignedInput>
            <<<batch, kBlockSize, 0, stream>>>(params);
    } else {
        error = launch_cluster(
            persistent_cluster_kernel<Cluster8, kAlignedInput>,
            params,
            kPersistentClusters,
            Cluster8::kClusterSize,
            stream);
    }
    return error == cudaSuccess ? cudaGetLastError() : error;
}

}  // namespace

void topk_v3(const torch::Tensor& scores,
                          const torch::Tensor& lengths,
                          torch::Tensor& output,
                          torch::Tensor& workspace,
                          int64_t k,
                          int64_t max_seq_len) {
    TORCH_CHECK(scores.is_cuda() && lengths.is_cuda() && output.is_cuda() &&
                    workspace.is_cuda(),
                "scores, lengths, output, and workspace must be CUDA tensors");
    TORCH_CHECK(lengths.device() == scores.device() &&
                    output.device() == scores.device() &&
                    workspace.device() == scores.device(),
                "scores, lengths, output, and workspace must be on the same CUDA device");
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
                "topk_v3 supports K=512, 1024, or 2048");
    TORCH_CHECK(max_seq_len > 0 && max_seq_len <= scores.size(1),
                "invalid max_seq_len");
    TORCH_CHECK(max_seq_len <= std::numeric_limits<int32_t>::max(),
                "max_seq_len exceeds int32 output-index range");
    TORCH_CHECK(scores.size(0) <= std::numeric_limits<uint32_t>::max(),
                "too many score rows for the CUDA dispatcher");
    TORCH_CHECK(workspace.dtype() == torch::kUInt8 &&
                    workspace.is_contiguous(),
                "workspace must be a contiguous CUDA uint8 tensor");

    if (scores.size(0) == 0) {
        return;
    }

    const c10::cuda::CUDAGuard device_guard(scores.device());

    const Params params{
        .scores = scores.data_ptr<float>(),
        .lengths = lengths.data_ptr<int32_t>(),
        .output = output.data_ptr<int32_t>(),
        .stride = scores.stride(0),
        .batch = static_cast<uint32_t>(scores.size(0)),
        .topk = static_cast<uint32_t>(k),
        .max_seq_len = static_cast<uint32_t>(max_seq_len),
    };
    const auto stream = at::cuda::getCurrentCUDAStream(scores.get_device());
    constexpr uintptr_t kVectorAlignment = 16;
    const bool aligned_input =
        (reinterpret_cast<uintptr_t>(params.scores) % kVectorAlignment == 0) &&
        ((params.stride * sizeof(float)) % kVectorAlignment == 0);
    const cudaError_t error = aligned_input
                                  ? launch_topk<true>(
                                        params, k, max_seq_len, stream)
                                  : launch_topk<false>(
                                        params, k, max_seq_len, stream);
    TORCH_CHECK(error == cudaSuccess,
                "DeepSeek V4 TopK launch failed: ", cudaGetErrorString(error));
}

void dsa_repair_topk_candidates(const torch::Tensor& scores,
                                const torch::Tensor& positions,
                                const torch::Tensor& raw_indices,
                                torch::Tensor& output_scores,
                                torch::Tensor& output_positions) {
    TORCH_CHECK(scores.is_cuda() && scores.dtype() == torch::kFloat32 && scores.dim() == 2 && scores.stride(1) == 1,
                "DSA candidate scores must be CUDA FP32 rows");
    TORCH_CHECK(raw_indices.dim() == 2, "Raw candidate indices must have two dimensions");
    const int64_t rows = scores.size(0), columns = scores.size(1), k = raw_indices.size(1);
    TORCH_CHECK(columns > 0 && columns <= INT_MAX && rows <= UINT_MAX && k > 0 && k <= 2048,
                "DSA candidate dimensions exceed the supported range");
    TORCH_CHECK(positions.dtype() == torch::kInt32 && positions.stride(-1) == 1 &&
                    ((positions.dim() == 1 && positions.size(0) == columns) ||
                     (positions.dim() == 2 && positions.sizes() == scores.sizes())), "Invalid global position layout");
    TORCH_CHECK(raw_indices.dim() == 2 && raw_indices.size(0) == rows && raw_indices.dtype() == torch::kInt32 && raw_indices.is_contiguous(),
                "Raw candidates must be contiguous INT32");
    TORCH_CHECK(output_scores.sizes() == raw_indices.sizes() && output_positions.sizes() == raw_indices.sizes() &&
                    output_scores.dtype() == torch::kFloat32 && output_positions.dtype() == torch::kInt32 &&
                    output_scores.is_contiguous() && output_positions.is_contiguous(), "Invalid bounded candidate output");
    TORCH_CHECK(positions.device() == scores.device() && raw_indices.device() == scores.device() &&
                    output_scores.device() == scores.device() && output_positions.device() == scores.device(),
                "DSA candidates must share a CUDA device");
    if (rows == 0) return;
    const c10::cuda::CUDAGuard guard(scores.device());
    repair_dsa_boundary<<<rows, 256, 0, at::cuda::getCurrentCUDAStream(scores.get_device())>>>(
        scores.data_ptr<float>(), positions.data_ptr<int32_t>(), raw_indices.data_ptr<int32_t>(),
        output_scores.data_ptr<float>(), output_positions.data_ptr<int32_t>(), scores.stride(0),
        positions.dim() == 1 ? 0 : positions.stride(0), columns, k);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

struct BoundedDsaTopKParams {
 const float* scores;const int32_t* positions;const int32_t* starts;const int32_t* ends;
 float* selected_scores;int32_t* selected_positions;
 int64_t score_stride,position_stride;uint32_t width,topk;
 __device__ device::k31_range_topk::TopKProblem problem(uint32_t row) const {
  const uint32_t start=min(static_cast<uint32_t>(max(starts ? starts[row] : 0,0)),width);
  const uint32_t end=max(start,min(static_cast<uint32_t>(max(ends ? ends[row] : static_cast<int32_t>(width),0)),width));
  return device::k31_range_topk::TopKProblem{scores+row*score_stride+start,positions+row*position_stride+start,
                           selected_scores+row*topk,selected_positions+row*topk,topk,end-start};
 }
};
template<class Implementation>
__global__ __launch_bounds__(device::topk::TopKConfig::kBlockSize,device::topk::TopKConfig::kOccupancy)
void select_bounded(const __grid_constant__ BoundedDsaTopKParams params) {
 device::enable_smem_spilling();
 const auto problem=params.problem(blockIdx.x);
 for(uint32_t i=threadIdx.x;i<problem.topk;i+=blockDim.x){problem.out[i]=-1;problem.output_scores[i]=-CUDART_INF_F;}
 __syncthreads();
 if(problem.seq_len<=problem.topk){
  for(uint32_t i=threadIdx.x;i<problem.seq_len;i+=blockDim.x)problem.emit(i,i);
  return;
 }
 __shared__ device::topk::MaxSmem<typename Implementation::Smem> smem;
 if((reinterpret_cast<uintptr_t>(problem.in)&15u)==0)Implementation::template forward<true>(problem,&smem);
 else Implementation::template forward<false>(problem,&smem);
}
void dsa_select_topk_candidates(torch::Tensor scores,torch::Tensor positions,
                 c10::optional<torch::Tensor> starts,c10::optional<torch::Tensor> ends,
                 torch::Tensor out_scores,torch::Tensor out_positions,int64_t topk){
 TORCH_CHECK(scores.is_cuda()&&scores.scalar_type()==torch::kFloat32&&scores.dim()==2&&scores.stride(1)==1,"FP32 CUDA score rows required");
 const int64_t rows=scores.size(0),width=scores.size(1);
 TORCH_CHECK(width<=INT32_MAX&&(topk==512||topk==1024||topk==2048),"Unsupported selector geometry");
 for(auto v:{positions,out_positions})TORCH_CHECK(v.is_cuda()&&v.device()==scores.device()&&v.scalar_type()==torch::kInt32,"INT32 metadata on the score device required");
 TORCH_CHECK((positions.dim()==1&&positions.size(0)==width)||(positions.dim()==2&&positions.sizes()==scores.sizes()),"Position shape mismatch");
 TORCH_CHECK(positions.stride(-1)==1,"Range/position layout mismatch");
 TORCH_CHECK(out_scores.is_cuda()&&out_scores.device()==scores.device()&&out_scores.scalar_type()==torch::kFloat32&&out_scores.is_contiguous()&&out_scores.size(0)==rows&&out_scores.size(1)==topk,"FP32 output shape mismatch");
 TORCH_CHECK(out_positions.is_contiguous()&&out_positions.size(0)==rows&&out_positions.size(1)==topk,"Index output shape mismatch");
 for(const auto& bound:{starts,ends}) if(bound.has_value())
  TORCH_CHECK(bound->is_cuda() && bound->device()==scores.device() &&
              bound->scalar_type()==torch::kInt32 && bound->dim()==1 &&
              bound->is_contiguous() && bound->numel()==rows,"INT32 range metadata required");
 TORCH_CHECK(rows<=UINT32_MAX,"Too many selector rows");
 if(!rows)return;
 TORCH_CHECK(out_scores.storage().data_ptr().get()!=scores.storage().data_ptr().get() &&
             out_positions.storage().data_ptr().get()!=positions.storage().data_ptr().get(),
             "Selector outputs must not alias input storage");
 const c10::cuda::CUDAGuard guard(scores.device());
 BoundedDsaTopKParams p{scores.data_ptr<float>(),positions.data_ptr<int32_t>(),starts ? starts->data_ptr<int32_t>() : nullptr,ends ? ends->data_ptr<int32_t>() : nullptr,out_scores.data_ptr<float>(),out_positions.data_ptr<int32_t>(),scores.stride(0),positions.dim()==2?positions.stride(0):0,static_cast<uint32_t>(width),static_cast<uint32_t>(topk)};
 const auto stream=at::cuda::getCurrentCUDAStream(scores.get_device());
 if(width<=device::k31_range_topk::TopKRegister<1>::kMaxSeqLen)select_bounded<device::k31_range_topk::TopKRegister<1>><<<rows,device::topk::TopKConfig::kBlockSize,0,stream>>>(p);
 else if(width<=device::k31_range_topk::TopKRegister<2>::kMaxSeqLen)select_bounded<device::k31_range_topk::TopKRegister<2>><<<rows,device::topk::TopKConfig::kBlockSize,0,stream>>>(p);
 else if(width<=device::k31_range_topk::TopKRegister<4>::kMaxSeqLen)select_bounded<device::k31_range_topk::TopKRegister<4>><<<rows,device::topk::TopKConfig::kBlockSize,0,stream>>>(p);
 else select_bounded<device::k31_range_topk::TopKStreaming><<<rows,device::topk::TopKConfig::kBlockSize,0,stream>>>(p);
 TORCH_CHECK(cudaGetLastError()==cudaSuccess,"Bounded selector launch failed");
}

}  // namespace torch_ext
