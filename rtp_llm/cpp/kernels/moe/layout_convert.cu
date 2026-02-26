/*
 * Efficient Contiguous → Masked conversion kernel
 * For Prefill stage: DeepEP Normal (contiguous) → DeepGemm Masked
 *
 * Avoids CPU-GPU sync, enables end-to-end GPU execution
 */

#include "rtp_llm/cpp/kernels/moe/layout_convert.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <stdio.h>

namespace rtp_llm {

// ============================================================================
// Kernel 1: Count tokens per expert
// ============================================================================
__global__ void count_tokens_per_expert_kernel(const int* grouped_layout,  // [total_tokens] expert ID for each token
                                               int*       token_counts,    // [num_experts] output: tokens per expert
                                               int        total_tokens,
                                               int        num_experts) {
    extern __shared__ int shared_counts[];

    // Initialize shared memory
    for (int i = threadIdx.x; i < num_experts; i += blockDim.x) {
        shared_counts[i] = 0;
    }
    __syncthreads();

    // Each thread processes some tokens
    for (int tid = blockIdx.x * blockDim.x + threadIdx.x; tid < total_tokens; tid += blockDim.x * gridDim.x) {
        int expert_id = grouped_layout[tid];
        atomicAdd(&shared_counts[expert_id], 1);
    }
    __syncthreads();

    // Write shared memory results back to global memory
    for (int i = threadIdx.x; i < num_experts; i += blockDim.x) {
        atomicAdd(&token_counts[i], shared_counts[i]);
    }
}

// ============================================================================
// Kernel 2: Compute write offsets for each expert
// ============================================================================
__global__ void
compute_write_offsets_kernel(const int* grouped_layout,  // [total_tokens]
                             int*       write_offsets,   // [total_tokens] output: write position for each token
                             int        total_tokens,
                             int        num_experts) {
    extern __shared__ int expert_counters[];

    // Initialize counters
    for (int i = threadIdx.x; i < num_experts; i += blockDim.x) {
        expert_counters[i] = 0;
    }
    __syncthreads();

    // Serial processing to maintain order
    // Only execute in first thread of first block
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        for (int tid = 0; tid < total_tokens; tid++) {
            int expert_id      = grouped_layout[tid];
            write_offsets[tid] = expert_counters[expert_id];
            expert_counters[expert_id]++;
        }
    }
}

// ============================================================================
// Kernel 3: Execute actual data conversion (FP16 version)
// ============================================================================
__global__ void contiguous_to_masked_fp16_kernel(const half* contiguous_data,  // [total_tokens, hidden_dim]
                                                 const int*  grouped_layout,   // [total_tokens]
                                                 const int*  write_offsets,    // [total_tokens]
                                                 half*       masked_data,      // [num_experts, max_tokens, hidden_dim]
                                                 int         total_tokens,
                                                 int         hidden_dim,
                                                 int         max_tokens) {
    int tid     = blockIdx.x;   // token index
    int dim_idx = threadIdx.x;  // dimension index

    if (tid >= total_tokens || dim_idx >= hidden_dim)
        return;

    // Read token information
    int expert_id       = grouped_layout[tid];
    int local_token_idx = write_offsets[tid];

    // Read from contiguous layout
    int  src_idx = tid * hidden_dim + dim_idx;
    half value   = contiguous_data[src_idx];

    // Write to masked layout
    int dst_idx          = expert_id * (max_tokens * hidden_dim) + local_token_idx * hidden_dim + dim_idx;
    masked_data[dst_idx] = value;
}

// ============================================================================
// Kernel 3: Execute actual data conversion (BF16 version)
// ============================================================================
__global__ void contiguous_to_masked_bf16_kernel(const __nv_bfloat16* contiguous_data,
                                                 const int*           grouped_layout,
                                                 const int*           write_offsets,
                                                 __nv_bfloat16*       masked_data,
                                                 int                  total_tokens,
                                                 int                  hidden_dim,
                                                 int                  max_tokens) {
    int tid     = blockIdx.x;
    int dim_idx = threadIdx.x;

    if (tid >= total_tokens || dim_idx >= hidden_dim)
        return;

    int expert_id       = grouped_layout[tid];
    int local_token_idx = write_offsets[tid];

    int           src_idx = tid * hidden_dim + dim_idx;
    __nv_bfloat16 value   = contiguous_data[src_idx];

    int dst_idx          = expert_id * (max_tokens * hidden_dim) + local_token_idx * hidden_dim + dim_idx;
    masked_data[dst_idx] = value;
}

// ============================================================================
// Kernel 3: Execute actual data conversion (FP8 version)
// ============================================================================
__global__ void contiguous_to_masked_fp8_kernel(const __nv_fp8_e4m3* contiguous_data,
                                                const int*           grouped_layout,
                                                const int*           write_offsets,
                                                __nv_fp8_e4m3*       masked_data,
                                                int                  total_tokens,
                                                int                  hidden_dim,
                                                int                  max_tokens) {
    int tid     = blockIdx.x;
    int dim_idx = threadIdx.x;

    if (tid >= total_tokens || dim_idx >= hidden_dim)
        return;

    int expert_id       = grouped_layout[tid];
    int local_token_idx = write_offsets[tid];

    int           src_idx = tid * hidden_dim + dim_idx;
    __nv_fp8_e4m3 value   = contiguous_data[src_idx];

    int dst_idx          = expert_id * (max_tokens * hidden_dim) + local_token_idx * hidden_dim + dim_idx;
    masked_data[dst_idx] = value;
}

// ============================================================================
// Kernel 4: Convert masked back to contiguous (FP16)
// ============================================================================
__global__ void masked_to_contiguous_fp16_kernel(const half* masked_data,      // [num_experts, max_tokens, hidden_dim]
                                                 const int*  grouped_layout,   // [total_tokens]
                                                 half*       contiguous_data,  // [total_tokens, hidden_dim]
                                                 int         total_tokens,
                                                 int         hidden_dim,
                                                 int         max_tokens,
                                                 int         num_experts) {
    int tid     = blockIdx.x;
    int dim_idx = threadIdx.x;

    if (tid >= total_tokens || dim_idx >= hidden_dim)
        return;

    int expert_id = grouped_layout[tid];

    // Count how many tokens before this one belong to the same expert
    int local_token_idx = 0;
    for (int i = 0; i < tid; i++) {
        if (grouped_layout[i] == expert_id) {
            local_token_idx++;
        }
    }

    // Read from masked layout
    int  src_idx = expert_id * (max_tokens * hidden_dim) + local_token_idx * hidden_dim + dim_idx;
    half value   = masked_data[src_idx];

    // Write to contiguous layout
    int dst_idx              = tid * hidden_dim + dim_idx;
    contiguous_data[dst_idx] = value;
}

// ============================================================================
// Kernel 4: Convert masked back to contiguous (BF16)
// ============================================================================
__global__ void masked_to_contiguous_bf16_kernel(const __nv_bfloat16* masked_data,
                                                 const int*           grouped_layout,
                                                 __nv_bfloat16*       contiguous_data,
                                                 int                  total_tokens,
                                                 int                  hidden_dim,
                                                 int                  max_tokens,
                                                 int                  num_experts) {
    int tid     = blockIdx.x;
    int dim_idx = threadIdx.x;

    if (tid >= total_tokens || dim_idx >= hidden_dim)
        return;

    int expert_id = grouped_layout[tid];

    int local_token_idx = 0;
    for (int i = 0; i < tid; i++) {
        if (grouped_layout[i] == expert_id) {
            local_token_idx++;
        }
    }

    int           src_idx = expert_id * (max_tokens * hidden_dim) + local_token_idx * hidden_dim + dim_idx;
    __nv_bfloat16 value   = masked_data[src_idx];

    int dst_idx              = tid * hidden_dim + dim_idx;
    contiguous_data[dst_idx] = value;
}

// ============================================================================
// Kernel 4: Convert masked back to contiguous (FP8)
// ============================================================================
__global__ void masked_to_contiguous_fp8_kernel(const __nv_fp8_e4m3* masked_data,
                                                const int*           grouped_layout,
                                                __nv_fp8_e4m3*       contiguous_data,
                                                int                  total_tokens,
                                                int                  hidden_dim,
                                                int                  max_tokens,
                                                int                  num_experts) {
    int tid     = blockIdx.x;
    int dim_idx = threadIdx.x;

    if (tid >= total_tokens || dim_idx >= hidden_dim)
        return;

    int expert_id = grouped_layout[tid];

    int local_token_idx = 0;
    for (int i = 0; i < tid; i++) {
        if (grouped_layout[i] == expert_id) {
            local_token_idx++;
        }
    }

    int           src_idx = expert_id * (max_tokens * hidden_dim) + local_token_idx * hidden_dim + dim_idx;
    __nv_fp8_e4m3 value   = masked_data[src_idx];

    int dst_idx              = tid * hidden_dim + dim_idx;
    contiguous_data[dst_idx] = value;
}

// ============================================================================
// Host functions
// ============================================================================

// FP16 three-step version (more efficient for large data)
void contiguous_to_masked_fp16(const half*  contiguous_data,
                               const int*   grouped_layout,
                               half*        masked_data,
                               int*         mask,
                               int          total_tokens,
                               int          hidden_dim,
                               int          max_tokens,
                               int          num_experts,
                               cudaStream_t stream) {
    // Allocate temporary buffer
    int* d_write_offsets;
    cudaMalloc(&d_write_offsets, total_tokens * sizeof(int));

    // Clear mask
    cudaMemsetAsync(mask, 0, num_experts * sizeof(int), stream);

    // Step 1: Count tokens per expert
    int block_size      = 256;
    int num_blocks      = (total_tokens + block_size - 1) / block_size;
    int shared_mem_size = num_experts * sizeof(int);

    count_tokens_per_expert_kernel<<<num_blocks, block_size, shared_mem_size, stream>>>(
        grouped_layout, mask, total_tokens, num_experts);

    // Step 2: Compute write offsets (serial to maintain order)
    compute_write_offsets_kernel<<<1, 256, shared_mem_size, stream>>>(
        grouped_layout, d_write_offsets, total_tokens, num_experts);

    // Step 3: Execute data conversion
    int threads_per_block = min(hidden_dim, 1024);

    contiguous_to_masked_fp16_kernel<<<total_tokens, threads_per_block, 0, stream>>>(
        contiguous_data, grouped_layout, d_write_offsets, masked_data, total_tokens, hidden_dim, max_tokens);

    cudaFree(d_write_offsets);
}

// BF16 three-step version
void contiguous_to_masked_bf16(const __nv_bfloat16* contiguous_data,
                               const int*           grouped_layout,
                               __nv_bfloat16*       masked_data,
                               int*                 mask,
                               int                  total_tokens,
                               int                  hidden_dim,
                               int                  max_tokens,
                               int                  num_experts,
                               cudaStream_t         stream) {
    int* d_write_offsets;
    cudaMalloc(&d_write_offsets, total_tokens * sizeof(int));

    cudaMemsetAsync(mask, 0, num_experts * sizeof(int), stream);

    int block_size      = 256;
    int num_blocks      = (total_tokens + block_size - 1) / block_size;
    int shared_mem_size = num_experts * sizeof(int);

    count_tokens_per_expert_kernel<<<num_blocks, block_size, shared_mem_size, stream>>>(
        grouped_layout, mask, total_tokens, num_experts);

    compute_write_offsets_kernel<<<1, 256, shared_mem_size, stream>>>(
        grouped_layout, d_write_offsets, total_tokens, num_experts);

    int threads_per_block = min(hidden_dim, 1024);

    contiguous_to_masked_bf16_kernel<<<total_tokens, threads_per_block, 0, stream>>>(
        contiguous_data, grouped_layout, d_write_offsets, masked_data, total_tokens, hidden_dim, max_tokens);

    cudaFree(d_write_offsets);
}

// FP8 three-step version
void contiguous_to_masked_fp8(const __nv_fp8_e4m3* contiguous_data,
                              const int*           grouped_layout,
                              __nv_fp8_e4m3*       masked_data,
                              int*                 mask,
                              int                  total_tokens,
                              int                  hidden_dim,
                              int                  max_tokens,
                              int                  num_experts,
                              cudaStream_t         stream) {
    int* d_write_offsets;
    cudaMalloc(&d_write_offsets, total_tokens * sizeof(int));

    cudaMemsetAsync(mask, 0, num_experts * sizeof(int), stream);

    int block_size      = 256;
    int num_blocks      = (total_tokens + block_size - 1) / block_size;
    int shared_mem_size = num_experts * sizeof(int);

    count_tokens_per_expert_kernel<<<num_blocks, block_size, shared_mem_size, stream>>>(
        grouped_layout, mask, total_tokens, num_experts);

    compute_write_offsets_kernel<<<1, 256, shared_mem_size, stream>>>(
        grouped_layout, d_write_offsets, total_tokens, num_experts);

    int threads_per_block = min(hidden_dim, 1024);

    contiguous_to_masked_fp8_kernel<<<total_tokens, threads_per_block, 0, stream>>>(
        contiguous_data, grouped_layout, d_write_offsets, masked_data, total_tokens, hidden_dim, max_tokens);

    cudaFree(d_write_offsets);
}

// ============================================================================
// Torch interface
// ============================================================================

std::tuple<torch::Tensor, torch::Tensor> convert_contiguous_to_masked_torch(const torch::Tensor& contiguous_data,
                                                                            const torch::Tensor& grouped_layout,
                                                                            int                  num_experts,
                                                                            int max_tokens_per_expert) {
    TORCH_CHECK(contiguous_data.is_cuda(), "contiguous_data must be a CUDA tensor");
    TORCH_CHECK(grouped_layout.is_cuda(), "grouped_layout must be a CUDA tensor");
    TORCH_CHECK(grouped_layout.dtype() == torch::kInt32, "grouped_layout must be int32");
    TORCH_CHECK(contiguous_data.dim() == 2, "contiguous_data must be 2D");
    TORCH_CHECK(grouped_layout.dim() == 1, "grouped_layout must be 1D");

    int total_tokens = contiguous_data.size(0);
    int hidden_dim   = contiguous_data.size(1);

    TORCH_CHECK(grouped_layout.size(0) == total_tokens, "grouped_layout size must match total_tokens");

    auto device = contiguous_data.device();
    auto dtype  = contiguous_data.dtype();

    // Allocate output tensors
    auto masked_data = torch::empty({num_experts, max_tokens_per_expert, hidden_dim},
                                    torch::TensorOptions().dtype(dtype).device(device));
    auto mask        = torch::zeros({num_experts}, torch::TensorOptions().dtype(torch::kInt32).device(device));

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // Call appropriate kernel based on dtype
    if (dtype == torch::kFloat16) {
        contiguous_to_masked_fp16(reinterpret_cast<const half*>(contiguous_data.data_ptr()),
                                  grouped_layout.data_ptr<int>(),
                                  reinterpret_cast<half*>(masked_data.data_ptr()),
                                  mask.data_ptr<int>(),
                                  total_tokens,
                                  hidden_dim,
                                  max_tokens_per_expert,
                                  num_experts,
                                  stream);
    } else if (dtype == torch::kBFloat16) {
        contiguous_to_masked_bf16(reinterpret_cast<const __nv_bfloat16*>(contiguous_data.data_ptr()),
                                  grouped_layout.data_ptr<int>(),
                                  reinterpret_cast<__nv_bfloat16*>(masked_data.data_ptr()),
                                  mask.data_ptr<int>(),
                                  total_tokens,
                                  hidden_dim,
                                  max_tokens_per_expert,
                                  num_experts,
                                  stream);
    } else if (dtype == torch::kFloat8_e4m3fn) {
        contiguous_to_masked_fp8(reinterpret_cast<const __nv_fp8_e4m3*>(contiguous_data.data_ptr()),
                                 grouped_layout.data_ptr<int>(),
                                 reinterpret_cast<__nv_fp8_e4m3*>(masked_data.data_ptr()),
                                 mask.data_ptr<int>(),
                                 total_tokens,
                                 hidden_dim,
                                 max_tokens_per_expert,
                                 num_experts,
                                 stream);
    } else {
        TORCH_CHECK(false, "Unsupported dtype for contiguous_to_masked conversion");
    }

    return std::make_tuple(masked_data, mask);
}

torch::Tensor convert_masked_to_contiguous_torch(const torch::Tensor& masked_data,
                                                 const torch::Tensor& grouped_layout,
                                                 const torch::Tensor& mask) {
    TORCH_CHECK(masked_data.is_cuda(), "masked_data must be a CUDA tensor");
    TORCH_CHECK(grouped_layout.is_cuda(), "grouped_layout must be a CUDA tensor");
    TORCH_CHECK(mask.is_cuda(), "mask must be a CUDA tensor");
    TORCH_CHECK(grouped_layout.dtype() == torch::kInt32, "grouped_layout must be int32");
    TORCH_CHECK(mask.dtype() == torch::kInt32, "mask must be int32");
    TORCH_CHECK(masked_data.dim() == 3, "masked_data must be 3D");
    TORCH_CHECK(grouped_layout.dim() == 1, "grouped_layout must be 1D");

    int num_experts  = masked_data.size(0);
    int max_tokens   = masked_data.size(1);
    int hidden_dim   = masked_data.size(2);
    int total_tokens = grouped_layout.size(0);

    auto device = masked_data.device();
    auto dtype  = masked_data.dtype();

    // Allocate output
    auto contiguous_data = torch::empty({total_tokens, hidden_dim}, torch::TensorOptions().dtype(dtype).device(device));

    cudaStream_t stream            = at::cuda::getCurrentCUDAStream();
    int          threads_per_block = min(hidden_dim, 1024);

    // Call appropriate kernel based on dtype
    if (dtype == torch::kFloat16) {
        masked_to_contiguous_fp16_kernel<<<total_tokens, threads_per_block, 0, stream>>>(
            reinterpret_cast<const half*>(masked_data.data_ptr()),
            grouped_layout.data_ptr<int>(),
            reinterpret_cast<half*>(contiguous_data.data_ptr()),
            total_tokens,
            hidden_dim,
            max_tokens,
            num_experts);
    } else if (dtype == torch::kBFloat16) {
        masked_to_contiguous_bf16_kernel<<<total_tokens, threads_per_block, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(masked_data.data_ptr()),
            grouped_layout.data_ptr<int>(),
            reinterpret_cast<__nv_bfloat16*>(contiguous_data.data_ptr()),
            total_tokens,
            hidden_dim,
            max_tokens,
            num_experts);
    } else if (dtype == torch::kFloat8_e4m3fn) {
        masked_to_contiguous_fp8_kernel<<<total_tokens, threads_per_block, 0, stream>>>(
            reinterpret_cast<const __nv_fp8_e4m3*>(masked_data.data_ptr()),
            grouped_layout.data_ptr<int>(),
            reinterpret_cast<__nv_fp8_e4m3*>(contiguous_data.data_ptr()),
            total_tokens,
            hidden_dim,
            max_tokens,
            num_experts);
    } else {
        TORCH_CHECK(false, "Unsupported dtype for masked_to_contiguous conversion");
    }

    return contiguous_data;
}

}  // namespace rtp_llm
