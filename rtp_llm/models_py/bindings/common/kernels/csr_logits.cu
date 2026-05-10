#include "rtp_llm/models_py/bindings/common/kernels/csr_logits.h"

#if USING_CUDA
#include <cuda_runtime.h>
#include "rtp_llm/models_py/bindings/cuda/cuda_host_utils.h"
#endif

#if USING_ROCM
#include <hip/hip_runtime.h>
#include "rtp_llm/models_py/bindings/rocm/hip_host_utils.h"
#endif

namespace rtp_llm {

// =============================================================================
// Kernel 1：csr_build_mask_kernel
//
// 每个线程负责一个 (batch_idx, vocab_idx) 位置。
// 判断 vocab_idx 对当前 beam 的状态是否合法：
//   - state == 0（根节点）：查 start_mask[vocab_idx]
//   - state  > 0          ：在 CSR 的 [indptr[state], indptr[state+1]) 区间
//                           线性扫描，判断 packed_csr_tokens 中是否包含 vocab_idx
//
// mask_out[batch_idx * vocab_size + vocab_idx]：
//   0 = 合法（不屏蔽），1 = 非法（屏蔽）
// =============================================================================
__global__ void csr_build_mask_kernel(
    const int*     indptr,            // [num_states + 2]
    const int*     packed_csr_tokens, // [num_transitions + vocab_size]
    const int*     current_states,    // [batch_size]
    const uint8_t* start_mask,        // [vocab_size]，state=0 时使用
    uint8_t*       mask_out,          // [batch_size, vocab_size]，输出
    int            batch_size,
    int            vocab_size)
{
    // grid.y = batch_size，grid.x * blockDim.x >= vocab_size
    const int batch_idx = blockIdx.y;
    const int vocab_idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (batch_idx >= batch_size || vocab_idx >= vocab_size) {
        return;
    }

    const int state      = current_states[batch_idx];
    const int global_idx = batch_idx * vocab_size + vocab_idx;

    if (state == 0) {
        // 根节点：直接查 start_mask（0=合法，1=非法，与 mask_out 语义一致）
        mask_out[global_idx] = start_mask[vocab_idx] ? 0 : 1;
    } else {
        // 非根节点：在 CSR 行 [row_start, row_end) 中线性查找 vocab_idx
        const int row_start = indptr[state];
        const int row_end   = indptr[state + 1];

        uint8_t found = 0;
        for (int k = row_start; k < row_end; ++k) {
            if (packed_csr_tokens[k] == vocab_idx) {
                found = 1;
                break;
            }
        }
        // found=1 表示合法（不屏蔽），found=0 表示非法（屏蔽）
        mask_out[global_idx] = found ? 0 : 1;
    }
}

// =============================================================================
// Kernel 2：csr_update_states_kernel
//
// 每个线程负责一个 beam（batch_idx）。
// 根据本步采样到的 token，在 CSR 中查找 next_state，原地更新 current_states。
//
//   - state == 0（根节点）：next_state = sampled_token + 1（第0层节点的固定映射）
//   - state  > 0          ：在 CSR 的 [indptr[state], indptr[state+1]) 区间
//                           线性扫描，找到 packed_csr_tokens[k] == sampled_token 时
//                           返回 packed_csr_states[k]；未找到则重置为 0（根节点）
// =============================================================================
__global__ void csr_update_states_kernel(
    const int* indptr,            // [num_states + 2]
    const int* packed_csr_tokens, // [num_transitions + vocab_size]
    const int* packed_csr_states, // [num_transitions + vocab_size]
    const int* sampled_tokens,    // [batch_size]
    int*       current_states,    // [batch_size]，原地更新
    int        batch_size,
    int        vocab_size)
{
    const int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (batch_idx >= batch_size) {
        return;
    }

    const int state         = current_states[batch_idx];
    const int sampled_token = sampled_tokens[batch_idx];

    if (state == 0) {
        // 根节点的快速路径：state_id = token_id + 1
        current_states[batch_idx] = sampled_token + 1;
    } else {
        // 在 CSR 行中线性扫描，找到采样 token 对应的 next_state
        const int row_start = indptr[state];
        const int row_end   = indptr[state + 1];

        int next_state = 0;  // 默认重置为根节点（未匹配 / 终止）
        for (int k = row_start; k < row_end; ++k) {
            if (packed_csr_tokens[k] == sampled_token) {
                next_state = packed_csr_states[k];
                break;
            }
        }
        current_states[batch_idx] = next_state;
    }
}

// =============================================================================
// Host 入口：invokeCsrBuildMask（CUDA 版）
// =============================================================================
template<>
void invokeCsrBuildMask<cudaStream_t>(
    const int*     indptr,
    const int*     packed_csr_tokens,
    const int*     current_states,
    const uint8_t* start_mask,
    uint8_t*       mask_out,
    int            batch_size,
    int            vocab_size,
    cudaStream_t   stream)
{
    constexpr int BLOCK_X = 128;
    dim3 block(BLOCK_X, 1, 1);
    dim3 grid((vocab_size + BLOCK_X - 1) / BLOCK_X, batch_size, 1);

    csr_build_mask_kernel<<<grid, block, 0, stream>>>(
        indptr, packed_csr_tokens, current_states, start_mask,
        mask_out, batch_size, vocab_size);

#if USING_CUDA
    check_cuda_value(cudaPeekAtLastError());
#endif
    check_cuda_error();
}

// =============================================================================
// Host 入口：invokeCsrUpdateStates（CUDA 版）
// =============================================================================
template<>
void invokeCsrUpdateStates<cudaStream_t>(
    const int*   indptr,
    const int*   packed_csr_tokens,
    const int*   packed_csr_states,
    const int*   sampled_tokens,
    int*         current_states,
    int          batch_size,
    int          vocab_size,
    cudaStream_t stream)
{
    // 每个 beam 一个线程，batch_size 通常很小（<=32），单 block 足够
    constexpr int BLOCK_X = 128;
    dim3 block(BLOCK_X, 1, 1);
    dim3 grid((batch_size + BLOCK_X - 1) / BLOCK_X, 1, 1);

    csr_update_states_kernel<<<grid, block, 0, stream>>>(
        indptr, packed_csr_tokens, packed_csr_states,
        sampled_tokens, current_states, batch_size, vocab_size);

#if USING_CUDA
    check_cuda_value(cudaPeekAtLastError());
#endif
    check_cuda_error();
}

// =============================================================================
// Kernel 3：csr_gather_tokens_kernel
//
// 每个线程负责一个 beam（batch_idx），从 new_tokens 矩阵中按 col_offsets[batch_idx]
// 取出对应 token，写入 sampled_out[batch_idx]。
// 消除原来的 D2H 大矩阵拷贝路径。
// =============================================================================
__global__ void csr_gather_tokens_kernel(
    const int* new_tokens,   // [batch_size, max_col]，GPU 上
    const int* col_offsets,  // [batch_size]，每个 beam 本步要读取的列索引
    int*       sampled_out,  // [batch_size]，输出采样 token
    int        batch_size,
    int        max_col)
{
    const int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (batch_idx >= batch_size) {
        return;
    }
    const int col = col_offsets[batch_idx];
    // 越界保护：col 超出范围时写 -1，调用方可检测
    sampled_out[batch_idx] = (col >= 0 && col < max_col)
                                 ? new_tokens[batch_idx * max_col + col]
                                 : -1;
}

// =============================================================================
// Host 入口：invokeCsrGatherTokens（CUDA 版）
// =============================================================================
template<>
void invokeCsrGatherTokens<cudaStream_t>(
    const int*   new_tokens,
    const int*   col_offsets,
    int*         sampled_out,
    int          batch_size,
    int          max_col,
    cudaStream_t stream)
{
    constexpr int BLOCK_X = 128;
    dim3 block(BLOCK_X, 1, 1);
    dim3 grid((batch_size + BLOCK_X - 1) / BLOCK_X, 1, 1);

    csr_gather_tokens_kernel<<<grid, block, 0, stream>>>(
        new_tokens, col_offsets, sampled_out, batch_size, max_col);

#if USING_CUDA
    check_cuda_value(cudaPeekAtLastError());
#endif
    check_cuda_error();
}

}  // namespace rtp_llm
