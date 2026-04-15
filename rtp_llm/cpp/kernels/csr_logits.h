#pragma once

#include <stdint.h>
#if USING_CUDA
#include <cuda_runtime.h>
#endif
#if USING_ROCM
#include <hip/hip_runtime.h>
#endif

namespace rtp_llm {

// ---------------------------------------------------------------------------
// invokeCsrBuildMask
//
// 在 GPU 上直接查 CSR 前缀树，为每个 beam 构建 vocab mask。
//
// 参数：
//   indptr           : [num_states + 2]  行指针数组（已在 GPU 上）
//   packed_csr_tokens: [num_transitions + vocab_size]  转移表 token 列（已在 GPU 上）
//   current_states   : [batch_size]  每个 beam 当前所处的前缀树状态（已在 GPU 上）
//   start_mask       : [vocab_size]  根节点（state=0）时的合法 token 掩码（已在 GPU 上）
//   mask_out         : [batch_size, vocab_size]  输出：0=合法，1=需要屏蔽
//   batch_size       : beam 数量
//   vocab_size       : 词表大小
//   stream           : CUDA/HIP 流
//
// 线程映射：grid(ceil(vocab/BX), batch)，block(BX, 1)
//   每个线程负责一个 (batch, vocab) 位置。
// ---------------------------------------------------------------------------
template<typename StreamType>
void invokeCsrBuildMask(const int*   indptr,
                        const int*   packed_csr_tokens,
                        const int*   current_states,
                        const uint8_t* start_mask,
                        uint8_t*     mask_out,
                        int          batch_size,
                        int          vocab_size,
                        StreamType   stream);

// ---------------------------------------------------------------------------
// invokeCsrUpdateStates
//
// 在 GPU 上并行查 CSR 前缀树，更新每个 beam 的状态。
//
// 参数：
//   indptr           : [num_states + 2]  行指针数组（已在 GPU 上）
//   packed_csr_tokens: [num_transitions + vocab_size]  转移表 token 列（已在 GPU 上）
//   packed_csr_states: [num_transitions + vocab_size]  转移表 next_state 列（已在 GPU 上）
//   sampled_tokens   : [batch_size]  每个 beam 本步采样到的 token（已在 GPU 上）
//   current_states   : [batch_size]  输入当前状态；原地更新为 next_state
//   batch_size       : beam 数量
//   vocab_size       : 词表大小（用于判断 state=0 的快速路径）
//   stream           : CUDA/HIP 流
//
// 线程映射：grid(ceil(batch/BX))，block(BX)
//   每个线程负责一个 beam（CSR 行通常很短，并行化 beam 维度即可）。
// ---------------------------------------------------------------------------
template<typename StreamType>
void invokeCsrUpdateStates(const int* indptr,
                           const int* packed_csr_tokens,
                           const int* packed_csr_states,
                           const int* sampled_tokens,
                           int*       current_states,
                           int        batch_size,
                           int        vocab_size,
                           StreamType stream);

// ---------------------------------------------------------------------------
// invokeCsrGatherTokens
//
// 在 GPU 上从 new_tokens[batch, max_col] 中，按各 beam 的列偏移 col_offsets[batch]
// gather 出本步采样 token，写入 sampled_out[batch]。
// 完全替代原来的 D2H → CPU 提取 → H2D 路径。
//
// 参数：
//   new_tokens  : [batch_size, max_col]  完整 token 矩阵（已在 GPU 上）
//   col_offsets : [batch_size]           每个 beam 本步要读取的列索引（已在 GPU 上）
//   sampled_out : [batch_size]           输出：每个 beam 本步的采样 token
//   batch_size  : beam 数量
//   max_col     : new_tokens 的列数
//   stream      : CUDA/HIP 流
//
// 线程映射：grid(ceil(batch/BX))，block(BX)，每个线程负责一个 beam。
// ---------------------------------------------------------------------------
template<typename StreamType>
void invokeCsrGatherTokens(const int* new_tokens,
                           const int* col_offsets,
                           int*       sampled_out,
                           int        batch_size,
                           int        max_col,
                           StreamType stream);

}  // namespace rtp_llm
