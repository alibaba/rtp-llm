#pragma once

#include "rtp_llm/cpp/devices/cuda_impl/CudaDevice.h"

namespace rtp_llm {

struct XQAParams: public ParamsBase {
    KVBlockArray  kv_block_array;
    size_t        batch_size;
    size_t        max_seq_len;
    torch::Tensor kv_cache_offset;
    torch::Tensor sequence_lengths;
};

using XQAParamsPtr = std::shared_ptr<XQAParams>;

/**
 * @brief
 *
 * @param input_type bf16, fp16
 * @param output_type bf16, fp16, fp8e4m3
 * @param kv_cache_type bf16, fp16, fp8e4m3
 * @param group_size 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16
 * @param head_dim 64, 128, 256
 * @param page_size 16, 32, 64, 128
 * @return true
 * @return false
 */
bool supportXqa(DataType input_type,
                DataType output_type,
                DataType kv_cache_type,
                size_t   group_size,
                size_t   head_dim,
                size_t   page_size);

/**
 * @brief run xqa for decoding attention
 *
 * @param input q [batch_size, beam_width, head_num, head_dim]
 *              spec q [batch_size, beam_width, max_q_len, head_num, head_dim]
 * @param is_input_bf16
 * @param output [batch_size, beam_width, head_num, head_dim]
 * @param head_num
 * @param kv_head_num
 * @param head_dim
 * @param batch_size
 * @param max_blocks_per_seq
 * @param max_seq_len max kv seq len
 * @param page_size
 * @param kv_cache_pool head ptr [block_nums, kv_head_num, page_size, head_dim]
 * @param kv_cache_page_list [batch_size, beam_width, 2, max_pages_per_seq]
 * @param is_kv_cache_fp8
 * @param sequence_lengths kv seq len [batch_size]
 * @param device
 * @param rcp_out_scale for fp8 output
 * @param max_q_len max q seqlen
 * @param q_cu_seqlens accumulate q seqlen [batch_size + 1]
 * @param max_batch_size for semaphores and spec q mask
 * @param q_scale
 * @param beam_width
 */
void runXqa(void*       input,
            bool        is_input_bf16,
            void*       output,
            size_t      head_num,
            size_t      kv_head_num,
            size_t      head_dim,
            size_t      batch_size,
            size_t      max_blocks_per_seq,
            size_t      max_seq_len,
            size_t      page_size,
            void*       kv_cache_pool,
            int32_t*    kv_cache_page_list,
            bool        is_kv_cache_fp8,
            uint32_t*   sequence_lengths,
            CudaDevice* device,
            float*      rcp_out_scale  = nullptr,
            size_t      max_q_len      = 2,
            void*       q_cu_seqlens   = nullptr,
            size_t      max_batch_size = 4096,
            float       q_scale        = 1.f,
            uint32_t    beam_width     = 1);

}  // namespace rtp_llm
